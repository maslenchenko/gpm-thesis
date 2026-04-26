import threading
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm_factory(
    norm_type: str,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    norm_groups: int = 1,
) -> Callable[[int], nn.Module]:
    if norm_type == "batch":
        def _factory(num_features: int) -> nn.Module:
            return nn.BatchNorm1d(num_features=num_features, eps=norm_eps, momentum=norm_momentum)
    elif norm_type == "instance":
        def _factory(num_features: int) -> nn.Module:
            return nn.InstanceNorm1d(num_features=num_features, eps=norm_eps, momentum=norm_momentum)
    elif norm_type == "group":
        def _closest_denominator(value: int, denominator: int) -> int:
            if value % denominator == 0:
                return denominator
            best = 1
            for i in range(2, denominator):
                if value % i == 0:
                    best = i
            return best

        def _factory(num_features: int) -> nn.Module:
            return nn.GroupNorm(
                num_groups=_closest_denominator(num_features, norm_groups),
                num_channels=num_features,
                eps=norm_eps,
            )
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    return _factory


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_sizes: Union[int, List[int]],
        stride: int,
        dilation: int,
        maxpooling: int,
        norm_type: str,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        norm_groups: int = 1,
        dropout: float = 0.0,
        cuda_devices: Optional[Sequence[Optional[int]]] = None,
        cuda_output_device: Optional[int] = None,
        out_channels: Optional[int] = None,
        data_dropout: Optional[float] = None,
        data_parallel: bool = True,
        model_parallel: bool = False,
    ):
        super().__init__()

        if not data_parallel and not model_parallel and cuda_devices is not None:
            raise ValueError('Please turn on "data parallel" or "model parallel" to use multiple devices.')
        if data_parallel and model_parallel:
            raise ValueError(
                'Cannot use model parallelism and data parallelism at the same time. '
                'Please turn on either "data parallel" or "model parallel".'
            )

        if not isinstance(cuda_devices, list) and cuda_devices is not None:
            cuda_devices = [cuda_devices]

        if cuda_output_device is not None:
            self.cuda_output_device = cuda_output_device
        elif cuda_devices is not None:
            self.cuda_output_device = cuda_devices[0]
        else:
            self.cuda_output_device = None

        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.cuda_devices = cuda_devices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self.norm_groups = norm_groups
        self.norm_type = norm_type
        self.norm_factory = _get_norm_factory(
            norm_type=norm_type,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            norm_groups=norm_groups,
        )

        if self.cuda_devices is not None:
            self.num_devices = len(self.cuda_devices)

        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes
        self.num_conv_layers_per_block = len(self.kernel_sizes)
        self.data_dropout = data_dropout
        self.dropout = dropout

        if maxpooling != 0:
            self.maxpooling = nn.MaxPool1d(maxpooling)
            if data_parallel and self.cuda_devices is not None:
                self.maxpooling = nn.DataParallel(self.maxpooling, device_ids=self.cuda_devices)
            self.add_module("maxpooling", self.maxpooling)
            if self.cuda_devices is not None:
                self.maxpooling.cuda(self.cuda_devices[0])
        else:
            self.maxpooling = None

    def initialize(self):
        conv_list = []
        channels_used = 0

        if self.data_dropout is not None:
            self.data_dropout = nn.Dropout(p=self.data_dropout)
            if self.cuda_devices is not None:
                self.data_dropout = nn.DataParallel(self.data_dropout, device_ids=self.cuda_devices)
            self.add_module("data_dropout", self.data_dropout)
        else:
            self.data_dropout = None

        for idx, kernel_size in enumerate(self.kernel_sizes):
            if idx == self.num_conv_layers_per_block - 1:
                channels = self.out_channels - channels_used
            else:
                channels = int(self.out_channels / self.num_conv_layers_per_block)

            if self.model_parallel and self.cuda_devices is not None:
                channels_per_device_used = 0
                conv = []
                for device_id in self.cuda_devices:
                    if device_id == self.cuda_devices[-1]:
                        channels_per_device = channels - channels_per_device_used
                    else:
                        channels_per_device = int(channels / self.num_devices)

                    conv_per_device = nn.Conv1d(
                        in_channels=self.in_channels,
                        out_channels=channels_per_device,
                        kernel_size=kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                    )
                    bn_per_device = self.norm_factory(channels_per_device)

                    self.add_module(
                        f"conv_per_device_device_id_{device_id}_kernel_size_{kernel_size}",
                        conv_per_device,
                    )
                    self.add_module(
                        f"bn_per_device_device_id_{device_id}_kernel_size_{kernel_size}",
                        bn_per_device,
                    )

                    conv_per_device.cuda(device_id)
                    bn_per_device.cuda(device_id)
                    conv.append([conv_per_device, bn_per_device])
                    channels_per_device_used += channels_per_device
            else:
                conv = nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    dilation=self.dilation,
                )
                self.add_module(f"conv_kernel_size_{kernel_size}", conv)

                if self.data_parallel and self.cuda_devices is not None:
                    conv = nn.DataParallel(conv, device_ids=self.cuda_devices)
                if self.cuda_devices is not None:
                    conv = conv.cuda(self.cuda_devices[0])

            conv_list.append(conv)
            channels_used += channels

        bn = self.norm_factory(self.out_channels)
        self.add_module("bn", bn)
        if self.data_parallel and self.cuda_devices is not None:
            bn = nn.DataParallel(bn, device_ids=self.cuda_devices)
        if self.cuda_devices is not None:
            bn.cuda(self.cuda_devices[0])

        self.conv_list = conv_list
        self.bn = bn
        self.norm_factory = None

    @staticmethod
    def _concat_activations(activation_list: List[torch.Tensor]) -> torch.Tensor:
        activation_lengths = [item.shape[2] for item in activation_list]
        max_activation_length = max(activation_lengths)
        for i in range(len(activation_list)):
            activation_list[i] = F.pad(
                activation_list[i],
                (max_activation_length - activation_list[i].shape[2], 0, 0, 0, 0, 0),
            )
        return torch.cat(activation_list, 1)

    def forward(self, input_tensor: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        if self.cuda_devices is not None:
            input_tensor = input_tensor.cuda(self.cuda_devices[0])
        else:
            input_tensor = input_tensor.float()

        if self.data_dropout is not None:
            input_tensor = self.data_dropout(input_tensor)

        activation_list: List[torch.Tensor] = []
        for conv in self.conv_list:
            if self.model_parallel:
                activation_list_per_device: List[torch.Tensor] = []
                for idx, device_id in enumerate(self.cuda_devices):
                    input_on_device = input_tensor.cuda(device_id)
                    conv_for_device = conv[idx][0]
                    bn_for_device = conv[idx][1]
                    activation_filters_per_device = conv_for_device(input_on_device)
                    activation_filters_per_device = bn_for_device(activation_filters_per_device)
                    if not return_logits:
                        activation_filters_per_device = F.dropout(
                            F.relu(activation_filters_per_device),
                            training=self.training,
                            p=self.dropout,
                        )
                    if self.maxpooling is not None:
                        activation_filters_per_device = self.maxpooling(activation_filters_per_device)
                    activation_filters_per_device = activation_filters_per_device.cuda(self.cuda_output_device)
                    activation_list_per_device.append(activation_filters_per_device)
                activation_list.append(self._concat_activations(activation_list_per_device))
            else:
                activation_list.append(conv(input_tensor))

        activation = self._concat_activations(activation_list)
        if not self.model_parallel:
            activation = self.bn(activation)
            if not return_logits:
                activation = F.dropout(F.relu(activation), training=self.training, p=self.dropout)
            if self.maxpooling is not None:
                activation = self.maxpooling(activation)
        return activation


class InputInterfaceWithAttention(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels_initial: int,
        channels_increase_rate: float,
        strides: Union[int, List[int]],
        kernel_sizes: Union[int, List[int], List[List[int]]],
        maxpooling: Union[int, List[int]],
        dilation: Union[int, List[int]],
        norm_type: str,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        norm_groups: int = 1,
        cuda_devices: Optional[Sequence[Optional[int]]] = None,
        cuda_output_device: Optional[int] = None,
        data_dropout: Optional[float] = None,
        block_dropout: Optional[float] = None,
        num_channels_output: Optional[int] = None,
        input_layer: Optional[InceptionBlock] = None,
        rnn_embedding: bool = False,
        model_parallel: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_channels_initial = num_channels_initial
        self.channels_increase_rate = channels_increase_rate
        self.num_channels_output = num_channels_output
        if not isinstance(cuda_devices, list) and cuda_devices is not None:
            cuda_devices = [cuda_devices]
        self.cuda_devices = cuda_devices
        if self.cuda_devices is not None:
            self.num_devices = len(self.cuda_devices)

        self.cuda_output_device = cuda_output_device
        if not isinstance(strides, list):
            strides = [strides for _ in range(num_layers)]
        self.strides = strides

        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes]
        if not isinstance(kernel_sizes[0], list):
            kernel_sizes = [[kernel_sizes[0]] for _ in range(num_layers)]
        self.kernel_sizes = kernel_sizes

        if not isinstance(dilation, list):
            dilation = [dilation for _ in range(num_layers)]
        self.dilation = dilation

        self.input_layer = input_layer
        if not isinstance(maxpooling, list):
            maxpooling = [maxpooling for _ in range(num_layers)]
        self.maxpooling = maxpooling

        self.data_dropout = data_dropout
        self.dropout = dropout
        self.rnn_embedding = rnn_embedding
        self.model_parallel = model_parallel
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self.norm_groups = norm_groups
        self.norm_type = norm_type
        self.rnn_initialized = False
        self.block_dropout = block_dropout

    def initialize(self):
        if self.input_layer is None:
            if self.cuda_devices is not None:
                cuda_output_device = self.cuda_devices[0]
            else:
                cuda_output_device = None
            self.input_layer = InceptionBlock(
                in_channels=4,
                out_channels=self.num_channels_initial,
                kernel_sizes=self.kernel_sizes[0],
                stride=self.strides[0],
                dilation=self.dilation[0],
                maxpooling=self.maxpooling[0],
                cuda_devices=self.cuda_devices,
                cuda_output_device=cuda_output_device,
                data_dropout=self.data_dropout,
                dropout=self.dropout,
                data_parallel=True,
                model_parallel=False,
                norm_eps=self.norm_eps,
                norm_momentum=self.norm_momentum,
                norm_groups=self.norm_groups,
                norm_type=self.norm_type,
            )
        else:
            self.input_layer.out_channels = self.num_channels_initial
            self.input_layer.data_dropout = self.data_dropout
            self.input_layer.stride = self.strides[0]

        self.input_layer.initialize()
        self.add_module("input_layer", self.input_layer)

        self.inception_blocks = [self.input_layer]
        current_num_filters = self.num_channels_initial
        output_channels = int(self.num_channels_initial * self.channels_increase_rate)

        for idx in range(1, self.num_layers):
            if self.model_parallel and self.cuda_devices is not None:
                devices = [self.cuda_devices[(idx - 1) % self.num_devices]]
                output_device = self.cuda_devices[idx % self.num_devices]
            elif self.cuda_devices is not None:
                devices = self.cuda_devices
                output_device = self.cuda_devices[0]
            else:
                devices = self.cuda_devices
                output_device = self.cuda_output_device

            block = InceptionBlock(
                in_channels=current_num_filters,
                out_channels=output_channels,
                kernel_sizes=self.kernel_sizes[idx],
                stride=self.strides[idx],
                dilation=self.dilation[idx],
                maxpooling=self.maxpooling[idx],
                cuda_devices=devices,
                cuda_output_device=output_device,
                data_parallel=True,
                dropout=self.dropout,
                norm_eps=self.norm_eps,
                norm_momentum=self.norm_momentum,
                norm_groups=self.norm_groups,
                norm_type=self.norm_type,
            )
            block.initialize()
            self.add_module(f"inception_block_{idx}", block)
            self.inception_blocks.append(block)
            current_num_filters = output_channels
            output_channels = int(output_channels * self.channels_increase_rate)

        if self.num_channels_output is None:
            self.num_channels_output = output_channels

        if self.model_parallel and self.cuda_devices is not None:
            devices = [self.cuda_devices[self.num_layers % self.num_devices]]
        else:
            devices = self.cuda_devices

        self.output_block = InceptionBlock(
            in_channels=current_num_filters,
            out_channels=self.num_channels_output,
            kernel_sizes=1,
            stride=1,
            dilation=1,
            maxpooling=0,
            dropout=self.dropout,
            cuda_devices=devices,
            cuda_output_device=self.cuda_output_device,
            data_parallel=True,
            norm_eps=self.norm_eps,
            norm_momentum=self.norm_momentum,
            norm_groups=self.norm_groups,
            norm_type=self.norm_type,
        )
        self.output_block.initialize()
        self.add_module("inception_block_output", self.output_block)

        if self.model_parallel and self.cuda_devices is not None:
            devices = [self.cuda_devices[(self.num_layers + 1) % self.num_devices]]
        else:
            devices = self.cuda_devices

        self.context_linear = nn.Conv1d(
            in_channels=current_num_filters,
            out_channels=self.num_channels_output,
            kernel_size=1,
        )
        if devices is not None:
            self.context_linear = nn.DataParallel(
                self.context_linear,
                device_ids=devices,
                output_device=self.cuda_output_device,
            ).cuda(devices[0])
        self.add_module("context_linear", self.context_linear)

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cuda_devices is not None:
            activation = input_tensor.cuda(self.cuda_devices[0])
        else:
            activation = input_tensor

        for block in self.inception_blocks:
            activation = block(activation, return_logits=False)

        interface_activation = self.output_block(activation, return_logits=True)
        context = self.context_linear(activation)
        return interface_activation, context


class InputInterfaceSplit(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels_initial: int,
        channels_increase_rate: float,
        strides: Union[int, List[int]],
        kernel_sizes: Union[int, List[int], List[List[int]]],
        maxpooling: Union[int, List[int]],
        dilation: Union[int, List[int]],
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        norm_groups: int = 1,
        context_separate: bool = True,
        cuda_devices: Optional[List[Optional[int]]] = None,
        cuda_output_device: Optional[int] = None,
        data_dropout: Optional[float] = None,
        block_dropout: Optional[float] = None,
        num_channels_output: Optional[int] = None,
        rnn_embedding: bool = False,
        dropout: float = 0.0,
        concat: bool = False,
        average_interfaces: bool = False,
        norm_type: str = "batch",
    ):
        super().__init__()

        if cuda_devices is None:
            raise ValueError("InputInterfaceSplit requires cuda_devices list")

        self.concat = concat
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self.norm_groups = norm_groups
        self.norm_type = norm_type
        self.context_separate = context_separate
        self.cuda_output_device = cuda_output_device
        self.cuda_devices = cuda_devices
        self.num_devices = len(cuda_devices)
        self.average_interfaces = average_interfaces
        self.device_interfaces: List[InputInterfaceWithAttention] = []

        for cuda_device in cuda_devices:
            if cuda_device is not None:
                cuda_device = [cuda_device]
                model_parallel = False
            else:
                model_parallel = True

            device_interface = InputInterfaceWithAttention(
                num_layers=num_layers,
                num_channels_initial=num_channels_initial // self.num_devices,
                channels_increase_rate=channels_increase_rate,
                strides=strides,
                kernel_sizes=kernel_sizes,
                maxpooling=maxpooling,
                dilation=dilation,
                data_dropout=data_dropout,
                cuda_devices=cuda_device,
                cuda_output_device=cuda_output_device,
                rnn_embedding=rnn_embedding,
                model_parallel=model_parallel,
                dropout=dropout,
                norm_eps=norm_eps,
                norm_momentum=norm_momentum,
                block_dropout=block_dropout,
                num_channels_output=num_channels_output,
                norm_type=norm_type,
            )
            self.device_interfaces.append(device_interface)
            self.add_module(f"device_interface_{cuda_device}", device_interface)

    def initialize(self):
        for device_interface in self.device_interfaces:
            device_interface.initialize()

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cuda_devices[0] is not None:
            input_tensor = input_tensor.cuda(self.cuda_devices[0])

        activation_list: List[torch.Tensor] = []
        context_list: List[torch.Tensor] = []

        def _activate_interface(interface: InputInterfaceWithAttention, input_data: torch.Tensor):
            activation, context = interface(input_data)
            activation_list.append(activation)
            context_list.append(context)

        threads = []
        for device_interface in self.device_interfaces:
            thread = threading.Thread(target=_activate_interface, args=(device_interface, input_tensor))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if self.concat:
            raise NotImplementedError("concat=True path is not implemented in InputInterfaceSplit")

        if self.cuda_devices[0] is not None:
            activation = activation_list[0].cuda(self.cuda_output_device)
            context = context_list[0].cuda(self.cuda_output_device)
        else:
            activation = activation_list[0]
            context = context_list[0]

        if self.average_interfaces:
            activation = activation / self.num_devices
            context = context / self.num_devices

        for i in range(1, len(activation_list)):
            activation_output = activation_list[i]
            context_output = context_list[i]

            if self.cuda_output_device is not None:
                activation_output = activation_output.cuda(self.cuda_output_device)
                context_output = context_output.cuda(self.cuda_output_device)

            if self.average_interfaces:
                activation_output = activation_output / self.num_devices
                context_output = context_output / self.num_devices

            activation = torch.add(activation, activation_output)
            context = torch.add(context, context_output)

            if self.cuda_devices[i] is not None:
                activation_output = activation_output.cuda(self.cuda_devices[i])
                context_output = context_output.cuda(self.cuda_devices[i])

        if not self.context_separate:
            context = activation

        return activation, context
