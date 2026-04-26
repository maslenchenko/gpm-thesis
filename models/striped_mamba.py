import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - optional dependency
    Mamba = None


def _activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    if name == "none" or name is None or name == "linear":
        return lambda x: x
    raise ValueError(f"Unknown activation: {name}")


def _pad_same_1d(length: int, kernel_size: int, stride: int, dilation: int):
    out_len = math.ceil(length / stride)
    pad_needed = max(0, (out_len - 1) * stride + (kernel_size - 1) * dilation + 1 - length)
    pad_left = pad_needed // 2
    pad_right = pad_needed - pad_left
    return pad_left, pad_right


def _max_pool1d_same(x: torch.Tensor, kernel_size: int, stride: int):
    # x: (B, C, L)
    pad_left, pad_right = _pad_same_1d(x.shape[-1], kernel_size, stride, dilation=1)
    if pad_left or pad_right:
        x = F.pad(x, (pad_left, pad_right))
    return F.max_pool1d(x, kernel_size=kernel_size, stride=stride)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class NormLayer(nn.Module):
    def __init__(self, norm_type: Optional[str], num_features: int, norm_groups: int = 1, eps: float = 1e-5, momentum: float = 0.9):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(num_features, eps=eps)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(norm_groups, num_features, eps=eps)
        elif norm_type == "rms":
            self.norm = RMSNorm(num_features, eps=eps)
        elif norm_type == "none" or norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor):
        if self.norm is None:
            return x
        if self.norm_type in ("batch", "group"):
            # (B, L, C) -> (B, C, L)
            x = x.transpose(1, 2)
            x = self.norm(x)
            return x.transpose(1, 2)
        return self.norm(x)


class Conv1dSame(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias, padding=0)

    def forward(self, x: torch.Tensor):
        pad_left, pad_right = _pad_same_1d(x.shape[-1], self.kernel_size, self.stride, self.dilation)
        if pad_left or pad_right:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class ConvDNA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        activation: str = "relu",
        stride: int = 1,
        dropout: float = 0.0,
        pool_size: int = 1,
        norm_type: Optional[str] = None,
        bn_momentum: float = 0.99,
    ):
        super().__init__()
        self.conv = Conv1dSame(in_channels, out_channels, kernel_size, stride=stride, bias=True)
        self.norm = NormLayer(norm_type, out_channels, momentum=bn_momentum)
        self.activation = _activation(activation)
        self.dropout = dropout
        self.pool_size = pool_size

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pool_size > 1:
            x = x.transpose(1, 2)
            x = _max_pool1d_same(x, self.pool_size, self.pool_size)
            x = x.transpose(1, 2)
        return x


class ConvNac(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        activation: str = "relu",
        stride: int = 1,
        dilation_rate: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
        pool_size: int = 1,
        norm_type: Optional[str] = None,
        bn_momentum: float = 0.99,
    ):
        super().__init__()
        self.norm = NormLayer(norm_type, in_channels, momentum=bn_momentum)
        self.activation = _activation(activation)
        self.conv = Conv1dSame(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_rate, bias=True)
        self.dropout = dropout
        self.residual = residual
        self.pool_size = pool_size

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)
        current = self.norm(x)
        current = self.activation(current)
        current = current.transpose(1, 2)
        current = self.conv(current)
        current = current.transpose(1, 2)
        if self.dropout > 0:
            current = F.dropout(current, p=self.dropout, training=self.training)
        if self.residual:
            current = current + x
        if self.pool_size > 1:
            current = current.transpose(1, 2)
            current = _max_pool1d_same(current, self.pool_size, self.pool_size)
            current = current.transpose(1, 2)
        return current


def make_filter_schedule(features_init: int, features_end: int, repeat: int, divisible_by: int):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    mul = math.exp(math.log(features_end / features_init) / (repeat - 1))
    schedule = []
    features = features_init
    for _ in range(repeat):
        schedule.append(_round(features))
        features *= mul
    return schedule


class ResLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: Optional[str] = "none",
        bn_momentum: float = 0.99,
    ):
        super().__init__()
        self.nac = ConvNac(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            norm_type=norm_type,
            bn_momentum=bn_momentum,
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = self.nac(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ResTower(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features_init: int = 384,
        features_end: int = 768,
        repeat: int = 4,
        layers_to_return: int = 1,
        divisible_by: int = 16,
        activation: str = "relu",
        kernel_size: int = 1,
        pool_size: int = 2,
        dropout: float = 0.0,
        norm_type: Optional[str] = "none",
        bn_momentum: float = 0.99,
    ):
        super().__init__()
        self.layers_to_return = layers_to_return
        self.pool_size = pool_size
        schedule = make_filter_schedule(features_init, features_end, repeat, divisible_by)
        layers = []
        prev = in_channels
        for filters in schedule:
            layers.append(
                ResLayer(
                    in_channels=prev,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                    norm_type=norm_type,
                    bn_momentum=bn_momentum,
                )
            )
            prev = filters
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        results = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.layers_to_return > 1 and idx > len(self.layers) - self.layers_to_return - 1:
                results.append(x)
            if self.pool_size > 1:
                x = x.transpose(1, 2)
                x = _max_pool1d_same(x, self.pool_size, self.pool_size)
                x = x.transpose(1, 2)
        if self.layers_to_return > 1:
            return results + [x]
        return x


class DRKTrunk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        repeat: int = 4,
        layers_to_return: int = 1,
        preconv_features: int = 320,
        features_init: int = 384,
        features: int = 768,
        norm_type: str = "batch",
        bn_momentum: float = 0.9,
    ):
        super().__init__()
        self.conv = ConvDNA(
            in_channels=in_channels,
            out_channels=preconv_features,
            kernel_size=9,
            activation="linear",
            pool_size=2,
            norm_type=None,
            bn_momentum=bn_momentum,
        )
        self.res = ResTower(
            in_channels=preconv_features,
            features_init=features_init,
            features_end=features,
            repeat=repeat,
            layers_to_return=layers_to_return,
            activation="gelu",
            kernel_size=5,
            pool_size=2,
            norm_type=norm_type,
            bn_momentum=bn_momentum,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.res(x)
        return x


class InputInterfaceSplitTrunk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 5,
        num_channels_initial: int = 320,
        channels_increase_rate: float = 1.5,
        strides: int = 1,
        kernel_sizes: int = 5,
        maxpooling: int = 2,
        dilation: int = 1,
        norm_type: str = "batch",
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        norm_groups: int = 1,
        context_separate: bool = False,
        average_interfaces: bool = False,
        dropout: float = 0.0,
        data_dropout: Optional[float] = None,
        block_dropout: Optional[float] = None,
        rnn_embedding: bool = False,
        concat: bool = False,
        cuda_devices: Optional[list] = None,
        cuda_output_device: Optional[int] = None,
    ):
        super().__init__()

        # Import lazily so the dependency is local to gpm and loaded only when used.
        from .input_interface import InputInterfaceSplit

        if cuda_devices is None:
            raise ValueError(
                "InputInterfaceSplitTrunk requires CUDA devices. "
                "Pass input_interface_args with cuda_devices=[0] (or your GPU ids)."
            )

        if not isinstance(cuda_devices, list):
            cuda_devices = [cuda_devices]

        # Force output channels to match downstream features.
        self._out_channels = out_channels

        self.interface = InputInterfaceSplit(
            num_layers=num_layers,
            num_channels_initial=num_channels_initial,
            channels_increase_rate=channels_increase_rate,
            strides=strides,
            kernel_sizes=kernel_sizes,
            maxpooling=maxpooling,
            dilation=dilation,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            norm_groups=norm_groups,
            context_separate=context_separate,
            cuda_devices=cuda_devices,
            cuda_output_device=cuda_output_device,
            data_dropout=data_dropout,
            block_dropout=block_dropout,
            num_channels_output=out_channels,
            rnn_embedding=rnn_embedding,
            dropout=dropout,
            concat=concat,
            average_interfaces=average_interfaces,
            norm_type=norm_type,
        )
        self.interface.initialize()

    def forward(self, x: torch.Tensor):
        # The vendored dt interface expects (B, C, L); gpm uses (B, L, C).
        x = x.transpose(1, 2)
        activation, _context = self.interface(x)
        activation = activation.transpose(1, 2)
        return activation


class Final(nn.Module):
    def __init__(self, in_channels: int, units: int = 1, activation: str = "linear"):
        super().__init__()
        self.dense = nn.Linear(in_channels, units)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.dense(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "softplus":
            return F.softplus(x)
        if self.activation == "linear" or self.activation is None:
            return x
        raise ValueError(f"Unknown activation: {self.activation}")


class DRKHead(nn.Module):
    def __init__(self, in_channels: int, features: int, crop: int = 2048):
        super().__init__()
        self.crop = crop
        self.final = Final(in_channels, units=features, activation="softplus")

    def forward(self, x: torch.Tensor):
        if self.crop > 0:
            x = x[:, self.crop:-self.crop, :]
        x = F.gelu(x)
        x = self.final(x)
        return x


class IsolateClassifierHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        units: int = 1,
        crop: int = 0,
        pool: str = "mean",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.crop = int(crop)
        self.pool = pool
        self.dropout_rate = float(dropout_rate)
        self.final = Final(in_channels, units=units, activation="linear")

    def forward(self, x: torch.Tensor):
        if self.crop > 0:
            x = x[:, self.crop:-self.crop, :]
        x = F.gelu(x)
        if self.pool == "mean":
            x = torch.mean(x, dim=1)
        elif self.pool == "max":
            x = torch.max(x, dim=1).values
        elif self.pool == "first":
            x = x[:, 0, :]
        else:
            raise ValueError(f"Unknown pool mode: {self.pool}")
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.final(x)
        # Keep [B, T, F] contract expected by current training/eval utilities.
        return x.unsqueeze(1)


class UNet(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_type: str = "none",
        activation: str = "relu",
        bn_momentum: float = 0.99,
        upsample_conv: bool = False,
    ):
        super().__init__()
        self.norm_x = NormLayer(norm_type, channels, momentum=bn_momentum)
        self.norm_u = NormLayer(norm_type, channels, momentum=bn_momentum)
        self.activate = _activation(activation)
        self.upsample_conv = upsample_conv
        if upsample_conv:
            self.x_proj = nn.Linear(channels, channels)
        else:
            self.x_proj = None
        self.u_proj = nn.Linear(channels, channels)
        self.depthwise = Conv1dSame(channels, channels, kernel_size, groups=channels, bias=True)
        self.pointwise = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        # x: pooled (B, Lp, C), u: skip (B, L, C)
        stride = u.shape[1] // x.shape[1]
        if x.shape[1] * stride != u.shape[1]:
            raise ValueError("UNet: pooled length does not evenly divide skip length")

        x = self.norm_x(x)
        u = self.norm_u(u)

        x = self.activate(x)
        u = self.activate(u)

        if self.x_proj is not None:
            x = self.x_proj(x)
        u = self.u_proj(u)

        x = x.repeat_interleave(stride, dim=1)
        x = x + u

        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.pointwise(x)

        return x


class RoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor):
        # x: (B, L, H, D)
        b, l, h, d = x.shape
        pos = torch.arange(l, device=x.device).float()
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, D)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos + x_rot * sin


def enformer_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor):
    # query/key: (B, L, H, D), value: (B, L, H, Dv)
    b, l, h, d = query.shape
    pos_emb_dim = w.shape[-1]

    pos = torch.arange(l, device=query.device)
    distance = pos[None, :] - pos[:, None]  # (L, L)

    pow_rate = torch.exp(torch.log(torch.tensor((l + 1) / 2.0, device=query.device)) / (pos_emb_dim // 2))
    center_widths = pow_rate ** torch.arange(1, (pos_emb_dim // 2) + 1, device=query.device)

    distance = distance[:, :, None].repeat(1, 1, pos_emb_dim // 2)
    unsigned_basis = (distance.abs() <= center_widths)  # (L, L, N/2)
    signed_basis = distance.sign() * unsigned_basis
    basis = torch.cat([unsigned_basis, signed_basis], dim=-1).float()  # (L, L, N)

    # r: (B, H, L, L, D)
    r = torch.einsum("hdn,ijn->hijd", w, basis)  # (H, L, L, D)
    r = r.unsqueeze(0).expand(b, -1, -1, -1, -1)

    qr_term = torch.einsum("bihd,bhijd->bhij", query, r)
    uk_term = torch.einsum("hd,bjhd->bhj", u, key).unsqueeze(2)
    vr_term = torch.einsum("hd,bhijd->bhij", v, r)

    bias = (qr_term + uk_term + vr_term) / math.sqrt(d)

    attn_scores = torch.einsum("bihd,bjhd->bhij", query, key) / math.sqrt(d)
    attn_scores = attn_scores + bias
    attn = torch.softmax(attn_scores, dim=-1)

    out = torch.einsum("bhij,bjhd->bihd", attn, value)
    return out


class EnformerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        qk_features: int,
        v_features: int,
        out_features: int,
        dropout_rate: float = 0.0,
        positional_encoding: str = "enformer",
        pos_emb_dim: Optional[int] = None,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_features = qk_features
        self.v_features = v_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.positional_encoding = positional_encoding
        self.pos_emb_dim = pos_emb_dim
        self.use_flash_attention = use_flash_attention

        self.q_proj = nn.Linear(out_features, num_heads * qk_features, bias=False)
        self.k_proj = nn.Linear(out_features, num_heads * qk_features, bias=False)
        self.v_proj = nn.Linear(out_features, num_heads * v_features, bias=False)
        self.out_proj = nn.Linear(num_heads * v_features, out_features, bias=True)

        if positional_encoding == "enformer":
            pos_dim = pos_emb_dim if pos_emb_dim is not None else qk_features
            self.u = nn.Parameter(torch.empty(num_heads, qk_features))
            self.v = nn.Parameter(torch.empty(num_heads, qk_features))
            self.w = nn.Parameter(torch.empty(num_heads, qk_features, pos_dim))
            nn.init.kaiming_normal_(self.u)
            nn.init.kaiming_normal_(self.v)
            nn.init.kaiming_normal_(self.w)
        elif positional_encoding == "rope":
            self.rope = RoPE(qk_features, num_heads=num_heads)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor):
        b, l, _ = x.shape
        q = self.q_proj(x).view(b, l, self.num_heads, self.qk_features)
        k = self.k_proj(x).view(b, l, self.num_heads, self.qk_features)
        v = self.v_proj(x).view(b, l, self.num_heads, self.v_features)

        if self.positional_encoding == "rope":
            q = self.rope(q)
            k = self.rope(k)
            out = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.qk_features)
            attn = torch.softmax(out, dim=-1)
            out = torch.einsum("bhij,bjhd->bihd", attn, v)
        elif self.positional_encoding == "enformer":
            out = enformer_attention(q, k, v, self.u, self.v, self.w)
        else:
            scores = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.qk_features)
            attn = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhij,bjhd->bihd", attn, v)

        out = out.reshape(b, l, self.num_heads * self.v_features)
        out = self.out_proj(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        return out


class TransNao(nn.Module):
    def __init__(
        self,
        transformer_features: int,
        heads: int = 4,
        key_size: int = 64,
        pos_emb_dim: int = 32,
        dropout_rate: float = 0.2,
        dense_expansion: int = 2,
        norm_type: str = "none",
        activation: str = "none",
        positional_encoding: str = "none",
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = NormLayer(norm_type, transformer_features)
        self.norm2 = NormLayer(norm_type, transformer_features)
        self.activation = _activation(activation)
        self.dropout_rate = dropout_rate
        self.attn = EnformerMultiHeadAttention(
            num_heads=heads,
            qk_features=key_size,
            v_features=transformer_features // heads,
            out_features=transformer_features,
            dropout_rate=dropout_rate,
            positional_encoding=positional_encoding,
            pos_emb_dim=pos_emb_dim,
            use_flash_attention=use_flash_attention,
        )
        self.mlp = nn.Sequential(
            nn.Linear(transformer_features, dense_expansion * transformer_features),
            nn.Dropout(dropout_rate),
            nn.GELU() if activation == "gelu" else nn.ReLU() if activation == "relu" else nn.SiLU() if activation == "silu" else nn.Identity(),
            nn.Linear(dense_expansion * transformer_features, transformer_features),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.norm1(x)
        x = self.attn(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + skip
        return x


class BidirectionalMamba(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        expansion_factor: float = 1.0,
        activation: str = "silu",
        norm_type: str = "rms",
        tie_in_proj: bool = True,
        tie_gate: bool = False,
        concatenate_fwd_rev: bool = False,
        mamba_args: Optional[dict] = None,
        mlp_layer: bool = False,
        dense_expansion: int = 2,
        mlp_dropout_rate: float = 0.1,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is required for BidirectionalMamba. Install it or replace this block.")
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.expansion_factor = expansion_factor
        self.tie_in_proj = tie_in_proj
        self.tie_gate = tie_gate
        self.concatenate_fwd_rev = concatenate_fwd_rev
        self.activation = _activation(activation)
        self.norm = NormLayer(norm_type, input_features)
        self.mlp_layer = mlp_layer
        self.mlp_dropout_rate = mlp_dropout_rate

        ed = math.ceil(expansion_factor * input_features)
        n_in_proj = 1 if tie_in_proj else 2
        n_gate = 1 if tie_gate else 2
        self.in_proj = nn.Linear(input_features, (n_in_proj + n_gate) * ed)

        self.mamba_fwd = Mamba(d_model=ed, d_state=hidden_features, expand=1, **(mamba_args or {}))
        self.mamba_rev = Mamba(d_model=ed, d_state=hidden_features, expand=1, **(mamba_args or {}))

        out_dim = ed * (2 if concatenate_fwd_rev else 1)
        self.out_proj = nn.Linear(out_dim, input_features)

        if mlp_layer:
            self.mlp = nn.Sequential(
                nn.Linear(input_features, dense_expansion * input_features),
                nn.Dropout(mlp_dropout_rate),
                nn.SiLU() if activation == "silu" else nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Linear(dense_expansion * input_features, input_features),
                nn.Dropout(mlp_dropout_rate),
            )
        else:
            self.mlp = None

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.norm(x)

        ed = math.ceil(self.expansion_factor * self.input_features)
        n_in_proj = 1 if self.tie_in_proj else 2
        proj = self.in_proj(x)

        if self.tie_in_proj:
            xf = proj[..., :ed]
            xr = xf
            gate_start = ed
        else:
            xf = proj[..., :ed]
            xr = proj[..., ed:2 * ed]
            gate_start = 2 * ed

        if self.tie_gate:
            zf = proj[..., gate_start:gate_start + ed]
            zr = zf
        else:
            zf = proj[..., gate_start:gate_start + ed]
            zr = proj[..., gate_start + ed:gate_start + 2 * ed]

        xf = self.mamba_fwd(xf)
        xr = torch.flip(xr, dims=[1])
        xr = self.mamba_rev(xr)
        xr = torch.flip(xr, dims=[1])

        if self.concatenate_fwd_rev:
            x = torch.cat([xf * self.activation(zf), xr * self.activation(zr)], dim=-1)
        else:
            x = xf * self.activation(zf) + xr * self.activation(zr)

        x = self.out_proj(x)
        x = x + skip

        if self.mlp is not None:
            skip = x
            x = self.mlp(x)
            x = x + skip

        return x


class StripedMamba(nn.Module):
    def __init__(
        self,
        seq_depth: int = 4,
        features: int = 1,
        crop: int = 2048,
        dropout_rate: float = 0.2,
        norm_type: str = "layer",
        bn_momentum: float = 0.9,
        activation: str = "gelu",
        mamba_features: int = 768,
        mamba_expansion_factor: int = 1,
        ssm_hidden_features: int = 8,
        mamba_layers: int = 3,
        positional_encoding: str = "enformer",
        trans_pool_size: int = 4,
        key_size: int = 64,
        heads: int = 4,
        pos_emb_dim: int = 32,
        transformer_layers: int = 2,
        trunk_norm_type: str = "batch",
        unet_norm_type: str = "batch",
        final_norm_type: Optional[str] = None,
        mamba_args: Optional[dict] = None,
        use_input_interface: bool = False,
        input_interface_args: Optional[dict] = None,
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        self.features = features
        self.crop = crop
        self.mamba_features = mamba_features
        self.mamba_layers = mamba_layers
        self.transformer_layers = transformer_layers
        self.trans_pool_size = trans_pool_size
        self.final_norm_type = final_norm_type
        self.checkpoint_blocks = checkpoint_blocks

        if use_input_interface:
            resolved_input_interface_args = {} if input_interface_args is None else dict(input_interface_args)
            if "cuda_devices" not in resolved_input_interface_args:
                if not torch.cuda.is_available():
                    raise ValueError(
                        "use_input_interface=True requires CUDA. "
                        "Pass input_interface_args with cuda_devices when running on GPU."
                    )
                resolved_input_interface_args["cuda_devices"] = [0]
            if "cuda_output_device" not in resolved_input_interface_args:
                resolved_input_interface_args["cuda_output_device"] = resolved_input_interface_args["cuda_devices"][0]
            self.trunk = InputInterfaceSplitTrunk(
                in_channels=seq_depth,
                out_channels=mamba_features,
                **resolved_input_interface_args,
            )
        else:
            self.trunk = DRKTrunk(
                in_channels=seq_depth,
                features=mamba_features,
                norm_type=trunk_norm_type,
                bn_momentum=bn_momentum,
            )

        self.mamba_blocks = nn.ModuleList([
            BidirectionalMamba(
                input_features=mamba_features,
                hidden_features=ssm_hidden_features,
                expansion_factor=mamba_expansion_factor,
                activation=activation,
                norm_type=norm_type,
                tie_in_proj=True,
                tie_gate=False,
                concatenate_fwd_rev=False,
                mamba_args=mamba_args,
            )
            for _ in range(transformer_layers * mamba_layers)
        ])

        self.trans_blocks = nn.ModuleList([
            TransNao(
                transformer_features=mamba_features,
                heads=heads,
                key_size=key_size,
                pos_emb_dim=pos_emb_dim,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                activation=activation,
                positional_encoding=positional_encoding,
            )
            for _ in range(transformer_layers)
        ])

        self.unet_blocks = nn.ModuleList([
            UNet(
                channels=mamba_features,
                kernel_size=3,
                norm_type=unet_norm_type,
                activation=activation,
                bn_momentum=bn_momentum,
            )
            for _ in range(transformer_layers)
        ])

        self.final_norm = NormLayer(final_norm_type, mamba_features) if final_norm_type else None
        self.head = DRKHead(in_channels=mamba_features, features=features, crop=crop)

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)
        x = self.trunk(x)

        block_idx = 0
        use_checkpoint = self.checkpoint_blocks and self.training
        for t in range(self.transformer_layers):
            for _ in range(self.mamba_layers):
                if use_checkpoint:
                    x = checkpoint(self.mamba_blocks[block_idx], x, use_reentrant=False)
                else:
                    x = self.mamba_blocks[block_idx](x)
                block_idx += 1

            x_pooled = x.transpose(1, 2)
            x_pooled = _max_pool1d_same(x_pooled, self.trans_pool_size, self.trans_pool_size)
            x_pooled = x_pooled.transpose(1, 2)

            if use_checkpoint:
                x_pooled = checkpoint(self.trans_blocks[t], x_pooled, use_reentrant=False)
                x = checkpoint(self.unet_blocks[t], x_pooled, x, use_reentrant=False)
            else:
                x_pooled = self.trans_blocks[t](x_pooled)
                x = self.unet_blocks[t](x_pooled, x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        x = self.head(x)
        return x


class StripedMambaIsolate(StripedMamba):
    def __init__(
        self,
        seq_depth: int = 4,
        features: int = 1,
        crop: int = 0,
        dropout_rate: float = 0.2,
        norm_type: str = "layer",
        bn_momentum: float = 0.9,
        activation: str = "gelu",
        mamba_features: int = 768,
        mamba_expansion_factor: int = 1,
        ssm_hidden_features: int = 8,
        mamba_layers: int = 3,
        positional_encoding: str = "enformer",
        trans_pool_size: int = 4,
        key_size: int = 64,
        heads: int = 4,
        pos_emb_dim: int = 32,
        transformer_layers: int = 2,
        trunk_norm_type: str = "batch",
        unet_norm_type: str = "batch",
        final_norm_type: Optional[str] = None,
        mamba_args: Optional[dict] = None,
        use_input_interface: bool = False,
        input_interface_args: Optional[dict] = None,
        checkpoint_blocks: bool = False,
        classifier_pool: str = "mean",
        classifier_dropout_rate: float = 0.0,
    ):
        super().__init__(
            seq_depth=seq_depth,
            features=features,
            crop=crop,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            bn_momentum=bn_momentum,
            activation=activation,
            mamba_features=mamba_features,
            mamba_expansion_factor=mamba_expansion_factor,
            ssm_hidden_features=ssm_hidden_features,
            mamba_layers=mamba_layers,
            positional_encoding=positional_encoding,
            trans_pool_size=trans_pool_size,
            key_size=key_size,
            heads=heads,
            pos_emb_dim=pos_emb_dim,
            transformer_layers=transformer_layers,
            trunk_norm_type=trunk_norm_type,
            unet_norm_type=unet_norm_type,
            final_norm_type=final_norm_type,
            mamba_args=mamba_args,
            use_input_interface=use_input_interface,
            input_interface_args=input_interface_args,
            checkpoint_blocks=checkpoint_blocks,
        )
        self.head = IsolateClassifierHead(
            in_channels=self.mamba_features,
            units=features,
            crop=crop,
            pool=classifier_pool,
            dropout_rate=classifier_dropout_rate,
        )


class StripedMambaInputInterface(nn.Module):
    def __init__(
        self,
        seq_depth: int = 4,
        features: int = 1,
        crop: int = 2048,
        dropout_rate: float = 0.2,
        norm_type: str = "layer",
        bn_momentum: float = 0.9,
        activation: str = "gelu",
        mamba_features: int = 768,
        mamba_expansion_factor: int = 1,
        ssm_hidden_features: int = 8,
        mamba_layers: int = 3,
        positional_encoding: str = "enformer",
        trans_pool_size: int = 4,
        key_size: int = 64,
        heads: int = 4,
        pos_emb_dim: int = 32,
        transformer_layers: int = 2,
        trunk_norm_type: str = "batch",
        unet_norm_type: str = "batch",
        final_norm_type: Optional[str] = None,
        mamba_args: Optional[dict] = None,
        input_interface_args: Optional[dict] = None,
        checkpoint_blocks: bool = False,
    ):
        super().__init__()
        self.features = features
        self.crop = crop
        self.mamba_features = mamba_features
        self.mamba_layers = mamba_layers
        self.transformer_layers = transformer_layers
        self.trans_pool_size = trans_pool_size
        self.final_norm_type = final_norm_type
        self.checkpoint_blocks = checkpoint_blocks

        if input_interface_args is None:
            input_interface_args = {}
        if "cuda_devices" not in input_interface_args:
            if not torch.cuda.is_available():
                raise ValueError(
                    "stripedmamba_input_interface requires CUDA. "
                    "Pass input_interface_args with cuda_devices when running on GPU."
                )
            input_interface_args["cuda_devices"] = [0]
        if "cuda_output_device" not in input_interface_args:
            input_interface_args["cuda_output_device"] = input_interface_args["cuda_devices"][0]

        self.trunk = InputInterfaceSplitTrunk(
            in_channels=seq_depth,
            out_channels=mamba_features,
            **input_interface_args,
        )

        self.mamba_blocks = nn.ModuleList([
            BidirectionalMamba(
                input_features=mamba_features,
                hidden_features=ssm_hidden_features,
                expansion_factor=mamba_expansion_factor,
                activation=activation,
                norm_type=norm_type,
                tie_in_proj=True,
                tie_gate=False,
                concatenate_fwd_rev=False,
                mamba_args=mamba_args,
            )
            for _ in range(transformer_layers * mamba_layers)
        ])

        self.trans_blocks = nn.ModuleList([
            TransNao(
                transformer_features=mamba_features,
                heads=heads,
                key_size=key_size,
                pos_emb_dim=pos_emb_dim,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                activation=activation,
                positional_encoding=positional_encoding,
            )
            for _ in range(transformer_layers)
        ])

        self.unet_blocks = nn.ModuleList([
            UNet(
                channels=mamba_features,
                kernel_size=3,
                norm_type=unet_norm_type,
                activation=activation,
                bn_momentum=bn_momentum,
            )
            for _ in range(transformer_layers)
        ])

        self.final_norm = NormLayer(final_norm_type, mamba_features) if final_norm_type else None
        self.head = DRKHead(in_channels=mamba_features, features=features, crop=crop)

    def forward(self, x: torch.Tensor):
        # x: (B, L, C)
        x = self.trunk(x)

        block_idx = 0
        use_checkpoint = self.checkpoint_blocks and self.training
        for t in range(self.transformer_layers):
            for _ in range(self.mamba_layers):
                if use_checkpoint:
                    x = checkpoint(self.mamba_blocks[block_idx], x, use_reentrant=False)
                else:
                    x = self.mamba_blocks[block_idx](x)
                block_idx += 1

            x_pooled = x.transpose(1, 2)
            x_pooled = _max_pool1d_same(x_pooled, self.trans_pool_size, self.trans_pool_size)
            x_pooled = x_pooled.transpose(1, 2)

            if use_checkpoint:
                x_pooled = checkpoint(self.trans_blocks[t], x_pooled, use_reentrant=False)
                x = checkpoint(self.unet_blocks[t], x_pooled, x, use_reentrant=False)
            else:
                x_pooled = self.trans_blocks[t](x_pooled)
                x = self.unet_blocks[t](x_pooled, x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        x = self.head(x)
        return x
