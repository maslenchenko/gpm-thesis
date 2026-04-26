import copy
import json
from typing import Dict

BORZOI_INPUT_INTERFACE_PRESET: Dict[str, object] = {
    # Geometry-preserving for seq_len=393216: 5x pool-2 => 12288 pre-head, then crop 2048 each side => 8192.
    "num_layers": 5,
    "num_channels_initial": 320,
    "channels_increase_rate": 1.244,
    "strides": 1,
    "kernel_sizes": 1,
    "maxpooling": 2,
    "dilation": 1,
    "norm_type": "batch",
    "context_separate": False,
    "average_interfaces": False,
    "concat": False,
}

ECOLI_INPUT_INTERFACE_PRESET: Dict[str, object] = {
    # Aggressive downsampling for long bacterial assemblies (~2^9 with maxpooling=2).
    "num_layers": 9,
    "num_channels_initial": 256,
    "channels_increase_rate": 1.2,
    "strides": 1,
    "kernel_sizes": 1,
    "maxpooling": 2,
    "dilation": 1,
    "norm_type": "batch",
    "context_separate": False,
    "average_interfaces": False,
    "concat": False,
}

_INPUT_INTERFACE_PRESETS: Dict[str, Dict[str, object]] = {
    "none": {},
    "borzoi": BORZOI_INPUT_INTERFACE_PRESET,
    "ecoli": ECOLI_INPUT_INTERFACE_PRESET,
}


def parse_json_dict(raw: str, arg_name: str) -> Dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid JSON in {arg_name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return value


def resolve_input_interface_args(preset: str, overrides_json: str) -> Dict[str, object]:
    if preset not in _INPUT_INTERFACE_PRESETS:
        allowed = ", ".join(sorted(_INPUT_INTERFACE_PRESETS.keys()))
        raise ValueError(f"Unknown input-interface preset '{preset}'. Allowed: {allowed}")
    resolved = copy.deepcopy(_INPUT_INTERFACE_PRESETS[preset])
    overrides = parse_json_dict(overrides_json, "--input_interface_args")
    resolved.update(overrides)
    return resolved


def inject_input_interface_model_args(
    model_name: str,
    model_args: Dict[str, object],
    use_input_interface: bool,
    preset: str,
    overrides_json: str,
) -> Dict[str, object]:
    out = dict(model_args)
    if not use_input_interface:
        return out

    resolved_args = resolve_input_interface_args(preset, overrides_json)
    existing = out.get("input_interface_args")
    if existing is None:
        merged = {}
    elif isinstance(existing, dict):
        merged = dict(existing)
    else:
        raise ValueError("model_args.input_interface_args must be a JSON object if provided")
    merged.update(resolved_args)
    out["input_interface_args"] = merged

    if model_name in ("stripedmamba", "stripedmamba_isolate"):
        out["use_input_interface"] = True
    elif model_name == "stripedmamba_input_interface":
        # Already input-interface based; only args injection is needed.
        pass
    else:
        raise ValueError(
            f"--use_input_interface is supported only for stripedmamba/stripedmamba_isolate/stripedmamba_input_interface, got '{model_name}'"
        )

    return out
