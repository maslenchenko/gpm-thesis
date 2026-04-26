import argparse
import json
import logging
import os

import torch

from gpm.data import SeqDataset, round_robin_iter
from gpm.models import models
from gpm.utils.input_interface import inject_input_interface_model_args


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single inference batch")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../../data"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="stripedmamba")
    parser.add_argument("--model_args", type=str, default="{}")
    parser.add_argument(
        "--use_input_interface",
        action="store_true",
        help="Enable the vendored input_interface_split trunk integration.",
    )
    parser.add_argument(
        "--input_interface_preset",
        type=str,
        choices=["none", "borzoi", "ecoli"],
        default="none",
        help="Preset for input-interface configuration.",
    )
    parser.add_argument(
        "--input_interface_args",
        type=str,
        default="{}",
        help="JSON object with input-interface config overrides (applied after preset).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=4,
        help="Shuffle buffer size (0 disables shuffling; ignored in eval mode).",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Prefetch buffer size (-1 uses AUTOTUNE, 0 disables prefetch).",
    )
    parser.add_argument(
        "--seq_length_crop",
        type=int,
        default=None,
        help="Crop input sequences to this centered length (seq_length - seq_length_crop must be even).",
    )
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    if args.model_name not in models:
        raise SystemExit(f"Model '{args.model_name}' not known. Available: {' '.join(models.keys())}")

    try:
        model_args = json.loads(args.model_args)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in --model_args: {exc}")
    if not isinstance(model_args, dict):
        raise SystemExit("--model_args must be a JSON object")
    try:
        model_args = inject_input_interface_model_args(
            model_name=args.model_name,
            model_args=model_args,
            use_input_interface=args.use_input_interface,
            preset=args.input_interface_preset,
            overrides_json=args.input_interface_args,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    dataset = SeqDataset(
        data_dir=args.data_dir,
        split_label=args.split,
        batch_size=1,
        mode="eval",
        seq_length_crop=args.seq_length_crop,
        shuffle_buffer=args.shuffle_buffer,
        prefetch=args.prefetch,
    )
    iterator = round_robin_iter([dataset], args.batch_size)
    batch_iter = next(iterator)
    x, y = next(batch_iter)

    model_cls = models[args.model_name]["new_model"]
    model = model_cls(seq_depth=dataset.seq_depth, features=dataset.num_targets, **model_args)

    if args.load:
        import pickle
        with open(args.load, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "model" in payload:
            model.load_state_dict(payload["model"], strict=False)

    model.eval()
    model.to(args.device)

    with torch.no_grad():
        x_t = torch.tensor(x, device=args.device)
        y_t = torch.tensor(y, device=args.device)
        y_pred = model(x_t)

    logging.warning(f"input shape: {tuple(x_t.shape)}")
    logging.warning(f"target shape: {tuple(y_t.shape)}")
    logging.warning(f"pred shape: {tuple(y_pred.shape)}")


if __name__ == "__main__":
    main()
