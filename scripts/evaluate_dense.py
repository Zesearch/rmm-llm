#!/usr/bin/env python3
"""Evaluate the paper's Attention, QKV, and MLP RMM variants on dense models."""

import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from rmm import RMMConfig, projection_pruning, rmm_patch
from rmm.eval.scoring import evaluate_multiple_choice
from rmm.eval.tasks import TASKS, iter_examples, load_task
from rmm.patch import MODEL_MODULES
from rmm.quantization import ModelLoadConfig, load_model_and_tokenizer


METHODS = ("attention", "qkv", "mlp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense-model evaluation for the paper's RMM pruning locations."
    )
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--keep-ratios", nargs="+", type=float, default=[1.0, 0.8, 0.5]
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=["copa"])
    parser.add_argument("--model-type", choices=sorted(MODEL_MODULES), default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", default="results/dense")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def detect_model_type(args):
    if args.model_type:
        return args.model_type
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if config.model_type not in MODEL_MODULES:
        supported = ", ".join(sorted(MODEL_MODULES))
        raise ValueError(
            f"Detected model_type={config.model_type!r}; supported: {supported}"
        )
    return config.model_type


def append_row(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"rmm_dense_{run_id}.csv"
    jsonl_path = output_dir / f"rmm_dense_{run_id}.jsonl"
    model_type = detect_model_type(args)

    datasets = {
        task: load_task(
            task,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        for task in args.tasks
    }
    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(
            model_name_or_path=args.model,
            quantization="none",
            dtype=args.dtype,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    )

    attention_config = RMMConfig(enabled=False)
    with rmm_patch(attention_config, model_types=[model_type]):
        for method in args.methods:
            for keep_ratio in args.keep_ratios:
                if method == "attention":
                    attention_config.set_keep_ratio(keep_ratio)
                    method_context = nullcontext(None)
                else:
                    attention_config.enabled = False
                    method_context = projection_pruning(
                        model, targets=[method], keep_ratio=keep_ratio
                    )

                with method_context as handle:
                    replaced = [] if handle is None else handle.names
                    for task in args.tasks:
                        started = time.perf_counter()
                        result = evaluate_multiple_choice(
                            model,
                            tokenizer,
                            iter_examples(task, datasets[task]),
                            max_samples=args.max_samples,
                            progress=lambda rows: tqdm(rows, desc=f"{method}:{task}"),
                        )
                        row = {
                            "run_id": run_id,
                            "model": args.model,
                            "model_type": model_type,
                            "revision": args.revision or "",
                            "method": method,
                            "keep_ratio": keep_ratio,
                            "task": task,
                            "accuracy": result["accuracy"],
                            "correct": result["correct"],
                            "total": result["total"],
                            "replaced_projection_count": len(replaced),
                            "elapsed_sec": time.perf_counter() - started,
                        }
                        append_row(csv_path, row)
                        with jsonl_path.open("a", encoding="utf-8") as handle_out:
                            handle_out.write(json.dumps(row, sort_keys=True) + "\n")
                        print(json.dumps(row, sort_keys=True))

    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
