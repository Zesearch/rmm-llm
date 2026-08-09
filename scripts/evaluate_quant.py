#!/usr/bin/env python3
"""Evaluate RMM across dense and weight-quantized Hugging Face models."""

import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from rmm import RMMConfig, rmm_patch
from rmm.eval.scoring import evaluate_multiple_choice
from rmm.eval.tasks import TASKS, iter_examples, load_task
from rmm.patch import MODEL_MODULES
from rmm.quantization import (
    QUANTIZATION_MODES,
    ModelLoadConfig,
    load_model_and_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RMM compatibility sweep for dense and quantized causal LMs."
    )
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument(
        "--quantization",
        nargs="+",
        default=["none"],
        choices=QUANTIZATION_MODES,
    )
    parser.add_argument(
        "--keep-ratios", nargs="+", type=float, default=[1.0, 0.8, 0.5]
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=["copa"])
    parser.add_argument("--model-type", choices=sorted(MODEL_MODULES), default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def append_row(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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
    model_type = config.model_type
    if model_type not in MODEL_MODULES:
        supported = ", ".join(sorted(MODEL_MODULES))
        raise ValueError(
            f"Detected model_type={model_type!r}, which has no RMM adapter. "
            f"Supported values: {supported}"
        )
    return model_type


def environment_metadata():
    import datasets
    import torch
    import transformers

    gpu_name = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "cuda_version": torch.version.cuda or "",
        "gpu_name": gpu_name,
    }


def main():
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"rmm_quant_{run_id}.csv"
    jsonl_path = output_dir / f"rmm_quant_{run_id}.jsonl"
    model_type = detect_model_type(args)
    environment = environment_metadata()
    datasets = {
        task: load_task(
            task,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        for task in args.tasks
    }

    runtime_config = RMMConfig(enabled=False)
    with rmm_patch(runtime_config, model_types=[model_type]):
        for quantization in args.quantization:
            load_config = ModelLoadConfig(
                model_name_or_path=args.model,
                quantization=quantization,
                dtype=args.dtype,
                device_map=args.device_map,
                cache_dir=args.cache_dir,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            )
            model, tokenizer = load_model_and_tokenizer(load_config)

            for keep_ratio in args.keep_ratios:
                runtime_config.set_keep_ratio(keep_ratio)
                for task in args.tasks:
                    start = time.perf_counter()
                    normalized = list(iter_examples(task, datasets[task]))
                    result = evaluate_multiple_choice(
                        model,
                        tokenizer,
                        normalized,
                        max_samples=args.max_samples,
                        progress=lambda rows: tqdm(rows, desc=task),
                    )
                    row = {
                        "run_id": run_id,
                        "model": args.model,
                        "model_type": model_type,
                        "revision": args.revision or "",
                        "quantization": quantization,
                        "task": task,
                        "dimension_keep_ratio": runtime_config.dimension_keep_ratio,
                        "token_keep_ratio": runtime_config.token_keep_ratio,
                        "accuracy": result["accuracy"],
                        "correct": result["correct"],
                        "total": result["total"],
                        "elapsed_sec": time.perf_counter() - start,
                        **environment,
                    }
                    append_row(csv_path, row)
                    with jsonl_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True))

            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
