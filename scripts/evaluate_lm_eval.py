#!/usr/bin/env python3
"""Run lm-evaluation-harness tasks with RMM and optional weight quantization."""

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from rmm import RMMConfig, rmm_patch
from rmm.patch import MODEL_MODULES
from rmm.quantization import (
    QUANTIZATION_MODES,
    ModelLoadConfig,
    load_model_and_tokenizer,
)


HISTORICAL_TASKS = (
    "mmlu",
    "gsm8k",
    "humaneval",
    "ruler_cwe",
    "ruler_qa_hotpot",
)


def parse_limit(value):
    if value is None:
        return None
    return float(value) if "." in value else int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="lm-evaluation-harness sweep for RMM and quantized models."
    )
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["mmlu", "gsm8k"],
        help="Any task names installed in lm-evaluation-harness",
    )
    parser.add_argument(
        "--quantization",
        nargs="+",
        default=["none"],
        choices=QUANTIZATION_MODES,
    )
    parser.add_argument(
        "--keep-ratios", nargs="+", type=float, default=[1.0, 0.8, 0.5]
    )
    parser.add_argument("--model-type", choices=sorted(MODEL_MODULES), default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=parse_limit, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", default="results/lm_eval")
    parser.add_argument("--use-cache", default=None)
    parser.add_argument(
        "--ruler-lengths",
        nargs="+",
        type=int,
        default=None,
        help="RULER generation lengths, e.g. 4096 8192 16384",
    )
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
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


def harness_batch_size(value):
    return value if value.startswith("auto") else int(value)


def serializable_default(value):
    try:
        from lm_eval.utils import handle_non_serializable

        return handle_non_serializable(value)
    except Exception:
        return str(value)


def run_harness(args, model, tokenizer, metadata):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    batch_size = harness_batch_size(args.batch_size)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=batch_size,
        max_batch_size=args.max_batch_size,
        max_length=args.max_length,
    )
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        use_cache=args.use_cache,
        log_samples=args.log_samples,
        apply_chat_template=args.apply_chat_template,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        metadata=metadata,
    )


def main():
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_type = detect_model_type(args)
    runtime_config = RMMConfig(enabled=False)
    metadata = {}
    if args.ruler_lengths:
        metadata["max_seq_lengths"] = args.ruler_lengths

    if "humaneval" in args.tasks and not args.confirm_run_unsafe_code:
        raise ValueError(
            "HumanEval may execute generated code. Re-run only in a sandbox with "
            "--confirm-run-unsafe-code after reviewing the risk."
        )

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
                started = time.perf_counter()
                results = run_harness(args, model, tokenizer, metadata)
                record = {
                    "run": {
                        "run_id": run_id,
                        "model": args.model,
                        "model_type": model_type,
                        "revision": args.revision,
                        "quantization": quantization,
                        "dimension_keep_ratio": runtime_config.dimension_keep_ratio,
                        "token_keep_ratio": runtime_config.token_keep_ratio,
                        "tasks": args.tasks,
                        "num_fewshot": args.num_fewshot,
                        "batch_size": args.batch_size,
                        "max_length": args.max_length,
                        "ruler_lengths": args.ruler_lengths,
                        "elapsed_sec": time.perf_counter() - started,
                    },
                    "lm_eval": results,
                }
                ratio_label = str(keep_ratio).replace(".", "p")
                output_path = output_dir / (
                    f"{run_id}_{quantization}_keep-{ratio_label}.json"
                )
                with output_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        record,
                        handle,
                        indent=2,
                        sort_keys=True,
                        default=serializable_default,
                    )
                print(f"Results: {output_path}")

            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


if __name__ == "__main__":
    main()

