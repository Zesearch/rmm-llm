#!/usr/bin/env python3
"""Sliding-window perplexity evaluation for RMM and quantized causal LMs."""

import argparse
import csv
import gc
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from rmm import RMMConfig, rmm_patch
from rmm.patch import MODEL_MODULES
from rmm.quantization import (
    QUANTIZATION_MODES,
    ModelLoadConfig,
    load_model_and_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standard strided perplexity sweep for RMM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--quantization", nargs="+", choices=QUANTIZATION_MODES, default=["none"]
    )
    parser.add_argument(
        "--keep-ratios", nargs="+", type=float, default=[1.0, 0.8, 0.5]
    )
    parser.add_argument("--model-type", choices=sorted(MODEL_MODULES), default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", default="results/perplexity")
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
        raise ValueError(f"No RMM adapter for model_type={config.model_type!r}")
    return config.model_type


def load_corpus(args):
    from datasets import load_dataset

    dataset_args = [args.dataset]
    if args.dataset_config:
        dataset_args.append(args.dataset_config)
    dataset = load_dataset(
        *dataset_args,
        split=args.split,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    texts = [text for text in dataset[args.text_field] if text and text.strip()]
    return "\n\n".join(texts)


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def strided_perplexity(model, input_ids, max_length, stride):
    import torch

    if stride <= 0 or stride > max_length:
        raise ValueError("stride must be in (0, max_length]")
    sequence_length = input_ids.shape[1]
    device = model_input_device(model)
    nll_sum = 0.0
    token_count = 0
    previous_end = 0

    for begin in tqdm(range(0, sequence_length, stride), desc="perplexity"):
        end = min(begin + max_length, sequence_length)
        target_length = end - previous_end
        window = input_ids[:, begin:end].to(device)
        labels = window.clone()
        labels[:, :-target_length] = -100

        with torch.inference_mode():
            output = model(input_ids=window, labels=labels, use_cache=False)

        valid_tokens = int((labels != -100).sum().item())
        loss_tokens = valid_tokens - labels.shape[0]
        if loss_tokens > 0:
            nll_sum += float(output.loss.item()) * loss_tokens
            token_count += loss_tokens
        previous_end = end
        if end == sequence_length:
            break

    if token_count == 0:
        raise RuntimeError("No predictable tokens were evaluated")
    mean_nll = nll_sum / token_count
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "evaluated_tokens": token_count,
        "corpus_tokens": sequence_length,
    }


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
    if args.max_length < 2:
        raise ValueError("max-length must be at least 2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"perplexity_{run_id}.csv"
    jsonl_path = output_dir / f"perplexity_{run_id}.jsonl"
    corpus = load_corpus(args)
    model_type = detect_model_type(args)
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
            input_ids = tokenizer(corpus, return_tensors="pt")["input_ids"]
            if args.max_tokens is not None:
                input_ids = input_ids[:, : args.max_tokens]

            for keep_ratio in args.keep_ratios:
                runtime_config.set_keep_ratio(keep_ratio)
                started = time.perf_counter()
                metrics = strided_perplexity(
                    model, input_ids, args.max_length, args.stride
                )
                row = {
                    "run_id": run_id,
                    "model": args.model,
                    "model_type": model_type,
                    "revision": args.revision or "",
                    "quantization": quantization,
                    "dataset": args.dataset,
                    "dataset_config": args.dataset_config or "",
                    "split": args.split,
                    "dimension_keep_ratio": runtime_config.dimension_keep_ratio,
                    "token_keep_ratio": runtime_config.token_keep_ratio,
                    "max_length": args.max_length,
                    "stride": args.stride,
                    **metrics,
                    "elapsed_sec": time.perf_counter() - started,
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
