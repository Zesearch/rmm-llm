#!/usr/bin/env python3
"""Generate summaries with dense, RMM, or quantized Hugging Face models."""

import argparse
import gc
import json
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
    parser = argparse.ArgumentParser(description="RMM summarization generation sweep.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="abisee/cnn_dailymail")
    parser.add_argument("--dataset-config", default="3.0.0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--article-field", default="article")
    parser.add_argument("--reference-field", default="highlights")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=1300)
    parser.add_argument("--max-new-tokens", type=int, default=80)
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
    parser.add_argument("--output-dir", default="results/summarization")
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


def batches(dataset, size):
    for start in range(0, len(dataset), size):
        end = min(start + size, len(dataset))
        yield start, dataset[start:end]


def main():
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
            tokenizer.padding_side = "left"

            for keep_ratio in args.keep_ratios:
                runtime_config.set_keep_ratio(keep_ratio)
                ratio_label = str(keep_ratio).replace(".", "p")
                output_path = output_dir / (
                    f"{run_id}_{quantization}_keep-{ratio_label}.jsonl"
                )

                with output_path.open("w", encoding="utf-8") as handle:
                    for start, batch in tqdm(
                        batches(dataset, args.batch_size),
                        total=(len(dataset) + args.batch_size - 1) // args.batch_size,
                        desc=f"summaries keep={keep_ratio}",
                    ):
                        articles = batch[args.article_field]
                        references = batch[args.reference_field]
                        prompts = [
                            f"Summarize the following article:\n\n{article}\n\nSummary:"
                            for article in articles
                        ]
                        encoded = tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=args.max_input_length,
                        )
                        device = model.get_input_embeddings().weight.device
                        encoded = {name: value.to(device) for name, value in encoded.items()}
                        prompt_width = encoded["input_ids"].shape[1]

                        started = time.perf_counter()
                        outputs = model.generate(
                            **encoded,
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        elapsed = time.perf_counter() - started
                        predictions = tokenizer.batch_decode(
                            outputs[:, prompt_width:], skip_special_tokens=True
                        )

                        for offset, (prediction, reference) in enumerate(
                            zip(predictions, references)
                        ):
                            record = {
                                "id": start + offset,
                                "prediction": prediction.strip(),
                                "reference": reference,
                                "model": args.model,
                                "revision": args.revision,
                                "quantization": quantization,
                                "dimension_keep_ratio": runtime_config.dimension_keep_ratio,
                                "token_keep_ratio": runtime_config.token_keep_ratio,
                                "generation_sec": elapsed / len(predictions),
                            }
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Predictions: {output_path}")

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
