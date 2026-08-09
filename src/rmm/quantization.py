"""Model loading helpers for RMM and weight-quantization compatibility tests."""

from dataclasses import dataclass
from typing import Optional


QUANTIZATION_MODES = (
    "none",
    "prequantized",
    "bnb-int8",
    "bnb-nf4",
    "bnb-fp4",
)


@dataclass
class ModelLoadConfig:
    model_name_or_path: str
    quantization: str = "none"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    local_files_only: bool = False

    def validate(self):
        if self.quantization not in QUANTIZATION_MODES:
            choices = ", ".join(QUANTIZATION_MODES)
            raise ValueError(
                f"Unknown quantization mode {self.quantization!r}; choose from {choices}"
            )


def _torch_dtype(name):
    import torch

    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def build_model_kwargs(config: ModelLoadConfig):
    """Build ``from_pretrained`` arguments without loading a model."""

    config.validate()
    dtype = _torch_dtype(config.dtype)
    kwargs = {
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
        "attn_implementation": "eager",
    }
    if config.cache_dir:
        kwargs["cache_dir"] = config.cache_dir
    if config.revision:
        kwargs["revision"] = config.revision

    if config.quantization in {"none", "prequantized"}:
        kwargs["torch_dtype"] = dtype
        return kwargs

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes quantization requires `pip install -e '.[quant]'`."
        ) from exc

    if config.quantization == "bnb-int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_type = config.quantization.removeprefix("bnb-")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    return kwargs


def load_model_and_tokenizer(config: ModelLoadConfig):
    """Load a causal LM for dense or quantized RMM evaluation."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = build_model_kwargs(config)
    tokenizer_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if config.cache_dir:
        tokenizer_kwargs["cache_dir"] = config.cache_dir
    if config.revision:
        tokenizer_kwargs["revision"] = config.revision

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, **tokenizer_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, **kwargs
    )
    model.eval()
    return model, tokenizer

