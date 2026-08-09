import torch


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def answer_nll(model, tokenizer, prompt, answer):
    """Mean teacher-forced NLL over answer tokens only."""

    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    answer_ids = tokenizer(
        answer, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    if answer_ids.numel() == 0:
        return float("inf")

    input_ids = torch.cat([prompt_tokens["input_ids"], answer_ids], dim=1)
    prompt_length = prompt_tokens["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    attention_mask = torch.ones_like(input_ids)
    device = model_input_device(model)

    with torch.inference_mode():
        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device),
            use_cache=False,
        )
    return float(output.loss.item())


def evaluate_multiple_choice(
    model,
    tokenizer,
    examples,
    max_samples=None,
    progress=None,
):
    if max_samples is not None:
        limit = min(max_samples, len(examples))
        if hasattr(examples, "select"):
            examples = examples.select(range(limit))
        else:
            examples = examples[:limit]
    iterator = progress(examples) if progress else examples

    correct = 0
    total = 0
    for prompt, choices, answer_index in iterator:
        scores = [answer_nll(model, tokenizer, prompt, choice) for choice in choices]
        prediction = min(range(len(scores)), key=scores.__getitem__)
        correct += int(prediction == answer_index)
        total += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }
