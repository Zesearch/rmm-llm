TASKS = (
    "copa",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "commonsense_qa",
)


TASK_SPECS = {
    "copa": ("pkavumba/balanced-copa", None, "test"),
    "piqa": ("baber/piqa", None, "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "validation"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation"),
    "commonsense_qa": ("tau/commonsense_qa", None, "validation"),
}


def load_task(task, cache_dir=None, trust_remote_code=False):
    if task not in TASK_SPECS:
        raise ValueError(f"Unknown task {task!r}; choose from {', '.join(TASKS)}")
    from datasets import load_dataset

    dataset_name, subset, split = TASK_SPECS[task]
    kwargs = {
        "split": split,
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    if subset is None:
        return load_dataset(dataset_name, **kwargs)
    return load_dataset(dataset_name, subset, **kwargs)


def _arc_answer_index(example):
    labels = list(example["choices"]["label"])
    answer_key = str(example["answerKey"])
    if answer_key in labels:
        return labels.index(answer_key)
    if answer_key.isdigit() and 1 <= int(answer_key) <= len(labels):
        return int(answer_key) - 1
    return ord(answer_key.upper()) - ord("A")


def iter_examples(task, dataset):
    """Yield normalized ``(prompt, choices, answer_index)`` tuples."""

    for example in dataset:
        if task == "copa":
            question = example["question"]
            is_cause = question == 0 or str(question).lower() == "cause"
            relation = "CAUSE" if is_cause else "EFFECT"
            prompt = (
                f"Question: What is the {relation} of the following? "
                f"{example['premise']}\nAnswer:"
            )
            choices = [f" {example['choice1']}", f" {example['choice2']}"]
            answer = int(example["label"])
        elif task == "piqa":
            prompt = f"Question: {example['goal']}\nAnswer:"
            choices = [f" {example['sol1']}", f" {example['sol2']}"]
            answer = int(example["label"])
        elif task in {"arc_easy", "arc_challenge"}:
            prompt = f"Question: {example['question']}\nAnswer:"
            choices = [f" {choice}" for choice in example["choices"]["text"]]
            answer = _arc_answer_index(example)
        elif task == "commonsense_qa":
            prompt = f"Question: {example['question']}\nAnswer:"
            labels = list(example["choices"]["label"])
            choices = [f" {choice}" for choice in example["choices"]["text"]]
            answer = labels.index(example["answerKey"])
        else:
            raise ValueError(f"Unsupported task {task!r}")
        yield prompt, choices, answer

