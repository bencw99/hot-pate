import numpy as np
import torch as ch
from torchtyping import TensorType as TT
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def logits_to_probs(
    logits: TT["num_teachers", "vocab_size", float],
    temperature: float = 0.6,
) -> TT["num_teachers", "vocab_size", float]:
    return ch.softmax(logits / temperature, dim=-1)


def create_histograms(
    samples: TT["num_samples", "num_teachers", int], vocab_size: int
) -> TT["num_samples", "vocab_size", int]:
    num_samples, _ = samples.shape
    histograms = ch.zeros(num_samples, vocab_size, device=samples.device, dtype=ch.long)
    histograms.scatter_add_(dim=1, index=samples, src=ch.ones_like(samples))
    return histograms


def estimate_transfer_support(
    histograms: TT["num_samples", "vocab_size", int],
    threshold: int,
) -> TT["num_teachers", int]:
    return np.where((histograms >= threshold).any(dim=0).cpu().numpy())[0]


def estimate_transfer_mass(
    histograms: TT["num_samples", "vocab_size", int],
    threshold: int,
) -> TT["num_teachers", int]:
    num_teachers = histograms[0].sum().item()
    histograms = ch.clone(histograms)
    histograms[histograms < threshold] = 0
    return (histograms.sum(dim=-1) / num_teachers).mean().cpu().numpy()


def get_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=ch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_instruction_dataset():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = dataset.filter(lambda x: len(x["context"]) == 0)
    dataset = dataset.filter(lambda x: len(x["instruction"]) <= 256)
    return dataset


def create_instruction_teacher_prompts(
    dataset, num_teachers, num_examples_per_teacher, seed=None
):
    random = np.random if seed is None else np.random.RandomState(seed)
    num_samples = num_teachers * num_examples_per_teacher
    if num_samples > len(dataset):
        raise ValueError(
            f"More samples needed ({num_samples}) than available ({len(dataset)})"
        )
    indices = random.choice(len(dataset), size=num_samples, replace=False)
    instructions = np.array(dataset["instruction"])
    prompts = []
    for i in range(num_teachers):
        start = i * num_examples_per_teacher
        end = start + num_examples_per_teacher
        examples = instructions[indices[start:end]]
        prompts.append("\n\n".join(examples) + "\n\n")
    return prompts


def create_stop_fn(tokenizer: AutoTokenizer, max_length: int = 128):
    def stop_fn(generated_tokens):
        if len(generated_tokens) >= max_length:
            return True
        elif len(generated_tokens) > 0 and generated_tokens[-1] == tokenizer.eos_token_id:
            return True
        generated_text = tokenizer.decode(generated_tokens)
        return "\n\n" in generated_text
    return stop_fn
