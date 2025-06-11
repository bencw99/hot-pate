from typing import List, Any, Dict, Callable
from tqdm.auto import tqdm
import torch as ch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from .samplers import Sampler
from .aggregators import Aggregator
from .utils import create_histograms, logits_to_probs


def initialize_prompt_dict(prompts: List[str], tokenizer: Any) -> Dict[str, List[int]]:
    prompt_dict = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for prompt in prompts:
        input_ids = tokenizer(prompt).input_ids
        prompt_dict["input_ids"].append(input_ids)
        prompt_dict["attention_mask"].append([1] * len(input_ids))
        prompt_dict["labels"].append(input_ids)
    return prompt_dict


def update_prompt_dict(prompt_dict: Dict[str, List[int]], token: int) -> None:
    for i in range(len(prompt_dict["attention_mask"])):
        prompt_dict["attention_mask"][i].append(1)
    for i in range(len(prompt_dict["input_ids"])):
        prompt_dict["input_ids"][i].append(token)
        # We don't need to append to labels because it's a reference to input_ids


def get_loader(
    prompt_dict: Dict[str, List[int]], tokenizer: Any, batch_size: int
) -> DataLoader:
    prompt_dataset = Dataset.from_dict(prompt_dict)
    collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        prompt_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return loader


def compute_logits(
    prompt_dict: Dict[str, List[int]],
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
    verbose: bool = False,
) -> ch.Tensor:
    loader = get_loader(prompt_dict, tokenizer, batch_size)
    num_teachers = len(prompt_dict["input_ids"])
    logits = ch.zeros((num_teachers, model.vocab_size), device=model.device)
    start_index = 0
    for batch in tqdm(loader, desc="Evaluating teachers", disable=not verbose):
        with ch.no_grad():
            outputs = model(**batch.to(model.device))
        logits[start_index : start_index + len(batch.input_ids)] = outputs.logits[
            :, -1, :
        ]
        start_index += len(batch.input_ids)
    return logits


def sample_token_from_average_distribution(
    prompt_dict: Dict[str, List[int]],
    model: Any,
    tokenizer: Any,
    temperature: float = 0.6,
    batch_size: int = 32,
    verbose: bool = False,
) -> int:
    logits = compute_logits(prompt_dict, model, tokenizer, batch_size, verbose)
    probs = logits_to_probs(logits, temperature).mean(dim=0)
    token = ch.multinomial(probs, num_samples=1).item()
    return token


def sample_token_with_pate(
    prompt_dict: Dict[str, List[int]],
    model: Any,
    tokenizer: Any,
    sampler: Sampler,
    aggregator: Aggregator,
    batch_size: int = 32,
    verbose: bool = False,
) -> int:
    logits = compute_logits(prompt_dict, model, tokenizer, batch_size, verbose)
    samples = sampler.sample(logits, 1)
    histograms = create_histograms(samples, model.vocab_size)
    tokens, votes = aggregator.aggregate(histograms)
    return tokens[0].item(), votes[0].item()


def generate(
    prompts: List[str],
    model: Any,
    tokenizer: Any,
    sampler: Sampler,
    aggregator: Aggregator,
    stop_fn: Callable[[List[int]], bool],
    batch_size: int = 32,
    verbose: bool = False,
):
    prompt_dict = initialize_prompt_dict(prompts, tokenizer)
    generated_tokens = []
    while not stop_fn(generated_tokens):
        token, votes = sample_token_with_pate(
            prompt_dict,
            model,
            tokenizer,
            sampler,
            aggregator,
            batch_size,
            verbose=verbose,
        )
        generated_tokens.append(token)
        update_prompt_dict(prompt_dict, token)
        cur_generation = tokenizer.decode(generated_tokens)
        print(votes, cur_generation)
    return generated_tokens
