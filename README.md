# hot-pate

This repository contains an implementation of [Hot PATE](https://arxiv.org/abs/2312.02132), a variant of the PATE (Private Aggregation of Teacher Ensembles) framework tailored towards tasks with diverse, open-ended outputs.

## Overview

Hot PATE enables privacy-preserving generation by aggregating tokens sampled by individual teachers (each teacher has access to a subset of the records from which we would like to generate).
For example, we might be interested in privately generating synthetic records from a given dataset of sensitive records.
To do so, we would divide the sensitive records among the teachers and generate records by aggregating tokens sampled by the individual teachers.
The key contribution of Hot PATE enabling private and yet *diverse* generation is that tokens are sampled across teachers in a *coordinated* manner (see [the paper](https://arxiv.org/abs/2312.02132) for additional details).

The code is organized as follows:

- `hot_pate/samplers.py` contains strategies for sampling tokens from teachers. Hot PATE uses *coordinated* sampling, while the standard PATE framework uses *independent* sampling.
- `hot_pate/aggregators.py` contains strategies for aggregating tokens sampled from teachers. For demonstration purposes, we provide just a simple "argmax" aggregation. For formal privacy guarantees, we would use a "noisy argmax" aggregation as in [Papernot et al.](https://arxiv.org/abs/1610.05755).
- `hot_pate/hot_pate.py` contains functions for extracting probability distributions from teachers and generating by repeatedly sampling and aggregating.
- `hot_pate/utils.py` contains utilities for the experiments in the paper (a synthetic instruction generation task).

## Quick Start

Begin by installing the requirements:

```bash
pip install -r requirements.txt
```

The following is a minimal pipeline for privately generating synthetic instructions with Hot PATE:

```python
from hot_pate.samplers import CoordinatedSampler
from hot_pate.aggregators import MaxAggregator
from hot_pate.hot_pate import generate
from hot_pate.utils import get_model_and_tokenizer, load_instruction_dataset, create_instruction_teacher_prompts, create_stop_fn

NUM_TEACHERS = 512
NUM_EXAMPLES_PER_TEACHER = 10

# Initialize the model
model_name = "meta-llama/Llama-3.1-8B"
model, tokenizer = get_model_and_tokenizer(model_name)

# Load a dataset of instructions and divide them among the teachers
dataset = load_instruction_dataset()
teacher_prompts = create_instruction_teacher_prompts(
    dataset, NUM_TEACHERS, NUM_EXAMPLES_PER_TEACHER, seed=0
)

# Create a sampler and aggregator (use CoordinateSampler for Hot PATE)
sampler = CoordinatedSampler(temperature=0.6)
aggregator = MaxAggregator()
stop_fn = create_stop_fn(tokenizer)


generated_tokens = generate(
    teacher_prompts,
    model,
    tokenizer,
    sampler,
    aggregator,
    stop_fn,
    verbose=True,
)

print(tokenizer.decode(generated_tokens))
```

## Citation

```bibtex
@article{cohen2023hot,
    title={Hot pate: Private aggregation of distributions for diverse task},
    author={Cohen, Edith and Cohen-Wang, Benjamin and Lyu, Xin and Nelson, Jelani and Sarlos, Tamas and Stemmer, Uri},
    journal={arXiv preprint arXiv:2312.02132},
    year={2023}
}
```
