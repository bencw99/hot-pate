from abc import ABC, abstractmethod
import torch as ch
from torchtyping import TensorType as TT
from torch.distributions.exponential import Exponential

from .utils import logits_to_probs


class Sampler(ABC):
    def __init__(self, temperature: float):
        self.temperature = temperature

    @abstractmethod
    def sample(
        self, logits: TT["num_teachers", "vocab_size", float], num_samples: int
    ) -> TT["num_samples", "num_teachers", int]:
        raise NotImplementedError


class IndependentSampler(Sampler):
    def sample(
        self, logits: TT["num_teachers", "vocab_size", float], num_samples: int
    ) -> TT["num_samples", "num_teachers", int]:
        probs = logits_to_probs(logits, self.temperature)
        samples = ch.multinomial(probs, num_samples=num_samples, replacement=True).T
        return samples.type(ch.int64)


class CoordinatedSampler(Sampler):
    def sample(
        self, logits: TT["num_teachers", "vocab_size", float], num_samples: int
    ) -> TT["num_samples", "num_teachers", int]:
        probs = logits_to_probs(logits, self.temperature)
        _, vocab_size = probs.shape
        m = Exponential(ch.tensor([1.0], device=logits.device, dtype=logits.dtype))
        shared_randomness = m.sample(sample_shape=(num_samples, vocab_size))[:, :, 0]
        samples = (probs / shared_randomness[:, None]).argmax(dim=-1)
        return samples.type(ch.int64)
