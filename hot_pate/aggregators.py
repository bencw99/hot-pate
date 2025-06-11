from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict
from torchtyping import TensorType as TT


class Aggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        histograms: TT["batch_size", "vocab_size", int],
    ) -> Tuple[TT["batch_size", int], TT["batch_size", int]]:
        raise NotImplementedError


class MaxAggregator(Aggregator):
    def aggregate(
        self,
        histograms: TT["batch_size", "vocab_size", int],
    ) -> Tuple[TT["batch_size", int], TT["batch_size", bool]]:
        max_outputs = histograms.max(dim=-1)
        return max_outputs.indices, max_outputs.values
