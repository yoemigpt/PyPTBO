from torch import Tensor, Size
from abc import ABC, abstractmethod
import torch

class Oracle(ABC):
    def __init__(self, inputs: Size, outputs: Size) -> None:
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
    
    @abstractmethod
    def solve(self, eta: Tensor) -> Tensor:
        pass


class RankOracle(Oracle):
    def solve(self, eta: Tensor) -> Tensor:
        return torch.argsort(eta, dim=1)