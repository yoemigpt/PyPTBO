import torch
from torch import Tensor
from abc import ABC, abstractmethod

class Perturbation(ABC):
    @abstractmethod
    def sample(self, theta: Tensor, n_samples: int = 1) -> Tensor:
        pass

    @abstractmethod
    def dlog(self, theta: Tensor, eta: Tensor) -> Tensor:
        pass
