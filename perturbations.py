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

class GaussianPerturbation(Perturbation):
    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma

    def sample(self, theta: Tensor, n_samples: int = 1) -> Tensor:
        device = theta.device
        return theta + self.sigma * torch.randn(n_samples, *theta.size(), device=device)

    def dlog(self, theta: Tensor, eta: Tensor) -> Tensor:
        return (eta - theta) / self.sigma ** 2
