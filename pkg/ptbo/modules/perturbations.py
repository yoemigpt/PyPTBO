from torch import rand, randn, exp
from torch import Tensor

from math import sqrt, pi
from abc import ABC, abstractmethod

class Perturbation(ABC):
    @abstractmethod
    def sample(
        self,
        theta: Tensor,
        samples: int = 1
    ) -> Tensor:
        pass

    @abstractmethod
    def pdf(
        self,
        theta: Tensor,
        eta: Tensor
    ) -> Tensor:
        pass

    @abstractmethod
    def dlog(
        self,
        theta: Tensor,
        eta: Tensor
    ) -> Tensor:
        pass

class NormalPerturbation(Perturbation):
    def __init__(
        self,
        sigma: float = 1.0
    ) -> None:
        self.sigma = sigma

    def sample(
        self,
        theta: Tensor,
        samples: int = 1
    ) -> Tensor:
        device = theta.device
        noises = randn(samples, *theta.size(), device=device)
        return theta + self.sigma * noises

    def pdf(
        self,
        theta: Tensor,
        eta: Tensor
    ) -> Tensor:
        eta_norm = (eta - theta) / self.sigma
        return exp(-0.5 * eta_norm ** 2) / (self.sigma * sqrt(2 * pi))

    def dlog(
        self,
        theta: Tensor,
        eta: Tensor
    ) -> Tensor:
        return (eta - theta) / self.sigma ** 2

class GumbelPerturbation(Perturbation):
    def __init__(
        self,
        alpha: float = 1.0
    ) -> None:
        self.alpha = alpha
    
    def sample(
        self,
        theta: Tensor,
        samples: int = 1
    ) -> Tensor:
        device = theta.device
        return theta + self.alpha * rand(samples, *theta.size(), device=device)

    def pdf(
        self,
        theta: Tensor,
        eta: Tensor
    ) -> Tensor:
        return exp(-self.alpha * (eta - theta) - exp(-self.alpha * (eta - theta)))

    def dlog(
        self,
        theta: Tensor,
        eta: Tensor
    ) -> Tensor:
        return self.alpha - self.alpha * exp(-self.alpha * (eta - theta))