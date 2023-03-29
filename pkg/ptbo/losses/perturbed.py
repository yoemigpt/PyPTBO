import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd import Function
from torch.nn.modules.loss import _Loss as Loss

from ptbo.random import Perturbation
from ptbo.oracles import Oracle

class PerturbedLoss(Module):
    def __init__(self, oracle: Oracle, rnd: Perturbation, loss: Loss, n_samples: int = 100) -> None:
        super().__init__()
        if not isinstance(oracle, Oracle):
            raise TypeError("oracle must be an instance of Oracle")
        
        self.oracle = oracle
        
        if not isinstance(rnd, Perturbation):
            raise TypeError("rnd must be an instance of Perturbation")

        self.rnd = rnd
        self.n_samples = n_samples
        self.loss = loss
        self.ptb = PerturbedLossFunction()

    def forward(self, theta: Tensor, y: Tensor):
        loss = self.ptb.apply(theta, y, self.oracle, self.rnd, self.loss, self.n_samples)
        return loss

class PerturbedLossFunction(Function):
    @staticmethod
    def forward(ctx, theta: Tensor, y: Tensor, oracle: Oracle, rnd: Perturbation, loss: Loss, n_samples: int) -> Tensor:
        device = theta.device
        eta = rnd.sample(theta, n_samples=n_samples)
        pi = torch.empty(*eta.size()[:2], *oracle.outputs, device=device)
        losses = torch.empty(*eta.size()[:2], device=device)
        for i in range(eta.size(0)):
            for j in range(eta.size(1)):
                pi[i, j] = oracle.solve(eta[i, j])
            losses[i] = loss(pi[i], y)
        ctx.save_for_backward(theta, eta, pi, losses)
        ctx.rnd = rnd
        return losses.mean(dim=(0,1))

    @staticmethod
    def backward(ctx, grad_output):
        theta, eta, _, losses = ctx.saved_tensors
        rnd = ctx.rnd
        dlog = rnd.dlog(theta, eta)
        dlog = dlog.permute(*torch.arange(dlog.ndim).flip(0))
        losses = losses.permute(*torch.arange(losses.ndim).flip(0))
        grad = dlog * losses
        grad = grad.permute(*torch.arange(grad.ndim).flip(0))
        grad = grad.mean(dim=0)
        return grad, None, None, None, None, None