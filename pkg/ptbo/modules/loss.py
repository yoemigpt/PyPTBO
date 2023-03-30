from torch import empty
from torch import Tensor
from torch.nn import Module
from torch.autograd import Function
from torch.nn.modules.loss import _Loss as Loss, _WeightedLoss

from .perturbations import Perturbation
from .oracles import Oracle
from .metrics import ndcg_score

class PerturbedLoss(Module):
    def __init__(
        self,
        perturb: Perturbation,
        samples: int,
        oracle: Oracle,
        loss: Loss
    ) -> None:
        super().__init__()
        if not isinstance(oracle, Oracle):
            raise TypeError("oracle must be an instance of Oracle")
        
        self.oracle = oracle
        
        if not isinstance(perturb, Perturbation):
            raise TypeError("perturb must be an instance of Perturbation")

        self.perturb = perturb
        self.samples = samples
        self.loss = loss
        self.f = PerturbedLossFunction()

    def forward(self, theta: Tensor, y: Tensor):
        result = self.f.apply(theta, y, self.perturb, self.samples, self.oracle, self.loss)
        return tuple(result)[0] # type: ignore

class PerturbedLossFunction(Function):
    @staticmethod
    def forward(
        theta: Tensor,
        y: Tensor,
        ptb: Perturbation,
        samples: int,
        oracle: Oracle,
        loss: Loss
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = theta.device
        eta = ptb.sample(theta, samples=samples)
        pi = empty(*eta.size()[:2], *oracle.outputs, device=device)
        losses = empty(*eta.size()[:2], device=device)
        for i in range(eta.size(0)):
            for j in range(eta.size(1)):
                pi[i, j] = oracle.call(eta[i, j])
            losses[i] = loss(pi[i], y)
        val = losses.mean(dim=(0, 1))
        return val, losses, eta, pi

    @staticmethod
    def setup_context(
        ctx,
        inputs: tuple[Tensor, Tensor, Perturbation, int, Oracle, Loss],
        output: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> None:
        theta, _, ptb, _, _, _ = inputs
        _, losses, eta, pi = output
        ctx.save_for_backward(theta, eta, pi, losses)
        ctx.mark_non_differentiable(losses)
        ctx.mark_non_differentiable(pi)
        ctx.mark_non_differentiable(eta)
        ctx.ptb = ptb

    @staticmethod
    def backward(
        ctx,
        *grads: Tensor
    ) -> tuple[Tensor, None, None, None, None, None]:
        theta, eta, _, losses = ctx.saved_tensors
        ptb = ctx.ptb
        grad = ptb.dlog(theta, eta)
        for i in range(grad.size(0)):
            for j in range(grad.size(1)):
                grad[i, j] *= losses[i, j]
        grad = grad.mean(dim=0)
        return grad, None, None, None, None, None

class NDCGLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Tensor,
        size_average=None,
        reduce=None,
        reduction='mean'
    ):
        super().__init__(weight, size_average, reduce, reduction)
        self.weight: Tensor
        self.weight = self.weight.sort(dim=0, descending=True)[0]

    def forward(self, perm: Tensor, rel: Tensor) -> Tensor:
        return self.loss(perm, rel)

    def loss(self, perm: Tensor, rel: Tensor) -> Tensor:
        device = perm.device
        perm = perm.long()
        w = self.weight.to(device)
        ndcg = empty(perm.size(0), device=device)
        for i in range(perm.size(0)):
            ndcg[i] = ndcg_score(perm[i], rel[i], w)
        return 1 - ndcg