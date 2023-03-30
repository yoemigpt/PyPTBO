from torch import empty
from torch import Tensor
from torch.nn import Module
from torch.autograd import Function
from torch.nn.modules.loss import _Loss as Loss, _WeightedLoss

from .perturbations import Perturbation
from .oracles import Oracle
from .metrics import ndcg_score

class PerturbedLoss(Module):
    def __init__(self, oracle: Oracle, ptb: Perturbation, loss: Loss, n_samples: int = 100) -> None:
        super().__init__()
        if not isinstance(oracle, Oracle):
            raise TypeError("oracle must be an instance of Oracle")
        
        self.oracle = oracle
        
        if not isinstance(ptb, Perturbation):
            raise TypeError("ptb must be an instance of Perturbation")

        self.ptb = ptb
        self.n_samples = n_samples
        self.loss = loss
        self.ptbf = PerturbedLossFunction()

    def forward(self, theta: Tensor, y: Tensor):
        loss, _, _ = self.ptbf.apply(theta, y, self.oracle, self.ptb, self.loss, self.n_samples) # type: ignore
        return Tensor(loss).mean(dim=(0, 1))

class PerturbedLossFunction(Function):
    @staticmethod
    def forward(theta: Tensor, y: Tensor, oracle: Oracle, ptb: Perturbation, loss: Loss, n_samples: int):
        device = theta.device
        eta = ptb.sample(theta, n_samples=n_samples)
        pi = empty(*eta.size()[:2], *oracle.outputs, device=device)
        losses = empty(*eta.size()[:2], device=device)
        for i in range(eta.size(0)):
            for j in range(eta.size(1)):
                pi[i, j] = oracle.call(eta[i, j])
            losses[i] = loss(pi[i], y)
        return losses, eta, pi

    @staticmethod
    def setup_context(
        ctx,
        inputs: tuple[Tensor, Tensor, Oracle, Perturbation, Loss, int],
        output: tuple[Tensor, Tensor, Tensor]):
        theta, _, _, ptb, _, _ = inputs
        losses, eta, pi = output
        ctx.save_for_backward(theta, eta, pi, losses)
        ctx.ptb = ptb

    @staticmethod
    def backward(ctx, *grad_outputs):
        theta, eta, _, losses = ctx.saved_tensors
        ptb = ctx.ptb
        grad = ptb.dlog(theta, eta)
        for i in range(grad.size(0)):
            for j in range(grad.size(1)):
                grad[i, j] *= losses[i, j]
        grad = grad.mean(dim=0)
        return grad, None, None, None, None, None

class NDCGLoss(_WeightedLoss):
    def __init__(self, weight: Tensor, size_average=None, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.weight: Tensor
        self.weight = self.weight.sort(dim=0, descending=True)[0]

    def forward(self, perm, rel: Tensor):
        return self.loss(perm, rel)

    def loss(self, perm, rel: Tensor):
        device = perm.device
        perm = perm.long()
        w = self.weight.to(device)
        ndcg = empty(perm.size(0), device=device)
        for i in range(perm.size(0)):
            ndcg[i] = ndcg_score(perm[i], rel[i], w)
        return 1 - ndcg