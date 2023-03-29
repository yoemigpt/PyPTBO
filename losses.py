import torch
from torch import Tensor, LongTensor
from torch.nn.modules.loss import _WeightedLoss


class NDGCLoss(_WeightedLoss):
    def __init__(self, weight: Tensor, size_average=None, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.weight: Tensor
        self.weight = self.weight.sort()[0]

    def forward(self, perm, rel: Tensor):
        return self.loss(perm, rel)

    def loss(self, perm, rel: Tensor):
        device = perm.device
        w = self.weight.to(device)
        val = torch.empty(perm.size(0), device=device)
        for i in range(perm.size(0)):
            val[i] = (w * rel[i, perm[i]]).sum()
        sorted_rel = rel.sort(dim=1)[0]
        N = (w * sorted_rel).sum(dim=1, keepdim=True)
        return 1 - val / N