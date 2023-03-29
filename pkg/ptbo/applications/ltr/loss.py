from torch import empty, Tensor
from torch.nn.modules.loss import _WeightedLoss

class NDGCLoss(_WeightedLoss):
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
        dcg = empty(perm.size(0), device=device)
        for i in range(perm.size(0)):
            dcg[i] = (w * rel[i, perm[i]]).sum()
        sorted_rel = rel.sort(dim=1)[0]
        N = (w * sorted_rel).sum(dim=1)
        return 1 - dcg / N