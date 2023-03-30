from torch import empty, Tensor
import warnings

def ndcg_score(perm: Tensor, rel: Tensor, weight: Tensor):
    perm = perm.long()
    dcg = empty(perm.size(0), device=perm.device)
    for i in range(perm.size(0)):
        dcg[i] = (weight * rel[i, perm[i]]).sum()
    sorted_rel = rel.sort(dim=1, descending=True)[0]
    idcg = (weight * sorted_rel).sum(dim=1)
    return dcg / idcg