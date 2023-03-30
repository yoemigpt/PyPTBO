from torch import Tensor

def ndcg_score(perm: Tensor, rel: Tensor, weight: Tensor):
    perm = perm.long()
    dcg = (weight * rel[perm]).sum()
    sorted_rel = rel.sort(descending=True)[0]
    idcg = (weight * sorted_rel).sum()
    return dcg / idcg