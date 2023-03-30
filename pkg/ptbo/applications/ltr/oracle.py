from torch import Tensor
from ptbo.oracles import Oracle

class RankingOracle(Oracle):
    def solve(self, eta: Tensor) -> Tensor:
        return eta.argsort(dim=0, descending=True).long()