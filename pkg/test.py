from torch import log2, arange
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import DataLoader

from ptbo import NormalPerturbation
from ptbo import PerturbedLoss, NDCGLoss
from ptbo import RankingOracle

from ptbo.datasets import salr

if __name__ == "__main__":
    dataset = salr.Wine()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    n = dataset[0][0].size(0)
    m = dataset[0][1].size(0)
    model = Linear(n, m)
    optimizer = Adam(model.parameters(), lr=0.01)

    weight = 1. / log2(2 + arange(m))
    ndgloss = NDCGLoss(weight)

    oracle = RankingOracle(inputs=m, outputs=m)
    ptb = NormalPerturbation()
    perturbed_loss = PerturbedLoss(oracle, ptb, ndgloss)

    for epoch in range(10):
        mean_loss = 0
        for X, y in dataloader:
            model.train()
            optimizer.zero_grad()
            theta = model(X)
            loss = perturbed_loss(theta, y)
            mean_loss += loss.item()
            loss.backward()
            optimizer.step()
        mean_loss /= len(dataloader)
        print(f"Epoch {epoch + 1:3}: loss={mean_loss:.4f}")