from torch import log2, arange
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import DataLoader
from ptbo.random import NormalPerturbation
from ptbo.applications.ltr import NDCGLoss, RankingOracle
from ptbo.losses import PerturbedLoss

from ptbo.applications.ltr.datasets import Sushi

if __name__ == "__main__":
    dataset = Sushi()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    n = dataset[0][0].size(0)
    m = dataset[0][1].size(0)
    model = Linear(n, m)
    optimizer = Adam(model.parameters(), lr=0.01)

    weight = 1. / log2(2 + arange(m))
    ndgloss = NDCGLoss(weight)

    oracle = RankingOracle(inputs=m, outputs=m)
    rnd = NormalPerturbation()
    perturbed_loss = PerturbedLoss(oracle, rnd, ndgloss)

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
        print(f"Epoch {epoch + 1}: loss={mean_loss / len(dataloader):.4f}")