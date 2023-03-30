import torch
from torch.optim import Adam
from ptbo.random import NormalPerturbation
from ptbo.applications.ltr import NDGCLoss, RankingOracle
from ptbo.losses import PerturbedLoss

if __name__ == "__main__":
    X = torch.rand(1000, 100)
    y = torch.rand(1000, 10)

    model = torch.nn.Linear(X.size(1), y.size(1))
    optimizer = Adam(model.parameters(), lr=0.01)

    weight = 1. / torch.log2(2 + torch.arange(y.size(1), dtype=torch.float))
    ndgloss = NDGCLoss(weight)

    oracle = RankingOracle(torch.Size([y.size(1)]), torch.Size([y.size(1)]))
    rnd = NormalPerturbation()
    loss = PerturbedLoss(oracle, rnd, ndgloss)

    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        theta = model(X)
        loss_val = loss(theta, y)
        loss_val.backward()
        optimizer.step()
        print(f"Epoch {epoch}: loss={loss_val}")