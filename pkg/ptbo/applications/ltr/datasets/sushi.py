from torch import tensor
from .dataset import Dataset, _read, _features, _labels

class Sushi(Dataset):
    path = 'sushi_one_hot.txt'
    def __init__(self) -> None:
        sushi = _read(self.path)
        features = _features(sushi)
        self.features = tensor(features).float()
        labels = _labels(sushi, 'full')
        self.labels = tensor(labels).long()