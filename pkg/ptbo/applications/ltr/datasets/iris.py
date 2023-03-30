
from torch import tensor

from .dataset import Dataset, _read, _features, _labels


class Iris(Dataset):
    path = 'iris.txt'

    def __init__(self):
        iris = _read(self.path)
        features = _features(iris)
        self.features = tensor(features).float()
        
        labels = _labels(iris, 'full')
        self.labels = tensor(labels).long()