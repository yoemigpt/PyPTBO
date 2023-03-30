import os
import pandas as pd
import numpy as np
import string

from torch import tensor
from torch import Tensor
from torch.utils.data import Dataset

root = os.path.dirname(os.path.abspath(__file__))
letters = string.ascii_lowercase

def _read(path):
    return pd.read_csv(os.path.join(root, 'data', path), sep=',')

def _relevance(ranking: str):
    groups = ranking.split('>')
    m = len(groups)
    weights = 1 / np.log2(1 + np.arange(1, m + 1))
    relevances = np.empty(m)
    for p, g in enumerate(groups):
        i = letters.index(g)
        relevances[i] = weights[p]
    return np.array(relevances)

def _features(frame):
    return frame.drop('ranking', axis=1).values

def _relevances(series):
    rankings = series.ranking.values
    return np.array([_relevance(r) for r in rankings], dtype=int)

class _SalrDataset(Dataset):
    path: str
    features: Tensor
    relevances: Tensor

    def __init__(self) -> None:
        data = _read(self.path)
        self.features = tensor(_features(data)).float()
        self.relevances = tensor(_relevances(data)).float()

    def __getitem__(self, index):
        return self.features[index], self.relevances[index]

    def __len__(self):
        return len(self.features)

class SalrDataset(_SalrDataset):
    def __init__(self) -> None:
        super().__init__()
    
class Authorship(SalrDataset):
    path = 'authorship.txt'

class Bodyfat(SalrDataset):
    path = 'bodyfat.txt'

class Callhousing(SalrDataset):
    path = 'callhousing.txt'

class CpuSmall(SalrDataset):
    path = 'cpu-small.txt'

class Fried(SalrDataset):
    path = 'fried.txt'

class Glass(SalrDataset):
    path = 'glass.txt'

class Housing(SalrDataset):
    path = 'housing.txt'

class Iris(SalrDataset):
    path = 'iris.txt'

class Pendigits(SalrDataset):
    path = 'pendigits.txt'

class Segment(SalrDataset):
    path = 'segment.txt'

class Stock(SalrDataset):
    path = 'stock.txt'

class Sushi(SalrDataset):
    path = 'sushi_one_hot.txt'

class Vehicle(SalrDataset):
    path = 'vehicle.txt'

class Vowel(SalrDataset):
    path = 'vowel.txt'

class Wine(SalrDataset):
    path = 'wine.txt'

class Wisconsin(SalrDataset):
    path = 'wisconsin.txt'
