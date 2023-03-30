import os
import pandas as pd
import numpy as np
import string

from typing import Callable

from torch import tensor
from torch import Tensor
from torch.utils.data import Dataset

root = os.path.dirname(os.path.abspath(__file__))
letters = string.ascii_lowercase

def _read(path):
    return pd.read_csv(os.path.join(root, 'data', path), sep=',')

def _features(frame):
    return frame.drop('ranking', axis=1).values

def _labels(series, ranking_format):
    rankings = series.ranking.values
    if ranking_format == 'full':
        return np.array([_full(r) for r in rankings], dtype=int)
    elif ranking_format == 'partial':
        return np.array([_partial(r) for r in rankings], dtype=int)
    else:
        raise ValueError('Unknown ranking format')

def _full(ranking: str):
    groups = ranking.split('>')
    level = len(groups)
    labels = np.empty(len(groups))
    for g in groups:
        i = letters.index(g)
        labels[i] = level
        level -= 1
    return np.array(labels)

def _partial(ranking):
    groups = ranking.split('>')
    level = len(groups)
    labels = np.empty(sum(map(len, groups)))
    for g in groups:
        for c in g:
            i = letters.index(c)
            labels[i] = level
        level -= 1
    return np.array(labels)
    
class _Salr(Dataset):
    path: str
    features: Tensor
    labels: Tensor

    def __init__(self, feature_transform: None | Callable = None, label_transform: None | Callable = None) -> None:
        data = _read(self.path)
        if feature_transform is None:
            feature_transform = _features
        if label_transform is None:
            label_transform = lambda data: _labels(data, 'full')
        self.features = tensor(feature_transform(data)).float()
        self.labels = tensor(label_transform(data)).long()

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.size(0)

class Salr(_Salr):
    def __init__(self) -> None:
        super().__init__()
    
class Authorship(Salr):
    path = 'authorship.txt'

class Bodyfat(Salr):
    path = 'bodyfat.txt'

class Callhousing(Salr):
    path = 'callhousing.txt'

class CpuSmall(Salr):
    path = 'cpu-small.txt'

class Fried(Salr):
    path = 'fried.txt'

class Glass(Salr):
    path = 'glass.txt'

class Housing(Salr):
    path = 'housing.txt'

class Iris(Salr):
    path = 'iris.txt'

class Pendigits(Salr):
    path = 'pendigits.txt'

class Segment(Salr):
    path = 'segment.txt'

class Stock(Salr):
    path = 'stock.txt'

class Sushi(Salr):
    path = 'sushi_one_hot.txt'

class Vehicle(Salr):
    path = 'vehicle.txt'

class Vowel(Salr):
    path = 'vowel.txt'

class Wine(Salr):
    path = 'wine.txt'

class Wisconsin(Salr):
    path = 'wisconsin.txt'
