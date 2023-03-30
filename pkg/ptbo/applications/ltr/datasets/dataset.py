import os
import pandas as pd
import numpy as np
import string

from torch import Tensor
from torch.utils.data import Dataset as _Dataset

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
    
class Dataset(_Dataset):
    path: str
    features: Tensor
    labels: Tensor

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.size(0)