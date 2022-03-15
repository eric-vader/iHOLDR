#!/usr/bin/env python3
import common
import logging

import numpy as np
import george
import math
from dataclasses import dataclass

from hpolib.benchmarks import synthetic_functions
# Hacks to get the import to work.
from hpolib.benchmarks.synthetic_functions.rosenbrock import Rosenbrock5D

@dataclass
class DataInstance:
    N: int
    X: np.ndarray
    fX: np.ndarray
    z: np.ndarray
    y: np.ndarray
    def reshape(self):
        self.X = self.X[:,None]
        self.y = self.y[:,None]

class Dataset(common.Component):
    def __init__(self, noise_kwargs, n_samples, fn_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)
        self.n_samples = n_samples

        self.noise_kwargs = noise_kwargs
        self.fn_kwargs = fn_kwargs

    def generate_noise(self, mean, variance):
        return self.rng.normal(mean, np.sqrt(variance), self.n_samples)
    def generate_data(self):
        X, fX = self.generate_X_fX(**self.fn_kwargs)
        z = self.generate_noise(**self.noise_kwargs)
        y = fX + z
        return DataInstance(N=self.n_samples, X=X, fX=fX, z=z, y=y)

class Function(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_X_fX(self, fn, bounds):

        if type(fn) == str:
            _fn = eval(fn)
        else:
            _fn = fn
        
        low, high = np.swapaxes(bounds, 1, 0)
        
        X = self.rng.uniform(low, high, self.n_samples)
        fX = _fn(X)
        return X, fX