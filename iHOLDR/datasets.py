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
    D: int
    N: int
    X: np.ndarray
    fX: np.ndarray
    z: np.ndarray
    y: np.ndarray
    def split(self, ratio):

        N_ratioed = int(ratio * self.N)

        return DataInstance(D=self.D, N=N_ratioed, X=self.X[:N_ratioed], fX=self.fX[:N_ratioed], z=self.z[:N_ratioed], y=self.y[:N_ratioed]), \
            DataInstance(D=self.D, N=self.N-N_ratioed, X=self.X[N_ratioed:], fX=self.fX[N_ratioed:], z=self.z[N_ratioed:], y=self.y[N_ratioed:])

class Dataset(common.Component):
    def __init__(self, noise_kwargs, n_samples, fn_kwargs, n_train_ratio, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)
        self.n_samples = n_samples

        self.noise_kwargs = noise_kwargs
        self.fn_kwargs = fn_kwargs

        self.n_train_ratio = n_train_ratio

    def generate_noise(self, mean, variance):
        # output is always 1D
        return self.rng.normal(mean, np.sqrt(variance), self.n_samples)
    def generate_data(self):
        D, X, fX = self.generate_X_fX(**self.fn_kwargs)
        z = self.generate_noise(**self.noise_kwargs)
        y = fX + z

        # Here we split the data up to train, test splits
        data = DataInstance(D=D, N=self.n_samples, X=X, fX=fX, z=z, y=y)
        train_data, test_data = data.split(self.n_train_ratio)
        return train_data, test_data, data

class Function(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_X_fX(self, fn, bounds):

        if type(fn) == str:
            _fn = eval(fn)
        else:
            _fn = fn
        
        D = len(bounds)
        low, high = np.swapaxes(bounds, 1, 0)
        
        X = self.rng.uniform(low, high, (self.n_samples, D))
        
        fX = np.fromiter((_fn(xi) for xi in X), X.dtype)
        return D, X, fX