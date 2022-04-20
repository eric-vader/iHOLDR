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
synthetic_functions.Rosenbrock5D = Rosenbrock5D
import GPy

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
        return self.apply_to_X_fX(_fn, bounds)

    def apply_to_X_fX(self, _fn, bounds):
        D = len(bounds)
        low, high = np.swapaxes(bounds, 1, 0)
        
        X = self.rng.uniform(low, high, (self.n_samples, D))
        
        fX = np.fromiter((_fn(xi) for xi in X), X.dtype)
        return D, X, fX

class HpolibFunction(Function):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_X_fX(self, hpo_fn_ref):

        hpo_fn = getattr(synthetic_functions, hpo_fn_ref)()
        bounds = hpo_fn.get_meta_information()['bounds']
        
        return self.apply_to_X_fX(hpo_fn, bounds)

class GPFunction(Function):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_X_fX(self, bounds, kernel, kernel_kwargs):

        D = len(bounds)
        X = self.generate_grid(bounds, D)

        Kernel = getattr(GPy.kern, kernel)
        ker = Kernel(input_dim=D, **kernel_kwargs)
        logging.info(f"Ground truth kernel_params = (var, ls) = {(kernel_kwargs['variance'], kernel_kwargs['lengthscale'])}")
        
        mu = np.zeros(self.n_samples) #(N*N)
        C = ker.K(X, X) #(N*N)
        # The following function will generate n_functions * (N*N)
        fX = self.rng.multivariate_normal(mu, C, (1), check_valid='raise').reshape(-1)
        return D, X, fX

    def generate_grid(self, bounds, D):
        samples_per_dim = int(np.ceil(np.power(self.n_samples, 1/D)))
        X_domain = np.swapaxes(np.linspace(*np.swapaxes(bounds,0,1), samples_per_dim),0,1)
        # This generates a N-Dim Grid of size approx. n_samples
        X_overflow = np.array(np.meshgrid(*X_domain)).T.reshape(-1, D)
        # Now we only choose subset of it, to fit the n_samples req
        choices = self.rng.choice(len(X_overflow), size=self.n_samples, replace=False)
        return X_overflow[choices]
