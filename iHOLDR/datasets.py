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
import csv

from uci_datasets import Dataset as UCIDataset

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

    def rearrange(self, idx):
        self.X[:] = self.X[idx]
        self.fX[:] = self.fX[idx]
        self.z[:] = self.z[idx]
        self.y[:] = self.y[idx]

    def clone(self):
        return DataInstance(D=self.D, N=self.N, X=self.X.copy(), fX=self.fX.copy(), z=self.z.copy(), y=self.y.copy())

class Dataset(common.Component):
    def __init__(self, noise_kwargs, fn_kwargs, n_train_ratio, n_samples=None, **kwargs):
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
        if self.n_samples == None:
            self.n_samples = len(X)

        z = self.generate_noise(**self.noise_kwargs)
        y = fX + z

        # Here we split the data up to train, test splits
        data = DataInstance(D=D, N=self.n_samples, X=X, fX=fX, z=z, y=y)
        train_data, test_data = data.split(self.n_train_ratio)
        return train_data, test_data, data
    def generate_X_fX(self, **fn_kwargs):
        raise NotImplementedError
        # Must return of shape X=(self.n_samples, D) and fX=(self.n_samples,)

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

# We follow the convention from Exact Gaussian Processes on a Million Data Points
class UCIFunction(Function):
    def __init__(self, is_pre_split, **kwargs):
        super().__init__(**kwargs)
        self.uci_loader = self.config.configs['uci']
        self.is_pre_split = is_pre_split

    def generate_data(self):

        # Use our own split
        if not self.is_pre_split:
            return super().generate_data()
        
        # From documentation that the 
        assert(self.random_seed<10)
        uci_dataset = UCIDataset(base_path=self.uci_loader.get_path('uci_datasets'), **self.fn_kwargs)
        X_train, fX_train, X_test, fX_test = uci_dataset.get_split(split=self.random_seed)

        if self.n_samples == None:
            self.n_samples = len(X_train) + len(X_test)
        else:
            total_samples = len(X_train) + len(X_test)

            n_train_samples = int(np.ceil(self.n_samples * len(X_train) / total_samples))
            n_test_samples = self.n_samples - n_train_samples

            train_choices = self.rng.choice(len(X_train), size=n_train_samples, replace=False)
            test_choices = self.rng.choice(len(X_test), size=n_test_samples, replace=False)

            X_train = X_train[train_choices]
            fX_train = fX_train[train_choices]
            X_test = X_test[test_choices]
            fX_test = fX_test[test_choices]

        assert(fX_train.shape[1]==1)
        # Flatten fX
        fX_train = fX_train.reshape(-1)
        fX_test = fX_test.reshape(-1)

        z = self.generate_noise(**self.noise_kwargs)
        z_train = z[:len(X_train)]
        z_test = z[len(X_train):]

        y_train = fX_train + z_train
        y_test = fX_test + z_test

        D = uci_dataset.x.shape[1]

        X = np.concatenate((X_train, X_test), axis=0)
        fX = np.concatenate((fX_train, fX_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        # Here we split the data up to train, test splits
        data = DataInstance(D=D, N=self.n_samples, X=X, fX=fX, z=z, y=y)

        train_data = DataInstance(D=D, N=len(X_train), X=X_train, fX=fX_train, z=z_train, y=y_train)
        test_data = DataInstance(D=D, N=len(X_test), X=X_test, fX=fX_test, z=z_test, y=y_test)

        return train_data, test_data, data

    def load_manual(self, filename, has_header, predict_index):
        dataset_path = self.uci_loader.get_path(filename)
        with open(dataset_path) as data_csvfile:
            reader = csv.reader(data_csvfile)
            dataset = [ row for row in reader ]

        if has_header:
            dataset = dataset[1:]

        dataset = np.array(dataset, dtype=np.float64)

        if self.n_samples != None:
            choices = self.rng.choice(len(dataset), size=self.n_samples, replace=False)
            dataset = dataset[choices]

        fX = dataset[:,predict_index]
        X = dataset[:,1:]
        D = len(X[0])

        return D, X, fX
    
    def generate_X_fX(self, **fn_kwargs):
        uci_dataset = UCIDataset(base_path=self.uci_loader.get_path('uci_datasets'), **fn_kwargs)
        
        assert(uci_dataset.y.shape[1]==1)

        X = uci_dataset.x
        D = X.shape[1]
        fX = uci_dataset.y.reshape(-1)

        if self.n_samples != None:
            choices = self.rng.choice(X.shape[0], size=self.n_samples, replace=False)

            X = X[choices]
            fX = fX[choices]

        return D, X, fX