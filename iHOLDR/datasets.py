#!/usr/bin/env python3
import common
import logging

import numpy as np
import george
import math
from dataclasses import dataclass

@dataclass
class DataInstance:
    X: np.ndarray
    y: np.ndarray
    N: int
    def reshape(self):
        self.X = self.X[:,None]
        self.y = self.y[:,None]
    def 

class Dataset(common.Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)

class ExampleDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)

        N = 200

        X = self.rng.uniform(-10, 10, N)
        noise = 0.3
        sigma_err = noise ** 2
        yerr = noise * np.ones_like(X)
        y = np.sin(X)
        var_y = np.var(y)

        self.data = DataInstance(X, y, N)
        

        print(len(yerr))
        
        self.N = N
        self.X = X
        self.y = y
        self.var_y = var_y
        self.yerr = yerr
