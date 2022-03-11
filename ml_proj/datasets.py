#!/usr/bin/env python3
import common
import logging

import numpy as np
import george
import math

class Dataset(common.Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)

class ExampleDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)

        N = 200

        x = self.rng.uniform(0, 10, N)
        noise = 0.3
        sigma_err = noise ** 2
        yerr = noise * np.ones_like(x)
        y = np.sin(x)
        var_y = np.var(y)

        self.N = N
        self.X = x
        self.y = y
        self.var_y = var_y
        self.yerr = yerr
