#!/usr/bin/env python3
import numpy as np
import logging

import common

class CommonGP(common.Component):
    kernel_kwargs_mapper = {}
    def kernel_kwargs_adaptor(self, kernel_kwargs):
        return { (self.kernel_kwargs_mapper[k] if k in self.kernel_kwargs_mapper else k) : v for k,v in kernel_kwargs.items()  }
    def __init__(self, datasets, kernel_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)
        
        self.datasets = datasets
        self.data = datasets.generate_data()

        if kernel_kwargs['scale_variance'] == 'population':
            kernel_kwargs['scale_variance'] = np.var(self.data.y)
        self.kernel_kwargs = self.kernel_kwargs_adaptor(kernel_kwargs)
       
    # Measured, but must return a python float
    def compute_log_likelihood(self):
        raise NotImplementedError

    # Measured, but must return a tuple of hypers (lengthscale, scale_variance, noise_variance)
    def optimize_hypers(self):
        raise NotImplementedError

    def __call__(self):

        # Time this and report the result
        nll = self.compute_log_likelihood()
        logging.info(nll)