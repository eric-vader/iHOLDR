#!/usr/bin/env python3
import gpflow
import numpy as np
import logging

from algorithms.commonGP import CommonGP

class GPflowGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'lengthscales',
        'noise_variance':'noise_variance',
        'scale_variance':'variance'
    }
    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.Kernel = getattr(gpflow.kernels, kernel)
        self.noise_variance = self.kernel_kwargs.pop('noise_variance')
    
        self.train_data.y = self.train_data.y[:,None]

    def compute_log_likelihood(self):
        kernel = self.Kernel(**self.kernel_kwargs)
        model = gpflow.models.GPR(data=(self.train_data.X, self.train_data.y), kernel=kernel)
        model.likelihood.variance.assign(self.noise_variance)
        return model.log_marginal_likelihood().numpy()