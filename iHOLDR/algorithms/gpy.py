#!/usr/bin/env python3
import GPy
import numpy as np
import logging

from algorithms.commonGP import CommonGP

class GPyGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'lengthscale',
        'noise_variance':'noise_variance',
        'scale_variance':'variance'
    }
    def adapt_data(self, data):
        data.y = data.y[:,None]
        return data
        
    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.Kernel = getattr(GPy.kern, kernel)
        self.kernel_kwargs['input_dim'] = self.train_data.D
        self.noise_variance = self.kernel_kwargs.pop('noise_variance')

    def compute_log_likelihood(self):
        kernel = self.Kernel(**self.kernel_kwargs)
        model = GPy.models.GPRegression(self.train_data.X, self.train_data.y, kernel)
        model.Gaussian_noise.fix(self.noise_variance)
        return model.log_likelihood()