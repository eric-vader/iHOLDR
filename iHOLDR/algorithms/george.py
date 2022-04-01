#!/usr/bin/env python3
import logging
import numpy as np
import george

from algorithms.commonGP import CommonGP

class GeorgeGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'metric'
    }
    def __init__(self, solver, kernel, **kwargs):
        super().__init__(**kwargs)

        self.scale_variance = self.kernel_kwargs.pop('scale_variance')
        self.yerr = np.sqrt(self.kernel_kwargs.pop('noise_variance')) * np.ones_like(self.train_data.y)

        self.Kernel = getattr(george.kernels, kernel)
        self.Solver = getattr(george, solver)

    def compute_log_likelihood(self):
        kernel = self.scale_variance * self.Kernel(**self.kernel_kwargs)
        idx = np.array(range(self.train_data.N))
        model = george.GP(kernel, solver=self.Solver)
        model.compute(self.train_data.X, self.yerr)

        return model.log_likelihood(self.train_data.y)

# https://george.readthedocs.io/en/latest/tutorials/hyper/