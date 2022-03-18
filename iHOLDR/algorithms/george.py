#!/usr/bin/env python3

import logging
import numpy as np

import george
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor

import torch
import gpytorch
import gc

from algorithms.commonGP import CommonGP

class GeorgeGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'metric'
    }
    def __init__(self, solver, kernel, **kwargs):
        super().__init__(**kwargs)

        N = self.data.N
        X = self.data.X
        y = self.data.y

        var_y = self.kernel_kwargs.pop('scale_variance')
        noise_variance = self.kernel_kwargs.pop('noise_variance')

        noise = np.sqrt(noise_variance)
        yerr = noise * np.ones_like(X)

        kernel = var_y * getattr(george.kernels, kernel)(**self.kernel_kwargs)

        idx = np.array(range(N))
        model = george.GP(kernel, solver=getattr(george, solver))
        model.compute(X[idx], yerr[idx])
        logging.info(model.log_likelihood(y[idx]))

        self.data.reshape()
        N = self.data.N
        X = self.data.X
        y = self.data.y

        kernel_skl = var_y * getattr(sklearn.gaussian_process.kernels, "RBF")(length_scale=1.0)
        gp_skl = GaussianProcessRegressor(kernel_skl,
                                        alpha=noise_variance,
                                        optimizer=None,
                                        copy_X_train=False)
        gp_skl.fit(X, y)
        logging.info(gp_skl.log_marginal_likelihood(kernel_skl.theta))

        kernel_gpy = GPy.kern.RBF(input_dim=1, variance=var_y, lengthscale=1.)
        gp_gpy = GPy.models.GPRegression(X, y, kernel_gpy)
        gp_gpy.Gaussian_noise.fix(noise_variance)
        logging.info(gp_gpy.log_likelihood())
        
    def compute_log_likelihood(self):
        pass