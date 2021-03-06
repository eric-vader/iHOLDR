#!/usr/bin/env python3
import GPy
import numpy as np
import logging

from algorithms.commonGP import CommonGP

class SVGPRegression(GPy.core.SVGP):
    def __init__(self, X, Y, kernel=None, Z=None, batchsize=None, num_latent_functions=None, mean_function=None, mpi_comm=None, name='SVGPRegression'):
        num_data, input_dim = X.shape

        # kern defaults to rbf (plus white for stability)
        if kernel is None:
            kernel = GPy.kern.RBF(input_dim)#  + kern.white(input_dim, variance=1e-3)

        likelihood = GPy.likelihoods.Gaussian()

        super(SVGPRegression, self).__init__(X, Y, Z, kernel, likelihood, mean_function=mean_function,
        batchsize=None, num_latent_functions=None, name=name)

GPy.models.SVGPRegression = SVGPRegression


class GPyGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'lengthscale',
        'noise_variance':'noise_variance',
        'scale_variance':'variance'
    }
    def adapt_data(self, data):
        data.y = data.y[:,None]
        return data
        
    def __init__(self, kernel, model='GPRegression', model_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self.Kernel = getattr(GPy.kern, kernel)
        self.kernel_kwargs['input_dim'] = self.train_data.D
        self.noise_variance = self.kernel_kwargs.pop('noise_variance')

        self.Model = getattr(GPy.models, model)
        self.model_kwargs = model_kwargs
        if 'Z' in self.model_kwargs:
            self.num_inducing = min(self.model_kwargs['Z'], self.train_data.N)
            ix = self.rng.permutation(self.train_data.N)[:self.num_inducing]
            self.model_kwargs['Z'] = self.train_data.X.view(np.ndarray)[ix].copy()
        
        if model != "GPRegression":
            # Sparse if not GPR, remove the limiter
            self.sufficient_resources = self.sufficient_resources_Sparse

    def sufficient_resources_Sparse(self):
        # O(nm)
        n_bytes = self.data.X.size * self.num_inducing * self.train_data.X.itemsize
        X_MB = int((n_bytes)/(10**6))
        return X_MB < self.free_MB

    def make_model(self):
        np.random.seed(self.random_seed)
        kernel = self.Kernel(**self.kernel_kwargs)
        model = self.Model(self.train_data.X, self.train_data.y, kernel, **self.model_kwargs)
        model.Gaussian_noise.fix(self.noise_variance)
        return model, kernel

    def compute_log_likelihood(self):
        model, kernel = self.make_model()
        return np.float64(model.log_likelihood())

    def predict(self, X, perform_opt):
        model, kernel = self.make_model()
        if perform_opt:
            model.optimize(ipython_notebook=False, **self.optimizer_kwargs)

        y_predicted, y_predicted_confidence = model.predict(X)
        opt_kernel_params = (np.float64(kernel.variance.values), np.float64(kernel.lengthscale.values))

        return y_predicted, np.float64(model.log_likelihood()), opt_kernel_params