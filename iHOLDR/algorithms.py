#!/usr/bin/env python3
import common
import logging
import numpy as np

import george
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
import GPy
import torch
import gpytorch


class ExampleAlgorithm(common.Component):
    def __init__(self, datasets, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)
        logging.info(datasets)

class CommonGP(common.Component):
    kernel_kwargs_mapper = {}
    def kernel_kwargs_adaptor(self, kernel_kwargs):
        return { (self.kernel_kwargs_mapper[k] if k in self.kernel_kwargs_mapper else k) : v for k,v in kernel_kwargs.items()  }
    def __init__(self, datasets, kernel_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)
        
        self.datasets = datasets
        self.data = datasets.generate_data()

        if kernel_kwargs['variance'] == 'population':
            kernel_kwargs['variance'] = np.var(self.data.y)
        self.kernel_kwargs = self.kernel_kwargs_adaptor(kernel_kwargs)

class GeorgeGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'metric'
    }
    def __init__(self, solver, kernel, noise_variance, **kwargs):
        super().__init__(**kwargs)

        noise = np.sqrt(noise_variance)

        N = self.data.N
        X = self.data.X
        y = self.data.y

        yerr = noise * np.ones_like(X)

        var_y = self.kernel_kwargs.pop('variance')

        kernel = var_y * getattr(george.kernels, kernel)(**self.kernel_kwargs)

        idx = np.array(range(N))
        _gp_hodlr = george.GP(kernel, solver=getattr(george, solver))
        _gp_hodlr.compute(X[idx], yerr[idx])
        logging.info(_gp_hodlr.log_likelihood(y[idx]))

        gp_basic = george.GP(kernel)
        gp_basic.compute(X, yerr)
        logging.info(gp_basic.log_likelihood(y))

        self.data.reshape()

        train_x = torch.from_numpy(X)
        train_y = torch.from_numpy(y)
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)
        hypers = {
            'likelihood.noise_covar.noise': noise_variance,
            'covar_module.base_kernel.lengthscale': 1.0,
            'covar_module.outputscale': var_y,
        }
        model.initialize(**hypers)
        with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=True):
            logging.info(model.likelihood(model(train_x)).log_prob(train_y))

        try:
            train_x = torch.from_numpy(X).float().cuda()
            train_y = torch.from_numpy(y).float().cuda()
            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            model = ExactGPModel(train_x, train_y, likelihood)
            hypers = {
                'likelihood.noise_covar.noise': noise_variance,
                'covar_module.base_kernel.lengthscale': 1.0,
                'covar_module.outputscale': var_y,
            }
            model.initialize(**hypers)
            model = model.cuda()
            with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=True):
                logging.info(model.likelihood(model(train_x)).log_prob(train_y))
        except RuntimeError as e:
            logging.exception(e)

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

