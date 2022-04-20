#!/usr/bin/env python3
import logging
import numpy as np
import george
import scipy.optimize as op

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
        kernel = self.scale_variance * self.Kernel(ndim=self.train_data.D, **self.kernel_kwargs)
        idx = np.array(range(self.train_data.N))
        model = george.GP(kernel, solver=self.Solver)
        model.compute(self.train_data.X, self.yerr)

        return model.log_likelihood(self.train_data.y)
    
    def predict(self, X):

        kernel = self.scale_variance * self.Kernel(ndim=self.train_data.D, **self.kernel_kwargs)
        idx = np.array(range(self.train_data.N))
        model = george.GP(kernel, solver=self.Solver)

        # https://george.readthedocs.io/en/latest/tutorials/hyper/

        # You need to compute the GP once before starting the optimization.
        model.compute(self.train_data.X, self.yerr)

        # Define the objective function (negative log-likelihood in this case).
        def nll(p):
            model.set_parameter_vector(p)
            ll = model.log_likelihood(self.train_data.y, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            model.set_parameter_vector(p)
            return -model.grad_log_likelihood(self.train_data.y, quiet=True)

        # Run the optimization routine.
        p0 = model.get_parameter_vector()
        results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

        # Update the kernel and print the final log-likelihood.
        model.set_parameter_vector(results.x)
        opt_kernel_params = tuple(np.exp(kernel.get_parameter_vector()))

        y_predicted, y_predicted_confidence = model.predict(self.train_data.y, X, return_var=False)

        return y_predicted, model.log_likelihood(self.train_data.y), opt_kernel_params