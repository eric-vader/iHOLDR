#!/usr/bin/env python3
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor

from algorithms.commonGP import CommonGP

class SklearnGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'length_scale'
    }
    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.Kernel = getattr(sklearn.gaussian_process.kernels, kernel)
        self.scale_variance = self.kernel_kwargs.pop('scale_variance')
        self.noise_variance = self.kernel_kwargs.pop('noise_variance')
        
    def compute_log_likelihood(self):

        kernel = self.scale_variance * self.Kernel(**self.kernel_kwargs)
        model = GaussianProcessRegressor(kernel,
                                        alpha=self.noise_variance,
                                        optimizer=None,
                                        copy_X_train=False)
        model.fit(self.data.X, self.data.y)
        return model.log_marginal_likelihood(kernel.theta)