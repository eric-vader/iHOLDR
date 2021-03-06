#!/usr/bin/env python3
import sklearn
import logging

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
        model = self.make_model()
        return model.log_marginal_likelihood(model.kernel.theta)

    def make_model(self, optimizer_kwargs={'optimizer':None}):
        kernel = self.scale_variance * self.Kernel(**self.kernel_kwargs)
        model = GaussianProcessRegressor(kernel,
                                        alpha=self.noise_variance,
                                        copy_X_train=False,
                                        random_state=self.random_seed,
                                        **optimizer_kwargs)
        model.fit(self.train_data.X, self.train_data.y)
        return model

    def predict(self, X, perform_opt):

        if perform_opt:
            model = self.make_model(self.optimizer_kwargs)
        else:
            model = self.make_model()
        
        # logging.info(f"Kernel parameters before fit: {kernel}")
        # logging.info(f"Kernel parameters after fit: {model.kernel_}")
        # mll for model after model.log_marginal_likelihood(model.kernel_.theta)
        # print(model.kernel_.k1.constant_value, model.kernel_.k2.length_scale)
        # print(type(model.kernel_.k2.length_scale))
        opt_kernel_params = (model.kernel_.k1.constant_value, model.kernel_.k2.length_scale)

        return model.predict(X), model.log_marginal_likelihood(model.kernel_.theta), opt_kernel_params