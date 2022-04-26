#!/usr/bin/env python3
import gpflow
import numpy as np

from gpflow.utilities import set_trainable

from algorithms.commonGP import CommonGP

class GPflowGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'lengthscales',
        'noise_variance':'noise_variance',
        'scale_variance':'variance'
    }
    def adapt_data(self, data):
        data.y = data.y[:,None]
        return data

    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.Kernel = getattr(gpflow.kernels, kernel)
        self.noise_variance = self.kernel_kwargs.pop('noise_variance')

    def make_model(self):
        kernel = self.Kernel(**self.kernel_kwargs)
        model = gpflow.models.GPR(data=(self.train_data.X, self.train_data.y), kernel=kernel, mean_function=None)
        model.likelihood.variance.assign(self.noise_variance)
        set_trainable(model.likelihood.variance, False)
        return model

    def compute_log_likelihood(self): 
        return self.make_model().log_marginal_likelihood().numpy()

    def predict(self, X, perform_opt):
        model = self.make_model()

        if perform_opt:
            scipy_opt = gpflow.optimizers.Scipy()
            scipy_opt.minimize(model.training_loss, model.trainable_variables, **self.optimizer_kwargs)

        y_predicted, y_predicted_confidence = model.predict_f(X)
        opt_kernel_params = (np.float64(model.kernel.variance.numpy()), np.float64(model.kernel.lengthscales.numpy()))
        
        return y_predicted.numpy(), model.log_marginal_likelihood().numpy(), opt_kernel_params
