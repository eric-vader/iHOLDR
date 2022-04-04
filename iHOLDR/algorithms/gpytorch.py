#!/usr/bin/env python3
import torch
import gpytorch
import gc
import numpy as np
import logging

# Implementation of LBFGS is from git@github.com:hjmshi/PyTorch-LBFGS.git
from algorithms.LBFGS import FullBatchLBFGS
from algorithms.commonGP import CommonGP

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, Kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(Kernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    def get_lengthscale(self):
        return self.covar_module.base_kernel.lengthscale
    def get_outputscale(self):
        return self.covar_module.outputscale
    def get_noisevariance(self):
        return self.likelihood.noise_covar.noise

class PyTorchGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'covar_module.base_kernel.lengthscale',
        'noise_variance':'likelihood.noise_covar.noise',
        'scale_variance':'covar_module.outputscale'
    }
    def __init__(self, output_device, kernel, **kwargs):
        super().__init__(**kwargs)
        self.output_device = torch.device(output_device)
        self.train_x = torch.from_numpy(self.train_data.X).to(self.output_device)
        self.train_y = torch.from_numpy(self.train_data.y).to(self.output_device)
        self.Kernel = getattr(gpytorch.kernels, kernel)
    def compute_log_likelihood(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.output_device)
        model = ExactGPModel(self.train_x, self.train_y, likelihood, self.Kernel).to(self.output_device)
        model.initialize(**self.kernel_kwargs)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            log_likelihood = model.likelihood(model(self.train_x)).log_prob(self.train_y).cpu().numpy()
            return float(log_likelihood)

    def predict(self, X):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.output_device)
        model = ExactGPModel(self.train_x, self.train_y, likelihood, self.Kernel).to(self.output_device)
        model.initialize(**self.kernel_kwargs)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            log_likelihood = model.likelihood(model(self.train_x)).log_prob(self.train_y).cpu().numpy()
            return float(log_likelihood)

    def train(self):
        pass
        

class ExactAlexanderGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, Kernel, n_devices, output_device):
        super(ExactAlexanderGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(Kernel())

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    def get_lengthscale(self):
        return self.covar_module.module.base_kernel.lengthscale
    def get_outputscale(self):
        return self.covar_module.module.outputscale
    def get_noisevariance(self):
        return self.likelihood.noise_covar.noise

# Author implementation of https://arxiv.org/abs/1903.08114
class PyTorchAlexanderGP(PyTorchGP):
    kernel_kwargs_mapper = {
        'lengthscale':'covar_module.module.base_kernel.lengthscale',
        'noise_variance':'likelihood.noise_covar.noise',
        'scale_variance':'covar_module.module.outputscale'
    }
    def __init__(self, output_device, preconditioner_size, n_devices, **kwargs):
        if output_device == "cpu":
            PyTorchAlexanderGP.kernel_kwargs_mapper = {
                'lengthscale':'covar_module.base_kernel.lengthscale',
                'noise_variance':'likelihood.noise_covar.noise',
                'scale_variance':'covar_module.outputscale'
            }
        super().__init__(output_device=output_device, **kwargs)

        self.n_devices = n_devices

        # Set a large enough preconditioner size to reduce the number of CG iterations run
        self.preconditioner_size = preconditioner_size
        self.checkpoint_size = self.find_best_partition_setting()

    def compute_log_likelihood(self):

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.output_device)
        model = self.create_model(likelihood)

        # gpytorch.settings.fast_pred_var(), 
        with torch.no_grad(), gpytorch.beta_features.checkpoint_kernel(self.checkpoint_size), \
            gpytorch.settings.max_preconditioner_size(self.preconditioner_size), gpytorch.settings.fast_computations(log_prob=True):

            log_likelihood = model.likelihood(model(self.train_x)).log_prob(self.train_y).cpu().numpy()
            return np.float64(log_likelihood)
    def predict(self, X):

        model, likelihood = self.train(checkpoint_size=self.checkpoint_size, n_training_iter=15000)

        with torch.no_grad(), gpytorch.beta_features.checkpoint_kernel(self.checkpoint_size), \
            gpytorch.settings.max_preconditioner_size(self.preconditioner_size), gpytorch.settings.fast_computations(log_prob=True):

            opt_log_likelihood = model.likelihood(model(self.train_x)).log_prob(self.train_y).cpu().numpy()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.beta_features.checkpoint_kernel(self.checkpoint_size), \
            gpytorch.settings.max_preconditioner_size(self.preconditioner_size), gpytorch.settings.fast_computations(log_prob=True):
            
            y_predicted_latent = model(torch.from_numpy(X).to(self.output_device))
            y_predicted = y_predicted_latent.mean.cpu().numpy()

            opt_kernel_params = np.float64(model.get_outputscale().cpu().numpy()), np.float64(model.get_lengthscale().cpu().numpy())

        return y_predicted, opt_log_likelihood, opt_kernel_params

    def create_model(self, likelihood):
        if str(self.output_device) == "cpu":
            model = ExactGPModel(self.train_x, self.train_y, likelihood, self.Kernel).to(self.output_device)
            model.initialize(**self.kernel_kwargs)
        else:
            model = ExactAlexanderGPModel(self.train_x, self.train_y, likelihood, self.Kernel, self.n_devices, self.output_device).to(self.output_device)
            model.initialize(**self.kernel_kwargs)
        return model

    def train(self, checkpoint_size, n_training_iter):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.output_device)
        model = self.create_model(likelihood)
        likelihood.train()
        model.train()

        parameters = []
        for name, param in model.named_parameters():
            if name.startswith('likelihood'):
                param.requires_grad = False
                continue
            parameters.append(param)

        optimizer = FullBatchLBFGS(parameters, lr=0.1)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
            gpytorch.settings.max_preconditioner_size(self.preconditioner_size):

            def closure():
                optimizer.zero_grad()
                output = model(self.train_x)
                loss = -mll(output, self.train_y)
                return loss

            loss = closure()
            loss.backward()

            i = -1
            logging.info('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   outputscale: %.3f   noise: %.5f' % (
                i + 1, n_training_iter, loss.item(),
                model.get_lengthscale().item(),
                model.get_outputscale().item(),
                model.likelihood.noise.item()
            ))

            for i in range(n_training_iter):
                options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                loss, _, _, _, _, _, _, fail = optimizer.step(options)

                logging.info('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   outputscale: %.3f   noise: %.5f' % (
                    i + 1, n_training_iter, loss.item(),
                    model.get_lengthscale().item(),
                    model.get_outputscale().item(),
                    model.likelihood.noise.item()
                ))

                if fail:
                    logging.info('Convergence reached!')
                    break

        logging.info(f"Finished training on {self.train_x.size(0)} data points using {self.n_devices} {self.output_device}s.")
        return model, likelihood
    def find_best_partition_setting(self):
        N = self.train_x.size(0)

        # Find the optimum partition/checkpoint size by decreasing in powers of 2
        # Start with no partitioning (size = 0)
        settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

        for checkpoint_size in settings:
            logging.info('Number of devices: {} -- Kernel partition size: {}'.format(self.n_devices, checkpoint_size))
            try:
                # Try a full forward and backward pass with this setting to check memory usage
                _, _ = self.train(checkpoint_size=checkpoint_size, n_training_iter=1)

                # when successful, break out of for-loop and jump to finally block
                break
            except RuntimeError as e:
                logging.error('RuntimeError: {}'.format(e))
            except AttributeError as e:
                logging.error('AttributeError: {}'.format(e))
            finally:
                # handle CUDA OOM error
                gc.collect()
                torch.cuda.empty_cache()
        return checkpoint_size