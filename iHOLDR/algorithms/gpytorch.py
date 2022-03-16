#!/usr/bin/env python3
import torch
import gpytorch
import gc
import numpy as np
import logging

# Implementation of LBFGS is from git@github.com:hjmshi/PyTorch-LBFGS.git
from LBFGS import FullBatchLBFGS
from algorithms.commonGP import CommonGP

class ExactAlexanderGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices, output_device):
        super(ExactAlexanderGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Author implementation of https://arxiv.org/abs/1903.08114
class PyTorchAlexanderGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'covar_module.module.base_kernel.lengthscale',
        'noise_variance':'likelihood.noise_covar.noise',
        'scale_variance':'covar_module.module.outputscale'
    }
    def __init__(self, output_device, n_devices, preconditioner_size, **kwargs):
        super().__init__(**kwargs)

        self.output_device = torch.device(output_device)
        self.train_x = torch.from_numpy(self.data.X).to(self.output_device)
        self.train_y = torch.from_numpy(self.data.y).to(self.output_device)
        self.n_devices = n_devices
        # Set a large enough preconditioner size to reduce the number of CG iterations run
        self.preconditioner_size = preconditioner_size
        self.checkpoint_size = self.find_best_gpu_setting()

    def compute_log_likelihood(self):

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.output_device)
        model = ExactAlexanderGPModel(self.train_x, self.train_y, likelihood, self.n_devices, self.output_device).to(self.output_device)
        model.initialize(**self.kernel_kwargs)

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(self.checkpoint_size), \
            gpytorch.settings.max_preconditioner_size(self.preconditioner_size), gpytorch.settings.fast_computations(log_prob=True):

            log_likelihood = model.likelihood(model(self.train_x)).log_prob(self.train_y).item()
            return log_likelihood

        # model, likelihood = self.train(checkpoint_size=self.checkpoint_size, n_training_iter=20)
        # logging.info(model.likelihood(model(self.train_x)).log_prob(self.train_y))

    def train(self, checkpoint_size, n_training_iter):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.output_device)
        model = ExactAlexanderGPModel(self.train_x, self.train_y, likelihood, self.n_devices, self.output_device).to(self.output_device)
        model.train()
        likelihood.train()

        optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
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

            for i in range(n_training_iter):
                options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                loss, _, _, _, _, _, _, fail = optimizer.step(options)

                logging.info('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, n_training_iter, loss.item(),
                    model.covar_module.module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))

                if fail:
                    logging.info('Convergence reached!')
                    break

        logging.info(f"Finished training on {self.train_x.size(0)} data points using {self.n_devices} GPUs.")
        return model, likelihood
    def find_best_gpu_setting(self):
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

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class PyTorchGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'covar_module.module.base_kernel.lengthscale',
        'noise_variance':'likelihood.noise_covar.noise',
        'scale_variance':'covar_module.module.outputscale'
    }
    def __init__(self, output_device, n_devices, preconditioner_size, **kwargs):
        super().__init__(**kwargs)
        self.output_device = torch.device(output_device)
        self.train_x = torch.from_numpy(self.data.X).to(self.output_device)
        self.train_y = torch.from_numpy(self.data.y).to(self.output_device)
        self.n_devices = n_devices
    def compute_log_likelihood(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.train_x, self.train_y, likelihood)
        model.initialize(**self.kernel_kwargs)
        with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=True):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            logging.info(model.likelihood(model(train_x)).log_prob(train_y))
