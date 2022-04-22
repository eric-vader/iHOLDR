#!/usr/bin/env python3
import logging
import numpy as np
import george
import scipy.optimize as op
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from algorithms.commonGP import CommonGP

class GeorgeGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'metric'
    }
    def __init__(self, solver, kernel, rearrange_fn='rearrange_placebo', rearrange_kwargs={}, **kwargs):
        super().__init__(**kwargs)

        self.scale_variance = self.kernel_kwargs.pop('scale_variance')
        self.yerr = np.sqrt(self.kernel_kwargs.pop('noise_variance')) * np.ones_like(self.train_data.y)

        self.Kernel = getattr(george.kernels, kernel)
        self.Solver = getattr(george, solver)

        self.rearrange_fn = getattr(self, rearrange_fn)
        self.rearrange_kwargs = rearrange_kwargs

    def compute_log_likelihood(self):

        kernel = self.scale_variance * self.Kernel(ndim=self.train_data.D, **self.kernel_kwargs)
        model = george.GP(kernel, solver=self.Solver)
        self.rearrange_fn(model, **self.rearrange_kwargs)

        model.compute(self.train_data.X, self.yerr)

        return model.log_likelihood(self.train_data.y)
    
    def predict(self, X):

        kernel = self.scale_variance * self.Kernel(ndim=self.train_data.D, **self.kernel_kwargs)
        model = george.GP(kernel, solver=self.Solver)
        self.rearrange_fn(model, **self.rearrange_kwargs)

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
        results = op.minimize(nll, p0, jac=grad_nll, **self.optimizer_kwargs)

        # Update the kernel and print the final log-likelihood.
        model.set_parameter_vector(results.x)
        opt_kernel_params = tuple(np.exp(kernel.get_parameter_vector()))

        y_predicted, y_predicted_confidence = model.predict(self.train_data.y, X, return_var=False)

        return y_predicted, model.log_likelihood(self.train_data.y), opt_kernel_params

    # No rearrangement
    def rearrange_placebo(self, model, **rearrange_kwargs):
        self.prev_idx = np.array(range(self.train_data.N))

    def rearrange_la_pca(self, model, **rearrange_kwargs):

        self.prev_idx = np.array(range(self.train_data.N))

        pca = PCA(n_components=1)
        _pca_X = pca.fit_transform(self.train_data.X)

        # print(pca.explained_variance_ratio_)
        # print(pca.singular_values_)
        # print(np.dot(_X[0],pca.components_))
        idx = np.argsort(_pca_X.reshape(-1))

        self.train_data.rearrange(idx)
        self.prev_idx[:] = self.prev_idx[idx]

    def rearrange_la_kpca(self, model, **rearrange_kwargs):

        # https://colab.research.google.com/github/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb#scrollTo=Ggr2A2Ew54k0
        # https://opendatascience.com/implementing-a-kernel-principal-component-analysis-in-python/

        self.prev_idx = np.array(range(self.train_data.N))
        data = self.train_data.X
        
        cov_mat = np.cov(data.T)
        
        if data.shape[1] == 1:
            eig_val, eig_vec = np.array([cov_mat]), np.array([[cov_mat]])
        else:
            eig_val, eig_vec = np.linalg.eig(cov_mat)
        
        eig_idx_sort = np.flip(eig_val.argsort()) 
        eig_components = eig_vec[:,eig_idx_sort]
        
        num_components = 1
        _pca_X = data @ eig_components[:,:num_components]
        
        idx = np.argsort(_pca_X.reshape(-1))
        self.train_data.rearrange(idx)
        self.prev_idx[:] = self.prev_idx[idx]

    def clean_up(self):
        self.train_data.rearrange(self.prev_idx)

    def plot_KXX(self, KXX, file_name):
        figure = plt.figure()
        axes = figure.add_subplot(111)
        
        caxes = axes.matshow(KXX, interpolation ='nearest')
        figure.colorbar(caxes)

        self.mlflow_logger.log_figure(figure, file_name)

    def visualize(self):

        kernel = self.scale_variance * self.Kernel(ndim=self.train_data.D, **self.kernel_kwargs)
        model = george.GP(kernel, solver=self.Solver)
        self.plot_KXX(model.get_matrix(self.train_data.X), "george/before_kXX.png")

        self.rearrange_fn(model, **self.rearrange_kwargs)
        self.plot_KXX(model.get_matrix(self.train_data.X), "george/after_kXX.png")