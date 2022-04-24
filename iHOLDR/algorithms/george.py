#!/usr/bin/env python3
import logging
import numpy as np
import george
import scipy.optimize as op
from sklearn.decomposition import PCA, KernelPCA
import sklearn
import matplotlib.pyplot as plt

from algorithms.sklearn import SklearnGP
from algorithms.commonGP import CommonGP

class GeorgeGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'metric'
    }
    def __init__(self, solver, kernel, sk_kwargs={}, rearrange_fn='rearrange_placebo', rearrange_kwargs={}, rearrange_opt_rerun=False, **kwargs):
        super().__init__(**kwargs)

        self.scale_variance = self.kernel_kwargs.pop('scale_variance')
        self.yerr = np.sqrt(self.kernel_kwargs.pop('noise_variance')) * np.ones_like(self.train_data.y)

        self.Kernel = getattr(george.kernels, kernel)
        self.Solver = getattr(george, solver)

        self.rearrange_fn = getattr(self, rearrange_fn)
        self.rearrange_kwargs = rearrange_kwargs
        self.rearrange_opt_rerun = rearrange_opt_rerun

        if sk_kwargs != {}:
            sk_gp = SklearnGP(**kwargs, **sk_kwargs)
            self.Sk_Kernel = sk_gp.Kernel

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
        opt_kernel_params = self.optimize_hypers(kernel, model)
        if self.rearrange_opt_rerun:
            self.rearrange_fn(model, **self.rearrange_kwargs)

        y_predicted, y_predicted_confidence = model.predict(self.train_data.y, X, return_var=False)

        return y_predicted, model.log_likelihood(self.train_data.y), opt_kernel_params
    def optimize_hypers(self, kernel, model):

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
        return opt_kernel_params

    # No rearrangement
    def rearrange_placebo(self, model, **rearrange_kwargs):
        pass

    def rearrange_la_pca(self, model, n_components=None):
        self.rearrange_la_pca_sk(model, n_components)

    def rearrange_la_kpca(self, model, n_components=None):
        self.rearrange_la_kpca_sk(model, n_components)
        #self.rearrange_la_kpca_np(model, n_components)

    def rearrange_la_pca_sk(self, model, n_components):

        pca = PCA(n_components=n_components)
        pca_X = pca.fit_transform(self.train_data.X)
        n_components = self.choose_n_eigvals(pca.singular_values_**2, n_components)
        
        idx = self.recursive_sort(pca_X, n_components)
        self.train_data.rearrange(idx)

    def rearrange_la_kpca_sk(self, model, n_components):

        scale_variance, length_scale = np.exp(model.get_parameter_vector())
        kernel = scale_variance * self.Sk_Kernel(length_scale=length_scale)

        pca = KernelPCA(kernel=kernel, n_components=n_components)
        pca_X = pca.fit_transform(self.train_data.X)
        n_components = self.choose_n_eigvals(pca.eigenvalues_, n_components)

        idx = self.recursive_sort(pca_X, n_components)
        self.train_data.rearrange(idx)

    def rearrange_la_pca_np(self, model, n_components):

        # https://colab.research.google.com/github/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb#scrollTo=Ggr2A2Ew54k0
        X = self.train_data.X

        X_centered = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(X_centered)
        
        num_components = 1
        n_components = self.choose_n_eigvals(s**2, n_components)
        eig_components = Vt.T[:, :num_components]
        _pca_X = X_centered.dot(eig_components)
        idx = np.argsort(_pca_X.reshape(-1))

        self.train_data.rearrange(idx)

    def rearrange_la_kpca_np(self, model, n_components):

        # https://opendatascience.com/implementing-a-kernel-principal-component-analysis-in-python/

        K = model.get_matrix(self.train_data.X)
        # Center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)    

        # Obtaining eigenpairs from the centered kernel matrix
        # scipy.linalg.eigh returns them in ascending order
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

        n_components = self.choose_n_eigvals(eigvals, n_components)
            
        # Collect the top k eigenvectors (projected examples)
        pca_X = np.column_stack([eigvecs[:, i] for i in range(n_components)])

        idx = self.recursive_sort(pca_X, n_components)
        
        self.train_data.rearrange(idx)
    
    def choose_n_eigvals(self, eigvals, n_components):
        if n_components == None:
            n_components = len(eigvals)
        elif n_components < 1:
            var_explained = np.cumsum(eigvals / np.sum(eigvals))
            n_components = np.searchsorted(var_explained, n_components) + 1
        else:
            assert(n_components <= len(eigvals))
        return n_components

    def recursive_sort(self, pca_X, n_components):
        submatrices_idx = [ np.argsort(pca_X[:, 0].reshape(-1)) ]
        for component_i in range(1, n_components):
            submatrices_idx = sum([ np.array_split(submatrix_ids, 2) for submatrix_ids in submatrices_idx ], [])
            # print([len(s) for s in submatrices_idx])
            submatrices_idx = [ submatrix_idx[np.argsort(pca_X[submatrix_idx, component_i].reshape(-1))] for submatrix_idx in submatrices_idx ]

        idx = np.concatenate(submatrices_idx)
        return idx

    def clean_up(self):
        self.train_data.restore()

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

        if self.rearrange_opt_rerun:
            self.optimize_hypers(kernel, model)
            self.rearrange_fn(model, **self.rearrange_kwargs)

            self.plot_KXX(model.get_matrix(self.train_data.X), "george/afteropt_kXX.png")