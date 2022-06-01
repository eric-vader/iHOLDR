#!/usr/bin/env python3
import logging
import numpy as np
import george
import scipy.optimize as op
from sklearn.decomposition import PCA, KernelPCA
import sklearn
import matplotlib.pyplot as plt
from functools import reduce
import networkx as nx
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import math
import copy
from scipy.spatial import cKDTree

from algorithms.sklearn import SklearnGP
from algorithms.commonGP import CommonGP
from algorithms.hsort import HSort

class GeorgeGP(CommonGP):
    kernel_kwargs_mapper = {
        'lengthscale':'metric'
    }
    def __init__(self, solver, kernel, re_rearrange=False, model_kwargs={}, rearrange_fn='rearrange_placebo', rearrange_kwargs={}, is_plot_KXX=True, **kwargs):
        super().__init__(**kwargs)

        self.scale_variance = self.kernel_kwargs.pop('scale_variance')
        noise_variance = self.kernel_kwargs.pop('noise_variance')
        # Kernel takes in ^2 here at George....
        self.kernel_kwargs['metric'] = self.kernel_kwargs['metric']**2
        self.white_noise = np.log(noise_variance)
        self.yerr = np.sqrt(noise_variance) * np.ones_like(self.train_data.y)

        self.Kernel = getattr(george.kernels, kernel)
        self.Solver = getattr(george, solver)

        self.rearrange_fn = getattr(self, rearrange_fn)
        self.rearrange_kwargs = rearrange_kwargs

        self.train_data_stash = self.train_data.clone()

        self.re_rearrange = re_rearrange
        if self.re_rearrange:
            self.hyper_rearrange_fn = self.rearrange_fn
        else:
            self.hyper_rearrange_fn = lambda *args, **kwargs: None
        
        self.model_kwargs = model_kwargs
        if solver == "HODLRSolver":
            self.model_kwargs['seed'] = self.random_seed
            self.sufficient_resources = self.sufficient_resources_HODLRSolver

        self.is_plot_KXX = is_plot_KXX
    def kernelparm_transform(self, params):
        exp_params = np.exp(params)
        return exp_params[0]*self.train_data.D, np.sqrt(exp_params[1])
    def make_model(self):
        kernel = self.scale_variance * self.Kernel(ndim=self.train_data.D, **self.kernel_kwargs)
        model = george.GP(kernel,solver=self.Solver, **self.model_kwargs) # min_size is the size of the leaf matrices , 
        self.rearrange_fn(model, **self.rearrange_kwargs)
        # You need to compute the GP once before starting the optimization.
        model.compute(self.train_data.X, self.yerr)
        self.model = model # KXX for visualization
        return model, kernel

    def sufficient_resources_HODLRSolver(self):
        n_bytes = self.data.X.size * self.train_data.X.itemsize
        X_MB = int((n_bytes)/(10**6))
        return X_MB < self.free_MB

    def compute_log_likelihood(self):

        model, _ = self.make_model()
        return model.log_likelihood(self.train_data.y, quiet=True)
    
    def predict(self, X, perform_opt):
        model, kernel = self.make_model()
        
        if perform_opt:
            # https://george.readthedocs.io/en/latest/tutorials/hyper/
            self.optimize_hypers(kernel, model)
        kernel_params = self.kernelparm_transform(kernel.get_parameter_vector())

        y_predicted, _ = model.predict(self.train_data.y, X, return_var=False)
        self.model = model # KXX for visualization

        return y_predicted, model.log_likelihood(self.train_data.y, quiet=True), kernel_params
    def optimize_hypers(self, kernel, model):

        # we need to implement batched LBFGS
        # https://docs.ray.io/en/latest/ray-core/examples/plot_lbfgs.html

        # Define the objective function (negative log-likelihood in this case).
        def nll(p):
            model.set_parameter_vector(p)
            try:
                self.hyper_rearrange_fn(model, **self.rearrange_kwargs)
                ll = model.log_likelihood(self.train_data.y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            except Exception as e:
                print(e)
                return 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            model.set_parameter_vector(p)
            try:
                self.hyper_rearrange_fn(model, **self.rearrange_kwargs)
                return model.grad_log_likelihood(self.train_data.y, quiet=True)
            except Exception as e:
                print(e)
                return model.get_parameter_vector() * 0.0

        # Run the optimization routine.
        p0 = model.get_parameter_vector()
        results = op.minimize(nll, p0, **self.optimizer_kwargs)

        # Update the kernel and print the final log-likelihood.
        model.set_parameter_vector(results.x)
        self.hyper_rearrange_fn(model, **self.rearrange_kwargs)

    # No rearrangement
    def rearrange_placebo(self, model, **rearrange_kwargs):
        pass

    def rearrange_dsort(self, model, metric='euclidean'):

        if metric == "kernel":
            metric = lambda x1, x2: model.kernel.get_value(x1.reshape(1,-1), x2.reshape(1,-1))

        idx = self.dsort(self.train_data.X, metric=metric)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist

        # idx = np.argsort(self.train_data.X.reshape(-1))

        # min_idx = np.argsort(means_dist.min(axis=1))
        # half_len = int(len(min_idx)/2)
        # first_half = min_idx[:half_len]
        # second_half = min_idx[half_len:]
        # first_half = first_half[np.argsort(means_dist[first_half].max(axis=1))]
        # idx = np.concatenate((first_half, second_half))

        self.train_data.rearrange(idx)
    def dsort(self, X, metric):
        mean_x = np.mean(X, axis=0, keepdims=True)
        dist_x = cdist(X, mean_x, metric=metric).reshape(-1)
        return np.argsort(dist_x)

    def rearrange_ksort(self, model, n_components=None):

        if n_components == None:
            n_components = self.train_data.X.shape[1]
        else:
            assert(n_components <= len(self.train_data.X.shape[1]))
        
        idx = self.recursive_ksort(self.train_data.X, n_components)
        self.train_data.rearrange(idx)

    def rearrange_eric(self, model):
        K = model.get_matrix(self.train_data.X)
        def largest_indices_tril(a):
            m = a.shape[0]
            r,c = np.tril_indices(m,-1)
            idx = a[r,c].argpartition(-1)[-1:]
            return (r[idx][0], c[idx][0])

        curr_id = largest_indices_tril(K)[0]
        mask = np.zeros(K.shape[0], dtype=bool)
        mask[curr_id] = True
        idx = [ curr_id ]
        for i in range(K.shape[0]-1):
            curr_k = np.ma.array(K[:,curr_id], mask=mask)
            curr_id = np.argmax(curr_k)
            idx.append(curr_id)
            mask[curr_id] = True
        
        self.train_data.rearrange(idx)

    def rearrange_la_pca(self, model, n_components=None):
        self.rearrange_la_pca_sk(model, n_components)

    def rearrange_la_kpca(self, model, n_components=None):
        self.rearrange_la_kpca_sk(model, n_components)
        #self.rearrange_la_kpca_np(model, n_components)

    def rearrange_la_pca_sk(self, model, n_components):

        pca = PCA(n_components=n_components, random_state=self.random_seed)
        pca_X = pca.fit_transform(self.train_data.X)
        n_components = self.choose_n_eigvals(pca.singular_values_, n_components)
        
        idx = self.recursive_ksort(pca_X, n_components)
        self.train_data.rearrange(idx)

    def rearrange_la_kpca_sk(self, model, n_components):

        def _kernel(x1, x2):
            return model.kernel.get_value(x1.reshape(1,-1), x2.reshape(1,-1))

        pca = KernelPCA(kernel=_kernel, n_components=n_components, random_state=self.random_seed)
        pca_X = pca.fit_transform(self.train_data.X)
        n_components = self.choose_n_eigvals(pca.eigenvalues_, n_components)

        idx = self.recursive_ksort(pca_X, n_components)
        self.train_data.rearrange(idx)
    def rearrange_la_kpca_tree(self, model, n_components):

        curr_id = np.argmax(np.sum(model.get_matrix(self.train_data.X), axis=0))

        def _kernel(x1, x2):
            return model.kernel.get_value(x1.reshape(1,-1), x2.reshape(1,-1))

        pca = KernelPCA(kernel=_kernel, n_components=n_components, random_state=self.random_seed)
        pca_X = pca.fit_transform(self.train_data.X)
        n_components = self.choose_n_eigvals(pca.eigenvalues_, n_components)

        tree = cKDTree(pca_X)
        d, c_idx = tree.query(pca_X[curr_id], k=len(pca_X))
        idx = [curr_id]
        for i in range(1, len(c_idx), 2):
            if i == len(c_idx)-1:
                idx = [ c_idx[i] ] + idx
            else:
                idx = [ c_idx[i] ] + idx + [ c_idx[i+1] ]
        # d, idx = tree.query(pca_X[curr_id], k=len(pca_X))
        
        self.train_data.rearrange(idx)
    def rearrange_la_pca_tree(self, model, n_components):

        curr_id = np.argmax(np.sum(model.get_matrix(self.train_data.X), axis=0))

        pca = PCA(n_components=n_components, random_state=self.random_seed)
        pca_X = pca.fit_transform(self.train_data.X)
        n_components = self.choose_n_eigvals(pca.singular_values_, n_components)
        
        tree = cKDTree(pca_X)
        d, c_idx = tree.query(pca_X[curr_id], k=len(pca_X))
        idx = [curr_id]
        for i in range(1, len(c_idx), 2):
            if i == len(c_idx)-1:
                idx = [ c_idx[i] ] + idx
            else:
                idx = [ c_idx[i] ] + idx + [ c_idx[i+1] ]
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

        idx = self.recursive_ksort(pca_X, n_components)
        
        self.train_data.rearrange(idx)
    
    def rearrange_graph_kernighan(self, model, k, max_iter):
        M = model.get_matrix(self.train_data.X)

        def get_sequence(idx):
            G = nx.Graph()
            for i, i1 in enumerate(idx):
                for i2 in idx[i+1:]:
                    G.add_edge(i1, i2, weight=M[i1][i2])

            return list(map(np.array,map(list, nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(G, max_iter=max_iter, seed=self.random_seed))))

        def sub_divide(idx, k=1):
            a, b = get_sequence(idx)
            if k == 1:
                return np.append(a,b)
            else:
                return np.array(reduce(np.append, np.append(sub_divide(a, k-1), sub_divide(b, k-1))))
        
        idx = np.array(sub_divide(idx = np.array(range(self.train_data.N)), k=k), copy=True)
        self.train_data.rearrange(idx)

    def rearrange_kmeans_eq(self, model, n_clusters):

        def get_even_clusters(X, n_clusters):
            cluster_size = int(np.ceil(len(X)/n_clusters))
            kmeans = KMeans(n_clusters, random_state=self.random_seed)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
            centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
            distance_matrix = cdist(X, centers)
            clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
            return clusters
        
        labels = get_even_clusters(self.train_data.X, n_clusters)
        idx = np.argsort(labels)
        self.train_data.rearrange(idx)

    def rearrange_kmeans(self, model, n_clusters):

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed).fit(self.train_data.X)
        idx = np.argsort(kmeans.labels_)
        self.train_data.rearrange(idx)

    def rearrange_ga(self, model, opt_N, init_N):
        N = self.train_data.N
        _x = self.train_data.X

        arg_idx =  np.array(range(self.train_data.N))

        matrix_N = len(arg_idx)
        matrix_half_N = math.ceil(matrix_N/2)
        def random_indices():
            return np.array([self.rng.integers(matrix_half_N), matrix_half_N + self.rng.integers(matrix_N - matrix_half_N)])

        def eval_fitness(idx):
            half_N = int(N/2)
            lr_mat = model.get_matrix(_x[idx[:half_N]], _x[idx[half_N:]])
            return lr_mat.sum()

        # def eval_fitness(idx):
        #     half_N = int(N/2)
        #     lr_mat = model.get_matrix(_x[idx[:half_N]], _x[idx[half_N:]])
        #     return np.linalg.matrix_rank(lr_mat)

        cand_pop = [  ]
        for i in range(init_N):
            candidate_arg_idx = arg_idx.copy()
            idx = random_indices()
            candidate_arg_idx[idx] = np.flip(candidate_arg_idx[idx])
            cand_pop.append((candidate_arg_idx, eval_fitness(candidate_arg_idx)))
        for i in range(opt_N):
            new_cand_pop = []
            for current_cand_pop in cand_pop:
                curr_candidate_arg_idx, old_fitness_val = current_cand_pop
                idx = random_indices()
                curr_candidate_arg_idx[idx] = np.flip(curr_candidate_arg_idx[idx])
                new_fitness_val = eval_fitness(curr_candidate_arg_idx)
                if new_fitness_val < old_fitness_val:
                    new_cand_pop.append((curr_candidate_arg_idx, new_fitness_val))
                else:
                    curr_candidate_arg_idx[idx] = np.flip(curr_candidate_arg_idx[idx])
                    new_cand_pop.append((curr_candidate_arg_idx, old_fitness_val))
            cand_pop = new_cand_pop

        best_arg_idx,_ = min(cand_pop, key=lambda x:x[1])
        self.train_data.rearrange(best_arg_idx)

    def choose_n_eigvals(self, eigvals, n_components):
        if n_components == None:
            n_components = len(eigvals)
        elif n_components < 1:
            var_explained = np.cumsum(eigvals / np.sum(eigvals))
            n_components = np.searchsorted(var_explained, n_components) + 1
        else:
            assert(n_components <= len(eigvals))
        return n_components

    def recursive_ksort(self, pca_X, n_components):
        submatrices_idx = [ np.argsort(pca_X[:, 0].reshape(-1)) ]
        for component_i in range(1, n_components):
            submatrices_idx = sum([ np.array_split(submatrix_ids, 2) for submatrix_ids in submatrices_idx ], [])
            # print([len(s) for s in submatrices_idx])
            submatrices_idx = [ submatrix_idx[np.argsort(pca_X[submatrix_idx, component_i].reshape(-1))] for submatrix_idx in submatrices_idx ]

        idx = np.concatenate(submatrices_idx)
        return idx

    def clean_up(self, status):
        
        # KXX space needed to compute the matrix, which the method in the super()
        if super().sufficient_resources() and self.is_plot_KXX:
            self.plot_KXX(self.model.get_matrix(self.train_data.X), f"george/{status}.png")
        # Reset train data.
        self.train_data = self.train_data_stash.clone()

    def plot_KXX(self, KXX, file_name):
        figure = plt.figure()
        axes = figure.add_subplot(111)
        
        caxes = axes.matshow(KXX, interpolation ='nearest')
        figure.colorbar(caxes)
        self.mlflow_logger.log_figure(figure, file_name)
        plt.cla()
        plt.close(figure)

    def visualize(self):

        # KXX space needed to compute the matrix, which the method in the super()
        if super().sufficient_resources() and self.is_plot_KXX:
            kernel = self.scale_variance * self.Kernel(ndim=self.train_data.D, **self.kernel_kwargs)
            model = george.GP(kernel, solver=self.Solver)
            self.plot_KXX(model.get_matrix(self.train_data.X), "george/kXX.png")

    def viz_graph1d(self):
        # tidx = np.array(list(range(self.test_data.N)))
        tidx = np.argsort(self.test_data.X.reshape(-1))
        self.test_data.rearrange(tidx)
        y_predicted, y_pred_var = model.predict(self.train_data.y, self.test_data.X, return_var=True)

        plt.fill_between(self.test_data.X.reshape(-1), y_predicted - np.sqrt(y_pred_var), y_predicted + np.sqrt(y_pred_var),
                        color="k", alpha=0.2)
        plt.plot(self.test_data.X.reshape(-1), y_predicted, "k", lw=1.5, alpha=0.5)
        plt.errorbar(self.train_data.X.reshape(-1), self.train_data.y, yerr=self.yerr, fmt=".k", capsize=0)
        plt.plot(self.test_data.X.reshape(-1), np.sin(self.test_data.X.reshape(-1)), "--g")
        plt.xlim(-10, 10)
        plt.ylim(-1.45, 1.45)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()