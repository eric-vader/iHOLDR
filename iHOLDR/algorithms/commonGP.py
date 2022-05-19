#!/usr/bin/env python3
import numpy as np
import scipy
import logging
from time import process_time_ns
import resource
import os
from sklearn.metrics import mean_squared_error

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, DotProduct

import common
from config import Config

class CommonGP(common.Component):
    kernel_kwargs_mapper = {}
    def kernel_kwargs_adaptor(self, kernel_kwargs):
        return { (self.kernel_kwargs_mapper[k] if k in self.kernel_kwargs_mapper else k) : v for k,v in kernel_kwargs.items()  }
    def __init__(self, datasets, kernel_kwargs, optimizer_kwargs, m_repeats=0, test_mode=False, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)
        self.mlflow_logger = self.config.configs['mlflow_logging']

        self.datasets = datasets
        dxs = datasets.generate_data()
        self.train_data, self.test_data, self.data = dxs
        logging.info(f"Dataset sizes - (Train, Test, Total) = {self.train_data.N, self.test_data.N, self.data.N}.")

        if kernel_kwargs['scale_variance'] == 'population':
            kernel_kwargs['scale_variance'] = np.var(self.train_data.y)
        self.kernel_kwargs_original = kernel_kwargs.copy()
        self.kernel_kwargs = self.kernel_kwargs_adaptor(kernel_kwargs)

        self.total_MB, self.used_MB, self.free_MB = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        self.mlflow_logger.log_params({
            "train_N":self.train_data.N, 
            "test_N":self.test_data.N, 
            "total_N":self.data.N
        })
        
        if CommonGP.sufficient_resources(self):
            # Perform any clean GT computation before going on
            logging.info("Computing gt_log_likelihood")
            self.gt_log_likelihood = self.groundtruth_log_likelihood()
            logging.info(f"gt_log_likelihood = {self.gt_log_likelihood}")
            # self.find_gt_kernel_params()
        else:
            logging.info("Skipping gt_log_likelihood due to insufficient resources.")
            self.gt_log_likelihood = 0

        # Now we adapt the data
        self.train_data, self.test_data, self.data = [ self.adapt_data(d) for d in dxs ]

        self.optimizer_kwargs = optimizer_kwargs

        self.m_repeats = m_repeats

        # Optional parameter for testing
        self.test_mode = test_mode

    def sufficient_resources(self):
        n_bytes = self.data.X.size * self.train_data.X.itemsize
        KXX_MB = int((n_bytes**2)/(10**6))
        return KXX_MB < self.free_MB

    def adapt_data(self, data):
        return data

    def compute_log_likelihood(self):
        raise NotImplementedError

    def predict(self, X, perform_opt):
        raise NotImplementedError

    def clean_up(self, status):
        pass

    def visualize(self):
        pass

    def run(self):
        if self.test_mode:
            self.run_test()
            return
    
        if not self.sufficient_resources():
            logging.info("Skipping experiment due to insufficient resources.")
            return

        if self.m_repeats > 0:
            logging.info("Measure timing for log-likelihood computation.")
            total_time_taken_ns = 0
            ru_maxrss_KB_list = []
            for i in range(self.m_repeats):
                start_time_ns = process_time_ns() 
                log_likelihood = self.compute_log_likelihood()
                time_taken_ns = process_time_ns()-start_time_ns
                # https://stackoverflow.com/questions/12050913/whats-the-unit-of-ru-maxrss-on-linux
                # maximum resident set size, maxrss kilobytes
                ru_maxrss_KB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                # Clean-up
                self.clean_up(f'compute_log_likelihood_{i:03}')
                
                total_time_taken_ns += time_taken_ns
                ea_metrics_dict = {
                    'time_taken_ns': time_taken_ns,
                    'ru_maxrss_KB': ru_maxrss_KB,
                    'log_likelihood': log_likelihood
                }
                self.mlflow_logger.log_metrics(ea_metrics_dict, i)
                ru_maxrss_KB_list.append(ru_maxrss_KB)
        else:
            metrics_dict = {}

            logging.info(f"Starting prediction (non-opt) using kernel with kernel_params = (var, ls) = {self.kernel_kwargs_original['scale_variance'], self.kernel_kwargs_original['lengthscale']}")
            y_predicted, log_likelihood, kernel_params = self.predict(self.test_data.X, False)
            rmse = mean_squared_error(self.test_data.y, y_predicted, squared=False)
            logging.info(f"rmse = {rmse}, log_likelihood = {log_likelihood}, kernel_params = (var, ls) = {kernel_params}")
            self.clean_up('prediction')

            gt_log_likelihood = self.gt_log_likelihood
            # abs_err = np.abs(log_likelihood+gt_log_likelihood)
            # rel_err = abs_err/np.abs(gt_log_likelihood)
            metrics_dict['gt_log_likelihood'] = -gt_log_likelihood
            # metrics_dict['abs_err_ll'] = abs_err
            # metrics_dict['rel_err_ll'] = rel_err

            metrics_dict['log_likelihood'] = log_likelihood
            metrics_dict['rmse'] = rmse

            logging.info(f"Starting prediction (opt) using kernel with kernel_params = (var, ls) = {self.kernel_kwargs_original['scale_variance'], self.kernel_kwargs_original['lengthscale']}")
            y_predicted, opt_log_likelihood, opt_kernel_params = self.predict(self.test_data.X, True)
            opt_rmse = mean_squared_error(self.test_data.y, y_predicted, squared=False)
            logging.info(f"opt_rmse = {opt_rmse}, opt_log_likelihood = {opt_log_likelihood}, opt_kernel_params = (var, ls) = {opt_kernel_params}")
            self.clean_up('prediction_opt')

            metrics_dict['opt_log_likelihood'] = opt_log_likelihood
            metrics_dict['opt_rmse'] = opt_rmse
            metrics_dict['opt_var'], metrics_dict['opt_ls'] = opt_kernel_params

            self.mlflow_logger.log_metrics(metrics_dict, None)
        self.visualize()

    def run_test(self):
        logging.info(f'Conducting tests on {self.__class__.__name__}')
        
        metrics_dict = {}
        self.test_type_compute_log_likelihood(metrics_dict)
        self.test_acc_compute_log_likelihood(metrics_dict)

        # Now compute overall test result
        test_all = 1
        for k,v in metrics_dict.items():
            if k.startswith('test'):
                test_all &= v
        metrics_dict['test'] = test_all

        self.mlflow_logger.log_metrics(metrics_dict, None)

    def test_type_compute_log_likelihood(self, metrics_dict):
        logging.info('Test: compute_log_likelihood() must return a np.float64')
        log_likelihood = self.compute_log_likelihood()
        logging.info(f'log_likelihood has type - {type(log_likelihood)}')
        metrics_dict['test_type_compute_log_likelihood'] = int(type(log_likelihood) == np.float64)

    def test_acc_compute_log_likelihood(self, metrics_dict):

        gt_log_likelihood = self.groundtruth_log_likelihood()
        log_likelihood = self.compute_log_likelihood()
        diff_log_likelihood = np.abs(log_likelihood+gt_log_likelihood)
        logging.info(f'Manual calculation using RBF yields {-gt_log_likelihood}, which log_likelihood differs by {diff_log_likelihood}')

        metrics_dict['test_acc_compute_log_likelihood'] = int(np.isclose(-gt_log_likelihood, log_likelihood))
        metrics_dict['diff_log_likelihood'] = diff_log_likelihood

    def groundtruth_log_likelihood(self):
        noise_variance = self.kernel_kwargs_original["noise_variance"]
        lengthscale = self.kernel_kwargs_original["lengthscale"]
        scale_variance = self.kernel_kwargs_original["scale_variance"]

        # Compute nll manually
        X_train = self.train_data.X
        Y_train = self.train_data.y
        N = self.train_data.N

        def gaussian_rbf(x1, x2, l=1, scale_variance=1):
            # distance between each rows
            dist_matrix = np.sum(np.square(x1), axis=1).reshape(-1, 1) + np.sum(np.square(x2), axis=1) - 2 * np.dot(x1, x2.T)
            return scale_variance * np.exp(-1 / (2 * np.square(l)) * dist_matrix)

        # using matrix inversion
        # K = gaussian_rbf(X_train, X_train, l=lengthscale, scale_variance=scale_variance) + \
        #     noise_variance * np.eye(N)
        # r = np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
        #        0.5 * Y_train.T @ np.linalg.inv(K) @ Y_train + \
        #        0.5 * N * np.log(2*np.pi)

        # using cholesky
        K = gaussian_rbf(X_train, X_train, l=lengthscale, scale_variance=scale_variance) + \
            noise_variance * np.eye(N)
        L_factors = scipy.linalg.cho_factor(K)
        alpha = scipy.linalg.cho_solve(L_factors,Y_train)
        r = np.sum(np.log(np.diagonal(L_factors[0]))) + \
               0.5 * Y_train.T @ alpha + \
               0.5 * N * np.log(2*np.pi)
        return r

    def find_gt_kernel_params(self):
        X_train = self.data.X
        fX_train = self.data.fX

        param_grid = [{
            "kernel": [ np.var(fX_train) * RBF(l, (1e-15,  1e5)) for l in np.logspace(-5, 3, 10)]
        }]

        gp = GaussianProcessRegressor()

        clf = GridSearchCV(estimator=gp, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
        clf.fit(X_train, fX_train)
        best_kernel = clf.best_params_['kernel']
        scale_variance, lengthscale = best_kernel.k1.constant_value, best_kernel.k2.length_scale
        logging.info(f"Ground Truth gt_kernel_params = (var, ls) = ({scale_variance}, {lengthscale})")

        return scale_variance, lengthscale