#!/usr/bin/env python3
import numpy as np
import scipy
import logging
from time import process_time_ns
import resource

import common
from config import Config

class CommonGP(common.Component):
    kernel_kwargs_mapper = {}
    def kernel_kwargs_adaptor(self, kernel_kwargs):
        return { (self.kernel_kwargs_mapper[k] if k in self.kernel_kwargs_mapper else k) : v for k,v in kernel_kwargs.items()  }
    def __init__(self, datasets, kernel_kwargs, m_repeats, test_mode=False, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)
        
        self.datasets = datasets
        self.train_data, self.test_data = datasets.generate_data()

        if kernel_kwargs['scale_variance'] == 'population':
            kernel_kwargs['scale_variance'] = np.var(self.train_data.y)
        self.kernel_kwargs_original = kernel_kwargs.copy()
        self.kernel_kwargs = self.kernel_kwargs_adaptor(kernel_kwargs)

        self.m_repeats = m_repeats

        # Optional parameter for testing
        self.test_mode = test_mode

    # Measured, but must return a python float
    def compute_log_likelihood(self) -> np.float64:
        raise NotImplementedError

    # Measured, but must return a tuple of hypers (lengthscale, scale_variance, noise_variance)
    def optimize_hypers(self):
        raise NotImplementedError

    def run(self):
        if self.test_mode:
            self.run_test()
            return

        metrics_dict = {}

        start_time_ns = process_time_ns() 
        for i in range(self.m_repeats):
            log_likelihood = self.compute_log_likelihood()

        time_taken_ns = process_time_ns()-start_time_ns
        metrics_dict['time_taken_ns'] = time_taken_ns / self.m_repeats
        # https://stackoverflow.com/questions/12050913/whats-the-unit-of-ru-maxrss-on-linux
        # maximum resident set size, maxrss kilobytes
        metrics_dict['ru_maxrss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics_dict['log_likelihood'] = log_likelihood

        mlflow_logger = self.config.configs['mlflow_logging']
        mlflow_logger.log_metrics(metrics_dict, None)

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

        mlflow_logger = self.config.configs['mlflow_logging']
        mlflow_logger.log_metrics(metrics_dict, None)

    def test_type_compute_log_likelihood(self, metrics_dict):
        logging.info('Test: compute_log_likelihood() must return a np.float64')
        log_likelihood = self.compute_log_likelihood()
        logging.info(f'log_likelihood has type - {type(log_likelihood)}')
        metrics_dict['test_type_compute_log_likelihood'] = int(type(log_likelihood) == np.float64)

    def test_acc_compute_log_likelihood(self, metrics_dict):
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

        log_likelihood = self.compute_log_likelihood()
        diff_log_likelihood = np.abs(log_likelihood+r)
        logging.info(f'Manual calculation using RBF yields {-r}, which log_likelihood differs by {diff_log_likelihood}')

        metrics_dict['test_acc_compute_log_likelihood'] = int(np.isclose(-r, log_likelihood))
        metrics_dict['diff_log_likelihood'] = diff_log_likelihood