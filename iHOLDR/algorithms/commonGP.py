#!/usr/bin/env python3
import numpy as np
import logging
from time import process_time_ns
import resource

import common
from config import Config

class CommonGP(common.Component):
    kernel_kwargs_mapper = {}
    def kernel_kwargs_adaptor(self, kernel_kwargs):
        return { (self.kernel_kwargs_mapper[k] if k in self.kernel_kwargs_mapper else k) : v for k,v in kernel_kwargs.items()  }
    def __init__(self, datasets, kernel_kwargs, m_repeats, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(self.random_seed)
        
        self.datasets = datasets
        self.data = datasets.generate_data()

        if kernel_kwargs['scale_variance'] == 'population':
            kernel_kwargs['scale_variance'] = np.var(self.data.y)
        self.kernel_kwargs = self.kernel_kwargs_adaptor(kernel_kwargs)

        self.m_repeats = m_repeats

    # Measured, but must return a python float
    def compute_log_likelihood(self):
        raise NotImplementedError

    # Measured, but must return a tuple of hypers (lengthscale, scale_variance, noise_variance)
    def optimize_hypers(self):
        raise NotImplementedError

    def run(self):

        metrics_dict = {}

        start_time_ns = process_time_ns() 
        for i in range(self.m_repeats):
            log_likelihood = self.compute_log_likelihood()
        logging.info(log_likelihood)

        time_taken_ns = process_time_ns()-start_time_ns
        metrics_dict['time_taken_ns'] = time_taken_ns / self.m_repeats
        # https://stackoverflow.com/questions/12050913/whats-the-unit-of-ru-maxrss-on-linux
        # maximum resident set size, maxrss kilobytes
        metrics_dict['ru_maxrss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics_dict['log_likelihood'] = log_likelihood

        mlflow_logger = self.config.configs['mlflow_logging']
        mlflow_logger.log_metrics(metrics_dict, None)