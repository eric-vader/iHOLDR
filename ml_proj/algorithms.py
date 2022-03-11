#!/usr/bin/env python3
import common
import logging
import george
import numpy as np

class ExampleAlgorithm(common.Component):
    def __init__(self, datasets, **kwargs):
        super().__init__(**kwargs)
        logging.info(kwargs)
        print(datasets)

class GeorgeGP(common.Component):
    def __init__(self, datasets, solver, kernel, kernel_kwargs, **kwargs):

        N = datasets.N
        X = datasets.X
        y = datasets.y
        var_y = datasets.var_y
        yerr = datasets.yerr

        kernel = var_y * getattr(george.kernels, kernel)(**kernel_kwargs)

        idx = np.array(range(N))
        _gp_hodlr = george.GP(kernel, solver=getattr(george, solver))
        _gp_hodlr.compute(X[idx], yerr[idx])
        print(_gp_hodlr.log_likelihood(y[idx]))

        gp_basic = george.GP(kernel)
        gp_basic.compute(X[:N], yerr[:N])
        print(gp_basic.log_likelihood(y[:N]))