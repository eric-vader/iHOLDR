#!/usr/bin/env python3
__all__ = ["PyTorchAlexanderGP", "PyTorchGP", "GeorgeGP", "GPyGP", "SklearnGP"]

from algorithms.gpytorch import PyTorchAlexanderGP
from algorithms.gpytorch import PyTorchGP
from algorithms.george import GeorgeGP
from algorithms.gpy import GPyGP
from algorithms.sklearn import SklearnGP