#!/usr/bin/env python3
__all__ = ["PyTorchAlexanderGP", "PyTorchGP", "GeorgeGP", "GPyGP", "SklearnGP", "GPflowGP"]

from algorithms.gpytorchGP import PyTorchAlexanderGP
from algorithms.gpytorchGP import PyTorchGP
from algorithms.george import GeorgeGP
from algorithms.sklearn import SklearnGP
from algorithms.gpy import GPyGP
from algorithms.gpflow import GPflowGP