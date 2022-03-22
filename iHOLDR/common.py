#!/usr/bin/env python3
from config import Config

class Component:
    def __init__(self, hash, random_seed, **kwargs):
        self.hash = hash
        self.random_seed = random_seed
        self.config = Config()
    def prepare(self):
        # cache anything needed
        pass
    def run(self):
        # cache anything needed
        pass