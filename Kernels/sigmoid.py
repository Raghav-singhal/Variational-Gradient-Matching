import numpy as np
import matplotlib.pyplot as plt


class Sigmoid:
    def __init__(self, theta=None, sigma=None, eps=1e-4):
        self.theta = theta
        self.sigma = sigma
        self.eps = eps

    def k(self, t1, t2):
        sigma, theta = self.sigma, self.theta
        a, b = theta
        raise NotImplementedError

    def CDash(self, t1, t2):
        raise NotImplementedError

    def DashC(self, t1, t2):
        raise NotImplementedError

    def CDoubleDash(self, t1, t2):
        raise NotImplementedError
