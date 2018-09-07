import numpy as np
import matplotlib.pyplot as plt


"""got the kernel params from n gorbach repo"""


class RBF:
    def __init__(self, theta=10, sigma=0.2, eps=1e-4):
        self.theta = theta
        self.sigma = sigma
        self.eps = eps

    def k(self, t1, t2):
        sigma, theta = self.sigma, self.theta
        r = t1 - t2
        return sigma * np.exp(-r**2 / (2 * theta**2))

    def CDash(self, t1, t2):
        sigma, theta = self.sigma, self.theta
        r = t1 - t2

        return r/(theta**2) * self.k(t1, t2)

    def DashC(self, t1, t2):
        return -self.CDash(t1, t2)

    def CDoubleDash(self, t1, t2):
        sigma, theta = self.sigma, self.theta
        r = t1 - t2

        part1 = 1/theta**2
        part2 = -r**2 / theta**4
        return (part1 + part2) * self.k(t1, t2)

    def getBounds(self, y, time):
        """
        creates the bounds for the optimization of the hyperparameters.

        Parameters
        ----------
        y:          vector
                    observation of the states. Target of the regression
        time:       vector
                    time points of the observations. Input of the regression
        Returns
        ----------
        bounds: list of theta.size + 1 pairs of the form
                (lowerBound, upperBound), representing the bounds on the
                kernel hyperparameters in theta, while the last one is the
                bound on sigma
        """
        upperBoundSigmaF = (np.max(y) - np.min(y))**2
        upperBoundLengthscale = time[1]*100
        upperBoundStd = np.max(y) - np.min(y)
        lowerBoundLengthscale = time[1]
        bounds = [(1e-4, upperBoundSigmaF),
                  (lowerBoundLengthscale, upperBoundLengthscale),
                  (1e-3, upperBoundStd)
                  ]
        return bounds
