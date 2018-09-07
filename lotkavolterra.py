import argparse
import numpy as np
import matplotlib.pyplot as plt

import svi_solvers as solvers
# from svi import SVI


class LotkaVolterra:
    def __init__(self, x0=[3, 5], theta=[2, 1, 4, 1], sigma=0.5):
        self.x0 = x0
        self.theta = theta
        self.sigma = sigma

    def f(self, x, theta):
        theta1, theta2, theta3, theta4 = theta
        x1, x2 = x
        return np.array([theta1*x1 - theta2*x1*x2,
                         -theta3*x2 + theta4*x1*x2])

    def get_trajectory(self, x0=None, T=4, eps=1e-2, theta=None):

        if theta is None:
            theta = self.theta

        if x0 is None:
            x0 = self.x0

        x0 = np.asarray(x0)

        path = [x0.copy()]
        steps = int(T/eps)
        for t in range(steps - 1):
            x0 = solvers.rk4(x0, self.f, eps=eps, params=theta)
            path.append(x0)

        t = np.arange(0, T, eps)
        return np.array(path), t

    def get_observations(self, x0=None, T=4, eps=1e-2,
                         theta=None, sigma=None,
                         t_obs=None):   ## figure out how to use t_obs

        x, t = self.get_trajectory(x0, T, eps, theta)

        if sigma is None:
            sigma = self.sigma

        x_noisy = x.copy()
        x_noisy += np.random.RandomState().normal(size=x.shape) * sigma

        """save experiment"""
        np.savetxt('x_lotkavolterra.txt', x_noisy)
        np.savetxt('t_lotkavolterra.txt', t)

        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].plot(t, x_noisy.T[0], '-r')
        ax[0].plot(t, x.T[0])
        ax[0].set_title('predator')

        ax[1].plot(t, x_noisy.T[1])
        ax[1].plot(t, x.T[1])
        ax[1].set_title('prey')

        plt.show()

        return x_noisy, t, x

    @staticmethod
    def get_BTheta(X, theta=None, f=None): # theta is for testing purposes only
        """test by multiplying the matrices right now with f_k(X, theta)"""
        dim = 2                 # number of states

        # check dimension of X
        assert X.shape[1] == dim
        nObs = X.shape[0]       # number of time points

        x1, x2 = X.T

        B_Theta_1 = np.array([x1, -x1*x2, np.zeros(nObs), np.zeros(nObs)]).T
        B_Theta_2 = np.array([np.zeros(nObs), np.zeros(nObs), -x2, x1*x2]).T

        B_Theta = [B_Theta_1, B_Theta_2]
        b_Theta = [np.zeros(nObs), np.zeros(nObs)]

        if theta:
            theta = np.array(theta)
            f = np.array([f(x_i, theta) for x_i in X])

            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].plot(f.T[0], label='true')
            ax[0].plot(B_Theta_1.dot(theta), label='B_theta')
            ax[0].legend(loc='upper right')

            ax[1].plot(f.T[1])
            ax[1].plot(B_Theta_2.dot(theta))

            plt.show()

        return B_Theta, b_Theta

    @staticmethod
    def get_BX(X, theta, f=None):
        """
        B_11 = nObs * nObs
        B_x = x_dim * x_dim = 2 * 2 = [B_11  B_12]
                                      [B_21  B_22]
        """

        dim = 2
        assert X.shape[1] == dim
        nObs = X.shape[0]

        B_X = [[] for k in range(2)]
        b_X = [[] for k in range(2)]

        x1, x2 = X.T
        theta1, theta2, theta3, theta4 = theta

        B_11, B_21 = np.zeros((nObs, nObs)), np.eye(nObs)
        B_22, B_12 = np.zeros((nObs, nObs)), np.zeros((nObs, nObs))

        B_11[np.arange(nObs), np.arange(nObs)] = -theta2 * x2 + theta1
        b_11 = np.zeros(nObs)

        B_12[np.arange(nObs), np.arange(nObs)] = theta4 * x2
        b_12 = -theta3 * x2

        B_21[np.arange(nObs), np.arange(nObs)] = -theta2 * x1
        b_21 = theta1 * x1

        B_22[np.arange(nObs), np.arange(nObs)] = theta4 * x1 - theta3
        b_22 = np.zeros(nObs)


        # sanity check
        if f:
            f = np.array([f(x_i, theta) for x_i in X])

            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].plot(f.T[0], label='f1')
            ax[0].plot(B_21.dot(x2) + b_21)
            ax[0].plot(B_11.dot(x1) + b_11)
            ax[0].legend(loc='best')

            ax[1].plot(f.T[1], label='f2')
            ax[1].plot(B_12.dot(x1) + b_12)
            ax[1].plot(B_22.dot(x2) + b_22)
            ax[1].legend(loc='best')

            plt.show()
            plt.close()

        return [[B_11, B_12], [B_21, B_22]], [[b_11, b_12], [b_21, b_22]]

def lotka_Experiment(args):

    Y_loc = 'x_lotkavolterra.txt'
    time_loc = 't_lotkavolterra.txt'
    nParams = 4

    ###### optimizae kernel hyperparameters
    ###### use an empirical bayes approach
    ###### log-likelihood maximization

    nStates = 2

    kernel_params = [[0.25, 0.1] for i in range(nStates)] # 1 rbf param for each x_i
    sigma = [(0.1) for i in range(nStates)]             # observation noise
    gamma = [(0.02) for i in range(nStates)]            # gradient noise

    lv_Exp = LotkaVolterra()

    epochs = args.epochs
    nSamples = args.nSamples
    exp = SVI(Y_loc=Y_loc, time_loc=time_loc,
              sigma=sigma,
              nParams=nParams,
              nHiddenStates=0,
              kernel_params=kernel_params,
              get_BTheta=lv_Exp.get_BTheta,
              get_BX=lv_Exp.get_BX, F=lv_Exp.f,
              gamma=gamma)

    rnd = np.random.RandomState()

    Y, time = exp.Y, exp.time

    mean, cov, X_mean, X_cov = exp.coordinate_ascent(epochs=epochs,
                                                     nSamples=nSamples)

    mean2, cov2, X_mean_2, X_cov_2 = exp.coordinate_ascent(epochs=epochs,
                                                           nSamples=nSamples)

    print(mean, mean2)
    nXSamples = args.nXSamples
    X_samples = exp.sample_X(X_mean, X_cov, nSamples=1)
    X_samples_2 = exp.sample_X(X_mean_2, X_cov_2, nXSamples)

    theta_samples = [mean]
    # theta_samples = rnd.multivariate_normal(mean, cov,
                                            # args.nThetaSamples)

    theta_samples_2 = [mean2]
    # theta_samples_2 = rnd.multivariate_normal(mean2, cov2,
                                              # args.nThetaSamples)

    T = 10
    path = []
    for theta_i in theta_samples:
        path_i, t = lv_Exp.get_trajectory(theta=theta_i, T=T)
        path.append(path_i)

    path2 = []
    for theta_i in theta_samples_2:
        path_i, t = lv_Exp.get_trajectory(theta=theta_i, T=T)
        path2.append(path_i)

    mean_path, t = lv_Exp.get_trajectory(theta=mean, T=T)
    mean_path_2, t = lv_Exp.get_trajectory(theta=mean2, T=T)
    true_path, t = lv_Exp.get_trajectory(theta=lv_Exp.theta, T=T)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

    for path_i, path_j in zip(path, path2):
        ax[0, 0].plot(t, path_i.T[0], label='sample 1')
        ax[0, 1].plot(t, path_i.T[1], label='sampler 1')

        ax[0, 0].plot(t, path_j.T[0], label='sample 2')
        ax[0, 1].plot(t, path_j.T[1], label='sample 2')

    ax[0, 0].plot(t, mean_path.T[0], label='mean')
    ax[0, 0].plot(t, mean_path_2.T[0], label='mean 2')
    ax[0, 0].plot(t, true_path.T[0], label='true')
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].set_title('predator')

    ax[0, 1].plot(t, mean_path.T[1], label='mean')
    ax[0, 1].plot(t, mean_path_2.T[1], label='mean 2')
    ax[0, 1].plot(t, true_path.T[1], label='true')
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].set_title('prey')

    for x, y in zip(X_samples, X_samples_2):
        ax[1, 0].plot(time, x.T[0], label='sample')
        ax[1, 1].plot(time, x.T[1], label='sample')

        ax[1, 0].plot(time, y.T[0], label='sample 2')
        ax[1, 1].plot(time, y.T[1], label='sample 2')

    ax[1, 0].plot(time, Y.T[0], 'x', label='observed')
    ax[1, 0].plot(time, X_mean[0], 'x', label='estimated')
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].set_title('predator')

    ax[1, 1].plot(time, Y.T[1], 'x', label='observed')
    ax[1, 1].plot(time, X_mean[1], 'x', label='estimate')
    ax[1, 1].legend(loc='upper right')
    ax[1, 1].set_title('prey')

    # bar plot for parameters
    ind = np.arange(4) * 2
    ind2 = [1, 3, 5, 7]

    ax[2, 0].bar(ind, np.array([2, 1, 4, 1]), label='true')
    ax[2, 0].bar(ind2, mean, label='estimate')
    ax[2, 0].legend(loc='upper right')

    ax[2, 1].bar(ind, np.array([2, 1, 4, 1]), label='true')
    ax[2, 1].bar(ind2, mean2, label='estimate')
    ax[2, 1].legend(loc='upper right')

    plt.savefig('lotka.pdf')
    plt.show()

    print(mean, '\n', cov, '\n', cov2)



if __name__ == '__main__':
    exp = LotkaVolterra(sigma=0.1)
    t_obs = np.array([1., 2.])
    eps = 0.1
    T = 3
    x_noisy, t, x = exp.get_observations(t_obs=t_obs, eps=eps, T=T)

    ## check whether linear ode params are right
    # exp.get_BTheta(x, theta=exp.theta, f=exp.f)
    # exp.get_BX(x, theta=exp.theta, f=exp.f)

    parser = argparse.ArgumentParser(description='VGM lotka volterra')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--nSamples', type=int, default=10)
    parser.add_argument('--nXSamples', type=int, default=1)
    parser.add_argument('--nThetaSamples', type=int, default=1)
    args = parser.parse_args()

    lotka_Experiment(args)
