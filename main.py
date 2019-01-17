import argparse
import numpy as np
import matplotlib.pyplot as plt

from lotkavolterra import LotkaVolterra
from svi import SVI


def lotka_Experiment(args):

    Y_loc = 'x_lotkavolterra.txt'
    time_loc = 't_lotkavolterra.txt'
    nParams = 4

    nStates = 2

    kernel_params = [[0.25, 0.1] for i in range(nStates)]
    sigma = [(0.1) for i in range(nStates)]
    gamma = [(0.02) for i in range(nStates)]

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

    Y, time = exp.Y, exp.time

    mean, cov, X_mean, X_cov = exp.coordinate_ascent(epochs=epochs,
                                                     nSamples=nSamples)

    mean2, cov2, X_mean_2, X_cov_2 = exp.coordinate_ascent(epochs=epochs,
                                                           nSamples=nSamples)

    print(mean, mean2)
    nXSamples = args.nXSamples
    X_samples = exp.sample_X(X_mean, X_cov, nSamples=1)
    X_samples_2 = exp.sample_X(X_mean_2, X_cov_2, nXSamples)

    # theta_samples = [mean]
    theta_samples = rnd.multivariate_normal(mean, cov,
                                            args.nThetaSamples)

    # theta_samples_2 = [mean2]
    theta_samples_2 = rnd.multivariate_normal(mean2, cov2,
                                              args.nThetaSamples)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VGM lotka volterra')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--nSamples', type=int, default=100)
    parser.add_argument('--nXSamples', type=int, default=1)
    parser.add_argument('--nThetaSamples', type=int, default=1)
    args = parser.parse_args()

    lotka_Experiment(args)
