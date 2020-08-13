import numpy as np
import matplotlib.pyplot as plt
from kernel import SpectralMixture
from utils import sm_init
from gpflow.utilities import print_summary
from gpflow.models import GPR
from gpflow.optimizers import Scipy
np.random.seed(123)


if __name__=='__main__':
    rng = np.random.RandomState(123)
    def f(X):
        return np.sin(X) + np.cos(3 * X)

    Xtrue = np.linspace(-5, 5, 500).reshape(-1, 1)
    ytrue = f(Xtrue)

    X = np.sort(rng.uniform(low=-5.0, high=5.0, size=100)).reshape(-1, 1)
    ynoise = f(X) + rng.normal(loc=0.0, scale=0.2, size=X.shape)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(X, ynoise, "o", label="Observations")
    # ax.plot(X, y, linewidth=2, label="Latent fn.")
    # ax.legend(loc="best")

    Q = 10
    weights, means, scales = sm_init(train_x=X, train_y=ynoise, num_mixtures=Q)
    k_sm = SpectralMixture(
        n_mixtures=Q,
        mixture_weights=weights,
        mixture_scales=scales,
        mixture_means=means + 1e-6,
    )

    m = GPR((X, ynoise), kernel=k_sm)
    print_summary(m)

    opt = Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
    print_summary(m)
    print('-'*80)
    print(opt_logs)

    xtest = np.linspace(-7, 7, 500).reshape(-1, 1)
    mu, sigma = m.predict_y(xtest)
    mun, sigman = mu.numpy().ravel(), sigma.numpy().ravel()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(X, ynoise, 'o', alpha=0.5, label='Training points')
    ax.plot(Xtrue, ytrue, label='Latent function')
    ax.plot(xtest, mun, label="Predictive mean")
    ax.fill_between(xtest.ravel(), mun-sigman, mun+sigman, alpha=0.5, label='predictive variance')
    ax.legend(loc='best')
    plt.show()