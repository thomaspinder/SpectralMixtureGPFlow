from gpflow.kernels import RBF, Cosine, Kernel
from gpflow import Parameter
from gpflow.utilities import positive
from gpflow.utilities.ops import square_distance
import tensorflow as tf
import numpy as np


class SpectralMixture(Kernel):
    def __init__(
        self,
        n_mixtures=1,
        mixture_weights=None,
        mixture_scales=None,
        mixture_means=None,
        active_dims=None,
        name=None,
    ):
        super().__init__(active_dims, name=name)
        if n_mixtures == 1:
            print("Using default mixture = 1")
        self.num_mixtures = n_mixtures
        self.mixture_weights = Parameter(mixture_weights, transform=positive())
        self.mixture_scales = Parameter(mixture_scales, transform=positive())
        self.mixture_means = Parameter(mixture_means, transform=positive())

    def K(self, X1, X2=None):
        if (
            self.mixture_weights == None
            or self.mixture_means == None
            or self.mixture_scales == None
        ):
            raise RuntimeError(
                "Parameters of spectral mixture kernel not initialized.\
                                    Run `sm_kern_object.initialize_(train_x,train_y)`."
            )
            # initialization can only be done by user as it needs target data as well.
        if X2 is None:
            X2 = X1

        # get absolute distances
        X1 = tf.transpose(tf.expand_dims(X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(X2, perm=[1, 0]), -2)  # D x 1 x N2

        r = tf.abs(tf.subtract(X1, X2)) # D x N1 x N2

        cos_term = tf.multiply(
            tf.tensordot(self.mixture_means, r, axes=((1), (0))), 2.0 * np.pi
        )
        # num_mixtures x N1 x N2

        scales_expand = tf.expand_dims(tf.expand_dims(self.mixture_scales, -2), -2)
        # D x 1 x 1 x num_mixtures
        r_tile = tf.tile(tf.expand_dims(r, -1), (1, 1, 1, self.num_mixtures))
        # D x N1 x N2 x num_mixtures
        exp_term = tf.multiply(
            tf.transpose(
                tf.reduce_sum(tf.square(tf.multiply(r_tile, scales_expand)), 0),
                perm=[2, 0, 1],
            ),
            -2.0 * np.pi ** 2,
        )
        # num_mixtures x N1 x N2

        weights = tf.expand_dims(tf.expand_dims(self.mixture_weights, -1), -1)
        weights = tf.tile(weights, (1, tf.shape(X1)[1], tf.shape(X2)[2]))
        return tf.reduce_sum(
            tf.multiply(weights, tf.multiply(tf.exp(exp_term), tf.cos(cos_term))), 0
        )

    def K_diag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]),tf.reduce_sum(self.mixture_weights,0))