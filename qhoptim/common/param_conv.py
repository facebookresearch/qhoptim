# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__doc__ = """

Utility methods for hyperparameter conversion from various optimization algorithms to QHM/QHAdam.

These methods need not be invoked directly. Rather, they can be invoked through classmethods of the PyTorch/TensorFlow
optimizer classes.

"""

import collections
import math


QHMParams = collections.namedtuple("QHMParams", ["alpha", "nu", "beta"])

QHAdamParams = collections.namedtuple("QHAdamParams", ["alpha", "nu1", "nu2", "beta1", "beta2"])


def from_pid(k_p, k_i, k_d):
    alpha = k_i
    nu = k_p * k_p / (k_i * k_d)
    beta = k_d / (k_d - k_p)
    return QHMParams(alpha=alpha, nu=nu, beta=beta)


def from_synthesized_nesterov(alpha, beta1, beta2):
    new_alpha = alpha / (1.0 - beta1)
    nu = 1.0 - ((1.0 - beta1) / beta1) * beta2
    beta = beta1
    return QHMParams(alpha=new_alpha, nu=nu, beta=beta)


def from_robust_momentum(l, kappa, rho):
    if rho is None:
        rho = 1.0 - 1.0 / math.sqrt(kappa)

    alpha = kappa * ((1.0 - rho) ** 2) * (1.0 + rho) / l
    beta1 = kappa * (rho ** 3) / (kappa - 1.0)
    beta2 = (rho ** 3) / ((kappa - 1.0) * ((1.0 - rho) ** 2) * (1.0 + rho))
    return from_synthesized_nesterov(alpha, beta1, beta2)


def from_accsgd(delta, kappa, xi, eps):
    alpha = (delta * eps * (1.0 + xi)) / (1.0 + eps)
    nu = (eps * xi - 1.0) / (eps * (1.0 + xi))
    beta = (kappa - (eps * eps) * xi) / (kappa + eps * xi)
    return QHMParams(alpha=alpha, nu=nu, beta=beta)


def from_two_state_optimizer(h, k, l, m, q, z):
    phi = math.sqrt((h - q) * (h - q) + 4.0 * k * m)
    psi = k * m - h * q
    xi = (h - q - phi) * (l * m - h * z) + 2.0 * m * (l * q - k * z)

    alpha = 0.5 * xi / (phi * psi)
    nu = 2.0 * m * (l * q - k * z) / xi
    beta = 0.5 * (h + q - phi)
    return QHMParams(alpha=alpha, nu=nu, beta=beta)


def from_nadam(lr, beta1, beta2):
    return QHAdamParams(alpha=lr, nu1=beta1, nu2=1.0, beta1=beta1, beta2=beta2)
