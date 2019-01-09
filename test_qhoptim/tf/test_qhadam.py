# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf

from qhoptim.tf import QHAdamOptimizer

from .util import assert_optimizers_equal


def test_adam_equiv():
    lr = 3e-4
    betas = (0.9, 0.999)
    eps = 1e-8

    def adam_ctor():
        return tf.train.AdamOptimizer(learning_rate=lr, beta1=betas[0], beta2=betas[1], epsilon=eps)

    def qhadam_ctor():
        return QHAdamOptimizer(learning_rate=lr, beta1=betas[0], beta2=betas[1], nu1=1.0, nu2=1.0, epsilon=eps)

    assert_optimizers_equal(adam_ctor, qhadam_ctor)
