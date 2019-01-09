# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf

from qhoptim.tf import QHMOptimizer

from .util import assert_optimizers_equal


def test_plain_sgd_equiv():
    lr = 0.01
    beta = 0.0

    def sgd_ctor():
        return tf.train.GradientDescentOptimizer(learning_rate=lr)

    def qhm_ctor():
        return QHMOptimizer(learning_rate=lr / (1.0 - beta), nu=1.0, momentum=beta)

    assert_optimizers_equal(sgd_ctor, qhm_ctor)


def test_momentum_equiv():
    lr = 0.01
    beta = 0.9

    def momentum_ctor():
        return tf.train.MomentumOptimizer(learning_rate=lr, momentum=beta)

    def qhm_ctor():
        return QHMOptimizer(learning_rate=lr / (1.0 - beta), nu=1.0, momentum=beta)

    assert_optimizers_equal(momentum_ctor, qhm_ctor)
