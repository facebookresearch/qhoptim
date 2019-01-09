# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD

from qhoptim.pyt import QHM

from .util import assert_optimizers_equal


def test_plain_sgd_equiv():
    lr = 0.01
    beta = 0.0
    l2 = 0.5e-4

    def sgd_ctor(params):
        return SGD(params, lr=lr, momentum=beta, weight_decay=l2 * 2.0)

    def qhm_ctor(params):
        return QHM(params, lr=lr / (1.0 - beta), nu=1.0, momentum=beta, weight_decay=l2 * 2.0)

    assert_optimizers_equal(sgd_ctor, qhm_ctor)


def test_momentum_equiv():
    lr = 0.01
    beta = 0.9
    l2 = 0.5e-4

    def sgd_ctor(params):
        return SGD(params, lr=lr, momentum=beta, weight_decay=l2 * 2.0)

    def qhm_ctor(params):
        return QHM(params, lr=lr / (1.0 - beta), nu=1.0, momentum=beta, weight_decay=l2 * 2.0)

    assert_optimizers_equal(sgd_ctor, qhm_ctor)


def test_nesterov_equiv():
    lr = 0.01
    beta = 0.9
    l2 = 0.5e-4

    def sgd_ctor(params):
        return SGD(params, lr=lr, momentum=beta, weight_decay=l2 * 2.0, nesterov=True)

    def qhm_ctor(params):
        return QHM(params, lr=lr / (1.0 - beta), nu=beta, momentum=beta, weight_decay=l2 * 2.0)

    assert_optimizers_equal(sgd_ctor, qhm_ctor)
