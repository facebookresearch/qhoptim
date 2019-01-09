# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import Adam

from qhoptim.pyt import QHAdam

from .util import assert_optimizers_equal


def test_adam_equiv():
    lr = 3e-4
    betas = (0.9, 0.999)
    l2 = 0.5e-4
    eps = 1e-8

    def adam_ctor(params):
        return Adam(params, lr=lr, betas=betas, weight_decay=l2 * 2.0, eps=eps)

    def qhadam_ctor(params):
        return QHAdam(params, lr=lr, betas=betas, weight_decay=l2 * 2.0, nus=(1.0, 1.0), eps=eps)

    assert_optimizers_equal(adam_ctor, qhadam_ctor)
