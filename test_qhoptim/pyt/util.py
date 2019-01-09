# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn


def assert_optimizers_equal(reference_optim_ctor, test_optim_ctor, n=16, k=4, iters=1024, tol=1e-7):
    reference_m = nn.Linear(k, 1).double()
    test_m = copy.deepcopy(reference_m)

    reference_optim = reference_optim_ctor(reference_m.parameters())
    test_optim = test_optim_ctor(test_m.parameters())

    coeffs = torch.randn(k, dtype=torch.float64)

    for _ in range(iters):
        x = torch.randn((n, k), dtype=torch.float64)
        y = torch.mm(x, coeffs.view(-1, 1)) + torch.randn(n, dtype=torch.float64) + 5.0

        reference_output = reference_m(x)
        test_output = test_m(x)

        reference_loss = torch.mean((reference_output - y) ** 2)
        test_loss = torch.mean((test_output - y) ** 2)

        reference_optim.zero_grad()
        test_optim.zero_grad()

        reference_loss.backward()
        test_loss.backward()

        reference_optim.step()
        test_optim.step()

        assert torch.allclose(test_m.weight, reference_m.weight, atol=tol, rtol=tol)
        assert torch.allclose(test_m.bias, reference_m.bias, atol=tol, rtol=tol)
