=======================================
Quasi-hyperbolic optimizers for PyTorch
=======================================

.. testsetup::

   import torch
   import torch.nn as nn

   import qhoptim.pyt

   model = nn.Linear(8, 1)

   def loss_fn(pred, y):
      return torch.mean((pred - y) ** 2)

   input = torch.randn(3, 8)

   target = torch.randn(3)

Getting started
===============

The PyTorch optimizer classes are :class:`qhoptim.pyt.QHM` and
:class:`qhoptim.pyt.QHAdam`.

Use these optimizers as you would any other PyTorch optimizer:

.. doctest::

    >>> from qhoptim.pyt import QHM, QHAdam

    # something like this for QHM
    >>> optimizer = QHM(model.parameters(), lr=1.0, nu=0.7, momentum=0.999)

    # or something like this for QHAdam
    >>> optimizer = QHAdam(
    ...     model.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.995, 0.999))

    # a single optimization step
    >>> optimizer.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()

QHM API reference
=================

.. autoclass:: qhoptim.pyt.QHM
   :members:

QHAdam API reference
====================

.. autoclass:: qhoptim.pyt.QHAdam
   :members:

.. autofunction:: qhoptim.pyt.QHAdamW
