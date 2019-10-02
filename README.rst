======================================
qhoptim: Quasi-hyperbolic optimization
======================================

This repository contains PyTorch and TensorFlow implementations of the
quasi-hyperbolic momentum (QHM) and quasi-hyperbolic Adam (QHAdam)
optimization algorithms from Facebook AI Research.

Quickstart
==========

Use this one-liner for installation::

    $ pip install qhoptim

Then, you can instantiate the optimizers in PyTorch:

.. code-block:: python

    >>> from qhoptim.pyt import QHM, QHAdam

    # something like this for QHM
    >>> optimizer = QHM(model.parameters(), lr=1.0, nu=0.7, momentum=0.999)

    # or something like this for QHAdam
    >>> optimizer = QHAdam(
    ...     model.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.995, 0.999))

Or in TensorFlow:

.. code-block:: python

    >>> from qhoptim.tf import QHMOptimizer, QHAdamOptimizer

    # something like this for QHM
    >>> optimizer = QHMOptimizer(
    ...     learning_rate=1.0, nu=0.7, momentum=0.999)

    # or something like this for QHAdam
    >>> optimizer = QHAdamOptimizer(
    ...     learning_rate=1e-3, nu1=0.7, nu2=1.0, beta1=0.995, beta2=0.999)


Documentation
=============

Please refer to the `documentation`__ for installation instructions, usage
information, and a Python API reference.

__ https://facebookresearch.github.io/qhoptim/

Direct link to installation instructions: `here`__.

__ https://facebookresearch.github.io/qhoptim/install

Reference
=========

QHM and QHAdam were proposed in the ICLR 2019 paper
`"Quasi-hyperbolic momentum and Adam for deep learning"`__. We recommend
reading the paper for both theoretical insights into and empirical analyses of
the algorithms.

__ https://arxiv.org/abs/1810.06801

If you find the algorithms useful in your research, we ask that you cite the
paper as follows:

.. code-block:: bibtex

    @inproceedings{ma2019qh,
      title={Quasi-hyperbolic momentum and Adam for deep learning},
      author={Jerry Ma and Denis Yarats},
      booktitle={International Conference on Learning Representations},
      year={2019}
    }

Contributing
============

Bugfixes and contributions are very much appreciated! Please see
``CONTRIBUTING.rst`` for more information.

License
=======

This source code is licensed under the MIT license found in the ``LICENSE`` file
in the root directory of this source tree.
