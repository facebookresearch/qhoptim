======================================
qhoptim: Quasi-hyperbolic optimization
======================================

This repository contains PyTorch and TensorFlow implementations of the
quasi-hyperbolic momentum (QHM) and quasi-hyperbolic Adam (QHAdam)
optimization algorithms from Facebook AI Research.

Documentation
=============

Please refer to the `documentation`__ for installation instructions, usage
information, and an API reference.

__ https://facebookresearch.github.io/qhoptim/

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
