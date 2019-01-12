======================================
qhoptim: Quasi-hyperbolic optimization
======================================

.. raw:: html

    <!-- Place this tag where you want the button to render. -->
    <a class="github-button" href="https://github.com/facebookresearch/qhoptim" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star facebookresearch/qhoptim on GitHub">Star</a>

    <!-- Place this tag where you want the button to render. -->
    <a class="github-button" href="https://github.com/facebookresearch/qhoptim/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork facebookresearch/qhoptim on GitHub">Fork</a>

    <!-- Place this tag where you want the button to render. -->
    <a class="github-button" href="https://github.com/facebookresearch/qhoptim/issues" data-icon="octicon-issue-opened" data-size="large" data-show-count="true" aria-label="Issue facebookresearch/qhoptim on GitHub">Issue</a>

----

The qhoptim library provides PyTorch and TensorFlow implementations of the
quasi-hyperbolic momentum (QHM) and quasi-hyperbolic Adam (QHAdam)
optimization algorithms from Facebook AI Research.

For those who use momentum or Nesterov's accelerated gradient with momentum
constant :math:`\beta = 0.9`, we recommend trying out QHM with
:math:`\nu = 0.7` and momentum constant :math:`\beta = 0.999`. You'll need to
normalize the learning rate by dividing by :math:`1 - \beta_{old}`.

Similarly, for those who use Adam with :math:`\beta_1 = 0.9`, we recommend
trying out QHAdam with :math:`\nu_1 = 0.7`, :math:`\beta_1 = 0.995`,
:math:`\nu_2 = 1`, and all other parameters unchanged.

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

GitHub
======

The project's GitHub repository can be found `here`__. Bugfixes and
contributions are very much appreciated!

__ https://github.com/facebookresearch/qhoptim

.. toctree::
    :maxdepth: 2
    :caption: Table of contents

    install
    pyt
    tf

Index
=====

:ref:`genindex`
