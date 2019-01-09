==========================================
Quasi-hyperbolic optimizers for TensorFlow
==========================================

.. testsetup::

    import qhoptim.tf

Getting started
===============

The TensorFlow optimizer classes are :class:`qhoptim.tf.QHMOptimizer` and
:class:`qhoptim.tf.QHAdamOptimizer`.

Use these optimizers as you would any other TensorFlow optimizer:

.. doctest::

    >>> from qhoptim.tf import QHMOptimizer, QHAdamOptimizer

    # something like this for QHM
    >>> optimizer = QHMOptimizer(
    ...     learning_rate=1.0, nu=0.7, momentum=0.999)

    # or something like this for QHAdam
    >>> optimizer = QHAdamOptimizer(
    ...     learning_rate=1e-3, nu1=0.7, nu2=1.0, beta1=0.995, beta2=0.999)

QHM API reference
=================

.. autoclass:: qhoptim.tf.QHMOptimizer
    :members:

QHAdam API reference
====================

.. autoclass:: qhoptim.tf.QHAdamOptimizer
    :members:
