# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

from ..common import param_conv
from .util import call_if_callable


class QHMOptimizer(optimizer.Optimizer):
    r"""Implements the quasi-hyperbolic momentum (QHM) optimization algorithm
    `(Ma and Yarats, 2019)`_.

    Note that many other optimization algorithms are accessible via specific
    parameterizations of QHM. See :func:`from_accsgd()`,
    :func:`from_robust_momentum()`, etc. for details.

    Args:
        learning_rate (float):
            learning rate (:math:`\alpha` from the paper)
        momentum (float):
            momentum factor (:math:`\beta` from the paper)
        nu (float):
            immediate discount factor (:math:`\nu` from the paper)
        use_locking (bool):
            whether or not to use locking parameter updates
        name (str):
            name of the optimizer

    Example:
        >>> optimizer = qhoptim.tf.QHMOptimizer(
        ...     learning_rate=1.0, nu=0.7, momentum=0.999)

    .. _`(Ma and Yarats, 2019)`: https://arxiv.org/abs/1810.06801

    .. note::

        Mathematically, QHM is a simple interpolation between plain SGD and
        momentum:

        .. math::

            \begin{align*}
                g_{t + 1} &\leftarrow
                    \beta \cdot g_t +
                    (1 - \beta) \cdot \nabla_t \\
                \theta_{t + 1} &\leftarrow
                    \theta_t + \alpha \left[ (1 - \nu) \cdot \nabla_t +
                                             \nu \cdot g_{t + 1} \right]
            \end{align*}

        Here, :math:`\alpha` is the learning rate, :math:`\beta` is the momentum
        factor, and :math:`\nu` is the "immediate discount" factor which
        controls the interpolation between plain SGD and momentum.
        :math:`g_t` is the momentum buffer, :math:`\theta_t` is the parameter
        vector, and :math:`\nabla_t` is the gradient with respect to
        :math:`\theta_t`.

    .. note::

        QHM uses **dampened** momentum. This means that when converting from
        plain momentum to QHM, the learning rate must be scaled by
        :math:`\frac{1}{1 - \beta}`. For example, momentum with learning rate
        :math:`\alpha = 0.1` and momentum :math:`\beta = 0.9` should be
        converted to QHM with learning rate :math:`\alpha = 1.0`.
    """

    def __init__(self, learning_rate, momentum, nu, use_locking=False, name="QHM"):
        super().__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._learning_rate_tensor = None
        self._momentum = momentum
        self._momentum_tensor = None
        self._nu = nu
        self._nu_tensor = None

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "momentum", self._name)

    def _prepare(self):
        learning_rate = call_if_callable(self._learning_rate)
        self._learning_rate_tensor = ops.convert_to_tensor(learning_rate, dtype=dtypes.float64, name="learning_rate")

        momentum = call_if_callable(self._momentum)
        self._momentum_tensor = ops.convert_to_tensor(momentum, dtype=dtypes.float64, name="momentum")

        nu = call_if_callable(self._nu)
        self._nu_tensor = ops.convert_to_tensor(nu, dtype=dtypes.float64, name="nu")

    def _apply_dense(self, grad, var):
        momentum_buffer = self.get_slot(var, "momentum")
        learning_rate = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        momentum = math_ops.cast(self._momentum_tensor, var.dtype.base_dtype)
        nu = math_ops.cast(self._nu_tensor, var.dtype.base_dtype)

        momentum_op = training_ops.apply_momentum(
            var,
            momentum_buffer,
            nu * (1.0 - momentum) * learning_rate,
            grad,
            momentum,
            use_locking=self._use_locking,
            use_nesterov=False,
        ).op

        with ops.control_dependencies([momentum_op]):
            gd_op = training_ops.apply_gradient_descent(
                var, (1.0 - nu) * learning_rate, grad, use_locking=self._use_locking
            ).op

        return control_flow_ops.group(momentum_op, gd_op)

    def _resource_apply_dense(self, grad, var):
        momentum_buffer = self.get_slot(var, "momentum")
        learning_rate = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        momentum = math_ops.cast(self._momentum_tensor, var.dtype.base_dtype)
        nu = math_ops.cast(self._nu_tensor, var.dtype.base_dtype)

        momentum_op = training_ops.resource_apply_momentum(
            var.handle,
            momentum_buffer.handle,
            nu * (1.0 - momentum) * learning_rate,
            grad,
            momentum,
            use_locking=self._use_locking,
            use_nesterov=False,
        )

        with ops.control_dependencies([momentum_op]):
            gd_op = training_ops.resource_apply_gradient_descent(
                var.handle, (1.0 - nu) * learning_rate, grad, use_locking=self._use_locking
            )

        return control_flow_ops.group(momentum_op, gd_op)

    def _apply_sparse(self, grad, var):
        momentum_buffer = self.get_slot(var, "momentum")
        learning_rate = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        momentum = math_ops.cast(self._momentum_tensor, var.dtype.base_dtype)
        nu = math_ops.cast(self._nu_tensor, var.dtype.base_dtype)

        momentum_op = training_ops.sparse_apply_momentum(
            var,
            momentum_buffer,
            nu * (1.0 - momentum) * learning_rate,
            grad.values,
            grad.indices,
            momentum,
            use_locking=self._use_locking,
            use_nesterov=False,
        ).op

        with ops.control_dependencies([momentum_op]):
            delta = ops.IndexedSlices((nu - 1.0) * learning_rate * grad.values, grad.indices, grad.dense_shape)
            gd_op = var.scatter_add(delta, use_locking=self._use_locking)

        return control_flow_ops.group(momentum_op, gd_op)

    def _resource_apply_sparse(self, grad, var, indices):
        momentum_buffer = self.get_slot(var, "momentum")
        learning_rate = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        momentum = math_ops.cast(self._momentum_tensor, var.dtype.base_dtype)
        nu = math_ops.cast(self._nu_tensor, var.dtype.base_dtype)

        momentum_op = training_ops.resource_sparse_apply_momentum(
            var.handle,
            momentum_buffer.handle,
            nu * (1.0 - momentum) * learning_rate,
            grad,
            indices,
            momentum,
            use_locking=self._use_locking,
            use_nesterov=False,
        )

        with ops.control_dependencies([momentum_op]):
            delta = (nu - 1.0) * learning_rate * grad
            gd_op = resource_variable_ops.resource_scatter_add(var.handle, indices, delta)

        return control_flow_ops.group(momentum_op, gd_op)

    @classmethod
    def _params_to_dict(cls, params):
        return {"learning_rate": params.alpha, "nu": params.nu, "momentum": params.beta}

    @classmethod
    def from_pid(cls, k_p, k_i, k_d):
        r"""Calculates the QHM hyperparameters required to recover a PID
        optimizer as described in `Recht (2018)`_.

        Args:
            k_p (float):
                proportional gain (see reference)
            k_i (float):
                integral gain (see reference)
            k_d (float):
                derivative gain (see reference)

        Returns:
            Three-element ``dict`` containing ``learning_rate``, ``momentum``,
            and ``nu`` to use in QHM.

        Example:
            >>> optimizer = qhoptim.tf.QHMOptimizer(
            ...     **qhoptim.tf.QHMOptimizer.from_pid(
            ...         k_p=-0.1, k_i=1.0, k_d=3.0))

        .. _`Recht (2018)`: https://web.archive.org/web/20181027184056/http://www.argmin.net/2018/04/19/pid/
        """
        return cls._params_to_dict(param_conv.from_pid(k_p, k_i, k_d))

    @classmethod
    def from_synthesized_nesterov(cls, alpha, beta1, beta2):
        r"""Calculates the QHM hyperparameters required to recover the
        synthesized Nesterov optimizer (Section 6 of `Lessard et al. (2016)`_).

        Args:
            alpha (float):
                learning rate
            beta1 (float):
                first momentum (see reference)
            beta2 (float):
                second momentum (see reference)

        Returns:
            Three-element ``dict`` containing ``learning_rate``, ``momentum``,
            and ``nu`` to use in QHM.

        Example:
            >>> optimizer = qhoptim.tf.QHMOptimizer(
            ...     **qhoptim.tf.QHMOptimizer.from_synthesized_nesterov(
            ...         alpha=0.1, beta1=0.9, beta2=0.6))

        .. _`Lessard et al. (2016)`: https://arxiv.org/abs/1408.3595
        """
        return cls._params_to_dict(param_conv.from_synthesized_nesterov(alpha, beta1, beta2))

    @classmethod
    def from_robust_momentum(cls, l, kappa, rho=None):
        r"""Calculates the QHM hyperparameters required to recover the Robust
        Momentum `(Cyrus et al., 2018)`_ or Triple Momentum
        `(Scoy et al., 2018)`_ optimizers.

        Args:
            l (float):
                Lipschitz constant of gradient (see reference)
            kappa (float):
                condition ratio (see reference)
            rho (float, optional):
                noise-free convergence rate. If None, will return the
                parameters for the Triple Momentum optimizer.

        Returns:
            Three-element ``dict`` containing ``learning_rate``, ``momentum``,
            and ``nu`` to use in QHM.

        Example:
            >>> optimizer = qhoptim.tf.QHMOptimizer(
            ...     **qhoptim.tf.QHMOptimizer.from_robust_momentum(
            ...         l=5.0, kappa=15.0))

        .. _`(Cyrus et al., 2018)`: https://arxiv.org/abs/1710.04753

        .. _`(Scoy et al., 2018)`: http://www.optimization-online.org/DB_FILE/2017/03/5908.pdf
        """
        return cls._params_to_dict(param_conv.from_robust_momentum(l, kappa, rho))

    @classmethod
    def from_accsgd(cls, delta, kappa, xi, eps=0.7):
        r"""Calculates the QHM hyperparameters required to recover the AccSGD
        optimizer `(Kidambi et al., 2018)`_.

        Args:
            delta (float):
                short step (see reference)
            kappa (float):
                long step parameter (see reference)
            xi (float):
                statistical advantage parameter (see reference)
            eps (float, optional):
                arbitrary value, between 0 and 1 exclusive (see reference)
                (default: 0.7)

        Returns:
            Three-element ``dict`` containing ``learning_rate``, ``momentum``,
            and ``nu`` to use in QHM.

        Example:
            >>> optimizer = qhoptim.tf.QHMOptimizer(
            ...     **qhoptim.tf.QHMOptimizer.from_accsgd(
            ...         delta=0.1, kappa=1000.0, xi=10.0))

        .. _`(Kidambi et al., 2018)`: https://arxiv.org/abs/1803.05591
        """
        return cls._params_to_dict(param_conv.from_accsgd(delta, kappa, xi, eps))

    @classmethod
    def from_two_state_optimizer(cls, h, k, l, m, q, z):
        r"""Calculates the QHM hyperparameters required to recover the
        following optimizer (named "TSO" in `Ma and Yarats (2019)`_):

        .. math::

            \begin{align*}
                a_{t + 1} &\leftarrow
                    h \cdot a_t + k \cdot \theta_t + l \cdot \nabla_t \\
                \theta_{t + 1} &\leftarrow
                    m \cdot a_t + q \cdot \theta_t + z \cdot \nabla_t
            \end{align*}

        Here, :math:`a_t` and :math:`\theta_t` are the two states and
        :math:`\nabla_t` is the gradient with respect to :math:`\theta_t`.

        Be careful that your coefficients satisfy the regularity conditions
        from the reference.

        Args:
            h (float):
                see description
            k (float):
                see description
            l (float):
                see description
            m (float):
                see description
            q (float):
                see description
            z (float):
                see description

        Returns:
            Three-element ``dict`` containing ``learning_rate``, ``momentum``,
            and ``nu`` to use in QHM.

        Example:
            >>> optimizer = qhoptim.tf.QHMOptimizer(
            ...     **qhoptim.tf.QHMOptimizer.from_two_state_optimizer(
            ...         h=0.9, k=0.0, l=0.1, m=-0.09, q=1.0, z=-0.01))

        .. _`Ma and Yarats (2019)`: https://arxiv.org/abs/1810.06801
        """
        return cls._params_to_dict(param_conv.from_two_state_optimizer(h, k, l, m, q, z))
