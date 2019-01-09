# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import slot_creator

from ..common import param_conv
from .util import call_if_callable


class QHAdamOptimizer(optimizer.Optimizer):
    r"""Implements the QHAdam optimization algorithm `(Ma and Yarats, 2019)`_.

    Note that the NAdam optimizer is accessible via a specific parameterization
    of QHAdam. See :func:`from_nadam()` for details.

    Args:
        learning_rate (float, optional):
            learning rate (:math:`\alpha` from the paper)
            (default: 1e-3)
        beta1 (float, optional):
            coefficient used for computing running average of gradient
            (default: 0.9)
        beta2 (float, optional): coefficients used for computing running average
            of squared gradient (default: 0.999)
        nu1 (float, optional): immediate discount factor used to estimate the
            gradient (default: 1.0)
        nu2 (float, optional): immediate discount factor used to estimate the
            squared gradient (default: 1.0)
        epsilon (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        use_locking (bool):
            whether or not to use locking parameter updates
        name (str):
            name of the optimizer

    Example:
        >>> optimizer = qhoptim.tf.QHAdamOptimizer(
        ...     learning_rate=3e-4, nu1=0.8, nu2=1.0,
        ...     beta1=0.99, beta2=0.999)

    .. _`(Ma and Yarats, 2019)`: https://arxiv.org/abs/1810.06801
    """

    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        nu1=1.0,
        nu2=1.0,
        epsilon=1e-8,
        use_locking=False,
        name="QHAdam",
    ):
        super().__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._learning_rate_tensor = None
        self._beta1 = beta1
        self._beta1_tensor = None
        self._beta2 = beta2
        self._beta2_tensor = None
        self._nu1 = nu1
        self._nu1_tensor = None
        self._nu2 = nu2
        self._nu2_tensor = None
        self._epsilon = epsilon
        self._epsilon_tensor = None

    def _get_beta_weights(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
        return (
            self._get_non_slot_variable("beta1_weight", graph=graph),
            self._get_non_slot_variable("beta2_weight", graph=graph),
        )

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        zero = ops.convert_to_tensor(0.0, dtype=dtypes.float64)
        self._create_non_slot_variable(initial_value=zero, name="beta1_weight", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=zero, name="beta2_weight", colocate_with=first_var)

        for v in var_list:
            self._zeros_slot(v, "exp_avg", self._name)
            self._zeros_slot(v, "exp_avg_sq", self._name)

    def _prepare(self):
        learning_rate = call_if_callable(self._learning_rate)
        self._learning_rate_tensor = ops.convert_to_tensor(learning_rate, dtype=dtypes.float64, name="learning_rate")

        beta1 = call_if_callable(self._beta1)
        self._beta1_tensor = ops.convert_to_tensor(beta1, dtype=dtypes.float64, name="beta1")

        beta2 = call_if_callable(self._beta2)
        self._beta2_tensor = ops.convert_to_tensor(beta2, dtype=dtypes.float64, name="beta2")

        nu1 = call_if_callable(self._nu1)
        self._nu1_tensor = ops.convert_to_tensor(nu1, dtype=dtypes.float64, name="nu1")

        nu2 = call_if_callable(self._nu2)
        self._nu2_tensor = ops.convert_to_tensor(nu2, dtype=dtypes.float64, name="nu2")

        epsilon = call_if_callable(self._epsilon)
        self._epsilon_tensor = ops.convert_to_tensor(epsilon, dtype=dtypes.float64, name="epsilon")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            beta1_weight, beta2_weight = self._get_beta_weights()
            with ops.colocate_with(beta1_weight):
                update_beta1 = beta1_weight.assign(
                    beta1_weight * self._beta1_tensor + 1.0, use_locking=self._use_locking
                )
                update_beta2 = beta2_weight.assign(
                    beta2_weight * self._beta2_tensor + 1.0, use_locking=self._use_locking
                )

                return control_flow_ops.group(*(update_ops + [update_beta1, update_beta2]), name=name_scope)

    def _apply_dense_shared(self, grad, var):
        beta1_weight, beta2_weight = self._get_beta_weights()

        learning_rate_tensor = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        beta1_tensor = math_ops.cast(self._beta1_tensor, var.dtype.base_dtype)
        beta2_tensor = math_ops.cast(self._beta2_tensor, var.dtype.base_dtype)
        nu1_tensor = math_ops.cast(self._nu1_tensor, var.dtype.base_dtype)
        nu2_tensor = math_ops.cast(self._nu2_tensor, var.dtype.base_dtype)
        epsilon_tensor = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)

        beta1_weight = math_ops.cast(beta1_weight, var.dtype.base_dtype) * beta1_tensor + 1.0
        beta2_weight = math_ops.cast(beta2_weight, var.dtype.base_dtype) * beta2_tensor + 1.0

        beta1_adj = 1.0 - (1.0 / beta1_weight)
        beta2_adj = 1.0 - (1.0 / beta2_weight)

        exp_avg = self.get_slot(var, "exp_avg")
        exp_avg_sq = self.get_slot(var, "exp_avg_sq")

        grad_sq = grad * grad

        exp_avg_tensor = state_ops.assign(
            exp_avg, beta1_adj * exp_avg + (1.0 - beta1_adj) * grad, use_locking=self._use_locking
        )
        exp_avg_sq_tensor = state_ops.assign(
            exp_avg_sq, beta2_adj * exp_avg_sq + (1.0 - beta2_adj) * grad_sq, use_locking=self._use_locking
        )

        avg_grad_tensor = nu1_tensor * exp_avg_tensor + (1.0 - nu1_tensor) * grad
        avg_grad_sq_tensor = nu2_tensor * exp_avg_sq_tensor + (1.0 - nu2_tensor) * grad_sq
        avg_grad_rms_tensor = math_ops.sqrt(avg_grad_sq_tensor)

        var_update = state_ops.assign_add(
            var,
            -learning_rate_tensor * avg_grad_tensor / (avg_grad_rms_tensor + epsilon_tensor),
            use_locking=self._use_locking,
        )

        return control_flow_ops.group(*[var_update, exp_avg_tensor, exp_avg_sq_tensor])

    def _apply_dense(self, grad, var):
        return self._apply_dense_shared(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self._apply_dense_shared(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_weight, beta2_weight = self._get_beta_weights()

        learning_rate_tensor = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        beta1_tensor = math_ops.cast(self._beta1_tensor, var.dtype.base_dtype)
        beta2_tensor = math_ops.cast(self._beta2_tensor, var.dtype.base_dtype)
        nu1_tensor = math_ops.cast(self._nu1_tensor, var.dtype.base_dtype)
        nu2_tensor = math_ops.cast(self._nu2_tensor, var.dtype.base_dtype)
        epsilon_tensor = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)

        beta1_weight = math_ops.cast(beta1_weight, var.dtype.base_dtype) * beta1_tensor + 1.0
        beta2_weight = math_ops.cast(beta2_weight, var.dtype.base_dtype) * beta2_tensor + 1.0

        beta1_adj = 1.0 - (1.0 / beta1_weight)
        beta2_adj = 1.0 - (1.0 / beta2_weight)

        exp_avg = self.get_slot(var, "exp_avg")
        exp_avg_sq = self.get_slot(var, "exp_avg_sq")

        grad_sq = grad * grad

        exp_avg_tensor = state_ops.assign(exp_avg, beta1_adj * exp_avg, use_locking=self._use_locking)
        with ops.control_dependencies([exp_avg_tensor]):
            exp_avg_tensor = scatter_add(exp_avg, indices, (1.0 - beta1_adj) * grad)

        exp_avg_sq_tensor = state_ops.assign(exp_avg_sq, beta2_adj * exp_avg_sq, use_locking=self._use_locking)
        with ops.control_dependencies([exp_avg_sq_tensor]):
            exp_avg_sq_tensor = scatter_add(exp_avg_sq, indices, (1.0 - beta2_adj) * grad_sq)

        avg_grad = slot_creator.create_zeros_slot(var, self._name)
        avg_grad_tensor = state_ops.assign(avg_grad, nu1_tensor * exp_avg_tensor, use_locking=self._use_locking)
        with ops.control_dependencies([avg_grad_tensor]):
            avg_grad_tensor = scatter_add(avg_grad, indices, (1.0 - nu1_tensor) * grad)

        avg_grad_sq = slot_creator.create_zeros_slot(var, self._name)
        avg_grad_sq_tensor = state_ops.assign(
            avg_grad_sq, nu2_tensor * exp_avg_sq_tensor, use_locking=self._use_locking
        )
        with ops.control_dependencies([avg_grad_sq_tensor]):
            avg_grad_sq_tensor = scatter_add(avg_grad_sq, indices, (1.0 - nu2_tensor) * grad_sq)

        avg_grad_rms_tensor = math_ops.sqrt(avg_grad_sq_tensor)

        var_update = state_ops.assign_add(
            var,
            -learning_rate_tensor * avg_grad_tensor / (avg_grad_rms_tensor + epsilon_tensor),
            use_locking=self._use_locking,
        )

        return control_flow_ops.group(*[var_update, exp_avg_tensor, exp_avg_sq_tensor])

    def _apply_sparse(self, grad, var):
        def scatter_add(x, i, v):
            return state_ops.scatter_add(x, i, v, use_locking=self._use_locking)

        return self._apply_sparse_shared(grad.values, var, grad.indices, scatter_add)

    def _resource_apply_sparse(self, grad, var, indices):
        def resource_scatter_add(x, i, v):
            with ops.control_dependencies([resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
                return x.value()

        return self._apply_sparse_shared(grad, var, indices, resource_scatter_add)

    @classmethod
    def _params_to_dict(cls, params):
        return {
            "learning_rate": params.alpha,
            "nu1": params.nu1,
            "nu2": params.nu2,
            "beta1": params.beta1,
            "beta2": params.beta2,
        }

    @classmethod
    def from_nadam(cls, learning_rate=1e-3, beta1=0.9, beta2=0.999):
        r"""Calculates the QHAdam hyperparameters required to recover the NAdam
        optimizer `(Dozat, 2016)`_.

        This is *not* an identical recovery of the formulation in the paper, due
        to subtle differences in the application of the bias correction in the
        first moment estimator. However, in practice, this difference is almost
        certainly irrelevant.

        Args:
            learning_rate(float, optional):
                learning rate (:math:`\alpha` from the paper)
                (default: 1e-3)
            beta1 (float, optional):
                coefficient used for computing running average of gradient
                (default: 0.9)
            beta2 (float, optional)
                coefficients used for computing running averages of squared
                gradient
                (default: 0.999)

        Returns:
            Five-element ``dict`` containing ``learning_rate``, ``beta1``,
            ``beta2``, ``nu1``, and ``nu2`` to use in QHAdam.

        Example:
            >>> optimizer = qhoptim.tf.QHAdamOptimizer(
            ...     **qhoptim.tf.QHAdamOptimizer.from_nadam(
            ...         learning_rate=1e-3, beta1=0.9, beta2=0.999))

        .. _`(Dozat, 2016)`: https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        """
        return cls._params_to_dict(param_conv.from_nadam(learning_rate, beta1, beta2))
