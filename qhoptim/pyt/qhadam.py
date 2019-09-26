# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer

from ..common import param_conv


class QHAdam(Optimizer):
    r"""Implements the QHAdam optimization algorithm `(Ma and Yarats, 2019)`_.

    Note that the NAdam optimizer is accessible via a specific parameterization
    of QHAdam. See :func:`from_nadam()` for details.

    Args:
        params (iterable):
            iterable of parameters to optimize or dicts defining parameter
            groups
        lr (float, optional): learning rate (:math:`\alpha` from the paper)
            (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of the gradient and its square
            (default: (0.9, 0.999))
        nus (Tuple[float, float], optional): immediate discount factors used to
            estimate the gradient and its square
            (default: (1.0, 1.0))
        eps (float, optional): term added to the denominator to improve
            numerical stability
            (default: 1e-8)
        weight_decay (float, optional): weight decay (default: 0.0)
        decouple_weight_decay (bool, optional): whether to decouple the weight
            decay from the gradient-based optimization step
            (default: False)

    Example:
        >>> optimizer = qhoptim.pyt.QHAdam(
        ...     model.parameters(),
        ...     lr=3e-4, nus=(0.8, 1.0), betas=(0.99, 0.999))
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. _`(Ma and Yarats, 2019)`: https://arxiv.org/abs/1810.06801
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        nus=(1.0, 1.0),
        weight_decay=0.0,
        decouple_weight_decay=False,
        eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "betas": betas,
            "nus": nus,
            "weight_decay": weight_decay,
            "decouple_weight_decay": decouple_weight_decay,
            "eps": eps,
        }
        super(QHAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nu1, nu2 = group["nus"]
            weight_decay = group["weight_decay"]
            decouple_weight_decay = group["decouple_weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError("QHAdam does not support sparse gradients")

                param_state = self.state[p]

                if weight_decay != 0:
                    if decouple_weight_decay:
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        d_p.add_(weight_decay, p.data)

                d_p_sq = d_p.mul(d_p)

                if len(param_state) == 0:
                    param_state["beta1_weight"] = 0.0
                    param_state["beta2_weight"] = 0.0
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)

                param_state["beta1_weight"] = 1.0 + beta1 * param_state["beta1_weight"]
                param_state["beta2_weight"] = 1.0 + beta2 * param_state["beta2_weight"]

                beta1_weight = param_state["beta1_weight"]
                beta2_weight = param_state["beta2_weight"]
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                beta1_adj = 1.0 - (1.0 / beta1_weight)
                beta2_adj = 1.0 - (1.0 / beta2_weight)
                exp_avg.mul_(beta1_adj).add_(1.0 - beta1_adj, d_p)
                exp_avg_sq.mul_(beta2_adj).add_(1.0 - beta2_adj, d_p_sq)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(1.0 - nu1, d_p)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(1.0 - nu2, d_p_sq)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)

                p.data.addcdiv_(-lr, avg_grad, avg_grad_rms)

        return loss

    @classmethod
    def _params_to_dict(cls, params):
        return {"lr": params.alpha, "nus": (params.nu1, params.nu2), "betas": (params.beta1, params.beta2)}

    @classmethod
    def from_nadam(cls, lr=1e-3, betas=(0.9, 0.999)):
        r"""Calculates the QHAdam hyperparameters required to recover the NAdam
        optimizer `(Dozat, 2016)`_.

        This is *not* an identical recovery of the formulation in the paper, due
        to subtle differences in the application of the bias correction in the
        first moment estimator. However, in practice, this difference is almost
        certainly irrelevant.

        Args:
            lr (float, optional):
                learning rate (:math:`\alpha` from the paper)
                (default: 1e-3)
            betas (Tuple[float, float], optional):
                coefficients used for computing running averages of the
                gradient and its square
                (default: (0.9, 0.999))

        Returns:
            Three-element ``dict`` containing ``lr``, ``betas``, and ``nus``
            to use in QHAdam.

        Example:
            >>> optimizer = qhoptim.pyt.QHAdam(
            ...     model.parameters(),
            ...     weight_decay=1e-4,
            ...     **qhoptim.pyt.QHAdam.from_nadam(
            ...         lr=1e-3, betas=(0.9, 0.999)))

        .. _`(Dozat, 2016)`: https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        """
        return cls._params_to_dict(param_conv.from_nadam(lr, betas[0], betas[1]))


def QHAdamW(params, *args, **kwargs):
    r"""Constructs the decoupled decay variant of the QHAdam optimization
    algorithm `(Ma and Yarats, 2019)`_,
    as proposed by `Loschilov and Hutter (2017)`_.

    Shares all arguments of the :class:`QHAdam` constructor –
    equivalent to constructing :class:`QHAdam` with
    ``decouple_weight_decay=True``.

    .. _`Loschilov and Hutter (2017)`: https://arxiv.org/abs/1711.05101
    .. _`(Ma and Yarats, 2019)`: https://arxiv.org/abs/1810.06801
    """
    return QHAdam(params, *args, decouple_weight_decay=True, **kwargs)
