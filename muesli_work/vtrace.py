from __future__ import annotations

import abc
import functools
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Callable, List, Union

import torch
from tensordict import TensorDictBase
from tensordict.nn import (
    CompositeDistribution,
    dispatch,
    ProbabilisticTensorDictModule,
    set_skip_existing,
    TensorDictModule,
    TensorDictModuleBase,
)
from tensordict.nn.probabilistic import interaction_type
from tensordict.utils import NestedKey
from torch import Tensor

from torchrl._utils import RL_WARNINGS
from torchrl.envs.utils import step_mdp

from torchrl.objectives.utils import _vmap_func, hold_out_net, RANDOM_MODULE_LIST
from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    td0_return_estimate,
    td_lambda_return_estimate,
    vec_generalized_advantage_estimate,
    vec_td1_return_estimate,
    vec_td_lambda_return_estimate,
    vtrace_advantage_estimate,
)
from torchrl.objectives.value import ValueEstimatorBase
try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling

try:
    from torch import vmap
except ImportError as err:
    try:
        from functorch import vmap
    except ImportError:
        raise ImportError(
            "vmap couldn't be found. Make sure you have torch>2.0 installed."
        ) from err

def _self_set_grad_enabled(fun):
    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        with torch.set_grad_enabled(self.differentiable):
            return fun(self, *args, **kwargs)

    return new_fun

def _self_set_skip_existing(fun):
    @functools.wraps(fun)
    def new_func(self, *args, **kwargs):
        if self.skip_existing is not None:
            with set_skip_existing(self.skip_existing):
                return fun(self, *args, **kwargs)
        return fun(self, *args, **kwargs)

    return new_func

def _call_actor_net(
    actor_net: ProbabilisticTensorDictModule,
    data: TensorDictBase,
    params: TensorDictBase,
    log_prob_key: NestedKey,
):
    dist = actor_net.get_dist(data.select(*actor_net.in_keys, strict=False))
    if isinstance(dist, CompositeDistribution):
        kwargs = {
            "aggregate_probabilities": True,
            "inplace": False,
            "include_sum": False,
        }
    else:
        kwargs = {}
    
    #s = actor_net._dist_sample(dist, interaction_type=interaction_type())
    return dist.log_prob(data["action"], **kwargs)

class VTrace(ValueEstimatorBase):
    """A class wrapper around V-Trace estimate functional.

    Refer to "IMPALA: Scalable Distributed Deep-RL with Importance Weighted  Actor-Learner Architectures"
    :ref:`here <https://arxiv.org/abs/1802.01561>`_ for more context.

    Keyword Args:
        gamma (scalar): exponential mean discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        actor_network (TensorDictModule): actor operator used to retrieve the log prob.
        rho_thresh (Union[float, Tensor]): rho clipping parameter for importance weights.
            Defaults to ``1.0``.
        c_thresh (Union[float, Tensor]): c clipping parameter for importance weights.
            Defaults to ``1.0``.
        average_adv (bool): if ``True``, the resulting advantage values will be standardized.
            Default is ``False``.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.
        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, i.e., the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
            Defaults to "state_value".
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        shifted (bool, optional): if ``True``, the value and next value are
            estimated with a single call to the value network. This is faster
            but is only valid whenever (1) the ``"next"`` value is shifted by
            only one time step (which is not the case with multi-step value
            estimation, for instance) and (2) when the parameters used at time
            ``t`` and ``t+1`` are identical (which is not the case when target
            parameters are to be used). Defaults to ``False``.
        device (torch.device, optional): the device where the buffers will be instantiated.
            Defaults to ``torch.get_default_device()``.
        time_dim (int, optional): the dimension corresponding to the time
            in the input tensordict. If not provided, defaults to the dimension
            markes with the ``"time"`` name if any, and to the last dimension
            otherwise. Can be overridden during a call to
            :meth:`~.value_estimate`.
            Negative dimensions are considered with respect to the input
            tensordict.

    VTrace will return an :obj:`"advantage"` entry containing the advantage value. It will also
    return a :obj:`"value_target"` entry with the V-Trace target value.

    .. note::
      As other advantage functions do, if the ``value_key`` is already present
      in the input tensordict, the VTrace module will ignore the calls to the value
      network (if any) and use the provided value instead.

    """

    def __init__(
        self,
        *,
        gamma: float | torch.Tensor,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        rho_thresh: float | torch.Tensor = 1.0,
        c_thresh: float | torch.Tensor = 1.0,
        average_adv: bool = False,
        differentiable: bool = False,
        skip_existing: bool | None = None,
        advantage_key: NestedKey | None = None,
        value_target_key: NestedKey | None = None,
        value_key: NestedKey | None = None,
        shifted: bool = False,
        device: torch.device | None = None,
        time_dim: int | None = None,
    ):
        super().__init__(
            shifted=shifted,
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
            device=device,
        )
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, device=self._device)
        if not isinstance(rho_thresh, torch.Tensor):
            rho_thresh = torch.tensor(rho_thresh, device=self._device)
        if not isinstance(c_thresh, torch.Tensor):
            c_thresh = torch.tensor(c_thresh, device=self._device)

        self.register_buffer("gamma", gamma)
        self.register_buffer("rho_thresh", rho_thresh)
        self.register_buffer("c_thresh", c_thresh)
        self.average_adv = average_adv
        self.actor_network = actor_network
        self.time_dim = time_dim

        if isinstance(gamma, torch.Tensor) and gamma.shape != ():
            raise NotImplementedError(
                "Per-value gamma is not supported yet. Gamma must be a scalar."
            )

    @property
    def in_keys(self):
        parent_in_keys = super().in_keys
        extended_in_keys = parent_in_keys + [self.tensor_keys.sample_log_prob]
        return extended_in_keys

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: List[Tensor] | None = None,
        target_params: List[Tensor] | None = None,
        time_dim: int | None = None,
    ) -> TensorDictBase:
        """Computes the V-Trace correction given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value estimates and the GAE.
                The data passed to this module should be structured as :obj:`[*B, T, F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).

        Keyword Args:
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.
            time_dim (int, optional): the dimension corresponding to the time
                in the input tensordict. If not provided, defaults to the dimension
                markes with the ``"time"`` name if any, and to the last dimension
                otherwise.
                Negative dimensions are considered with respect to the input
                tensordict.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> value_net = TensorDictModule(nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"])
            >>> actor_net = TensorDictModule(nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"])
            >>> actor_net = ProbabilisticActor(
            ...     module=actor_net,
            ...     in_keys=["logits"],
            ...     out_keys=["action"],
            ...     distribution_class=OneHotCategorical,
            ...     return_log_prob=True,
            ... )
            >>> module = VTrace(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ...     actor_network=actor_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> sample_log_prob = torch.randn(1, 10, 1)
            >>> tensordict = TensorDict({
            ...     "obs": obs,
            ...     "done": done,
            ...     "terminated": terminated,
            ...     "sample_log_prob": sample_log_prob,
            ...     "next": {"obs": next_obs, "reward": reward, "done": done, "terminated": terminated},
            ... }, batch_size=[1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"])
            >>> actor_net = TensorDictModule(nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"])
            >>> actor_net = ProbabilisticActor(
            ...     module=actor_net,
            ...     in_keys=["logits"],
            ...     out_keys=["action"],
            ...     distribution_class=OneHotCategorical,
            ...     return_log_prob=True,
            ... )
            >>> module = VTrace(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ...     actor_network=actor_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> sample_log_prob = torch.randn(1, 10, 1)
            >>> tensordict = TensorDict({
            ...     "obs": obs,
            ...     "done": done,
            ...     "terminated": terminated,
            ...     "sample_log_prob": sample_log_prob,
            ...     "next": {"obs": next_obs, "reward": reward, "done": done, "terminated": terminated},
            ... }, batch_size=[1, 10])
            >>> advantage, value_target = module(
            ...     obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated, sample_log_prob=sample_log_prob
            ... )

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got "
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device

        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        # Make sure we have the value and next value
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network):
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        # Make sure we have the log prob computed at collection time
        if self.tensor_keys.sample_log_prob not in tensordict.keys():
            raise ValueError(
                f"Expected {self.tensor_keys.sample_log_prob} to be in tensordict"
            )
        log_mu = tensordict.get(self.tensor_keys.sample_log_prob).view_as(value)

        # Compute log prob with current policy
        with hold_out_net(self.actor_network):
            log_pi = _call_actor_net(
                actor_net=self.actor_network,
                data=tensordict,
                params=None,
                log_prob_key=self.tensor_keys.sample_log_prob,
            )
            log_pi = log_pi.view_as(value)

        # Compute the V-Trace correction
        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated))

        time_dim = self._get_time_dim(time_dim, tensordict)
        adv, value_target = vtrace_advantage_estimate(
            gamma,
            log_pi,
            log_mu,
            value,
            next_value,
            reward,
            done,
            terminated,
            rho_thresh=self.rho_thresh,
            c_thresh=self.c_thresh,
            time_dim=time_dim,
        )

        if self.average_adv:
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-5)
            adv = adv - loc
            adv = adv / scale

        tensordict.set(self.tensor_keys.advantage, adv)
        tensordict.set(self.tensor_keys.value_target, value_target)

        return tensordict
