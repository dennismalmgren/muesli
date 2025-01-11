# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict import (
    is_tensor_collection,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.nn import (
    CompositeDistribution,
    dispatch,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl.objectives.common import LossModule

from torchrl.objectives.utils import (
    _cache_values,
    _clip_value_loss,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    _sum_td_features,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import (
    GAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)

from dynamics_prediction import compute_ensemble_disagreement, compute_jacobian_norm_ensemble

class PPOLoss(LossModule):
    """A parent PPO loss class.

    PPO (Proximal Policy Optimization) is a model-free, online RL algorithm
    that makes use of a recorded (batch of)
    trajectories to perform several optimization steps, while actively
    preventing the updated policy to deviate too
    much from its original parameter configuration.

    PPO loss can be found in different flavors, depending on the way the
    constrained optimization is implemented: ClipPPOLoss and KLPENPPOLoss.
    Unlike its subclasses, this class does not implement any regularization
    and should therefore be used cautiously.

    For more details regarding PPO, refer to: "Proximal Policy Optimization Algorithms",
    https://arxiv.org/abs/1707.06347

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.

    Keyword Args:
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``. Set ``critic_coef`` to ``None`` to exclude the value
            loss from the forward outputs.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.
        functional (bool, optional): whether modules should be functionalized.
            Functionalizing permits features like meta-RL, but makes it
            impossible to use distributed models (DDP, FSDP, ...) and comes
            with a little cost. Defaults to ``True``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (:obj:`float`, optional): If provided, it will be used to compute a clipped version of the value
            prediction with respect to the input tensordict value estimate and use it to calculate the value loss.
            The purpose of clipping is to limit the impact of extreme value predictions, helping stabilize training
            and preventing large updates. However, it will have no impact if the value estimate was done by the current
            version of the value estimator. Defaults to ``None``.

    .. note::
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> ppo_loss = PPOLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = ppo_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> ppo_loss = PPOLoss(actor, critic)
        >>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)

    .. note::
      If the actor and the value function share parameters, one can avoid
      calling the common module multiple times by passing only the head of the
      value network to the PPO loss module:

        >>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
        >>> actor_head = SomeActor(in_keys=["hidden"])
        >>> value_head = SomeValue(in_keys=["hidden"])
        >>> # first option, with 2 calls on the common module
        >>> model = ActorValueOperator(common, actor_head, value_head)
        >>> loss_module = PPOLoss(model.get_policy_operator(), model.get_value_operator())
        >>> # second option, with a single call to the common module
        >>> loss_module = PPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)

      This will work regardless of whether separate_losses is activated or not.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data.tensor_specs import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.ppo import PPOLoss
        >>> from tensordict import TensorDict
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> base_layer = nn.Linear(n_obs, 5)
        >>> net = nn.Sequential(base_layer, nn.Linear(5, 2 * n_act), NormalParamExtractor())
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> actor = ProbabilisticActor(
        ...     module=module,
        ...     distribution_class=TanhNormal,
        ...     in_keys=["loc", "scale"],
        ...     spec=spec)
        >>> module = nn.Sequential(base_layer, nn.Linear(5, 1))
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation"])
        >>> loss = PPOLoss(actor, value)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> data = TensorDict({"observation": torch.randn(*batch, n_obs),
        ...         "action": action,
        ...         "sample_log_prob": torch.randn_like(action[..., 1]),
        ...         ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...         ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...         ("next", "reward"): torch.randn(*batch, 1),
        ...         ("next", "observation"): torch.randn(*batch, n_obs),
        ...     }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_critic: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_objective: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "sample_log_prob", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor and value network.
    The return value is a tuple of tensors in the following order:
    ``["loss_objective"]`` + ``["entropy", "loss_entropy"]`` if entropy_bonus is set + ``"loss_critic"`` if critic_coef is not ``None``.
    The output keys can also be filtered using :meth:`PPOLoss.select_out_keys` method.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data.tensor_specs import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.ppo import PPOLoss
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> base_layer = nn.Linear(n_obs, 5)
        >>> net = nn.Sequential(base_layer, nn.Linear(5, 2 * n_act), NormalParamExtractor())
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> actor = ProbabilisticActor(
        ...     module=module,
        ...     distribution_class=TanhNormal,
        ...     in_keys=["loc", "scale"],
        ...     spec=spec)
        >>> module = nn.Sequential(base_layer, nn.Linear(5, 1))
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation"])
        >>> loss = PPOLoss(actor, value)
        >>> loss.set_keys(sample_log_prob="sampleLogProb")
        >>> _ = loss.select_out_keys("loss_objective")
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> loss_objective = loss(
        ...         observation=torch.randn(*batch, n_obs),
        ...         action=action,
        ...         sampleLogProb=torch.randn_like(action[..., 1]) / 10,
        ...         next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...         next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
        ...         next_reward=torch.randn(*batch, 1),
        ...         next_observation=torch.randn(*batch, n_obs))
        >>> loss_objective.backward()

    .. note::
      There is an exception regarding compatibility with non-tensordict-based modules.
      If the actor network is probabilistic and uses a :class:`~tensordict.nn.distributions.CompositeDistribution`,
      this class must be used with tensordicts and cannot function as a tensordict-independent module.
      This is because composite action spaces inherently rely on the structured representation of data provided by
      tensordicts to handle their actions.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            sample_log_prob (NestedKey): The input tensordict key where the
               sample log probability is expected.  Defaults to ``"sample_log_prob"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        sample_log_prob: NestedKey = "sample_log_prob"
        action: NestedKey = "action"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.GAE

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        advantage_key: str = None,
        value_target_key: str = None,
        value_key: str = None,
        functional: bool = True,
        actor: ProbabilisticTensorDictSequential = None,
        critic: ProbabilisticTensorDictSequential = None,
        reduction: str = None,
        clip_value: float | None = None,
        **kwargs,
    ):
        if actor is not None:
            actor_network = actor
            del actor
        if critic is not None:
            critic_network = critic
            del critic
        if actor_network is None or critic_network is None:
            raise TypeError(
                "Missing positional arguments actor_network or critic_network."
            )
        if reduction is None:
            reduction = "mean"

        self._functional = functional
        self._in_keys = None
        self._out_keys = None
        super().__init__()
        if functional:
            self.convert_to_functional(actor_network, "actor_network")
        else:
            self.actor_network = actor_network
            self.actor_network_params = None
            self.target_actor_network_params = None

        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        if functional:
            self.convert_to_functional(
                critic_network, "critic_network", compare_against=policy_params
            )
        else:
            self.critic_network = critic_network
            self.critic_network_params = None
            self.target_critic_network_params = None

        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus
        self.separate_losses = separate_losses
        self.reduction = reduction

        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")

        self.register_buffer("entropy_coef", torch.tensor(entropy_coef, device=device))
        if critic_coef is not None:
            self.register_buffer(
                "critic_coef", torch.tensor(critic_coef, device=device)
            )
        else:
            self.critic_coef = None
        self.loss_critic_type = loss_critic_type
        self.normalize_advantage = normalize_advantage
        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        self._set_deprecated_ctor_keys(
            advantage=advantage_key,
            value_target=value_target_key,
            value=value_key,
        )

        if clip_value is not None:
            if isinstance(clip_value, float):
                clip_value = torch.tensor(clip_value)
            elif isinstance(clip_value, torch.Tensor):
                if clip_value.numel() != 1:
                    raise ValueError(
                        f"clip_value must be a float or a scalar tensor, got {clip_value}."
                    )
            else:
                raise ValueError(
                    f"clip_value must be a float or a scalar tensor, got {clip_value}."
                )
        self.register_buffer("clip_value", clip_value)

    @property
    def functional(self):
        return self._functional

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            self.tensor_keys.sample_log_prob,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
            *self.critic_network.in_keys,
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if hasattr(self, "_value_estimator") and self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self.tensor_keys.advantage,
                value_target=self.tensor_keys.value_target,
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )
        self._set_in_keys()

    def reset(self) -> None:
        pass

    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        try:
            if isinstance(dist, CompositeDistribution):
                kwargs = {"aggregate_probabilities": False, "include_sum": False}
            else:
                kwargs = {}
            entropy = dist.entropy(**kwargs)
            if is_tensor_collection(entropy):
                entropy = _sum_td_features(entropy)
        except NotImplementedError:
            x = dist.rsample((self.samples_mc_entropy,))
            log_prob = dist.log_prob(x)
            if is_tensor_collection(log_prob):
                log_prob = log_prob.get(self.tensor_keys.sample_log_prob)
            entropy = -log_prob.mean(0)
        return entropy.unsqueeze(-1)

    def _log_weight(
        self, tensordict: TensorDictBase
    ) -> Tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        action = tensordict.get(self.tensor_keys.action)

        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            dist = self.actor_network.get_dist(tensordict)

        prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.sample_log_prob} requires grad."
            )

        if action.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.action} requires grad."
            )
        if isinstance(action, torch.Tensor):
            log_prob = dist.log_prob(action)
        else:
            if isinstance(dist, CompositeDistribution):
                is_composite = True
                kwargs = {
                    "inplace": False,
                    "aggregate_probabilities": False,
                    "include_sum": False,
                }
            else:
                is_composite = False
                kwargs = {}
            log_prob = dist.log_prob(tensordict, **kwargs)
            if is_composite and not isinstance(prev_log_prob, TensorDict):
                log_prob = _sum_td_features(log_prob)
                log_prob.view_as(prev_log_prob)

        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        kl_approx = (prev_log_prob - log_prob).unsqueeze(-1)

        return log_weight, dist, kl_approx

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Returns the critic loss multiplied by ``critic_coef``, if it is not ``None``."""
        # TODO: if the advantage is gathered by forward, this introduces an
        # overhead that we could easily reduce.
        if self.separate_losses:
            tensordict = tensordict.detach()
        target_return = tensordict.get(
            self.tensor_keys.value_target, None
        )  # TODO: None soon to be removed
        if target_return is None:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )

        if self.clip_value:
            old_state_value = tensordict.get(
                self.tensor_keys.value, None
            )  # TODO: None soon to be removed
            if old_state_value is None:
                raise KeyError(
                    f"clip_value is set to {self.clip_value}, but "
                    f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                    f"Make sure that the value_key passed to PPO exists in the input tensordict."
                )

        with self.critic_network_params.to_module(
            self.critic_network
        ) if self.functional else contextlib.nullcontext():
            state_value_td = self.critic_network(tensordict)

        state_value = state_value_td.get(
            self.tensor_keys.value, None
        )  # TODO: None soon to be removed
        if state_value is None:
            raise KeyError(
                f"the key {self.tensor_keys.value} was not found in the critic output tensordict. "
                f"Make sure that the value_key passed to PPO is accurate."
            )

        loss_value = distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )

        clip_fraction = None
        if self.clip_value:
            loss_value, clip_fraction = _clip_value_loss(
                old_state_value,
                state_value,
                self.clip_value.to(state_value.device),
                target_return,
                loss_value,
                self.loss_critic_type,
            )

        if self.critic_coef is not None:
            return self.critic_coef * loss_value, clip_fraction
        return loss_value, clip_fraction

    @property
    @_cache_values
    def _cached_critic_network_params_detached(self):
        if not self.functional:
            return None
        return self.critic_network_params.detach()

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale

        log_weight, dist, kl_approx = self._log_weight(tensordict)
        if is_tensor_collection(log_weight):
            log_weight = _sum_td_features(log_weight)
            log_weight = log_weight.view(advantage.shape)
        neg_loss = log_weight.exp() * advantage
        td_out = TensorDict({"loss_objective": -neg_loss}, batch_size=[])
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.critic_coef is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.GAE:
            self._value_estimator = GAE(value_network=self.critic_network, **hp)
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.VTrace:
            # VTrace currently does not support functional call on the actor
            if self.functional:
                actor_with_params = deepcopy(self.actor_network)
                self.actor_network_params.to_module(actor_with_params)
            else:
                actor_with_params = self.actor_network
            self._value_estimator = VTrace(
                value_network=self.critic_network, actor_network=actor_with_params, **hp
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value": self.tensor_keys.value,
            "value_target": self.tensor_keys.value_target,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
            "sample_log_prob": self.tensor_keys.sample_log_prob,
        }
        self._value_estimator.set_keys(**tensor_keys)


class ClipPPOLoss(PPOLoss):
    """Clipped PPO loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.

    Keyword Args:
        clip_epsilon (scalar, optional): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``. Set ``critic_coef`` to ``None`` to exclude the value
            loss from the forward outputs.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.
        functional (bool, optional): whether modules should be functionalized.
            Functionalizing permits features like meta-RL, but makes it
            impossible to use distributed models (DDP, FSDP, ...) and comes
            with a little cost. Defaults to ``True``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (bool or float, optional): If a ``float`` is provided, it will be used to compute a clipped
            version of the value prediction with respect to the input tensordict value estimate and use it to
            calculate the value loss. The purpose of clipping is to limit the impact of extreme value predictions,
            helping stabilize training and preventing large updates. However, it will have no impact if the value
            estimate was done by the current version of the value estimator. If instead ``True`` is provided, the
            ``clip_epsilon`` parameter will be used as the clipping threshold. If not provided or ``False``, no
            clipping will be performed. Defaults to ``False``.

    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = ppo_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)

    .. note::
      If the actor and the value function share parameters, one can avoid
      calling the common module multiple times by passing only the head of the
      value network to the PPO loss module:

        >>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
        >>> actor_head = SomeActor(in_keys=["hidden"])
        >>> value_head = SomeValue(in_keys=["hidden"])
        >>> # first option, with 2 calls on the common module
        >>> model = ActorValueOperator(common, actor_head, value_head)
        >>> loss_module = ClipPPOLoss(model.get_policy_operator(), model.get_value_operator())
        >>> # second option, with a single call to the common module
        >>> loss_module = ClipPPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)

      This will work regardless of whether separate_losses is activated or not.

    """

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    dynamics_network: TensorDictModule
    kalman_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    kalman_network_params: TensorDictParams
    dynamics_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams
    target_dynamics_network_params: TensorDictParams
    target_kalman_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        kalman_network: TensorDictModule | None = None,
        dynamics_network: TensorDictModule | None = None,
        policy_calculation_module: TensorDictModule | None = None,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        kalman_coef: float = 1.0,
        dynamics_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        reduction: str = None,
        clip_value: bool | float | None = None,
        functional: bool = True,
        **kwargs,
    ):
        # Define clipping of the value loss
        if isinstance(clip_value, bool):
            clip_value = clip_epsilon if clip_value else None

        super(ClipPPOLoss, self).__init__(
            actor_network,
            critic_network,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            reduction=reduction,
            clip_value=clip_value,
            **kwargs,
        )
        for p in self.parameters():
            device = p.device
            break
        else:
            device = None
        self.register_buffer("clip_epsilon", torch.tensor(clip_epsilon, device=device))
        # if functional:
        #     self.convert_to_functional(kalman_network, "kalman_network")
        # else:
        #     self.kalman_network = kalman_network
        #     self.kalman_network_params = None
        #     self.target_kalman_network_params = None
        if functional:
            self.convert_to_functional(dynamics_network, "dynamics_network")
        else:
            self.dynamics_network = dynamics_network
            self.dynamics_network_params = None
            self.target_dynamics_network_params = None
        self.policy_calculation_module = policy_calculation_module

        if kalman_coef is not None:
            self.register_buffer(
                "kalman_coef", torch.tensor(kalman_coef, device=device)
            )
        else:
            self.kalman_coef = None

        if dynamics_coef is not None:
            self.register_buffer(
                "dynamics_coef", torch.tensor(dynamics_coef, device=device)
            )
        else:
            self.dynamics_coef = None

    @property
    def _clip_bounds(self):
        return (
            (-self.clip_epsilon).log1p(),
            self.clip_epsilon.log1p(),
        )

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective", "clip_fraction", "loss_kalman"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            keys.append("ESS")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values
    
    def smooth_deadband_loss(self, y, y_star, epsilon=0.1, delta=0.01):
        error = torch.abs(y - y_star)
        smooth_factor = torch.nn.functional.sigmoid((error - epsilon) / delta)
        loss = 0.5 * (smooth_factor * (error - epsilon))**2
        return torch.where(error <= epsilon, torch.zeros_like(loss), loss)

    def asymmetric_smooth_deadband_loss(self, y, y_star, epsilon=0.1, delta_below=0.05, delta_above=0.01):
        error = y - y_star
        abs_error = torch.abs(error)
        
        # Smooth factor for below and above
        smooth_below = torch.nn.functional.sigmoid((y_star - y - epsilon) / delta_below)
        smooth_above = torch.nn.functional.sigmoid((y - y_star - epsilon) / delta_above)
        
        # Loss computation
        below_loss = 0.5 * (smooth_below * (y_star - y - epsilon))**2
        above_loss = 0.5 * (smooth_above * (y - y_star - epsilon))**2
        
        # Apply conditions
        loss = torch.where(abs_error <= epsilon, 
                        torch.zeros_like(error), 
                        torch.where(y < y_star, below_loss, above_loss))
        return loss

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        import math
        tensordict = tensordict.clone(False)
        #add the dynamics loss first.
        with self.dynamics_network_params.to_module(
            self.dynamics_network
        ) if self.functional else contextlib.nullcontext():
            dynamics_td = self.dynamics_network(tensordict.select("observation", "action"))
            dynamics_jacobian_norm_unnormed = compute_jacobian_norm_ensemble(self.dynamics_network, tensordict).detach().unsqueeze(-1)
            dynamics_jacobian_norm = dynamics_jacobian_norm_unnormed * 1 / math.sqrt(tensordict["observation"].shape[-1])
            dynamics_variance = compute_ensemble_disagreement(dynamics_td["predicted_state"].detach()) / tensordict["observation"].shape[-1]
        update_index = torch.randint(5, size=()).item()
        target_state = tensordict["next", "observation"]
        predicted_state = dynamics_td["predicted_state"][..., update_index]
        dynamics_loss = torch.nn.functional.mse_loss(predicted_state, target_state, reduction="none").mean(-1, keepdim=True)

        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale

        
        log_weight, dist, kl_approx = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        ratio = log_weight_clip.exp()
        gain2 = ratio * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        dynamics_jacobian_norm_inv = 1 / (dynamics_jacobian_norm + 1e-4)
        dynamics_jacobian_norm_unnormed_inv = 1 / (dynamics_jacobian_norm_unnormed + 1e-4)
        state = tensordict.select("observation")
        state["observation"].requires_grad_(True)
        output = self.policy_calculation_module(state)

        jacobian = torch.autograd.grad(
            outputs=output["loc"],  # sum to handle batch-wise differentiation
            inputs=state["observation"],
            grad_outputs=torch.ones_like(output["loc"]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

            #         outputs=output_td["predicted_state"],
            # inputs=input_td["observation"],
            # grad_outputs=torch.ones_like(output_td["predicted_state"]),
            # create_graph=True,
            # retain_graph=True,
            # allow_unused=True
        
        policy_jacobian_norm_unnormed = jacobian.norm(p='fro', dim=-1, keepdim=True) 
        policy_jacobian_norm = policy_jacobian_norm_unnormed * 1 / math.sqrt(tensordict["action"].shape[-1])
        #reg_loss = 0.01 * 1 / torch.sqrt(disagreement) * torch.nn.functional.mse_loss(lambda_reg * jacobian_norm, torch.ones_like(gain), reduction="none") #ok, next is to scale with expected uncertainty. scaled by dims, again.
        #reg_loss = 0.0001 * 1 / torch.sqrt(disagreement) * torch.nn.functional.mse_loss(torch.clamp(lambda_reg * jacobian_norm, min=1), torch.ones_like(gain), reduction="none") #ok, next is to scale with expected uncertainty. scaled by dims, again.
        jacobian_norm_ratio = dynamics_jacobian_norm_inv * policy_jacobian_norm 
        jacobian_norm_ratio_unnormed = dynamics_jacobian_norm_unnormed_inv * policy_jacobian_norm_unnormed
        jacobian_norm_error = jacobian_norm_ratio #- 0.02 * torch.ones_like(gain)

        #delta = self.smooth_deadband_loss(lambda_reg * jacobian_norm, torch.ones_like(gain), epsilon=0.5)
        #delta = self.asymmetric_smooth_deadband_loss(lambda_reg * jacobian_norm, torch.ones_like(gain), epsilon=0.05)
        ##### CURRENTLY RUNNING ##### Best "compromise" so far. Both environments reach acceptable results.
        # try with lower/higher alpha
        # try without uncertainty factor
        #transformed = torch.nn.functional.elu(delta, alpha=0.2)
        #reg_loss = 0.0001 * 1 / torch.sqrt(disagreement) * error.pow(2)
        reg_loss = 0.1 * jacobian_norm_error.pow(2)
        ##### END OF CURRENTLY RUNNING #####
#        reg_loss = 0.0001 * 1 / torch.sqrt(disagreement) * torch.nn.functional.mse_loss(torch.nn.functional.elu(torch.clamp(lambda_reg * jacobian_norm, min=1), torch.ones_like(gain), reduction="none") #ok, next is to scale with expected uncertainty. scaled by dims, again.

        gain = gain - reg_loss

        td_out = TensorDict({"loss_objective": -gain}, batch_size=[])
        td_out.set("loss_reg", reg_loss)
        td_out.set("loss_dynamics", dynamics_loss)
        td_out.set("clip_fraction", clip_fraction)
        td_out.set("loss_jacobian_norm", reg_loss)
        td_out.set("jacobian_norm_ratio", jacobian_norm_ratio)
        td_out.set("jacobian_norm_dynamics", dynamics_jacobian_norm)
        td_out.set("jacobian_norm_policy", policy_jacobian_norm)
        td_out.set("jacobian_norm_dynamics_unnormed", dynamics_jacobian_norm_unnormed)
        td_out.set("jacobian_norm_policy_unnormed", policy_jacobian_norm_unnormed)
        td_out.set("jacobian_norm_ratio_unnormed", jacobian_norm_ratio_unnormed)
        td_out.set("dynamics_variance", dynamics_variance)

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.critic_coef is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
        # if self.kalman_coef is not None:
        #     loss_kalman = self.loss_kalman(tensordict)
        #     td_out.set("loss_kalman", loss_kalman)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

    def loss_kalman(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Returns the kalman loss multiplied by ``kalman_coef``, if it is not ``None``."""
        # TODO: if the advantage is gathered by forward, this introduces an
        # overhead that we could easily reduce.
        with self.kalman_network_params.to_module(
            self.kalman_network
        ) if self.functional else contextlib.nullcontext():
            kalman_td = self.kalman_network(tensordict)
        #lets just use the current values, for now.
        loss_value = torch.nn.functional.mse_loss(kalman_td["next", "prediction_error"], torch.zeros_like(kalman_td["next", "prediction_error"]), reduction="none")

        return loss_value

class KLPENPPOLoss(PPOLoss):
    """KL Penalty PPO loss.

    The KL penalty loss has the following formula:
        loss = loss - beta * KL(old_policy, new_policy)
    The "beta" parameter is adapted on-the-fly to match a target KL divergence between the new and old policy, thus
    favouring a certain level of distancing between the two while still preventing them to be too much apart.

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.

    Keyword Args:
        dtarg (scalar, optional): target KL divergence. Defaults to ``0.01``.
        samples_mc_kl (int, optional): number of samples used to compute the KL divergence
            if no analytical formula can be found. Defaults to ``1``.
        beta (scalar, optional): initial KL divergence multiplier.
            Defaults to ``1.0``.
        decrement (scalar, optional): how much beta should be decremented if KL < dtarg. Valid range: decrement <= 1.0
            default: ``0.5``.
        increment (scalar, optional): how much beta should be incremented if KL > dtarg. Valid range: increment >= 1.0
            default: ``2.0``.
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies. Defaults to ``True``.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.
        functional (bool, optional): whether modules should be functionalized.
            Functionalizing permits features like meta-RL, but makes it
            impossible to use distributed models (DDP, FSDP, ...) and comes
            with a little cost. Defaults to ``True``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (:obj:`float`, optional): If provided, it will be used to compute a clipped version of the value
            prediction with respect to the input tensordict value estimate and use it to calculate the value loss.
            The purpose of clipping is to limit the impact of extreme value predictions, helping stabilize training
            and preventing large updates. However, it will have no impact if the value estimate was done by the current
            version of the value estimator. Defaults to ``None``.

    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> ppo_loss = KLPENPPOLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = ppo_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> ppo_loss = KLPENPPOLoss(actor, critic)
        >>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)

    .. note::
      If the actor and the value function share parameters, one can avoid
      calling the common module multiple times by passing only the head of the
      value network to the PPO loss module:

        >>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
        >>> actor_head = SomeActor(in_keys=["hidden"])
        >>> value_head = SomeValue(in_keys=["hidden"])
        >>> # first option, with 2 calls on the common module
        >>> model = ActorValueOperator(common, actor_head, value_head)
        >>> loss_module = KLPENPPOLoss(model.get_policy_operator(), model.get_value_operator())
        >>> # second option, with a single call to the common module
        >>> loss_module = KLPENPPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)

      This will work regardless of whether separate_losses is activated or not.

    """

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        dtarg: float = 0.01,
        beta: float = 1.0,
        increment: float = 2,
        decrement: float = 0.5,
        samples_mc_kl: int = 1,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        reduction: str = None,
        clip_value: float | None = None,
        **kwargs,
    ):
        super(KLPENPPOLoss, self).__init__(
            actor_network,
            critic_network,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            reduction=reduction,
            clip_value=clip_value,
            **kwargs,
        )

        self.dtarg = dtarg
        self._beta_init = beta
        self.register_buffer("beta", torch.tensor(beta))

        if increment < 1.0:
            raise ValueError(
                f"increment should be >= 1.0 in KLPENPPOLoss, got {increment:4.4f}"
            )
        self.increment = increment
        if decrement > 1.0:
            raise ValueError(
                f"decrement should be <= 1.0 in KLPENPPOLoss, got {decrement:4.4f}"
            )
        self.decrement = decrement
        self.samples_mc_kl = samples_mc_kl

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            self.tensor_keys.sample_log_prob,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
            *self.critic_network.in_keys,
        ]
        # Get the parameter keys from the actor dist
        actor_dist_module = None
        for module in self.actor_network.modules():
            # Ideally we should combine them if there is more than one
            if isinstance(module, ProbabilisticTensorDictModule):
                if actor_dist_module is not None:
                    raise RuntimeError(
                        "Actors with one and only one distribution are currently supported "
                        f"in {type(self).__name__}. If you need to use more than one "
                        f"distribtuion over the action space please submit an issue "
                        f"on github."
                    )
                actor_dist_module = module
        if actor_dist_module is None:
            raise RuntimeError("Could not find the probabilistic module in the actor.")
        keys += list(actor_dist_module.in_keys)
        self._in_keys = list(set(keys))

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective", "kl"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        tensordict_copy = tensordict.copy()
        try:
            previous_dist = self.actor_network.build_dist_from_params(tensordict)
        except KeyError as err:
            raise KeyError(
                "The parameters of the distribution were not found. "
                f"Make sure they are provided to {type(self).__name__}."
            ) from err
        advantage = tensordict_copy.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict_copy,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict_copy.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale
        log_weight, dist, kl_approx = self._log_weight(tensordict_copy)
        neg_loss = log_weight.exp() * advantage

        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            current_dist = self.actor_network.get_dist(tensordict_copy)
        try:
            kl = torch.distributions.kl.kl_divergence(previous_dist, current_dist)
        except NotImplementedError:
            x = previous_dist.sample((self.samples_mc_kl,))
            if isinstance(previous_dist, CompositeDistribution):
                kwargs = {
                    "aggregate_probabilities": False,
                    "inplace": False,
                    "include_sum": False,
                }
            else:
                kwargs = {}
            previous_log_prob = previous_dist.log_prob(x, **kwargs)
            current_log_prob = current_dist.log_prob(x, **kwargs)
            if is_tensor_collection(current_log_prob):
                previous_log_prob = _sum_td_features(previous_log_prob)
                current_log_prob = _sum_td_features(current_log_prob)
            kl = (previous_log_prob - current_log_prob).mean(0)
        kl = kl.unsqueeze(-1)
        neg_loss = neg_loss - self.beta * kl
        if kl.mean() > self.dtarg * 1.5:
            self.beta.data *= self.increment
        elif kl.mean() < self.dtarg / 1.5:
            self.beta.data *= self.decrement
        td_out = TensorDict(
            {
                "loss_objective": -neg_loss,
                "kl": kl.detach(),
            },
            batch_size=[],
        )

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.critic_coef is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict_copy)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )

        return td_out

    def reset(self) -> None:
        self.beta = self._beta_init