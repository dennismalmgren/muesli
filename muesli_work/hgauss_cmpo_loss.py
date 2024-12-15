
from __future__ import annotations

import contextlib

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple
from torchrl.modules import TanhNormal

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

from torchrl.objectives.ppo import PPOLoss
from dataclasses import dataclass
from retrace import ReTrace

class HGaussCMPOLoss(LossModule):
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
        reward_value: NestedKey = "reward_value"

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
        reward_network: TensorDictModule | None = None,
        *,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        reward_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        reduction: str = None,
        clip_value: bool | float | None = None,
        z_num_samples: int = 16,
        support: torch.Tensor = None,
        **kwargs,
    ):
        # Define clipping of the value loss
        super().__init__()
        self._functional = True
        self.entropy_bonus = entropy_bonus
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.normalize_advantage = normalize_advantage
        self.gamma = gamma
        self.reduction = reduction

        self.reward_network = reward_network
        self.reward_coef = reward_coef
        self.z_num_samples = z_num_samples

        for p in self.parameters():
            device = p.device
            break
        else:
            device = None

        self.convert_to_functional(actor_network,
                                   "actor_network",
                                   create_target_params=True)
        self.convert_to_functional(critic_network,
                                   "critic_network",
                                   create_target_params=True)
        
        self.make_value_estimator()
        self.register_buffer("support", support.to(device))
        atoms = self.support.numel()
        Vmin = self.support.min()
        Vmax = self.support.max()
        delta_z = (Vmax - Vmin) / (atoms - 1)
        self.register_buffer(
            "stddev", (0.75 * delta_z).unsqueeze(-1)
        )
        self.register_buffer(
            "support_plus",
            self.support + delta_z / 2
        )
        self.register_buffer(
            "support_minus",
            self.support - delta_z / 2
        )
        self._set_in_keys()

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

    @property
    def _clip_bounds(self):
        return (
            (-self.clip_epsilon).log1p(),
            self.clip_epsilon.log1p(),
        )

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective", "clip_fraction"]
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

    def make_value_estimator(self):
        # self._value_estimator = ReTrace(gamma = 0.99,
        #                value_network=self.critic_network,
        #                actor_network=self.actor_network,
        #                reward_network=self.reward_network,
        #                average_adv=True)
        self._value_estimator = VTrace(gamma = self.gamma,
                value_network=self.critic_network,
                actor_network=self.actor_network,
                average_adv=True,
                device=self._default_device)
        # TODO: not sure what to set for the value_target
        tensor_keys = {
            "value_target": "value_target",
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)

    @property
    @_cache_values
    def _cached_critic_network_params_detached(self):
        if not self.functional:
            return None
        return self.critic_network_params.detach()
    
    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        #1. Form a minibatch (done)
        #2. Use VTrace to estimate each return bootstrapping from v-hat prior 
        # TODO: Move to ReTrace. Figure out what q_prior is.
        self.value_estimator(
            tensordict,
            params=self._cached_critic_network_params_detached,
            target_params=self.target_critic_network_params,
        )

        #now advantage is set up. Let's study it thoroughly.
        #3. bias correct the advantages for small batches. TODO
        #4. Normalize the advantages (already done, for now)

        advantage = tensordict.get(self.tensor_keys.advantage)
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

        # Time to compute the losses
        log_weight, dist, kl_approx = self._log_weight(tensordict)
        
        gain = log_weight.exp() * advantage #loss # 1 done.

                #1. Get prior dist.
        with self.target_actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            prior_dist = self.actor_network.get_dist(tensordict)
        
        with torch.no_grad():
            #2. Sample N actions.
            prior_actions = prior_dist.sample(sample_shape=(self.z_num_samples,)).movedim(0, -2)
            z_cmpo_td = tensordict.select(*["observation", ("next", "observation"), ("next", "terminated"), "state_value"]).unsqueeze(-1).expand(*tensordict.batch_size, self.z_num_samples)
            z_cmpo_td.set(self.tensor_keys.action, prior_actions)
            self.reward_network(z_cmpo_td) 
            z_cmpo_td_next = z_cmpo_td["next"]
            with self.target_critic_network_params.to_module( #TODO: This is for q-prior, so unsure of whether to use v or v_target
                self.critic_network
            ) if self.functional else contextlib.nullcontext():
                self.critic_network(z_cmpo_td_next)
            z_cmpo_td["next", "state_value"] = z_cmpo_td_next["state_value"]
            terminateds = z_cmpo_td['next', 'terminated']
            predicted_rewards = z_cmpo_td['reward_value']
            predicted_values = z_cmpo_td["state_value"]
            predicted_qvalues = predicted_rewards + self.gamma * predicted_values * (1 - terminateds.float())
            values = tensordict['state_value'].unsqueeze(-1)
            cmpo_advantages = predicted_qvalues - values
            cmpo_loc = cmpo_advantages.mean(dim=-2, keepdim=True)
            cmpo_scale = cmpo_advantages.std(dim=-2, keepdim=True).clamp_min(1e-6)
            cmpo_advantages = torch.clip(cmpo_advantages, torch.tensor(-1.0, device=cmpo_advantages.device), torch.tensor(1.0, device=cmpo_advantages.device))
            cmpo_advantages = torch.exp(cmpo_advantages)
            z_cmpo = (1 + torch.sum(cmpo_advantages, dim=-2, keepdim=True) - cmpo_advantages) / self.z_num_samples
        
        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            dist = self.actor_network.get_dist(z_cmpo_td)
        regularization = -cmpo_advantages / z_cmpo * dist.log_prob(prior_actions).unsqueeze(-1)
        regularization = regularization.mean(dim=-2)
        print('ok')
        


        with torch.no_grad():
            z_cmpo_td = tensordict.clone(False)
            prior_actions = z_cmpo_td["sampled_actions"]
            N = prior_actions.shape[1]

            z_cmpo_td['observation'] = z_cmpo_td['observation'].unsqueeze(-2).expand(-1, N, -1)
            predicted_rewards = self.reward_network.module(z_cmpo_td['observation'], prior_actions)

            z_cmpo_td['next', 'observation'] = z_cmpo_td['next', 'observation'].unsqueeze(-2).expand(-1, N, -1)
            next_td = z_cmpo_td["next"].clone()
            predicted_values = self.critic_network(next_td)["state_value"]
            predicted_qvalues = predicted_rewards + 0.99 * predicted_values
            values = self.critic_network(z_cmpo_td.clone())["state_value"]
            cmpo_advantages = predicted_qvalues - values
            cmpo_loc = cmpo_advantages.mean(dim=1, keepdim=True)
            cmpo_scale = cmpo_advantages.std(dim=1, keepdim=True).clamp_min(1e-6)
            cmpo_advantages = (cmpo_advantages - cmpo_loc) / cmpo_scale
            #cmpo_advantages = cmpo_advantages + 0.02 * torch.rand_like(cmpo_advantages) - 0.01
            cmpo_advantages = torch.clip(cmpo_advantages, torch.tensor(-1.0, device=cmpo_advantages.device), torch.tensor(1.0, device=cmpo_advantages.device))
            cmpo_advantages = torch.exp(cmpo_advantages)
            z_cmpo = (1 + torch.sum(cmpo_advantages, dim=1, keepdim=True) - cmpo_advantages) / N
            z_cmpo_td["loc"] = z_cmpo_td["loc"].unsqueeze(-2)
            z_cmpo_td["scale"] = z_cmpo_td["scale"].unsqueeze(-2)
            dist = self.actor_network.get_dist(z_cmpo_td)

            #now it's B x 16
        regularization = -cmpo_advantages / z_cmpo * dist.log_prob(prior_actions).unsqueeze(-1)
        regularization = regularization.mean(dim=1)
        #regularization = -torch.exp(regularization) / z_cmpo * dist.log_prob(tensordict['action']).unsqueeze(-1)
        
        #log_weight_clip = log_weight.clamp(*self._clip_bounds)
        #clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        #ratio = log_weight_clip.exp()
        #gain2 = ratio * advantage

        #gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain}, batch_size=[])
        #td_out.set("clip_fraction", clip_fraction)
        td_out.set("loss_regularization", regularization)
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.reward_coef is not None:
            loss_reward_predictor = self.loss_reward_predictor(tensordict)
            td_out.set("loss_reward_predictor", loss_reward_predictor)

        if self.critic_coef is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out


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
        state_value_logits = state_value_td.get("state_value_logits")

        if state_value is None:
            raise KeyError(
                f"the key {self.tensor_keys.value} was not found in the critic output tensordict. "
                f"Make sure that the value_key passed to PPO is accurate."
            )

        target_return_logits = self.construct_gauss_dist(target_return)
        loss_value = torch.nn.functional.cross_entropy(state_value_logits, target_return_logits, reduction="none")

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

    def construct_gauss_dist(self, loc):
        loc = loc.clamp(self.support.min(), self.support.max())
        stddev_expanded = self.stddev.expand_as(loc)
        dist = torch.distributions.Normal(loc, stddev_expanded)
        cdf_plus = dist.cdf(self.support_plus)
        cdf_minus = dist.cdf(self.support_minus)
        m = cdf_plus - cdf_minus
            #m[..., 0] = cdf_plus[..., 0]
            #m[..., -1] = 1 - cdf_minus[..., -1]
        m = m / m.sum(dim=-1, keepdim=True)  #this should be handled differently. check the paper
        assert torch.allclose(m.sum(dim=-1), torch.ones_like(m.sum(dim=-1)))
        return m
    
    def loss_reward_predictor(self, tensordict):
        target_reward = tensordict.get(("next", self.tensor_keys.reward), None)
        reward_predictor_td = self.reward_network(tensordict)

        reward_prediction = reward_predictor_td.get(self.tensor_keys.reward_value)
        loss_reward_value = distance_loss(reward_prediction, target_reward, 
                                          loss_function="l2")
        return loss_reward_value
    