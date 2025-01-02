# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch.nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule, TensorDictSequential
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
    InitTracker,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder

from layers import SupportOperator, OrdinalLogitsModule, OrdinalLogitsKernelModule, GaussianLogitsKernelModule

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(InitTracker())
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment, device, cfg):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec_unbatched.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low.to(device),
        "high": proof_environment.action_spec_unbatched.space.high.to(device),
        "tanh_loc": False,
    }

    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[64, 64],
        device=device,
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            proof_environment.action_spec_unbatched.shape[-1], scale_lb=1e-8
        ).to(device),
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    #no need for margins with this setup
    vnbins = cfg.vnbins
    vmin = cfg.vmin
    vmax = cfg.vmax
    dk = (vmax - vmin) / (vnbins - 4)
    ktot = dk * vnbins
    vmax_support = math.ceil(vmin + ktot)

    value_support = torch.linspace(cfg.vmin, vmax_support, vnbins)

    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Mish, #todo: replace with something better
        out_features=vnbins,
        num_cells=[64, 64],
        device=device,
    )

    # Define value module
    support_network = SupportOperator(value_support)
    support_network = support_network.to(device)
    if cfg.ordinal_logits:
        value_module_1 = TensorDictModule(
            module=value_mlp,
            in_keys=["observation"],
            out_keys=["state_value_orig_logits"]
        )
        if cfg.window_size > 0:
            ordinal_logits_module = GaussianLogitsKernelModule(window_size=cfg.window_size, stddev_scale=0.75)
        else:
            ordinal_logits_module = OrdinalLogitsModule()
        ordinal_logits_module = ordinal_logits_module.to(device)
        value_module_2 = TensorDictModule(
            module = ordinal_logits_module,
            in_keys=["state_value_orig_logits"],
            out_keys=["state_value_logits"]
        )

        value_module_3 = TensorDictModule(
            module=support_network,
            in_keys=["state_value_logits"],
            out_keys=["state_value"]
        )

        value_module = TensorDictSequential(
            value_module_1, 
            value_module_2,
            value_module_3
        )
    else:
        value_module_1 = TensorDictModule(
            module=value_mlp,
            in_keys=["observation"],
            out_keys=["state_value_logits"]
        )
        #ordinal_logits_module = OrdinalLogitsModule()
        #ordinal_logits_module = ordinal_logits_module.to(device)
        #value_module_2 = TensorDictModule(
        #    module = ordinal_logits_module,
        #    in_keys=["state_value_orig_logits"],
        #    out_keys=["state_value_logits"]
        #)

        value_module_3 = TensorDictModule(
            module=support_network,
            in_keys=["state_value_logits"],
            out_keys=["state_value"]
        )

        value_module = TensorDictSequential(
            value_module_1, 
            #value_module_2,
            value_module_3
        )
    return policy_module, value_module, value_support


def make_ppo_models(env_name, device, cfg):
    proof_environment = make_env(env_name, device=device)
    actor, critic, support = make_ppo_models_state(proof_environment, device=device, cfg=cfg)
    return actor, critic, support


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
        test_env.apply(dump_video)
    del td_test
    return torch.cat(test_rewards, 0).mean()