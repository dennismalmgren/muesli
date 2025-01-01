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
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder
from torchrl.envs.utils import Composite
from torchrl.envs import ParallelEnv

# ====================================================================
# Environment utils
# --------------------------------------------------------------------

def make_single_env(env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env

def make_env(env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False):
    def make_gym_env():
        env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
        return env
    env = ParallelEnv(20, create_env_fn=make_gym_env)
    env = TransformedEnv(env)
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------
class SupportOperator(torch.nn.Module):
    def __init__(self, support):
        super().__init__()
        self.register_buffer("support", support)

    def forward(self, x):
        return (x.softmax(-1) * self.support).sum(-1, keepdim=True)
    

def make_ppo_models_state(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low,
        "high": proof_environment.action_spec_unbatched.space.high,
        "tanh_loc": False,
    }

    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[256, 256],
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
            proof_environment.action_spec.shape[-1], scale_lb=1e-8
        ),
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=Composite(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    nbins = 101
    Vmin = -10.0
    Vmax = 500.0
    dk = (Vmax - Vmin) / (nbins - 4)
    Ktot = dk * nbins
    Vmax = math.ceil(Vmin + Ktot)

    support = torch.linspace(Vmin, Vmax, nbins)

    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=nbins,
        num_cells=[256, 256],
    )


    last_layer: torch.nn.Linear = list(value_mlp.modules())[-1]
    desired_mean = 2.5
    std = 0.75
    class_indices = torch.arange(nbins).float()
    bias = -((class_indices - desired_mean)**2) / (2 * std **2)
    # # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            layer.bias.data.zero_()
    with torch.no_grad():
        last_layer.bias.copy_(bias)
    value_module_1 = TensorDictModule(
        in_keys=["observation"],
        out_keys=["state_value_logits"],
        module=value_mlp,
    )
    
    support_network = SupportOperator(support)
    value_module_2 = TensorDictModule(support_network, in_keys=["state_value_logits"], out_keys=["state_value"])
    value_module = TensorDictSequential(value_module_1, value_module_2)
    
    # Define reward architecture
    reward_mlp = MLP(
        in_features = input_shape[-1] + num_outputs,
        activation_class = torch.nn.Tanh,
        out_features = 1,
        num_cells = [64, 64]
    )

    # Define reward weights
    for layer in reward_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define reward module
    reward_module = ValueOperator(
        reward_mlp,
        in_keys=["observation", "action"],
        out_keys=["reward_value"]
    )

    return policy_module, value_module, reward_module, support


def make_ppo_models(env_name, device):
    proof_environment = make_single_env(env_name, device="cpu")
    actor, critic, reward_predictor, support = make_ppo_models_state(proof_environment)
    actor = actor.to(device)
    critic = critic.to(device)
    reward_predictor = reward_predictor.to(device)
    support = support.to(device)
    return actor, critic, reward_predictor, support



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