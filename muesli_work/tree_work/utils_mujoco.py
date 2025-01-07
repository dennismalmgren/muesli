# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch.nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule, TensorDictSequential
from torch.nn import Sequential
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
    InitTracker
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder
from torch.distributions import Categorical
from torchrl.modules import OneHotCategorical
from torchrl.modules import GRU, GRUModule, LSTMModule
from gru import GRU
# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False, rnn = None):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(InitTracker())
    #env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    #env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    if rnn is not None:
        env.append_transform(rnn.make_tensordict_primer())
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment, device):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec_unbatched.space.n
    distribution_kwargs = {}
    distribution_class = OneHotCategorical

    # Define policy architecture
    encoder_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=64,  # predict only loc
        num_cells=[64, 64],
        device=device,
    )
    
    encoder_module = TensorDictModule(
        module=encoder_mlp,
        in_keys=["observation"],
        out_keys=["encoded_observation"]
    )
    recurrent_body = GRUModule(
        input_size = 64,
        hidden_size = 32,
        in_key="encoded_observation",#, "recurrent_state", "is_init"],
        out_key="intermediate",# ("next", "recurrent_state")],
       # default_recurrent_mode=True,
        device=device
    )

    policy_mlp = MLP(
        in_features = 32,
       # activation_class=torch.nn.Tanh,
        out_features = num_outputs,
        num_cells = [64,],
        device=device
    )
    policy_module = TensorDictModule(
        module=policy_mlp,
        in_keys=["intermediate"],
        out_keys=["logits"]
    )
    policy_module = TensorDictSequential(encoder_module, recurrent_body, policy_module)
    # Initialize policy weights
    # for layer in policy_mlp.modules():
    #     if isinstance(layer, torch.nn.Linear):
    #         torch.nn.init.orthogonal_(layer.weight, 0.01)
    #         layer.bias.data.zero_()

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
        device=device,
    )

    #Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = TensorDictModule(
        value_mlp,
        in_keys=["observation"],
        out_keys=["state_value"]
    )

    return policy_module, value_module, encoder_module, recurrent_body


def make_ppo_models(env_name, device):
    proof_environment = make_env(env_name, device=device)
    actor, critic, encoder_module, rnn = make_ppo_models_state(proof_environment, device=device)
    return actor, critic, encoder_module, rnn


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