# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
import itertools

import torch.nn
import torch.optim
from torch.distributions import Categorical
from torch.nn import Embedding, Flatten

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
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator, OneHotCategorical
from torchrl.record import VideoRecorder

from layers import OrdinalLogitsModule, ClampOperator

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
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

def define_uniform_grid_anchors(action_low, action_high, K_actions):
    """
    Defines K_actions anchor points evenly across a 4D continuous space.
    
    Args:
        action_min (array): Minimum bounds for each action dimension (length 4).
        action_max (array): Maximum bounds for each action dimension (length 4).
        K_actions (int): Number of desired anchors.
    
    Returns:
        np.array: Anchor points of shape (K_actions, 4).
    """
    # Create evenly spaced grid values for each dimension
    grid_values = [torch.linspace(action_low[i], action_high[i], K_actions) for i in range(action_high.shape[-1])]

    # Select the first K_actions points
    anchors = torch.stack(grid_values, dim=0)
    return anchors



def make_ppo_models_state(proof_environment, device, cfg):

    action_embedding_dim=8

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    action_dims = proof_environment.action_spec_unbatched.shape[-1]
    # Define policy output distribution class
    action_k = cfg.k_actions
    anchors = define_uniform_grid_anchors(proof_environment.action_spec_unbatched.space.low, proof_environment.action_spec_unbatched.space.high, action_k).to(device)
    half_bin_sizes = 0.5 * (anchors[:, 1] - anchors[:, 0])
    distribution_class = Categorical
    distribution_kwargs = {

    }
    #start with a giant one.
    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=torch.Size((action_dims, action_k)),  # for now.
        num_cells=[64, 64],
        device=device,
    )

    # Initialize policy weights
    # for layer in policy_mlp.modules():
    #     if isinstance(layer, torch.nn.Linear):
    #         torch.nn.init.orthogonal_(layer.weight, 1.0)
    #         layer.bias.data.zero_()

    policy_mlp_module = TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["orig_logits"],
        )
    policy_logits_module = TensorDictModule(
        module = OrdinalLogitsModule(),
        in_keys=["orig_logits"],
        out_keys=["logits"]
    )

    policy_module = TensorDictSequential(
        policy_mlp_module,
        policy_logits_module
    )
    # # Add state-independent normal scale
    # policy_mlp = torch.nn.Sequential(
    #     policy_mlp,
    #     AddStateIndependentNormalScale(
    #         proof_environment.action_spec_unbatched.shape[-1], scale_lb=1e-8
    #     ).to(device),
    # )
    # Add probabilistic sampling of the actions
    discrete_policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["discrete_action"],
        #spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )



    action_embedding = Embedding(num_embeddings=action_k, embedding_dim=action_embedding_dim, device=device)

    action_embedding_module = TensorDictModule(
        module=torch.nn.Sequential(action_embedding, Flatten(-2, -1)),
        in_keys=["discrete_action"],
        out_keys=["embedded_action"]
    )

    continuous_action_network = MLP(
        in_features = input_shape[-1] + action_dims * action_embedding_dim,
        activation_class = torch.nn.Tanh,
        num_cells=[64, 64],
        out_features = action_dims,
        device=device
    )
    # Initialize policy weights
    for layer in continuous_action_network.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    continuous_action_module = TensorDictSequential(
        action_embedding_module,
        TensorDictModule(module=continuous_action_network, in_keys=["observation", "embedded_action"], out_keys=["loc"]),
        TensorDictModule(module=
                         torch.nn.Sequential(
                             ClampOperator(vmin=-100.0, vmax=100.0),
        AddStateIndependentNormalScale(
            action_dims, scale_lb=1e-8,
        ).to(device)),
        in_keys=["loc"], out_keys=["loc", "scale"]),
    )

    continuous_action_distribution_class = TanhNormal
    continuous_action_distribution_kwargs = {
        "low": -torch.ones(action_dims, device=device),
        "high": torch.ones(action_dims, device=device),
        "tanh_loc": False,
    }

    continuous_policy_module = ProbabilisticActor(
        module=continuous_action_module,
        in_keys=["loc", "scale"],
        out_keys=["continuous_action"],
        #spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=continuous_action_distribution_class,
        distribution_kwargs=continuous_action_distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
        log_prob_key="sample_continuous_log_prob"
    )

    policy_projection = TensorDictModule(
        lambda x, y: torch.clamp(anchors.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1) + y * half_bin_sizes, -1.0, 1.0),
        in_keys=["discrete_action", "continuous_action"],
        out_keys=["action"]
    )

    policy_module = TensorDictSequential(discrete_policy_module, continuous_policy_module, policy_projection)
    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
        device=device,
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module, discrete_policy_module, continuous_policy_module


def make_ppo_models(env_name, device, cfg):
    proof_environment = make_env(env_name, device=device)
    actor, critic, discrete_policy_module, continuous_policy_module = make_ppo_models_state(proof_environment, device=device, cfg=cfg)
    return actor, critic, discrete_policy_module, continuous_policy_module


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