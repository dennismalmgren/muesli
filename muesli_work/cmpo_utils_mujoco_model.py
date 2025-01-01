# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn
import torch.optim

from tensordict.nn import (
    AddStateIndependentNormalScale, 
    TensorDictModule, 
    NormalParamExtractor, 
    TensorDictSequential
)

from torchrl.data import Composite
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

from modules.layers import SoftmaxLayer, ClampOperator, SupportOperator

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


def make_ppo_models_state(proof_environment, cfg):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    latent_dim = 1024
    softmax_dim = 8
    enc_dim = 1024
    policy_dim = 1024
    value_dim = 2048
    reward_dim = 1024
    dynamics_dim = 1024

    softmax_activation_kwargs = {
        "internal_dim":softmax_dim
    }

    # Encoder
    encoder_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=SoftmaxLayer,
        activation_kwargs=softmax_activation_kwargs,
        out_features=latent_dim,  # predict only loc
        num_cells=[enc_dim, latent_dim],
        activate_last_layer=True,
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [enc_dim, latent_dim, latent_dim]],
    )

    for layer in encoder_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()


    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low,
        "high": proof_environment.action_spec_unbatched.space.high,
        "tanh_loc": False,
    }

    # Define policy architecture
    policy_mlp = MLP(
        in_features=latent_dim,
        activation_class=torch.nn.Mish,
        out_features=2 * num_outputs,  # predict only loc
        num_cells=[policy_dim, policy_dim],
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [policy_dim, policy_dim]],
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()
    #todo: use normal param extractor. for realzies

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        NormalParamExtractor()
        # AddStateIndependentNormalScale(
        #     proof_environment.action_spec.shape[-1], scale_lb=1e-8
        # ),
    )

    encoder_module = TensorDictModule(
        encoder_mlp,
        in_keys=["observation"],
        out_keys=["observation_encoded"]
    )
    policy_module = TensorDictModule(
        policy_mlp,
        in_keys=["observation_encoded"],
        out_keys=["loc", "scale"],
    )

    latent_actor_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        spec=Composite(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Add probabilistic sampling of the actions
    actor_module = ProbabilisticActor(
        module=TensorDictSequential(
            encoder_module,
            policy_module
        ),
        in_keys=["loc", "scale"],
        spec=Composite(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    
    vnbins = cfg.network.vnbins
    Vmin = cfg.network.vmin
    Vmax = cfg.network.vmax
    dk = (Vmax - Vmin) / (vnbins - 4)
    Ktot = dk * vnbins
    Vmax = math.ceil(Vmin + Ktot)

    value_support = torch.linspace(Vmin, Vmax, vnbins)

    # Define value architecture
    value_mlp = MLP(
        in_features=latent_dim,
        activation_class=torch.nn.Mish,
        out_features=vnbins,
        num_cells=[value_dim, value_dim],
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [value_dim, value_dim, value_dim]],
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()
 
    value_module_1 = TensorDictModule(
        in_keys=["observation_encoded"],
        out_keys=["state_value_logits"],
        module=value_mlp,
    )

    value_support_network = SupportOperator(value_support)
    value_module_2 = TensorDictModule(value_support_network, in_keys=["state_value_logits"], out_keys=["state_value"])
    value_module = TensorDictSequential(value_module_1, value_module_2)

    rnbins = cfg.network.rnbins
    Rmin = cfg.network.rmin
    Rmax = cfg.network.rmax
    dk = (Rmax - Rmin) / (rnbins - 4)
    Ktot = dk * rnbins
    Rmax = math.ceil(Rmin + Ktot)

    reward_support = torch.linspace(Rmin, Rmax, rnbins)

    # Define reward architecture
    reward_mlp = MLP(
        in_features = latent_dim + num_outputs,
        activation_class = torch.nn.Mish,
        out_features = rnbins,
        num_cells=[reward_dim, reward_dim],
        activate_last_layer=False,
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [reward_dim, reward_dim]],
    )

    # Define reward weights
    for layer in reward_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()

    # Define reward module
    reward_module_1 = TensorDictModule(
        in_keys=["observation_encoded", "action"],
        out_keys=["next_reward_logits"],
        module=reward_mlp,
    )
    
    reward_support_network = SupportOperator(reward_support)
    reward_module_2 = TensorDictModule(reward_support_network, in_keys=["next_reward_logits"], out_keys=["next_reward"])
    reward_module = TensorDictSequential(reward_module_1, reward_module_2)

    # Define dynamics architecture
    dynamics_mlp = MLP(
        in_features = latent_dim + num_outputs,
        activation_class = SoftmaxLayer,
        activation_kwargs=softmax_activation_kwargs,
        num_cells=[dynamics_dim, dynamics_dim],
        out_features = latent_dim,
        activate_last_layer=True,
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                    "normalized_shape": hidden_size} for hidden_size in [dynamics_dim, dynamics_dim, latent_dim]],
    )

    # Define reward weights
    for layer in dynamics_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.7)
            layer.bias.data.zero_()

    # Define reward module
    dynamics_module = TensorDictModule(
        dynamics_mlp,
        in_keys=["observation_encoded", "action"],
        out_keys=["next_observation_encoded"],
    )


    return latent_actor_module, value_module, reward_module, encoder_module, dynamics_module, actor_module, reward_support, value_support


def make_ppo_models(env_name, cfg):
    proof_environment = make_env(env_name, device="cpu")
    latent_actor_module, value_module, reward_module, encoder_module, dynamics_module, actor_module, reward_support, value_support= make_ppo_models_state(proof_environment, cfg)
    return latent_actor_module, value_module, reward_module, encoder_module, dynamics_module, actor_module, reward_support, value_support


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