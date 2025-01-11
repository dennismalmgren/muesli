# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymEnv,
    GymWrapper,
    NoopResetEnv,
    ParallelEnv,
    RenameTransform,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
    InitTracker,
    default_info_dict_reader
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.record import VideoRecorder
from torchrl.modules import GRU, GRUModule, LSTMModule
import gym
import gym_sokoban

from torchrl.envs.transforms import TensorDictPrimer
from torchrl.data.tensor_specs import Unbounded
from prediction_test.minlstm_fast import MinGRU, MinLSTM

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_base_env(env_name: str, device="cpu", from_pixels: bool = True, is_test=False):
    env = gym.make(env_name)
    env = GymWrapper(env, device="cpu")
    reader = default_info_dict_reader(["boxes_on_target", "num_boxes"])  
    env.set_info_dict_reader(reader)

    #env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    return env


def make_parallel_env(num_envs, env_name, device, is_test=False, transform = None):
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(env_name)),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    env.append_transform(InitTracker())
    env.append_transform(RenameTransform(in_keys=["pixels"], out_keys=["observed_room_orig"], create_copy=True))
    env.append_transform(ToTensorImage(in_keys=["observed_room_orig"], out_keys=["observed_room_img"]))
    env.append_transform(GrayScale("observed_room_img"))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    env.append_transform(VecNorm(in_keys=["observed_room_img"]))
    if transform is not None:
        env.append_transform(transform)
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_modules_pixels(proof_environment, device):

    # Define input shape
    input_shape = proof_environment.observation_spec["observed_room_img"].shape

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec_unbatched.space, CategoricalBox):
        num_outputs = proof_environment.action_spec_unbatched.space.n
        distribution_class = torch.distributions.OneHotCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec_unbatched.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": proof_environment.action_spec_unbatched.space.low.to(device),
            "high": proof_environment.action_spec_unbatched.space.high.to(device),
        }

    # Define input keys
    in_keys = ["observed_room_img"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    common_cnn = ConvNet(
        activation_class=torch.nn.ReLU,
        num_cells=[16, 32],
        kernel_sizes=[3, 3],
        strides=[1, 1],
        paddings=[1, 1],
        device=device,
    )
    #pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    #out = torch.nn.Linear(32, 32)
    #common_cnn = torch.nn.Sequential(common_cnn, pool, out)

    common_cnn_output = common_cnn(torch.ones(input_shape, device=device))
    
    encoder_module = TensorDictModule(
        module=common_cnn,
        in_keys=in_keys,
        out_keys=["encoded_observation"]
    )

    recurrent_net = MinGRU(input_dim=common_cnn_output.shape[-1], hidden_dim=32, output_dim=32, device=device)
    recurrent_body = TensorDictModule(
        module=recurrent_net,
        in_keys=["encoded_observation", "is_init", "recurrent_state"],
        out_keys=["intermediate", ("next", "recurrent_state")],
    )

    # recurrent_body = GRUModule(
    #     input_size = common_cnn_output.shape[-1],
    #     hidden_size = 32,
    #     in_key="encoded_observation",#, "recurrent_state", "is_init"],
    #     out_key="intermediate",# ("next", "recurrent_state")],
    #    # default_recurrent_mode=True,
    #     device=device
    # )

    feature_mlp = MLP(
        in_features=32,
        activation_class=torch.nn.ReLU,
        activate_last_layer=True,
        out_features=64,
        num_cells=[],
        device=device,
    )

    # Define shared net as TensorDictModule
    feature_module = TensorDictModule(
        module=feature_mlp,
        in_keys=["intermediate"],
        out_keys=["common_features"],
    )

    common_module = TensorDictSequential(encoder_module, recurrent_body, feature_module)

    # Define on head for the policy
    policy_net = MLP(
        in_features=64,
        out_features=num_outputs,
        activation_class=torch.nn.ReLU,
        num_cells=[],
        device=device,
    )
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

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

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU,
        in_features=64,
        out_features=1,
        num_cells=[],
        device=device,
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module, recurrent_body


def make_ppo_models(cfg, device):
    proof_environment = make_parallel_env(1, cfg.env.env_name, device=device)
    common_module, policy_module, value_module, recurrent_body = make_ppo_modules_pixels(
        proof_environment,
        device=device,
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    with torch.no_grad():
        td = proof_environment.fake_tensordict().expand(10)
        actor_critic(td)
        del td

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    del proof_environment

    return actor, critic, recurrent_body


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    episode_completion_list = []
    episode_partial_completion_list = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        episode_initial_status = td_test['num_boxes'].unsqueeze(-1)[td_test["next", "done"]]
        episode_completion_status = td_test['next', 'boxes_on_target'].unsqueeze(-1)[td_test["next", "done"]]
        episode_to_go = episode_initial_status - episode_completion_status
        episodes_completed = (episode_to_go == 0).float().sum()
        episode_partial_completion = episode_completion_status / episode_initial_status
        episode_completion_list.append(episodes_completed.cpu())
        episode_partial_completion_list.append(episode_partial_completion.cpu())
        test_rewards.append(reward.cpu())
    del td_test
    episode_completion_percentage = (torch.stack(episode_completion_list, 0).sum() / len(test_rewards)) * 100.0
    episode_partial_completion_percentage = (torch.stack(episode_partial_completion_list, 0).sum()/ len(test_rewards)) * 100.0

    return torch.cat(test_rewards, 0).mean(), episode_completion_percentage, episode_partial_completion_percentage