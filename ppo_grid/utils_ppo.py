# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import copy

import torch.nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule, TensorDictSequential
from torchrl.data import CompositeSpec
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
    CatFrames,
    TensorDictPrimer
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, SliceSampler, RandomSampler
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from tensordict import TensorDict
from torch import nn
from energy_predictor_module import EnergyPredictor

# ====================================================================
# Environment utils
# --------------------------------------------------------------------
def get_project_root_path_vscode():
    trace_object = getattr(sys, 'gettrace', lambda: None)() #vscode debugging
    if trace_object is not None:
        return "../../../"
    else:
        return "../../../../"

def make_env(env_name="HalfCheetah-v4", device="cpu"):
    env = GymEnv(env_name, device=device)
    env = TransformedEnv(env)
    #env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    #env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(CatFrames(2, in_keys=["observation"], dim=-1, padding="constant"))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    action_spec = env.action_spec
    env.append_transform(TensorDictPrimer(prev_action=action_spec, default_value=0.0))
    return env


def transfer_weights(source, target):
    for ind, layer_target in enumerate(target):
        if isinstance(layer_target, torch.nn.Linear):
            layer_target.weight.data.copy_(source[str(ind), 'weight'].clone().data)
            layer_target.bias.data.copy_(source[str(ind), 'bias'].clone().data)

def make_energy_prediction_module(input_shape, cfg) -> EnergyPredictor:
    model_save_base_path = cfg.energy_prediction.model_save_base_path
    model_save_dir = cfg.energy_prediction.model_save_dir
    metadata = TensorDict({}).load_memmap(get_project_root_path_vscode() + 
                                        f"{model_save_base_path}/{model_save_dir}/model_metadata")
    params = TensorDict({}).load_memmap(get_project_root_path_vscode() + 
                                      f"{model_save_base_path}/{model_save_dir}/model_params")
    cfg_num_place_cells = metadata["num_place_cells"].item()
    cfg_num_head_cells = metadata["num_head_cells"].item()
    cfg_num_energy_heads = metadata["num_energy_heads"].item()
    cfg_model_num_cells = metadata["num_cells"].item()
    cfg_num_cat_frames = metadata["num_cat_frames"].item()
    cfg_predict_heading = cfg_num_head_cells > 0
    cfg_predict_place = cfg_num_place_cells > 0
    cfg_predict_state = not (cfg_predict_heading or cfg_predict_place)
    cfg_from_source = metadata["from_source"]
    cfg_use_dropout = metadata["use_dropout"].item()
    cfg_include_action = metadata["include_action"].item()

    predictor_module = EnergyPredictor(in_features = input_shape[-1], 
                                       num_cat_frames=cfg_num_cat_frames,
                                       num_place_cells=cfg_num_place_cells, 
                                       num_head_cells=cfg_num_head_cells, 
                                       num_energy_heads=cfg_num_energy_heads,
                                       num_cells=cfg_model_num_cells,
                                       from_source=cfg_from_source,
                                       use_state_dropout = cfg_use_dropout,
                                       include_action = cfg_include_action)
    
    predictor_module.eval()
    out_keys = ["integration_prediction"]
    if cfg_predict_heading:
        out_keys.append("head_energy_prediction")
    if cfg_predict_place:
        out_keys.append("place_energy_prediction")
    if cfg_predict_state:
        out_keys.append("state_prediction")

    energy_prediction_module = TensorDictModule(
        predictor_module,
        in_keys=["observation"],
        out_keys=out_keys,
    )
    
    transfer_weights(params['module', 'path_integration_model'], energy_prediction_module.path_integration_model)
    if cfg_predict_place:
        transfer_weights(params['module', 'place_energy_output_model'], energy_prediction_module.place_energy_output_model)
    if cfg_predict_heading:
        transfer_weights(params['module', 'head_energy_output_model'], energy_prediction_module.head_energy_output_model)
    if cfg_predict_state:
        transfer_weights(params['module', 'state_output_model'], energy_prediction_module.state_output_model)
    energy_prediction_module.requires_grad_(False)
    energy_prediction_module.eval()
    return energy_prediction_module

class CopyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
def make_ppo_models_state(proof_environment, cfg):

    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": proof_environment.action_spec.space.low,
        "max": proof_environment.action_spec.space.high,
        "tanh_loc": False,
    }

    energy_prediction_module = make_energy_prediction_module(input_shape, cfg)
    
    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1] + energy_prediction_module.path_integration_model.out_features,
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  
        num_cells=[64, 64],
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    mean_predict_module = TensorDictModule(
        policy_mlp,
        in_keys=["observation", "integration_prediction"],
        out_keys=["loc"],
    )
    scale_predict_module = TensorDictModule(
        AddStateIndependentNormalScale(
            proof_environment.action_spec.shape[-1], scale_lb=1e-8
        ),
        in_keys=["loc"],
        out_keys=["loc", "scale"],
    )

    actor_module = TensorDictSequential(energy_prediction_module, mean_predict_module, scale_predict_module)
    
    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        actor_module,
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    copyModule = CopyModule()
    copyModule = TensorDictModule(
        copyModule,
        in_keys=["action"],
        out_keys=[("next", "prev_action")],
    )
    policy_module = TensorDictSequential(policy_module, copyModule)
    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
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

    return policy_module, value_module, energy_prediction_module, mean_predict_module, scale_predict_module


def make_ppo_models(env_name, cfg):
    proof_environment = make_env(env_name, device="cpu")
    actor, critic, energy_prediction_module, mean_predict_module, scale_predict_module = make_ppo_models_state(proof_environment, cfg)
    return actor, critic, energy_prediction_module, mean_predict_module, scale_predict_module


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def save_buffer_to_disk(td_list, num_episodes):
    trajectory_length = 1000
    trajectory_count = trajectory_length * num_episodes
    num_slices = 2 #per sample, how many trajectories.
    #batch_size = num_slices * trajectory_length
    storage_size = trajectory_count * trajectory_length 
    save_sampler = SliceSampler(num_slices=num_slices)
    save_replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(storage_size),
        sampler=save_sampler,
    )
    for ind, td in enumerate(td_list):
        td['episode'] = torch.ones(len(td)) * (ind + 1)
        save_replay_buffer.extend(td)
    save_replay_buffer.dumps("traj_buffer")

def eval_model(actor, test_env, num_episodes=3, save_buffer=False):
    test_rewards = []
    td_list = []
    for episode_ind in range(num_episodes):
        if save_buffer:
            print('Generating episode: ', (episode_ind + 1))
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
        if save_buffer:
            if len(td_test) < 1000:
                print('Too short episode')
            else:
                td_list.append(td_test.detach().cpu().clone())
        del td_test
    if save_buffer:
        save_buffer_to_disk(td_list, num_episodes)
    return torch.cat(test_rewards, 0).mean()
#what are we looking at here?