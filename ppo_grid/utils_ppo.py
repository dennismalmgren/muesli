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
    CatFrames
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, SliceSampler, RandomSampler
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from tensordict import TensorDict
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
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


# class ObservationModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         stored_model = MLP(in_features = 8 + 1 + 28, 
#                 out_features = 256 + 12, 
#                 num_cells = [256, 256],
#                 dropout=0.5)
#         params = TensorDict.from_module(stored_model)
#         params = params.load_memmap(get_project_root_path_vscode() + "model_state_dropout")
#         params.to_module(stored_model)
#         self.path_integration_model = MLP(in_features = 8 + 1 + 28,
#                                 out_features = 256,
#                                 num_cells = [256],
#                                 activate_last_layer=True,
#                                 dropout=0.5)
#         for layer_source, layer_target in zip(stored_model, self.path_integration_model):
#             if isinstance(layer_source, torch.nn.Linear):
#                 layer_target.weight.data.copy_(layer_source.weight.clone().data)
#                 layer_target.bias.data.copy_(layer_source.bias.clone().data)
#         self.path_integration_model.requires_grad_(False)

#     def create_path_integration_input(self, dat, device):
#         batch_dims = dat.shape[:-1]
#         t0_state = dat[..., :dat.shape[-1]//2] #t0
#         t1_state = dat[..., dat.shape[-1]//2:] #t1
#         u = t0_state / torch.linalg.vector_norm(t0_state, dim=-1).unsqueeze(-1)
#         v = t1_state / torch.linalg.vector_norm(t1_state, dim=-1).unsqueeze(-1)
#         I = torch.eye(u.shape[-1], device=device)
#         if len(batch_dims) > 0:
#             I = I.unsqueeze(0)
#         u_plus_v = u + v
#         u_plus_v = u_plus_v.unsqueeze(-1)
#         uv = torch.linalg.vecdot(u, v)
#         uv = uv.unsqueeze(-1).unsqueeze(-1)
#         u_extended = u.unsqueeze(-1)
#         v_extended = v.unsqueeze(-1)
#         uvtranspose = torch.transpose(u_plus_v, -2, -1)
#         vut = 2 * v_extended * torch.transpose(u_extended, -2, -1)
#         R = I - u_plus_v / (1 + uv) * uvtranspose + vut
#         indices = torch.triu_indices(R.shape[-2], R.shape[-1], offset=1)
#         R_input = R[..., indices[0], indices[1]] #Bx28
#         T_input = torch.linalg.vector_norm(t1_state - t0_state, dim=-1)
#         T_input = T_input.unsqueeze(-1)
#         # State + Vel + Rot
#         source = torch.cat((t0_state, T_input, R_input), dim=-1)

#         return source

#     def forward(self, observation):
#         t1_state = observation[..., observation.shape[-1] //2:]
#         integration_input = self.create_path_integration_input(observation, device=observation.device)
#         integration_prediction = self.path_integration_model(integration_input) #256
#         integration_prediction = torch.nan_to_num(integration_prediction, nan=0.0, posinf=0.0, neginf=0.0)
#         #how to get the 256 items from the second-to-last layer?
#         policy_input = torch.cat((integration_prediction, t1_state), dim=-1) #+8
#         return policy_input

def transfer_weights(source, target):
    for ind, layer_target in enumerate(target):
        if isinstance(layer_target, torch.nn.Linear):
            layer_target.weight.data.copy_(source[str(ind), 'weight'].clone().data)
            layer_target.bias.data.copy_(source[str(ind), 'bias'].clone().data)

def make_ppo_models_state(proof_environment):

    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": proof_environment.action_spec.space.low,
        "max": proof_environment.action_spec.space.high,
        "tanh_loc": False,
    }

    num_energy_prediction_cells = 512
    predictor_module = EnergyPredictor(in_features = input_shape[-1] // 2, num_place_cells=384, 
                                       num_head_cells=24, 
                                       num_cells=num_energy_prediction_cells)
    #predictor_module.eval()
    
    energy_prediction_module = TensorDictModule(
        predictor_module,
        in_keys=["observation"],
        out_keys=["integration_prediction", "place_energy_prediction", "head_energy_prediction"],
    )
    
    #energy_prediction_module = energy_prediction_module.to("cuda")

    params = TensorDict.from_module(energy_prediction_module, as_module=True)
    params = params.load_memmap(get_project_root_path_vscode() + "models/model_state_halfcheetahv4_384").to_tensordict()

    transfer_weights(params['module', 'path_integration_model'], energy_prediction_module.path_integration_model)
    transfer_weights(params['module', 'place_energy_output_model'], energy_prediction_module.place_energy_output_model)
    transfer_weights(params['module', 'head_energy_output_model'], energy_prediction_module.head_energy_output_model)
   
    energy_prediction_module.requires_grad_(False)
    energy_prediction_module.eval()
    #energy_prediction_module = energy_prediction_module.to("cuda")
    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1] + 256,
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


def make_ppo_models(env_name):
    proof_environment = make_env(env_name, device="cpu")
    actor, critic, energy_prediction_module, mean_predict_module, scale_predict_module = make_ppo_models_state(proof_environment)
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