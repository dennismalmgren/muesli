# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch.nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule, TensorDictSequential#, EnsembleModule
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
    TensorDictPrimer,
    CatFrames,
    RenameTransform,
    InitTracker,
    RewardScaling
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder
from dynamics_prediction import DynamicsModel
from ensemble_module import EnsembleModule


# ====================================================================
# Environment utils
# --------------------------------------------------------------------

def make_env(env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2)) #skipping this for easier comparison
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(InitTracker())
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    #env.append_transform(TensorDictPrimer(state_prediction=env.observation_spec["observation"],
    #                                      previous_action=env.action_spec))
    return env


# ====================================================================
# Saving and loading
# --------------------------------------------------------------------
def load_model_state(model_name, run_folder_name=""):
    #debug outputs is at the root.
    #commandline outputs is at scripts/patrol/outputs
    load_from_saved_models = run_folder_name == ""
    if load_from_saved_models:
        outputs_folder = "../../../saved_models/"
    else:
        outputs_folder = "../../"

    model_load_filename = f"{model_name}.pt"
    load_model_dir = outputs_folder + run_folder_name
    print('Loading model from ' + load_model_dir)
    loaded_state = torch.load(load_model_dir + f"{model_load_filename}")
    return loaded_state

# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment, device):

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

    policy_calculation_module = TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_calculation_module,
        in_keys=["loc", "scale"],
        spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    actor_module = policy_module
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

    
    dynamics_model = DynamicsModel(input_shape[-1], num_outputs, 32)
    dynamics_module = TensorDictModule(
        module=dynamics_model,
        in_keys=["observation", "action"],
        out_keys=["predicted_state"]
    )

    ensemble = EnsembleModule(
        module=dynamics_module,
        num_copies=5,
    )

    return actor_module, value_module, policy_module, ensemble, policy_calculation_module


def make_ppo_models(env_name, device):
    proof_environment = make_env(env_name, device=device)
    actor, critic, policy, dynamics, policy_calculation_module = make_ppo_models_state(proof_environment, device=device)
    return actor, critic, policy, dynamics, policy_calculation_module


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()

def calculate_sm(signal, time_step=1):
    import numpy as np
    signal = signal.numpy()
    magnitude_signal = np.linalg.norm(signal, axis=-1)

    fft_result = np.fft.fft(magnitude_signal)
    amplitudes = np.abs(fft_result)  # Magnitude of the FFT
    frequencies = np.fft.fftfreq(len(magnitude_signal), time_step)  # Frequency corresponding to each FFT bin

    # Step 3: Filter positive frequencies (since FFT is symmetric)
    positive_frequencies = frequencies[frequencies >= 0]
    positive_amplitudes = amplitudes[frequencies >= 0]

    # Step 4: Calculate the smoothness measure (Sm)
    n = len(positive_frequencies)
    sampling_frequency = 1 / time_step  # Sampling frequency = 5 Hz

# Calculate the smoothness measure using the formula
    Sm = (2 / (n * sampling_frequency)) * np.sum(positive_amplitudes * positive_frequencies)
    return Sm

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
        sm = calculate_sm(td_test["action"])
        print("Sm: ", sm)
        action_sgns = td_test["action"][:, 0] / torch.abs(td_test["action"][:, 0])
        delta_sgns = action_sgns[:-1] * action_sgns[1:]
        delta_sgns[delta_sgns == 1.] = 0
        change = -torch.sum(delta_sgns) / len(delta_sgns) * 100.0
        print("Sign change action dim 0 (percentage): ", change)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
        test_env.apply(dump_video)
    del td_test
    return torch.cat(test_rewards, 0).mean()