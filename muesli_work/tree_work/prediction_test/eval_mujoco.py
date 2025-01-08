# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm
results from Schulman et al. 2017 for the on MuJoCo Environments.
"""
from __future__ import annotations

import warnings

import hydra

from torchrl._utils import compile_with_warmup
import numpy as np



@hydra.main(config_path="", config_name="config_mujoco", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from tensordict.nn import CudaGraphModule

    from torchrl._utils import timeit
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import group_optimizers
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record import VideoRecorder
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_mujoco import eval_model, make_env, make_ppo_models, load_model_state
    from ppo_loss import ClipPPOLoss
    torch.set_float32_matmul_precision("high")

    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // cfg.collector.frames_per_batch)
        * cfg.loss.ppo_epochs
        * num_mini_batches
    )

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create models (check utils_mujoco.py)
    actor, critic, policy, kalman = make_ppo_models(cfg.env.env_name, device=device)
    

    load_model = True
    if load_model:
        model_dir="2025-01-08/03-10-45/"
        model_name = "training_snapshot_462848"
        loaded_state = load_model_state(model_name, model_dir)

        policy_state = loaded_state['model_policy']
        critic_state = loaded_state['model_critic']
        kalman_state = loaded_state['model_kalman']
        #optim_state = loaded_state['optimizer_state']

        #collected_frames = loaded_state['collected_frames']['collected_frames']
        #loaded_frames = collected_frames
        policy.load_state_dict(policy_state)
        critic.load_state_dict(critic_state)
        kalman.load_state_dict(kalman_state)
        #optim.load_state_dict(optim_state)

    #frames_remaining = cfg.collector.total_frames - collected_frames

    # Create logger
    logger = None
    if cfg.logger.backend:
        # exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        # logger = get_logger(
        #     cfg.logger.backend,
        #     logger_name="ppo",
        #     experiment_name=exp_name,
        #     wandb_kwargs={
        #         "config": dict(cfg),
        #         "project": cfg.logger.project_name,
        #         "group": cfg.logger.group_name,
        #         "mode": cfg.logger.mode
        #     },
        # )
        logger_video = cfg.logger.video
    else:
        logger_video = False
    
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes

    # Create test environment
    test_env = make_env(cfg.env.env_name, device, from_pixels=logger_video)
    if logger_video:
        test_env = test_env.append_transform(
            VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    test_env.eval()
    
    metrics_to_log = {}

    # Get test rewards
    with torch.no_grad(), set_exploration_type(
        ExplorationType.DETERMINISTIC
    ), timeit("eval"):
            actor.eval()
            test_rewards = eval_model(
                actor, test_env, num_episodes=cfg_logger_num_test_episodes
            )
            metrics_to_log.update(
                {
                    "eval/reward": test_rewards.mean(),
                }
            )
            actor.train()

    if not test_env.is_closed:
        test_env.close()


if __name__ == "__main__":
    main()