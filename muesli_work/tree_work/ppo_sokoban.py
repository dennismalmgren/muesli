# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm
results from Schulman et al. 2017 for the Atari Environments.
"""
from __future__ import annotations

import warnings

import hydra

from torchrl._utils import compile_with_warmup


@hydra.main(config_path="", config_name="ppo_config_sokoban", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from tensordict.nn import CudaGraphModule

    from torchrl._utils import timeit
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    #from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record import VideoRecorder
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_sokoban import eval_model, make_parallel_env, make_ppo_models
    from torchrl.envs.utils import step_mdp
    import gym_sokoban
    from ppo_loss import ClipPPOLoss
    from tensordict import pad
    from torchrl.envs.transforms import TensorDictPrimer
    from torchrl.data.tensor_specs import Unbounded

    torch.set_float32_matmul_precision("high")

    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Correct for frame_skip
    total_frames = cfg.collector.total_frames 
    frames_per_batch = cfg.collector.frames_per_batch 
    mini_batch_size = cfg.loss.mini_batch_size 
    test_interval = cfg.logger.test_interval 

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create models (check utils_atari.py)
    actor, critic, recurrent_module = make_ppo_models(cfg, device=device)
    # p_env = make_parallel_env(cfg.env.num_envs, device)
    # td_ = p_env.reset()
    # done = td_["done"]
    # for i in range(2000):
    #     action = p_env.action_spec.sample()
    #     td_['action'] = action
    #     td, td_ = p_env.step_and_maybe_reset(td_)
    #     done = td['next', 'done']
    #TODO: Add primer support to the minGRU
    primer = TensorDictPrimer(
            {
                
                "recurrent_state": Unbounded(shape=(32)),
                #"recurrent_state_2": Unbounded(shape=(32)),
            },
            expand_specs=True,
        )
    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_parallel_env(cfg.env.num_envs, cfg.env.env_name, device, 
                                        transform=primer),
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        max_frames_per_traj=-1,
        compile_policy={"mode": compile_mode, "warmup": 1} if compile_mode else False,
        cudagraph_policy=cfg.compile.cudagraphs,
    )

    # Create data buffer
    sampler = SliceSamplerWithoutReplacement(num_slices=32, strict_length=False)
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, compilable=cfg.compile.compile, device=device
        ),
        sampler=sampler,
        batch_size=mini_batch_size,
        compilable=cfg.compile.compile,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
        vectorized=not cfg.compile.compile,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
    )
    # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_sokoban7x7")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
                "mode": cfg.logger.mode,
            },
        )
        logger_video = cfg.logger.video
    else:
        logger_video = False

    # Create test environment
    test_env = make_parallel_env(1, cfg.env.env_name, device, is_test=True, transform=primer.clone())
    if logger_video:
        test_env = test_env.append_transform(
            VideoRecorder(logger, tag="rendering/test", in_keys=["pixels_int"])
        )
    test_env.eval()

    # Main loop
    collected_frames = 0
    num_network_updates = torch.zeros((), dtype=torch.int64, device=device)
    pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    )

    def update(batch, num_network_updates):
        optim.zero_grad(set_to_none=True)
        # Linearly decrease the learning rate and clip epsilon
        alpha = torch.ones((), device=device)
        if cfg_optim_anneal_lr:
            alpha = 1 - (num_network_updates / total_network_updates)
            for group in optim.param_groups:
                group["lr"] = cfg_optim_lr * alpha
        if cfg_loss_anneal_clip_eps:
            loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
        num_network_updates = num_network_updates + 1
        # Get a data batch
        batch = batch.to(device, non_blocking=True)

        # Forward pass PPO loss
        loss = loss_module(batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
        # Backward pass
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_norm=cfg_optim_max_grad_norm
        )

        # Update the networks
        optim.step()
        return loss.detach().set("alpha", alpha), num_network_updates

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)
        adv_module = compile_with_warmup(adv_module, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)
        adv_module = CudaGraphModule(adv_module)

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    cfg_optim_max_grad_norm = cfg.optim.max_grad_norm
    cfg.loss.clip_epsilon = cfg_loss_clip_epsilon
    losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            data = next(collector_iter)

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # Get training rewards and episode lengths['params', 'num_boxes']
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            episode_initial_status = data['num_boxes'].unsqueeze(-1)[data["next", "done"]]
            episode_completion_status = data['next', 'boxes_on_target'].unsqueeze(-1)[data["next", "done"]]
            episode_to_go = episode_initial_status - episode_completion_status
            episodes_completed = (episode_to_go == 0).float().sum().item()
            episode_partial_completion = episode_completion_status / episode_initial_status
            metrics_to_log.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                    "train/episode_completion_percentage": (episodes_completed / len(episode_rewards)) * 100.0,
                    "train/episode_partial_completion_percentage": (episode_partial_completion.sum().item() / len(episode_rewards)) * 100.0,
                }
            )

        with timeit("training"):
            for j in range(cfg_loss_ppo_epochs):

                # Compute GAE
                with torch.no_grad(), timeit("adv"):
                    torch.compiler.cudagraph_mark_step_begin()
                    data = adv_module(data)
                    if compile_mode:
                        data = data.clone()
                with timeit("rb - extend"):
                    # Update the data buffer
                    data_reshape = data.reshape(-1)
                    data_buffer.extend(data_reshape)

                for k, batch in enumerate(data_buffer):
                    batch["valid_samples"] = torch.ones(batch.shape, dtype=torch.bool, device=device)
                    batch = pad(batch, [mini_batch_size - batch.shape[0], 0]) 
                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss, num_network_updates = update(
                            batch, num_network_updates=num_network_updates
                        )
                    loss = loss.clone()
                    num_network_updates = num_network_updates.clone()
                    losses[j, k] = loss.select(
                        "loss_critic", "loss_entropy", "loss_objective"
                    )

        # Get training losses and times
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            metrics_to_log.update({f"train/{key}": value.item()})
        metrics_to_log.update(
            {
                "train/lr": loss["alpha"] * cfg_optim_lr,
                "train/clip_epsilon": loss["alpha"] * cfg_loss_clip_epsilon,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            if ((i - 1) * frames_in_batch) // test_interval < (
                i * frames_in_batch
            ) // test_interval:
                actor.eval()
                test_rewards, episode_completion_percentage, episode_partial_completion_percentage = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes
                )
                metrics_to_log.update(
                    {
                        "eval/reward": test_rewards.mean(),
                        "eval/episode_completion_percentage": episode_completion_percentage.mean(),
                        "eval/episode_partial_completion_percentage": episode_partial_completion_percentage.mean()
                    }
                )
                actor.train()
        if logger:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            for key, value in metrics_to_log.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()

    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()


if __name__ == "__main__":
    main()