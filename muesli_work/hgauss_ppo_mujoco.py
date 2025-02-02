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
    from torchrl.objectives import ClipPPOLoss, group_optimizers
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record import VideoRecorder
    from torchrl.record.loggers import generate_exp_name, get_logger
    from hgauss_utils_mujoco import eval_model, make_env, make_ppo_models
    from rs_ppo_loss import RSClipPPOLoss

    from torchrl.objectives.value.utils import _split_and_pad_sequence, _get_num_per_traj

    torch.set_float32_matmul_precision("high")

    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    #num_episodes_per_batch = cfg.collector.frames_per_batch // 1000
    #num_episodes_per_batch = cfg.collector.env_per_collector
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
    actor, critic, reward_predictor, support = make_ppo_models(cfg.env.env_name, device=device)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, device),
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        compile_policy={"mode": compile_mode, "warmup": 1} if compile_mode else False,
        cudagraph_policy=cfg.compile.cudagraphs,
    )
    import math
    nbins = 101
    Vmin = -10.0
    Vmax = 500.0
    dk = (Vmax - Vmin) / (nbins - 4)
    Ktot = dk * nbins
    Vmax = math.ceil(Vmin + Ktot)

    support = torch.linspace(Vmin, Vmax, nbins).to(device)

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            cfg.collector.frames_per_batch,
            compilable=cfg.compile.compile,
            device=device,
        ),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
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

    loss_module = RSClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        support=support,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5
    )
    optim = group_optimizers(actor_optim, critic_optim)
    del actor_optim, critic_optim

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
                "mode": cfg.logger.mode
            },
        )
        logger_video = cfg.logger.video
    else:
        logger_video = False

    # Create test environment
    test_env = make_env(cfg.env.env_name, device, from_pixels=logger_video)
    if logger_video:
        test_env = test_env.append_transform(
            VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    test_env.eval()

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

        # Forward pass PPO loss
        loss = loss_module(batch)
        critic_loss = loss["loss_critic"]
        actor_loss = loss["loss_objective"] + loss["loss_entropy"]
        total_loss = critic_loss + actor_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
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

    # Main loop
    collected_frames = 0
    num_network_updates = torch.zeros((), dtype=torch.int64, device=device)
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = torch.tensor(cfg.optim.lr, device=device)
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_test_interval = cfg.logger.test_interval
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            data = next(collector_iter)
        
        if cfg.loss.bootstrap_rs_value_target == "episode":
            vals = torch.linspace(0, num_episodes_per_batch, num_episodes_per_batch + 1) / num_episodes_per_batch
            scale = torch.ones_like(vals)
            loc = torch.zeros_like(vals)
            w = torch.distributions.normal.Normal(loc, scale)
            inv = w.icdf(vals)
            
            inv_plus_eta = inv + 0.25
            w_final = w.cdf(inv_plus_eta)
            #trim the edges
            w_deltas = w_final[1:] - w_final[:-1]
            w_deltas = w_deltas / torch.sum(w_deltas) * num_episodes_per_batch

        elif cfg.loss.bootstrap_rs_value_target == "parallel":
            vals = torch.linspace(0, cfg.collector.env_per_collector, cfg.collector.env_per_collector + 1, device=device) / cfg.collector.env_per_collector
            scale = torch.ones_like(vals)
            loc = torch.zeros_like(vals)
            w = torch.distributions.normal.Normal(loc, scale)
            inv = w.icdf(vals)
            
            inv_plus_eta = inv + 0.25
            w_final = w.cdf(inv_plus_eta)
            #trim the edges
            w_deltas = w_final[1:] - w_final[:-1]
            w_deltas = w_deltas / torch.sum(w_deltas) * w_deltas.numel()
            # vals = torch.linspace(0, cfg.collector.frames_per_batch, cfg.collector.frames_per_batch + 1) / cfg.collector.frames_per_batch
            # scale = torch.ones_like(vals)
            # loc = torch.zeros_like(vals)
            # w = torch.distributions.normal.Normal(loc, scale)
            # inv = w.icdf(vals)
            
            # inv_plus_eta = inv + 0.25
            # w_final = w.cdf(inv_plus_eta)
            # #trim the edges
            # w_deltas = w_final[1:] - w_final[:-1]
            # w_deltas = w_deltas / torch.sum(w_deltas) * num_episodes_per_batch

        log_info = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )
    # # _get_num_per_traj and _split_and_pad_sequence need
    # # time dimension at last position
    # done = done.transpose(-2, -1)
    # terminated = terminated.transpose(-2, -1)
    # reward = reward.transpose(-2, -1)
    # state_value = state_value.transpose(-2, -1)
    # next_state_value = next_state_value.transpose(-2, -1)

    # gammalmbda = gamma * lmbda
    # not_terminated = (~terminated).int()
    # td0 = reward + not_terminated * gamma * next_state_value - state_value

    # num_per_traj = _get_num_per_traj(done)
    # td0_flat, mask = _split_and_pad_sequence(td0, num_per_traj, return_mask=True)
    #sort data
        if cfg.loss.bootstrap_rs_value_target == "episode":
            data = data.reshape((num_episodes_per_batch, 1000))
            episode_rewards = data["next", "episode_reward"][data["next", "done"]]
            sorted_indices = torch.sort(episode_rewards)[1]
            data = data[sorted_indices]
            data["rs_weight"] = w_deltas.unsqueeze(-1).expand(-1, 1000).unsqueeze(-1)

        with timeit("training"):
            for j in range(cfg_loss_ppo_epochs):

                # Compute GAE
                with torch.no_grad(), timeit("adv"):
                    torch.compiler.cudagraph_mark_step_begin()
                    data = adv_module(data)
                    if compile_mode:
                        data = data.clone()

                if cfg.loss.bootstrap_rs_value_target == "single":
                    value_targets = data["value_target"].squeeze(-1)
                    sorted_indices = torch.sort(value_targets)[1]
                    data = data[sorted_indices]
                    data["rs_weight"] = w_deltas.unsqueeze(-1).expand(cfg.collector.frames_per_batch, 1)
                elif cfg.loss.bootstrap_rs_value_target == "parallel":
                    value_targets = data["value_target"].squeeze(-1)
                    value_targets = value_targets[..., -1]
                    sorted_indices = torch.sort(value_targets)[1]
                    w_deltas_batch = w_deltas[sorted_indices]
                    data["rs_weight"] = w_deltas_batch.unsqueeze(-1).unsqueeze(-1).expand(cfg.collector.env_per_collector, 100, 1)

                with timeit("rb - extend"):
                    # Update the data buffer
                    data_reshape = data.reshape(-1)
                    data_buffer.extend(data_reshape)

                for k, batch in enumerate(data_buffer):
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
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": loss["alpha"] * cfg_optim_lr,
                "train/clip_epsilon": loss["alpha"] * cfg_loss_clip_epsilon
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                actor.eval()
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes
                )
                log_info.update(
                    {
                        "eval/reward": test_rewards.mean(),
                    }
                )
                actor.train()

        if logger:
            log_info.update(timeit.todict(prefix="time"))
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()

    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()


if __name__ == "__main__":
    main()