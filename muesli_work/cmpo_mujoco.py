# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm
results from Schulman et al. 2017 for the on MuJoCo Environments.
"""
import hydra
from torchrl._utils import logger as torchrl_logger
from torchrl.record import VideoRecorder


@hydra.main(config_path="", config_name="config_mujoco", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import time

    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger
    from cmpo_utils_mujoco import eval_model, make_env, make_ppo_models
    from cmpo_loss import CMPOLoss
    from torchrl.modules import TanhNormal
#    from torchrl.objectives.value import VTrace
    from vtrace import VTrace
    #from retrace import ReTrace
    from torchrl.objectives.value.functional import vtrace_advantage_estimate

    device = "cpu" if not torch.cuda.device_count() else "cuda"
    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // cfg.collector.frames_per_batch)
        * cfg.loss.ppo_epochs
        * num_mini_batches
    )

    # Create models (check utils_mujoco.py)
    actor, critic, reward_predictor = make_ppo_models(cfg.env.env_name)
    actor, critic, reward_predictor = actor.to(device), critic.to(device), reward_predictor.to(device)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, device),
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
    )

    # Create loss and adv modules
    estimator = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    loss_module = CMPOLoss(
        actor_network=actor,
        critic_network=critic,
        reward_network=reward_predictor,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
        functional=False,
        #use_targets = False
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.optim.lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.optim.lr, eps=1e-5)
    reward_predictor_optim = torch.optim.Adam(reward_predictor.parameters(), lr=cfg.optim.lr, eps=1e-5)

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
                "mode": cfg.logger.mode,
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

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    sampling_start = time.time()

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_test_interval = cfg.logger.test_interval
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])
    estimator = VTrace(gamma = 0.99,
                       value_network=critic,
                       actor_network=actor,
                       average_adv=False)
    
    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

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

        training_start = time.time()
        N = 24
        prior_dist = TanhNormal(
            low = test_env.action_spec_unbatched.space.low,
            high = test_env.action_spec_unbatched.space.high,
            tanh_loc=False,
            loc = data["loc"],
            scale = data["scale"]
        )
       
        #######################
        with torch.no_grad():
            sampled_actions = prior_dist.sample(sample_shape=(N,)).permute(1, 0, 2)
            data['sampled_actions'] = sampled_actions
            z_cmpo_td = data.clone(False)
            prior_actions = z_cmpo_td["sampled_actions"]
            N = prior_actions.shape[1]

            z_cmpo_td['observation'] = z_cmpo_td['observation'].unsqueeze(-2).expand(-1, N, -1)
            predicted_rewards = reward_predictor(z_cmpo_td['observation'], prior_actions)

            z_cmpo_td['next', 'observation'] = z_cmpo_td['next', 'observation'].unsqueeze(-2).expand(-1, N, -1)

            predicted_values = critic(z_cmpo_td['next', 'observation'])

            predicted_qvalues = predicted_rewards + cfg.loss.gamma * predicted_values
            values = critic(z_cmpo_td['observation'])
            cmpo_advantages = predicted_qvalues - values
            cmpo_loc = cmpo_advantages.mean(dim=1, keepdim=True)
            cmpo_scale = cmpo_advantages.std(dim=1, keepdim=True).clamp_min(1e-6)
            cmpo_advantages = (cmpo_advantages - cmpo_loc) / cmpo_scale
            #cmpo_advantages = cmpo_advantages + 0.02 * torch.rand_like(cmpo_advantages) - 0.01
            cmpo_advantages = torch.clip(cmpo_advantages, torch.tensor(-1.0, device=cmpo_advantages.device), torch.tensor(1.0, device=cmpo_advantages.device))
            cmpo_advantages = torch.exp(cmpo_advantages)
            z_cmpo = (1 + torch.sum(cmpo_advantages, dim=1, keepdim=True) - cmpo_advantages) / N
            regularization = cmpo_advantages / z_cmpo
            data["cmpo_regularization"] = regularization
        #print('ok')

        ################################
        for j in range(cfg_loss_ppo_epochs):                     
            with torch.no_grad():
                data = estimator(data)

            
                
            data_reshape = data.reshape(-1)
            data_buffer.extend(data_reshape)
            for k, batch in enumerate(data_buffer):
                # Get a data batch
                batch = batch.to(device)

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in actor_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                    for group in critic_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                    for group in reward_predictor_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha                        
                # if cfg_loss_anneal_clip_eps:
                #     loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective", "loss_reward_predictor", "loss_regularization"
                ).detach()
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_regularization"] + loss["loss_entropy"]
                reward_predictor_loss = loss["loss_reward_predictor"]
                # Backward pass
                actor_optim.zero_grad()
                critic_optim.zero_grad()
                reward_predictor_optim.zero_grad()

                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                actor_optim.step()
                critic_loss.backward()
                reward_predictor_loss.backward()
                reward_grad = torch.nn.utils.clip_grad_norm_(reward_predictor.parameters(), 10.0)
                critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)

                # Update the networks
                critic_optim.step()
                reward_predictor_optim.step()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * cfg_optim_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": alpha * cfg_loss_clip_epsilon
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                actor.eval()
                eval_start = time.time()
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "eval/reward": test_rewards.mean(),
                        "eval/time": eval_time,
                    }
                )
                actor.train()

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        sampling_start = time.time()

    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()