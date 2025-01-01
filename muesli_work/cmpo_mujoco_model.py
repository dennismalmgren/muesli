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
from modules.layers import SoftmaxLayer, ClampOperator

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
    from cmpo_utils_mujoco_model import eval_model, make_env, make_ppo_models
    from cmpo_loss_model import CMPOLoss
    from torchrl.modules import TanhNormal
#    from torchrl.objectives.value import VTrace
    from vtrace import VTrace
    #from retrace import ReTrace


    device = "cpu" if not torch.cuda.device_count() else "cuda"
    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // cfg.collector.frames_per_batch)
        * cfg.loss.ppo_epochs
        * num_mini_batches
    )

    # Create models (check utils_mujoco.py)
    latent_actor_module, value_module, reward_module, encoder_module, dynamics_module, actor_module, reward_support, value_support = make_ppo_models(cfg.env.env_name, cfg)
    latent_actor_module, value_module, reward_module, encoder_module, dynamics_module, actor_module, reward_support, value_support = \
        latent_actor_module.to(device), value_module.to(device), reward_module.to(device), encoder_module.to(device), dynamics_module.to(device), actor_module.to(device), reward_support.to(device), value_support.to(device)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, device),
        policy=actor_module,
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
    # estimator = GAE(
    #     gamma=cfg.loss.gamma,
    #     lmbda=cfg.loss.gae_lambda,
    #     value_network=value_module,
    #     average_gae=False,
    # )

    loss_module = CMPOLoss(
        actor_network=latent_actor_module,
        critic_network=value_module,
        reward_network=reward_module,
        encoder_network=encoder_module,
#        policy_network=policy_module,
        dynamics_network=dynamics_module,
        value_support = value_support,
        reward_support = reward_support,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
        functional=False,
        #use_targets = False
    )

    critic_params = list(value_module.parameters()) + \
                                    list(reward_module.parameters()) + \
                                    list(encoder_module.parameters()) + \
                                    list(dynamics_module.parameters())
    # Create optimizers
    policy_optim = torch.optim.Adam(latent_actor_module.parameters(), lr=cfg.optim.lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic_params, lr=cfg.optim.lr, eps=1e-5)

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
    grad_norms = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])
    estimator = VTrace(gamma = 0.99,
                       value_network=value_module,
                       actor_network=actor_module,
                       average_adv=False)
    target_params_value = TensorDict.from_module(value_module)
    target_params_actor = TensorDict.from_module(latent_actor_module)
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
       
        #print('ok')

        ################################
        for j in range(cfg_loss_ppo_epochs):  

            with torch.no_grad():
                sampled_actions = prior_dist.sample(sample_shape=(N,)).permute(1, 0, 2)
                data['sampled_actions'] = sampled_actions
                z_cmpo_td = data.clone(False)
                prior_actions = z_cmpo_td["sampled_actions"]
                N = prior_actions.shape[1]

                z_cmpo_td['observation_encoded'] = z_cmpo_td['observation_encoded'].unsqueeze(-2).expand(-1, N, -1)
                predicted_rewards = reward_module(z_cmpo_td['observation_encoded'], prior_actions)[1]
                predicted_next_observation_encoded = dynamics_module(z_cmpo_td['observation_encoded'], prior_actions)
                #z_cmpo_td['next', 'observation'] = z_cmpo_td['next', 'observation'].unsqueeze(-2).expand(-1, N, -1)
                #next_observation_encoded = encoder_module(z_cmpo_td['next', 'observation']) #todo: should be via dynamics module
                #actually fix it right away...
                with target_params_value.to_module(value_module):
                    predicted_values = value_module(predicted_next_observation_encoded)[1]
                    values = value_module(z_cmpo_td['observation_encoded'])[1]
                predicted_qvalues = predicted_rewards + cfg.loss.gamma * predicted_values
                cmpo_advantages = predicted_qvalues - values
                cmpo_loc = cmpo_advantages.mean(dim=1, keepdim=True)
                cmpo_scale = cmpo_advantages.std(dim=1, keepdim=True).clamp_min(1e-6)
                cmpo_advantages = (cmpo_advantages - cmpo_loc) / cmpo_scale
                #cmpo_advantages = cmpo_advantages + 0.02 * torch.rand_like(cmpo_advantages) - 0.01
                cmpo_advantages = torch.clip(cmpo_advantages, torch.tensor(-1.0, device=cmpo_advantages.device), torch.tensor(1.0, device=cmpo_advantages.device))
                cmpo_advantages = torch.exp(cmpo_advantages)
                z_cmpo = (1 + torch.sum(cmpo_advantages, dim=1, keepdim=True) - cmpo_advantages)
                regularization = cmpo_advantages / z_cmpo
                data["cmpo_regularization"] = regularization
                
            data = encoder_module(data)
            data["next", "observation_encoded"] = encoder_module(data["next", "observation"])
            with torch.no_grad():
                current_params_value = TensorDict.from_module(value_module)
                data = estimator(data, 
                                 params=current_params_value,
                                 target_params=target_params_value)
                
                target_params_value.lerp_(current_params_value, 0.1)
            data_reshape = data.reshape(-1)
            data_buffer.extend(data_reshape)
            for k, batch in enumerate(data_buffer):
                # Get a data batch
                batch = batch.to(device)

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in policy_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                    for group in critic_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                   
                # if cfg_loss_anneal_clip_eps:
                #     loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective", "loss_regularization", "loss_reward", "loss_consistency"
                ).detach()
                critic_loss = loss["loss_critic"] + loss["loss_reward"] + 20 * loss["loss_consistency"]
                actor_loss = loss["loss_objective"] + loss["loss_regularization"] #+ loss["loss_entropy"]

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_grad = torch.nn.utils.clip_grad_norm_(critic_params, 10.0)
                critic_optim.step()

                policy_optim.zero_grad()
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(latent_actor_module.parameters(), 10.0)
                policy_optim.step()
                grad_norms[j, k] = TensorDict({
                    "grad_norm_actor": actor_grad,
                    "grad_norm_critic": critic_grad
                })

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        grad_norms_mean = grad_norms.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in grad_norms_mean.items():
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
                actor_module.eval()
                eval_start = time.time()
                test_rewards = eval_model(
                    actor_module, test_env, num_episodes=cfg_logger_num_test_episodes
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "eval/reward": test_rewards.mean(),
                        "eval/time": eval_time,
                    }
                )
                actor_module.train()

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