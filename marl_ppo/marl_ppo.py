import torch
import hydra
from torchrl._utils import logger as torchrl_logger
from torch import multiprocessing
import tqdm
from torchrl.envs.libs import PettingZooWrapper
from envs.matching_pennies_env import MatchingPenniesEnv
from torchrl.envs import (
   check_env_specs,
   # ExplorationType,
   # PettingZooEnv,
    RewardSum,
   # set_exploration_type,
    TransformedEnv,
    ParallelEnv,
    EnvCreator
)
import copy
from torchrl.modules import (
    MultiAgentMLP,
    ProbabilisticActor,
    TanhNormal
)
import time
from tensordict.nn import TensorDictModule, TensorDictSequential, AddStateIndependentNormalScale
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from tensordict import TensorDictBase

def process_batch(group_map, batch: TensorDictBase) -> TensorDictBase:
    """
    If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
    `"terminated"` and `"done"`.
    This is needed to present them with the same shape as the reward to the loss.
    """
    for group in group_map.keys():
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch

@hydra.main(config_path=".", config_name="marl_ppo", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    torch.manual_seed(cfg.seed)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    base_env = MatchingPenniesEnv()
    base_env = PettingZooWrapper(base_env)
    
    print(f"group_map: {base_env.group_map}")
    print("action_spec:", base_env.full_action_spec)
    print("reward_spec:", base_env.full_reward_spec)
    print("done_spec:", base_env.full_done_spec)
    print("observation_spec:", base_env.observation_spec)


    print("action_keys:", base_env.action_keys)
    print("reward_keys:", base_env.reward_keys)
    print("done_keys:", base_env.done_keys)
    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"] * len(base_env.group_map.keys()),
        ),
    )

    def make_env(cfg):
        def env_maker(cfg):
            base_env = MatchingPenniesEnv()
            base_env = PettingZooWrapper(base_env, device=device)
            return base_env
        
        parallel_env = ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(lambda cfg=cfg: env_maker(cfg)),
        )
        env = TransformedEnv(
            parallel_env,
            RewardSum( 
                in_keys=base_env.reward_keys,
                reset_keys=["_reset"] * len(base_env.group_map.keys()),
            ),
        )

        return env
    
    check_env_specs(env)

    n_rollout_steps = 5
    rollout = env.rollout(n_rollout_steps)
    print(f"rollout of {n_rollout_steps} steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)
    
    policy_modules = {}
    for group, agents in env.group_map.items():
        share_parameters_policy = True  # Can change this based on the group

        policy_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=env.full_action_spec[group, "action"].n,  # n_actions_per_agents
            n_agents=len(agents),  # Number of agents in the group
            centralised=False,  # the policies are decentralised (i.e., each agent will act from its local observation)
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.ReLU,
        )

        policy_module = TensorDictModule(
                policy_net,
                in_keys=[(group, "observation")],
                out_keys=[(group, "logits")],
            )  # We just name the input and output that the network will read and write to the input tensordict
        policy_modules[group] = policy_module

    policies = {}
    for group, _agents in env.group_map.items():
        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "logits")],
            out_keys=[(group, "action")],
            distribution_class=Categorical,
            distribution_kwargs={
             
            },
            return_log_prob=True,
        )
        policy = policy.to(device)
        policies[group] = policy

    critics = {}
    for group, agents in env.group_map.items():
        share_parameters_critic = True  # Can change for each group
        MADDPG = True  # IDDPG if False, can change for each group
        # This module applies the lambda function: reading the action and observation entries for the group
        # and concatenating them in a new ``(group, "obs_action")`` entry
        # cat_module = TensorDictModule(
        #     lambda obs, action: torch.cat([obs, action], dim=-1),
        #     in_keys=[(group, "observation"), (group, "action")],
        #     out_keys=[(group, "obs_action")],
        # )

        critic_module = TensorDictModule(
            module=MultiAgentMLP(
                n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],
           #     + env.full_action_spec[group, "action"].shape[-1],
                n_agent_outputs=1,  # 1 value per agent
                n_agents=len(agents),
                centralised=MADDPG,
                share_params=share_parameters_critic,
                device=device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            in_keys=[(group, "observation")],  # Read ``(group, "obs_action")``
            out_keys=[
                (group, "state_value")
            ],  # Write ``(group, "state_action_value")``
        )

        critics[group] = critic_module
        
    reset_td = env.reset().to(device)
    for group, _agents in env.group_map.items():
        print(f"Running value and policy for group '{group}':")
        td_step = policies[group](reset_td)
        td_res = critics[group](td_step)
        print(td_res)
    
    agents_exploration_policy = TensorDictSequential(*policies.values())
    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    frames_per_batch = cfg.collector.frames_per_batch
    total_frames = cfg.collector.total_frames
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs

    collector = SyncDataCollector(
        create_env_fn=make_env(cfg),
        policy=agents_exploration_policy,
        device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        storing_device=device,
        max_frames_per_traj = -1
    )

    replay_buffers = {}
    for group, _agents in env.group_map.items():
        # Create data buffer
        sampler = SamplerWithoutReplacement()
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
            sampler=sampler,
            batch_size=cfg.loss.mini_batch_size,
        )
        replay_buffers[group] = replay_buffer

    losses = {}
    for group, _agents in env.group_map.items():
        # Create loss and adv modules
        adv_module = GAE(
            gamma=cfg.loss.gamma,
            lmbda=cfg.loss.gae_lambda,
            value_network=critics[group],
            average_gae=True,
            device=device
        )

        loss_module = ClipPPOLoss(
            actor_network=policies[group],
            critic_network=critics[group],
            clip_epsilon=cfg.loss.clip_epsilon,
            loss_critic_type=cfg.loss.loss_critic_type,
            entropy_coef=cfg.loss.entropy_coef,
            critic_coef=cfg.loss.critic_coef,
            normalize_advantage=True,
        )

        adv_module.set_keys(
            value=(group, "state_value"),
            value_target=(group, "value_target"),
            advantage=(group, "advantage"),
            reward = (group, "reward"),
            done = (group, "done"),
            terminated = (group, "terminated"),
        )
        loss_module.set_keys(
            value=(group, "state_value"),
            action=(group, "action"),
            value_target=(group, "value_target"),
            advantage=(group, "advantage"),
            reward = (group, "reward"),
            done = (group, "done"),
            terminated = (group, "terminated"),
        )
        losses[group] = (adv_module, loss_module) #this is pretty cool
    optimizers = {
        group: {
            "loss_actor": torch.optim.Adam(
                loss_module.actor_network_params.flatten_keys().values(), lr=cfg.optim.lr
            ),
            "loss_critic": torch.optim.Adam(
                loss_module.critic_network_params.flatten_keys().values(), lr=cfg.optim.lr
            ),
        }
        for group, (adv_module, loss_module) in losses.items()
    }

    pbar = tqdm.tqdm(
        total=cfg.collector.total_frames,
        desc=", ".join(
            [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
        ),
    )
    sampling_start = time.time()

    episode_reward_mean_map = {group: [] for group in env.group_map.keys()}
    train_group_map = copy.deepcopy(env.group_map)
    collected_frames = 0
    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())
        batch = process_batch(train_group_map, data)
        # Get training rewards and episode lengths

        training_start = time.time()
        for group in train_group_map.keys():
            group_batch = batch.exclude(
                *[
                    key
                    for _group in train_group_map.keys()
                    if _group != group
                    for key in [_group, ("next", _group)]
                ]
            )  # Exclude data from other groups
            for j in range(cfg_loss_ppo_epochs):
                with torch.no_grad():
                    group_batch = losses[group][0](group_batch)
                group_batch = group_batch.reshape(
                    -1
                ) 
                replay_buffers[group].extend(group_batch)
                for k, sampled_batch in enumerate(replay_buffers[group]):
                    sampled_batch = sampled_batch.to(device)
                    loss_vals = losses[group][1](sampled_batch)
                    critic_loss = loss_vals["loss_critic"]
                    critic_optimizer = optimizers[group]["loss_critic"]
                    actor_loss = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
                    actor_optimizer = optimizers[group]["loss_actor"]
                    actor_loss.backward()
                    critic_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
        #logging
        for group in train_group_map.keys(): 
            episode_reward_mean = (
                data.get(("next", group, "episode_reward"))[
                    data.get(("next", group, "done"))
                ]
                .mean()
                .item()
            )
            episode_reward_mean_map[group].append(episode_reward_mean)
        pbar.set_description(
            ", ".join(
                [
                    f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                    for group in env.group_map.keys()
                ]
            ),
            refresh=False,
        )
        pbar.update()
if __name__ == "__main__":
    main()