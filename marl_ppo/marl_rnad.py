import torch
import hydra
from torchrl._utils import logger as torchrl_logger
from torch import multiprocessing
import tqdm
from torchrl.envs.libs import PettingZooWrapper
from envs.matching_pennies_env import MatchingPenniesEnv
from torchrl.envs import (
    check_env_specs,
    RewardSum,
    DTypeCastTransform,
    DoubleToFloat,
    FlattenObservation,
    TransformedEnv,
    ParallelEnv,
    EnvCreator,
    Compose
)
import copy
from torchrl.modules import (
    MultiAgentMLP,
    ProbabilisticActor,
    TanhNormal,
    MLP
)
import time
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from tensordict import TensorDictBase
from pettingzoo.classic import tictactoe_v3


def process_batch(group_map, batch: TensorDictBase) -> TensorDictBase:
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
 
def reward_transformation(rewards, policies, reg_param):
    transformed_rewards = {}
    for group, reward in rewards.items():
        transformed_rewards[group] = reward - reg_param * torch.log(policies[group])
    return transformed_rewards

@hydra.main(config_path=".", config_name="marl_ppo", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    torch.manual_seed(cfg.seed)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    base_env = env = tictactoe_v3.env()
    base_env = PettingZooWrapper(base_env,
                                 use_mask=True)
    
    env = TransformedEnv(
        base_env,
        
        Compose(
            RewardSum( 
                in_keys=base_env.reward_keys,
                reset_keys=["_reset"] * len(base_env.group_map.keys()),
            ),
            DTypeCastTransform(dtype_in=torch.int8, dtype_out=torch.float32,
                           in_keys=[("player_1", "observation", "observation"),
                            ("player_2", "observation", "observation")]),
            FlattenObservation(first_dim=-3, last_dim=-1,
                in_keys=[("player_1", "observation", "observation"),
                            ("player_2", "observation", "observation")])
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
            Compose(
            RewardSum( 
                in_keys=base_env.reward_keys,
                reset_keys=["_reset"] * len(base_env.group_map.keys()),
            ),
            DTypeCastTransform(dtype_in=torch.int8, dtype_out=torch.float32,
                           in_keys=[("player_1", "observation", "observation"),
                            ("player_2", "observation", "observation")]),
        ),
        )

        return env
    
    check_env_specs(env)

    policy_modules = {}

    for group, agents in env.group_map.items():

        # share_parameters_policy = True
        # policy_net = MultiAgentMLP(
        #     n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],
        #     n_agent_outputs=env.full_action_spec[group, "action"].n,
        #     n_agents=len(agents),
        #     centralised=False,
        #     share_params=share_parameters_policy,
        #     device=device,
        #     depth=2,
        #     num_cells=256,
        #     activation_class=torch.nn.ReLU,
        # )
        policy_net = MLP(
            in_features = env.observation_spec[group, "observation", "observation"].shape[-1],
            out_features = env.full_action_spec[group, "action"].n,
#            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],
#            n_agent_outputs=env.full_action_spec[group, "action"].n,
#            n_agents=len(agents),
#            centralised=False,
            #share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.ReLU,
        )
        policy_module = TensorDictModule(
                policy_net,
                in_keys=[(group, "observation", "observation")],
                out_keys=[(group, "logits")],
            )
        policy_modules[group] = policy_module

    policies = {}
    for group, _agents in env.group_map.items():
        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "logits")],
            out_keys=[(group, "action")],
            distribution_class=Categorical,
            return_log_prob=True,
        )
        policy = policy.to(device)
        policies[group] = policy

    critics = {}

    for group, agents in env.group_map.items():

        critic_net = MLP(
            in_features = env.observation_spec[group, "observation", "observation"].shape[-1],
            out_features = 1,
#            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],
#            n_agent_outputs=1,
#            n_agents=len(agents),
#            centralised=MADDPG,
#            share_params=share_parameters_critic,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )
        # share_parameters_critic = True
        # MADDPG = True
        # critic_net = MultiAgentMLP(
        #     n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],
        #     n_agent_outputs=1,
        #     n_agents=len(agents),
        #     centralised=MADDPG,
        #     share_params=share_parameters_critic,
        #     device=device,
        #     depth=2,
        #     num_cells=256,
        #     activation_class=torch.nn.Tanh,
        # )
        critic_module = TensorDictModule(
            module=critic_net,
            in_keys=[(group, "observation", "observation")],
            out_keys=[(group, "state_value")],
        )

        critics[group] = critic_module
        
    reset_td = env.reset().to(device)
    reset_td.batch_size = torch.Size([1])
    #process_batch(env.group_map, reset_td)
    for group, _agents in env.group_map.items():
        td_step = policies[group](reset_td)
        td_res = critics[group](td_step)
    
    agents_exploration_policy = TensorDictSequential(*policies.values())
    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    frames_per_batch = cfg.collector.frames_per_batch
    total_frames = cfg.collector.total_frames
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs

    # collector = SyncDataCollector(
    #     create_env_fn=make_env(cfg),
    #     policy=agents_exploration_policy,
    #     device=device,
    #     frames_per_batch=frames_per_batch,
    #     total_frames=total_frames,
    #     storing_device=device,
    #     max_frames_per_traj=-1
    # )

    # replay_buffers = {}
    # for group, _agents in env.group_map.items():
    #     sampler = SamplerWithoutReplacement()
    #     replay_buffer = TensorDictReplayBuffer(
    #         storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
    #         sampler=sampler,
    #         batch_size=cfg.loss.mini_batch_size, 
    #     )
    #     replay_buffers[group] = replay_buffer

    losses = {}
    for group, _agents in env.group_map.items():
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
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.set_keys(
            value=(group, "state_value"),
            action=(group, "action"),
            value_target=(group, "value_target"),
            advantage=(group, "advantage"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        losses[group] = (adv_module, loss_module)
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

    reg_param = 0.2  # Initial regularization parameter
    for i in range(2):
        td0 = env.reset()
        policies['player_1'](td0)
        td1 = env.step(td0)
        policies['player_2'](td1)
        td2 = env.step(td1)
        data = td2

    #for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())
        batch = process_batch(train_group_map, data)

        #transformed_rewards = reward_transformation(batch.get(("next", group, "reward")), policies, reg_param)
        #batch.update({"reward": transformed_rewards})

        training_start = time.time()
        for group in train_group_map.keys():
            group_batch = batch.exclude(
                *[
                    key
                    for _group in train_group_map.keys()
                    if _group != group
                    for key in [_group, ("next", _group)]
                ]
            )
            for j in range(cfg_loss_ppo_epochs):
                with torch.no_grad():
                    group_batch = losses[group][0](group_batch)
                group_batch = group_batch.reshape(-1)
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
