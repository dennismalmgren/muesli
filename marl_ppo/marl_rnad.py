from typing import Tuple, Dict

import torch
import hydra
from torchrl._utils import logger as torchrl_logger
from torch import multiprocessing
import tqdm
from torchrl.envs import step_mdp
from torchrl.envs.libs import PettingZooWrapper
from envs.matching_pennies_env import MatchingPenniesEnv
from torchrl.envs import (
    check_env_specs,
    RewardSum,
    DTypeCastTransform,
    DoubleToFloat,
    FlattenObservation,
    RenameTransform,
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
from torchrl.objectives.common import _make_target_param
import time
from tensordict.nn import TensorDictModule, TensorDictSequential, TensorDictParams
from torch.distributions import Categorical
from torchrl.modules.distributions import MaskedCategorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from tensordict import TensorDictBase
from pettingzoo.classic import tictactoe_v3
from tensordict import TensorDict

class ParamsUpdater:
    def __init__(self, params, target_params, tau):
        self.params = params
        self.target_params = target_params
        self.tau = tau

    def update(self):
        for target_param, param in zip(self.target_params, self.params):
            target_param.data = self.tau * param.data + (1 - self.tau) * target_param.data

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

def calculate_estimators(trajectory: TensorDict) -> TensorDict:
    eta = 0.2
    trajectory_in = trajectory.clone()
    # with critic_params['player_1'].to_module(critics['player_1']):
    #     trajectory_in = critics['player_1'](trajectory_in)
    n_actions = trajectory_in['logits'].shape[-1]

    #returns v_1, v_2, Q_1, Q_2
    t_effective = len(trajectory)
    v_hat = torch.zeros((2, t_effective), dtype=torch.float32, device=trajectory.device)
    Q_hat = torch.zeros((2, t_effective, n_actions), dtype=torch.float32, device=trajectory.device)
   #chi_t_p1 = 1.0
    v_hat_t = torch.zeros((2,), dtype=torch.float32, device=trajectory.device)
    v_hat_t_p1 = torch.zeros((2,), dtype=torch.float32, device=trajectory.device)
    
    V_next_t_p1 = torch.zeros((2,), dtype=torch.float32, device=trajectory.device)
    r_hat_t = torch.zeros((2,), dtype=torch.float32, device=trajectory.device)
    r_hat_t_p1 = torch.zeros((2,), dtype=torch.float32, device=trajectory.device)
    V_next_t = torch.zeros((2,), dtype=torch.float32, device=trajectory.device)
    Q_hat_t = torch.zeros((2, n_actions), dtype=torch.float32, device=trajectory.device)

    #current version does not do v-trace.
    for t_ind in range(t_effective - 1, -1, -1):
        acting_player_ind = (trajectory_in['acting_player'][t_ind].long() - 1).item()
        not_acting_player_ind = 1 - acting_player_ind
        acting_player_id = str(acting_player_ind + 1)
        not_acting_player_id = str(not_acting_player_ind + 1)

        #for not acting player
        v_hat_t[not_acting_player_ind] = v_hat_t_p1[not_acting_player_ind]
        V_next_t[not_acting_player_ind] = V_next_t_p1[not_acting_player_ind]
        r_hat_t[not_acting_player_ind] = trajectory_in['reward_' + not_acting_player_id + '_reg'][t_ind] + \
                                                r_hat_t_p1[not_acting_player_ind]
        Q_hat_t[not_acting_player_ind] = torch.zeros(n_actions, dtype=torch.float32, device=trajectory.device)
        #Q_hat_t.scatter_(-1, ~trajectory_in['action_mask'], -float('inf'))

        #for acting player
        v_hat_t[acting_player_ind] = trajectory_in['reward_' + acting_player_id + '_reg'][t_ind] + \
                                    r_hat_t_p1[acting_player_ind] + v_hat_t_p1[acting_player_ind]
        V_next_t[acting_player_ind] = trajectory_in['state_value'][t_ind]
        r_hat_t[acting_player_ind] = 0
        Q_hat_t[acting_player_ind].unsqueeze(0)[trajectory_in['action_mask'][t_ind]] = \
                         -eta * (torch.gather(trajectory_in["logits"][t_ind], -1, trajectory_in['action'][t_ind].unsqueeze(-1)) - 
                                 torch.gather(trajectory_in["reg_m_logits"][t_ind], -1, trajectory_in['action'][t_ind].unsqueeze(-1))) \
                         + trajectory_in['state_value'][t_ind]
        Q_add = 1.0 / torch.gather(torch.softmax(trajectory_in['logits'][t_ind].detach(), -1), -1, trajectory_in['action'][t_ind].unsqueeze(0)).squeeze(-1) \
            * (trajectory_in['reward_' + acting_player_id][t_ind] + \
               eta * (torch.gather(trajectory_in["logits"][t_ind], -1, trajectory_in['action'][t_ind].unsqueeze(-1)) - 
                                 torch.gather(trajectory_in["reg_m_logits"][t_ind], -1, trajectory_in['action'][t_ind].unsqueeze(-1))) + \
                                 r_hat_t_p1[acting_player_ind] + v_hat_t_p1[acting_player_ind] - trajectory_in['state_value'][t_ind])
        Q_hat_t[acting_player_ind][trajectory_in['action'][t_ind]] += Q_add.squeeze(-1)
            
        #Now update results before next round
        v_hat[:, t_ind] = v_hat_t
        Q_hat[:, t_ind] = Q_hat_t
        v_hat_t_p1 = v_hat_t
        V_next_t_p1 = V_next_t
        r_hat_t_p1 = r_hat_t

    trajectory.update(
        {
            "v_hat_player_1": v_hat[0].unsqueeze(-1),
            "v_hat_player_2": v_hat[1].unsqueeze(-1),
            "Q_hat_player_1": Q_hat[0].unsqueeze(-1),
            "Q_hat_player_2": Q_hat[1].unsqueeze(-1),
        }
    )
    
    return trajectory

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
                            ("player_2", "observation", "observation")]),
            
        ),
    )
                     
    
    
    check_env_specs(env)

    eta = 0.2

    policy_modules = {}
    policy_net = None
    for group, agents in env.group_map.items():
        if policy_net is None:
            policy_net = MLP(
                in_features = env.observation_spec[group, "observation", "observation"].shape[-1],
                out_features = env.full_action_spec[group, "action"].n,
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
    policy_params = {}
    policy_target_params = {}
    policy_reg_params = {}
    policy_params_updater = {}
    for group, _agents in env.group_map.items():
        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys={
                "logits": (group, "logits"),
                "mask": (group, "action_mask")
            },
            out_keys=[(group, "action")],
            distribution_class=MaskedCategorical,
            return_log_prob=True,
        )
        policy = policy.to(device)
        policies[group] = policy
        policy_params[group] = TensorDict.from_module(policy, as_module=True)
        policy_target_params[group] = TensorDictParams(
                policy_params[group].apply(
                    _make_target_param(clone=True), filter_empty=False
                ),
                no_convert=True,
            )
        policy_reg_params[group] = {}
        policy_reg_params[group]["m"] = TensorDictParams(
                policy_params[group].apply(
                    _make_target_param(clone=True), filter_empty=False
                ),
                no_convert=True,
            )
        policy_reg_params[group]["m_m1"] = TensorDictParams(
            policy_params[group].apply(
                    _make_target_param(clone=True), filter_empty=False
                ),
                no_convert=True,
            )
        policy_params_updater[group] = ParamsUpdater(policy_params[group], policy_target_params[group], cfg.optim.tau)

    critics = {}
    critic_net = None
    critic_params = {}
    critic_target_params = {}
    critic_params_updater = {}
    for group, agents in env.group_map.items():
        if critic_net is None:
            critic_net = MLP(
                in_features = env.observation_spec[group, "observation", "observation"].shape[-1],
                out_features = 1,
                device=device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            )
        critic_module = TensorDictModule(
            module=critic_net,
            in_keys=[(group, "observation", "observation")],
            out_keys=[(group, "state_value")],
        )

        critics[group] = critic_module
        critic_params[group] = TensorDict.from_module(critic_module, as_module=True)
        critic_target_params[group] = TensorDictParams(
                critic_params[group].apply(
                    _make_target_param(clone=True), filter_empty=False
                ),
                no_convert=True,
            )
        critic_params_updater[group] = ParamsUpdater(critic_params[group], critic_target_params[group], cfg.optim.tau)
    

    reset_td = env.reset().to(device)
    reset_td.batch_size = torch.Size([1])
    #process_batch(env.group_map, reset_td)
    for group, _agents in env.group_map.items():
        td_step = policies[group](reset_td)
        td_res = critics[group](td_step)
    
    agents_exploration_policy = TensorDictSequential(*policies.values())
   
    optimizers = {
        group: {
            "loss_actor": torch.optim.Adam(
                policy_params[group].flatten_keys().values(), lr=cfg.optim.lr
            ),
            "loss_critic": torch.optim.Adam(
                critic_params[group].flatten_keys().values(), lr=cfg.optim.lr
            ),
        }
        for group in env.group_map.keys()
    }

    pbar = tqdm.tqdm(
        total=cfg.collector.total_frames,
        desc=", ".join(
            [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
        ),
    )
    episode_reward_mean_map = {group: [] for group in env.group_map.keys()}
    train_group_map = copy.deepcopy(env.group_map)
    reward_keys = [(group, "reward") for group in env.group_map.keys()]
    done_keys = [(group, "done") for group in env.group_map.keys()]
    action_keys = [(group, "action") for group in env.group_map.keys()]
    groups = list(env.group_map.keys())
    n = 0
    m = 0
    delta_m = 100
    for i in range(200):
        #collect episode
        episode_tds = []
        td = env.reset()
        done = False
        with policy_params[groups[0]].to_module(policies[groups[0]]), policy_params[groups[1]].to_module(policies[groups[1]]):
            while not done:
                td = td.to(device)
                td = agents_exploration_policy(td)
                td = td.to('cpu')
                td = env.step(td)
                episode_tds.append(td)
                td = step_mdp(td, reward_keys=reward_keys, done_keys=done_keys, action_keys=action_keys)
                done = torch.any(td['done'])
        
        episode = torch.stack(episode_tds)

        pbar.update(1) #count episodes instead
        batch = process_batch(train_group_map, episode)
        batch = batch.to(device)
        with critic_params[groups[0]].to_module(critics[groups[0]]), critic_params[groups[1]].to_module(critics[groups[1]]):
            for critic in critics.values():
                critic(batch)

        batch_clone = batch.clone()
        # with critic_params[groups[0]].to_module(critics[groups[0]]), critic_params[groups[1]].to_module(critics[groups[1]]):
        #     for critic in critics.values():
        #         critic(batch_clone)
        #construct trajectory
        with critic_target_params[groups[0]].to_module(critics[groups[0]]), critic_target_params[groups[1]].to_module(critics[groups[1]]):
            for critic in critics.values():
                critic(batch_clone)
        #observations
        loss_in_td = TensorDict({}, batch_size=batch_clone.batch_size, device=device)

        batch_obs = torch.zeros_like(batch['player_1', 'observation', 'observation'], device=device)
        batch_obs[batch['player_1', 'mask']] = batch['player_1', 'observation', 'observation'][batch['player_1', 'mask']]
        batch_obs[batch['player_2', 'mask']] = batch['player_2', 'observation', 'observation'][batch['player_2', 'mask']]

        
        #actions
        batch_act = torch.zeros_like(batch['player_1', 'action'], device=device)
        batch_act[batch['player_1', 'mask']] = batch['player_1', 'action'][batch['player_1', 'mask']]
        batch_act[batch['player_2', 'mask']] = batch['player_2', 'action'][batch['player_2', 'mask']]
        
        #acting player
        batch_acting_player = torch.zeros_like(batch['player_1', 'mask'], dtype=torch.float32, device=device)
        batch_acting_player[batch['player_1', 'mask']] = torch.tensor(1, dtype=torch.float32, device=device)
        batch_acting_player[batch['player_2', 'mask']] = torch.tensor(2, dtype=torch.float32, device=device)

        batch_action_mask = torch.zeros_like(batch['player_1', 'action_mask'])
        batch_action_mask[batch['player_1', 'mask']] = batch['player_1', 'action_mask'][batch['player_1', 'mask']]
        batch_action_mask[batch['player_2', 'mask']] = batch['player_2', 'action_mask'][batch['player_2', 'mask']]

        #logits
        batch_player_logits = torch.zeros_like(batch['player_1', 'logits'], device=device)
        batch_player_logits[batch['player_1', 'mask']] = batch['player_1', 'logits'][batch['player_1', 'mask']]
        batch_player_logits[batch['player_2', 'mask']] = batch['player_2', 'logits'][batch['player_2', 'mask']]
        
        #state_value
        batch_state_value = torch.zeros_like(batch['player_1', 'state_value'], device=device)
        batch_state_value[batch_clone['player_1', 'mask']] = batch_clone['player_1', 'state_value'][batch_clone['player_1', 'mask']]
        batch_state_value[batch_clone['player_2', 'mask']] = batch['player_2', 'state_value'][batch_clone['player_2', 'mask']]

        loss_in_td['observation'] = batch_obs
        loss_in_td['action'] = batch_act
        loss_in_td['acting_player'] = batch_acting_player
        loss_in_td['reward_1'] = batch['next', 'player_1', 'reward'].clone()
        loss_in_td['reward_2'] = batch['next', 'player_2', 'reward'].clone()
        loss_in_td['logits'] = batch_player_logits
        loss_in_td['action_mask'] = batch_action_mask
        loss_in_td['logits'][~loss_in_td['action_mask']] = -float('inf')
        loss_in_td['state_value'] = batch_state_value
        
        reg_m_logits_batch = batch_clone.clone()
        reg_m_m1_logits_batch = batch_clone.clone()
        with policy_reg_params[groups[0]]["m"].to_module(policies[groups[0]]), policy_reg_params[groups[1]]["m"].to_module(policies[groups[1]]):
            agents_exploration_policy(reg_m_logits_batch)
        with policy_reg_params[groups[0]]["m_m1"].to_module(policies[groups[0]]), policy_reg_params[groups[1]]["m_m1"].to_module(policies[groups[1]]):
            agents_exploration_policy(reg_m_m1_logits_batch)

        loss_in_td['reg_m_logits'] = torch.zeros_like(batch['player_1', 'logits'], device=device)
        loss_in_td['reg_m_logits'][batch['player_1', 'mask']] = reg_m_logits_batch['player_1', 'logits'][batch['player_1', 'mask']]
        loss_in_td['reg_m_logits'][batch['player_2', 'mask']] = reg_m_logits_batch['player_2', 'logits'][batch['player_2', 'mask']]

        loss_in_td['reg_m_m1_logits'] = torch.zeros_like(batch['player_1', 'logits'], device=device)
        loss_in_td['reg_m_m1_logits'][batch['player_1', 'mask']] = reg_m_m1_logits_batch['player_1', 'logits'][batch['player_1', 'mask']]
        loss_in_td['reg_m_m1_logits'][batch['player_2', 'mask']] = reg_m_m1_logits_batch['player_2', 'logits'][batch['player_2', 'mask']]
        
        #now it's time for..dumdumdum...reward transformation
        alpha = min(1, 2 * n / delta_m)
        with torch.no_grad():
            policy_log_prob = torch.gather(loss_in_td['logits'], -1, loss_in_td['action'].long().unsqueeze(-1))
            policy_reg_m_log_prob = torch.gather(loss_in_td['reg_m_logits'], -1, loss_in_td['action'].long().unsqueeze(-1))
            policy_reg_m_m1_log_prob = torch.gather(loss_in_td['reg_m_m1_logits'], -1, loss_in_td['action'].long().unsqueeze(-1))
            r_1_m_reg = loss_in_td['reward_1'] + (1 - 2 * loss_in_td['acting_player'] == 1).unsqueeze(-1) * eta * (policy_log_prob - policy_reg_m_log_prob)
            r_2_m_reg = loss_in_td['reward_2'] + (1 - 2 * loss_in_td['acting_player'] == 2).unsqueeze(-1) * eta * (policy_log_prob - policy_reg_m_log_prob)
            r_1_m_m1_reg = loss_in_td['reward_1'] + (1 - 2 * loss_in_td['acting_player'] == 1).unsqueeze(-1) * eta * (policy_log_prob - policy_reg_m_m1_log_prob)
            r_2_m_m1_reg = loss_in_td['reward_2'] + (1 - 2 * loss_in_td['acting_player'] == 2).unsqueeze(-1) * eta * (policy_log_prob - policy_reg_m_m1_log_prob)

        r_1_reg = alpha * r_1_m_reg + (1 - alpha) * r_1_m_m1_reg
        r_2_reg = alpha * r_2_m_reg + (1 - alpha) * r_2_m_m1_reg
        loss_in_td["reward_1_reg"] = r_1_reg
        loss_in_td["reward_2_reg"] = r_2_reg

        loss_in_td = calculate_estimators(loss_in_td)
        #now calculate the losses.

        l_critic = torch.cat(
                        (
                            ((loss_in_td['state_value'].squeeze(-1) - loss_in_td['v_hat_player_1'])**2)[loss_in_td['acting_player'] == 1], 
                            ((loss_in_td['state_value'].squeeze(-1) - loss_in_td['v_hat_player_2'])**2)[loss_in_td['acting_player'] == 2]
                        ), 
                        dim=0)
        
        l_critic = l_critic.mean()
        print(l_critic.item())
        optimizers["player_1"]["loss_critic"].zero_grad()
        l_critic.backward()
        #"grad_norm = torch.nn.utils.clip_grad_norm_(critic_params["player_1"].flatten_values(), 1e6)
        optimizers["player_1"]["loss_critic"].step()


        n += 1
        if n == delta_m + 1:
            n = 0
            m += 1
            #here you can check m to determine the new delta_m.
            delta_m = 100

            #TODO: update regularization policies

        #    policy_params_updater.update()
        critic_params_updater["player_1"].update()
        critic_params_updater["player_2"].update()
        
        #calculate estimators
        
        for group in train_group_map.keys(): 
            episode_reward_mean = (
                batch.get(("next", group, "episode_reward"))[
                    batch.get(("next", group, "done"))
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
if __name__ == "__main__":
    main()
