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
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from tensordict import TensorDictBase

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

def regularize_reward(reward, prob_i, prob_i_reg, prob_adv, prob_adv_reg):
    eta = 0.2
    return reward - eta * (torch.log(prob_i) - torch.log(prob_i_reg)) + \
                    eta * (torch.log(prob_adv) - torch.log(prob_adv_reg))

def sample_policy(policy_module, observation):
    probs = torch.softmax(policy_module, dim=-1)
    action = torch.multinomial(probs, num_samples=1)
    return action

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
    
    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"] * len(base_env.group_map.keys()),
        ),
    )

    check_env_specs(env)

    policy_modules = {}
    policy_reg_modules = {}
    for group, agents in env.group_map.items():
        policy_group = torch.tensor([0.999, 0.001])
        policy_modules[group] = policy_group
        policy_reg_group = torch.tensor([0.999, 0.001])
        policy_reg_modules[group] = policy_reg_group

    critics = {}
    
    for group, agents in env.group_map.items():
        critic_group = torch.tensor([0.0])
        critics[group] = critic_group

    num_iters = 5
    train_iters = 1000
    lr_critic = 0.001
    lr_policy = 0.001
    groups = list(env.group_map.keys())
    print(f"Initial policy_modules: {policy_modules}")
    for fix_iter in range(num_iters):
        policy_reg_modules = {
            group: policy_modules[group].clone()
            for group in env.group_map.keys()}
        for train_iter in range(train_iters):
            episodes = []
            for episode_id in range(10):            
                td = env.reset()
                group0 = groups[0]
                group1 = groups[1]
                action0 = sample_policy(policy_modules[group0], td[group0]['observation'])
                action1= sample_policy(policy_modules[group1], td[group1]['observation'])
                td[group0]["action"] = action0.reshape((1, -1))
                td[group1]["action"] = action1.reshape((1, -1))
                td = env.step(td)
                episodes.append(td)
            td = torch.stack(episodes)
            #perform an update
            value0 = critics[group0]
            value1 = critics[group1]
            #this does not include reward regularization
            reward0 = td['next'][group0, 'reward']
            reward1 = td['next'][group1, 'reward']
            action0 = td[group0, 'action']
            action1 = td[group1, 'action']
            
            action0_prob = torch.gather(torch.softmax(policy_modules[group0], dim=-1), 0, action0.squeeze())
            action1_prob = torch.gather(torch.softmax(policy_modules[group1], dim=-1), 0, action1.squeeze())
            action_prob0_reg = torch.gather(torch.softmax(policy_reg_modules[group0], dim=-1), 0, action0.squeeze())
            action_prob1_reg = torch.gather(torch.softmax(policy_reg_modules[group1], dim=-1), 0, action1.squeeze())

            reward0 = regularize_reward(reward0.squeeze(), action0_prob, action_prob0_reg, action1_prob, action_prob1_reg)
            reward1 = regularize_reward(reward1.squeeze(), action1_prob, action_prob1_reg, action0_prob, action_prob0_reg)
            advantages0 = reward0 - value0
            advantages1 = reward1 - value1
            sum_advantages0 = torch.zeros_like(policy_modules[group0])
            sum_advantages1 = torch.zeros_like(policy_modules[group1])
            count_actions0 = torch.zeros_like(policy_modules[group0])
            count_actions1 = torch.zeros_like(policy_modules[group1])
            sum_advantages0.index_add_(0, action0.squeeze(), advantages0)
            sum_advantages1.index_add_(0, action1.squeeze(), advantages1)
            count_actions0.index_add_(0, action0.squeeze(), torch.ones_like(action0.squeeze(), dtype=torch.float32))
            count_actions0[count_actions0 == 0] = 1
            count_actions1.index_add_(0, action1.squeeze(), torch.ones_like(action1.squeeze(), dtype=torch.float32))
            count_actions1[count_actions1 == 0] = 1
            
            mean_advantages0 = sum_advantages0 / (count_actions0)
            mean_advantages1 = sum_advantages1 / (count_actions1)
            
            #policy_gradient0 = action_logit0 * advantage0
            #policy_gradient1 = action_logit1 * advantage1                
            policy_gradient0 = mean_advantages0
            policy_gradient1 = mean_advantages1

            policy_modules[group0] += lr_policy * policy_gradient0
            #policy_modules[group0][1 - action0.item()] -= lr_policy * policy_gradient0.mean()
            
            policy_modules[group1] += lr_policy * policy_gradient1
            #policy_modules[group1][1 - action1.item()] -= lr_policy * policy_gradient1.mean()
            critics[group0] -= lr_critic * ((value0 - reward0)).mean()
            critics[group1] -= lr_critic * ((value1 - reward1)).mean()
            if train_iter % 100 == 0:
                print(f"Fix iter: {fix_iter}, train iter: {train_iter}, policy_modules: {policy_modules}")

        print('Ok')
       
if __name__ == "__main__":
    main()
