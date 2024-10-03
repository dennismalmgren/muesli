import torch
import hydra
from torchrl._utils import logger as torchrl_logger
from torch import multiprocessing
import tqdm
from torchrl.envs.libs import PettingZooWrapper
from envs.matching_pennies_env import MatchingPenniesEnv
from envs.colonel_blotto_env import ColonelBlottoParallelEnv
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
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchrl.envs.utils import MarlGroupMapType


class CriticModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden_dim = 10

        self.embed = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        torch.nn.init.kaiming_normal_(self.embed.weight)
        torch.nn.init.zeros_(self.embed.bias)
        torch.nn.init.kaiming_normal_(self.hidden.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.kaiming_normal_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)       


    def forward(self, x):
        hidden = F.elu(self.embed(x))
        hidden = F.elu(self.hidden(hidden))
        output = self.output(hidden)
        return output
    
class PolicyModule(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = action_dim
        self.hidden_dim = 10

        self.embed = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        torch.nn.init.kaiming_normal_(self.embed.weight)
        torch.nn.init.zeros_(self.embed.bias)
        torch.nn.init.kaiming_normal_(self.hidden.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.kaiming_normal_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)     
        self.variance =  nn.Buffer(0.1 * torch.ones(self.output_dim), requires_grad=False)

    def forward(self, x):
        hidden = F.elu(self.embed(x))
        hidden = F.elu(self.hidden(hidden))
        output = self.output(hidden)
        return output, self.variance

def regularize_reward(reward, action_i_logprob, action_i_reg_logprob, action_adv_logprob, action_adv_reg_logprob):
    eta = 0.2
    return reward - eta * (action_i_logprob - action_i_reg_logprob) + \
                    eta * (action_adv_logprob - action_adv_reg_logprob)

def policy_logprob(policy_module, observation, action):
    mean, variance = policy_module(observation)
    dist = torch.distributions.MultivariateNormal(mean, torch.diag(variance))
    log_probs = dist.log_prob(action)
    return log_probs

def sample_policy(policy_module, observation):
    mean, variance = policy_module(observation)
    dist = torch.distributions.MultivariateNormal(mean, torch.diag(variance))
    sampled_action = dist.sample()
    action_logprob = dist.log_prob(sampled_action)
    return sampled_action, action_logprob, torch.exp(action_logprob)

def copy_weights(source_policy, target_policy):
    target_policy.load_state_dict(source_policy.state_dict())

@hydra.main(config_path=".", config_name="marl_ppo", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    torch.manual_seed(cfg.seed)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    base_env = ColonelBlottoParallelEnv(num_players=2, num_battlefields=3)
    base_env = PettingZooWrapper(base_env, group_map=MarlGroupMapType.ONE_GROUP_PER_AGENT, device=device)

    
    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"] * len(base_env.group_map.keys()),
        ),
    )

    check_env_specs(env)

    num_iters = 5
    train_iters = 3500
    lr_steps = num_iters * train_iters
    policy_lr = 0.005
    critic_lr = 0.005
    
    policy_modules = {}
    policy_reg_modules = {}
    policy_optimizers = {}
    for group, agents in env.group_map.items():
        policy_group = PolicyModule(1, 3)
        policy_modules[group] = policy_group
        policy_reg_group = PolicyModule(1, 3)
        copy_weights(policy_group, policy_reg_group)
        policy_reg_modules[group] = policy_reg_group
        policy_optimizers[group] = optim.SGD(policy_group.parameters(), lr=policy_lr)

    critics = {}
    critic_optimizers = {}
    for group, agents in env.group_map.items():
        critic_group = CriticModule(1)
        critics[group] = critic_group
        critic_optimizers[group] = optim.SGD(critic_group.parameters(), lr=critic_lr)

    
    groups = list(env.group_map.keys())
    print(f"Initial policy_modules: {policy_modules}")
    lr_step = 0
    for fix_iter in range(num_iters):
        for group in env.group_map.keys():
            copy_weights(policy_modules[group], policy_reg_modules[group])
            
        for train_iter in range(train_iters):
            episodes = []
            for episode_id in range(128):            
                td = env.reset()
                group0 = groups[0]
                group1 = groups[1]
                action0, action0_logprob, action0_probdist = sample_policy(policy_modules[group0], td[group0]['observation'])
                action1, action1_logprob, action1_probdist = sample_policy(policy_modules[group1], td[group1]['observation'])
                td[group0]["action"] = action0
                td[group1]["action"] = action1
                td[group0]["action_logprob"] = action0_logprob
                td[group1]["action_logprob"] = action1_logprob
                
                td = env.step(td)
                episodes.append(td)
            td = torch.stack(episodes)
            #perform an update
            observations0 = td[group0, 'observation']
            observations1 = td[group1, 'observation']
            value0 = critics[group0](observations0)
            value1 = critics[group1](observations1)
            #this does not include reward regularization
            reward0 = td['next'][group0, 'reward']
            reward1 = td['next'][group1, 'reward']
            action0 = td[group0, 'action']

            action1 = td[group1, 'action']
            action0_logprob = td[group0]["action_logprob"]
            action1_logprob = td[group1]["action_logprob"]
            
            with torch.no_grad():
                action0_reg_logprob = policy_logprob(policy_reg_modules[group0], observations0, action0)
                action1_reg_logprob = policy_logprob(policy_reg_modules[group1], observations1, action1)

                reward0 = regularize_reward(reward0, action0_logprob, action0_reg_logprob, action1_logprob, action1_reg_logprob)
                reward1 = regularize_reward(reward1, action1_logprob, action1_reg_logprob, action0_logprob, action0_reg_logprob)
                advantages0 = reward0 - value0
                advantages1 = reward1 - value1
            # for g in policy_optimizers[group0].param_groups:
            #     g['lr'] = policy_lr * (1.0 - lr_step / lr_steps)
            # for g in policy_optimizers[group1].param_groups:
            #     g['lr'] = policy_lr * (1.0 - lr_step / lr_steps)
            # for g in critic_optimizers[group0].param_groups:
            #     g['lr'] = critic_lr * (1.0 - lr_step / lr_steps)    
            # for g in critic_optimizers[group1].param_groups:
            #     g['lr'] = critic_lr * (1.0 - lr_step / lr_steps)      

            lr_step += 1
            policy0_loss = torch.mean(-torch.exp(action0_logprob) * advantages0) #how do we construct the policy loss?
            #policy0_loss = torch.mean(torch.exp(action0_logprob) * advantages0)
            policy_optimizers[group0].zero_grad()
            policy0_loss.backward()
            policy_optimizers[group0].step()

            policy1_loss = torch.mean(-torch.exp(action1_logprob) * advantages1)
            #policy1_loss = torch.mean(torch.exp(action1_logprob) * advantages1)
            policy_optimizers[group1].zero_grad()
            policy1_loss.backward()
            policy_optimizers[group1].step()

            #Train critics
            critic0_loss = F.mse_loss(value0, reward0)
            critic_optimizers[group0].zero_grad()
            critic0_loss.backward()
            critic_optimizers[group0].step()
            
            critic1_loss = F.mse_loss(value1, reward1)
            critic_optimizers[group1].zero_grad()
            critic1_loss.backward()
            critic_optimizers[group1].step()
          
            if train_iter % 10 == 0:
                print(f"Fix iter: {fix_iter}, train iter: {train_iter}, policy_0_loss: {policy0_loss.item()}, policy_0: {action0_probdist.detach()}, policy_1: {action1_probdist.detach()}")
                print(f"mean critic prediction: {value0.mean()} mean reward: {reward0.mean()}")
        print('Ok')
       
if __name__ == "__main__":
    main()
