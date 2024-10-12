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
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.observation_dim = observation_dim 
        self.output_dim = 1
        self.input_dim = observation_dim + action_dim
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

    def forward(self, x, a):
        input = torch.cat((x, a), dim=-1)
        hidden = F.elu(self.embed(input))
        hidden = F.elu(self.hidden(hidden))
        output = self.output(hidden)
        return output

def regularize_reward(reward, action_i_logprob, action_i_reg_logprob, action_adv_logprob, action_adv_reg_logprob):
    eta = 0.2
    return reward - eta * (action_i_logprob - action_i_reg_logprob) + \
                    eta * (action_adv_logprob - action_adv_reg_logprob)

def regularize_reward_energy(reward, action_i_energy, action_i_reg_energy, action_adv_energy, action_adv_reg_energy):
    eta = 0.2
    return reward - eta * (-action_i_energy + action_i_reg_energy) + \
                    eta * (-action_adv_energy + action_adv_reg_energy)

# def policy_logprob(policy_module, observation, action):
#     mean, variance = policy_module(observation)
#     dist = torch.distributions.MultivariateNormal(mean, torch.diag(variance))
#     log_probs = dist.log_prob(action)
#     return log_probs

def sample_policy(policy_module, observation, a_init = None, action_dim=3, steps=20, step_size=0.01):
    if a_init is None:
        a_init = torch.randn((*observation.shape[:-1], action_dim))  # Assuming zero mean and unit variance
        # Initialize action from base distribution
    a = a_init.clone()
    a.requires_grad = True
    for _ in range(steps):
        energy = policy_module(observation, a)
        energy.backward()
        noise = torch.randn_like(a) * torch.sqrt(torch.tensor(step_size))
        with torch.no_grad():
            a -= (step_size / 2) * a.grad
            a += noise
            a.grad.zero_()
    a_logits = a.detach()
    a_sampled = torch.softmax(a.detach(), dim=-1)
    return a_sampled, a_logits

def copy_weights(source_policy, target_policy):
    target_policy.load_state_dict(source_policy.state_dict())

def gather_samples(base_env, policy_0, policy_1, n_samples):
    td_policy = base_env.reset()
    sample_actions_policy_0 = torch.zeros((n_samples, 3))
    sample_actions_policy_1 = torch.zeros((n_samples, 3))
    for i in range(n_samples):
        policy_0_action, _ = sample_policy(policy_0, td_policy['player_0', 'observation'])
        policy_1_action, _ = sample_policy(policy_1, td_policy['player_1', 'observation'])
        sample_actions_policy_0[i, :] = policy_0_action
        sample_actions_policy_1[i, :] = policy_1_action
    return sample_actions_policy_0, sample_actions_policy_1

def plot_samples(action_samples_policy_0, action_samples_policy_1, iteration):
    #tbd
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

    def to_barycentric(point):
        # The point should sum to 1, or be normalized
        return point @ vertices

    points_2d_policy_0 = np.array([to_barycentric(p) for p in action_samples_policy_0])
    points_2d_policy_1 = np.array([to_barycentric(p) for p in action_samples_policy_1])
    
    # Plot the simplex (triangle)
    fig, ax = plt.subplots(1, 2, figsize=(6, 6))

    ax[0].plot([vertices[0, 0], vertices[1, 0]], [vertices[0, 1], vertices[1, 1]], 'k-', lw=2)
    ax[0].plot([vertices[1, 0], vertices[2, 0]], [vertices[1, 1], vertices[2, 1]], 'k-', lw=2)
    ax[0].plot([vertices[2, 0], vertices[0, 0]], [vertices[2, 1], vertices[0, 1]], 'k-', lw=2)
    
    ax[1].plot([vertices[0, 0], vertices[1, 0]], [vertices[0, 1], vertices[1, 1]], 'k-', lw=2)
    ax[1].plot([vertices[1, 0], vertices[2, 0]], [vertices[1, 1], vertices[2, 1]], 'k-', lw=2)
    ax[1].plot([vertices[2, 0], vertices[0, 0]], [vertices[2, 1], vertices[0, 1]], 'k-', lw=2)
    
   

    # Annotate the vertices to show actions corresponding to each corner
    ax[0].text(vertices[0, 0], vertices[0, 1] - 0.05, 'Action 1', fontsize=12, ha='center')
    ax[0].text(vertices[1, 0], vertices[1, 1] - 0.05, 'Action 2', fontsize=12, ha='center')
    ax[0].text(vertices[2, 0], vertices[2, 1] + 0.05, 'Action 3', fontsize=12, ha='center')

    ax[1].text(vertices[0, 0], vertices[0, 1] - 0.05, 'Action 1', fontsize=12, ha='center')
    ax[1].text(vertices[1, 0], vertices[1, 1] - 0.05, 'Action 2', fontsize=12, ha='center')
    ax[1].text(vertices[2, 0], vertices[2, 1] + 0.05, 'Action 3', fontsize=12, ha='center')

    ax[0].set_xlim(-0.1, 1.1)
    ax[0].set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
    ax[0].set_aspect('equal', adjustable='box')

    ax[1].set_xlim(-0.1, 1.1)
    ax[1].set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
    ax[1].set_aspect('equal', adjustable='box')

    # Scatter plot the points inside the triangle
    ax[0].scatter(points_2d_policy_0[:, 0], points_2d_policy_0[:, 1], color='blue', alpha=0.6)
    ax[1].scatter(points_2d_policy_1[:, 0], points_2d_policy_1[:, 1], color='blue', alpha=0.6)
    plt.savefig(f'policy_distribution_iter_{iteration}.png')
    plt.close()

@hydra.main(config_path=".", config_name="marl_ppo", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    torch.manual_seed(cfg.seed)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    device = torch.device("cpu")
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
    train_iters = 3000
    lr_steps = num_iters * train_iters
    policy_lr = 0.005
    critic_lr = 0.005
    
    policy_modules = {}
    policy_reg_modules = {}
    policy_optimizers = {}
    for group, agents in env.group_map.items():
        policy_group = PolicyModule(3, 3)
        policy_modules[group] = policy_group
        policy_reg_group = PolicyModule(3, 3)
        copy_weights(policy_group, policy_reg_group)
        policy_reg_modules[group] = policy_reg_group
        policy_optimizers[group] = optim.SGD(policy_group.parameters(), lr=policy_lr)

    critics = {}
    critic_optimizers = {}
    for group, agents in env.group_map.items():
        critic_group = CriticModule(3)
        critics[group] = critic_group
        critic_optimizers[group] = optim.SGD(critic_group.parameters(), lr=critic_lr)

    groups = list(env.group_map.keys())
    group0 = groups[0]
    group1 = groups[1]

    action_samples_policy_0, action_samples_policy_1 = gather_samples(base_env, policy_modules[group0], policy_modules[group1], 1000)
    action_samples_policy_0 = action_samples_policy_0.detach().numpy()
    action_samples_policy_1 = action_samples_policy_1.detach().numpy()
    plot_samples(action_samples_policy_0, action_samples_policy_1, 0)

    lr_step = 0
    for fix_iter in range(num_iters):
        for group in env.group_map.keys():
            copy_weights(policy_modules[group], policy_reg_modules[group])
            
        for train_iter in range(train_iters):
            episodes = []
            for episode_id in range(64):            
                td = env.reset()

                action0_sampled, action0_logits = sample_policy(policy_modules[group0], td[group0]['observation'])
                action1_sampled, action1_logits = sample_policy(policy_modules[group1], td[group1]['observation'])
                td[group0]["action"] = action0_sampled
                td[group1]["action"] = action1_sampled
                td[group0]["action_logits"] = action0_logits
                td[group1]["action_logits"] = action1_logits
                
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
            # action0_logprob = td[group0]["action_logprob"]
            # action1_logprob = td[group1]["action_logprob"]
            energy0 = policy_modules[group0](observations0, action0)
            energy1 = policy_modules[group1](observations1, action1)                 
            with torch.no_grad():
   

                #reg_energy0 = policy_reg_modules[group0](observations0, action0)
                #reg_energy1 = policy_reg_modules[group1](observations1, action1)


                #reward0 = regularize_reward_energy(reward0, energy0, reg_energy0, energy1, reg_energy1)
                #reward1 = regularize_reward_energy(reward1, energy1, reg_energy1, energy0, reg_energy0)
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
            policy0_loss = torch.mean(-energy0 * advantages0) #how do we construct the policy loss?
            #policy0_loss = torch.mean(torch.exp(action0_logprob) * advantages0)
            policy_optimizers[group0].zero_grad()
            policy0_loss.backward()
            policy_optimizers[group0].step()

            policy1_loss = torch.mean(-energy1 * advantages1)
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
                print(f"Fix iter: {fix_iter}, train iter: {train_iter}, policy_0_loss: {policy0_loss.item()}")
                print(f"mean critic prediction: {value0.mean()} mean reward: {reward0.mean()}")
            if train_iter % 100 == 0:
                action_samples_policy_0, action_samples_policy_1 = gather_samples(base_env, policy_modules[group0], policy_modules[group1], 1000)
                action_samples_policy_0 = action_samples_policy_0.detach().numpy()
                action_samples_policy_1 = action_samples_policy_1.detach().numpy()
                plot_samples(action_samples_policy_0, action_samples_policy_1, fix_iter * train_iters + train_iter + 1)
        print('Ok')
       
if __name__ == "__main__":
    main()
