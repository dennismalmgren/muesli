import torch
import hydra
from torchrl._utils import logger as torchrl_logger
from torch import multiprocessing
import tqdm
from torchrl.envs.libs import PettingZooWrapper
from envs.matching_pennies_env import MatchingPenniesEnv
from envs.colonel_blotto_env_torchrl import ColonelBlottoParallelEnv
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
from tensordict import TensorDict
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
import wandb

class QCriticModule(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.input_dim = observation_dim + action_dim
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


    def forward(self, observation, action):
        input = torch.cat([observation, action], dim=-1)
        hidden = F.elu(self.embed(input))
        hidden = F.elu(self.hidden(hidden))
        output = self.output(hidden)
        return output
    

class CriticModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden_dim = 64

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
    def __init__(self, observation_dim, action_dim, min_energy, max_energy):
        super().__init__()
        self.observation_dim = observation_dim 
        self.output_dim = 1
        self.input_dim = observation_dim + action_dim
        self.hidden_dim = 64
        self.min_energy = min_energy
        self.max_energy = max_energy
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
        output = torch.clamp(output, self.min_energy, self.max_energy)
        return output

def regularize_reward(reward, action_i_logprob, action_i_reg_logprob, action_adv_logprob, action_adv_reg_logprob):
    eta = 0.2
    return reward - eta * (action_i_logprob - action_i_reg_logprob) + \
                    eta * (action_adv_logprob - action_adv_reg_logprob)

def regularize_reward_energy(reward, action_i_energy, action_i_reg_energy, action_adv_energy, action_adv_reg_energy, 
                             energy_i_z, reg_energy_i_z, energy_adv_z, reg_energy_adv_z):
    eta = 0.2
    own_policy_reward = action_i_reg_energy - action_i_energy + energy_i_z - reg_energy_i_z
    adv_policy_reward = action_adv_reg_energy - action_adv_energy + energy_adv_z - reg_energy_adv_z
    return reward - eta * own_policy_reward + \
                    eta * adv_policy_reward

# def project_onto_simplex(a):
#     """
#     Projects the input 'a' onto the probability simplex (i.e., makes sure the elements sum to 1 and are >= 0).
#     """
#     sorted_a, _ = torch.sort(a, descending=True)
#     cumsum = torch.cumsum(sorted_a, dim=-1) - 1
#     rho = torch.arange(1, len(a) + 1, device=a.device).float()
#     delta_dims = a.dim() - rho.dim()
#     for _ in range(delta_dims):
#         rho = rho.unsqueeze(-1) #note, this should match all dimensions of a.
#     theta = (cumsum / rho).max(dim=-1, keepdim=True)[0]
#     epsilon = 1e-10
#     projected = torch.clamp(a - theta, min=epsilon)
#     projected_sum = projected.sum(-1, keepdim=True)
#     #projected_sum[projected_sum < epsilon] = epsilon

#    return projected / projected_sum  # Ensure exact sum-to-1 constraint
def sample_policy(policy_module, observation, x_init=None, action_dim=3, steps=10, step_size=0.01):
    if x_init is None:
        x_init = torch.randn((*observation.shape[:-1], action_dim), device=observation.device)
    noise_factor = torch.sqrt(2 * torch.tensor(step_size)).to(observation.device)
    x = x_init.clone()
    for _ in range(steps):
        x.requires_grad = True
        a = torch.softmax(x, dim=-1)
        energy = policy_module(observation, a)
        # Adjust energy with log-det Jacobian of softmax
        log_jacobian = -torch.sum(torch.log(a + 1e-10), dim=-1)
        adjusted_energy = energy - log_jacobian
        adjusted_energy = adjusted_energy.sum()
        adjusted_energy.backward()
        noise = torch.randn_like(x) * noise_factor
        with torch.no_grad():
            x -= step_size * x.grad
            x += noise
    a = torch.softmax(x, dim=-1).detach()
    a_energy = policy_module(observation, a)
    return a, a_energy

# def sample_policy(policy_module, observation, a_init = None, action_dim=3, steps=10, step_size=0.01):
#     if a_init is None:
#         a_init = torch.distributions.Dirichlet(torch.ones(3)).sample(observation.shape[:-1]).to(observation.device)
# #        a_init = torch.randn((*observation.shape[:-1], action_dim))  # Assuming zero mean and unit variance
#         # Initialize action from base distribution
#     noise_factor = torch.sqrt(2 * torch.tensor(step_size)).to(observation.device)
#     a = a_init.clone()
#     for _ in range(steps):
#         a.requires_grad = True
#         energy = policy_module(observation, a)
# #        grad_a = torch.autograd.grad(outputs=energy.sum(), inputs=a, create_graph=True)[0]

#         energy = energy.sum()
#         energy.backward()
#         noise = torch.randn_like(a) * noise_factor
#         with torch.no_grad():
#             a -= step_size * a.grad
#             a += noise
#             a = project_onto_simplex(a)

#             #a.grad.zero_()

#     a = a.detach()
#     #a_unconstrained_energy = policy_module(observation, a_unconstrained)
    
#     a_energy = policy_module(observation, a)   
#     return a, a_energy

def copy_weights(source_policy, target_policy):
    target_policy.load_state_dict(source_policy.state_dict())
    
def gather_qval_samples(base_env, 
                   qval_module_0, qval_module_1, 
                   action_dim=1,
                   n_samples=1000):
    reset_td = TensorDict({
        "player0": TensorDict({},
                              batch_size=(n_samples,)),
        "player1": TensorDict({},
                              batch_size=(n_samples,))

    },
                          batch_size=(n_samples,),
                          device=base_env.device)
    td_qval = base_env.reset(reset_td)
    
    sample_action = torch.distributions.Dirichlet(torch.ones(action_dim)).sample((n_samples, 1,)).to(reset_td.device)
    qval_0 = qval_module_0(td_qval["player0", "observation"], sample_action)
    qval_1 = qval_module_1(td_qval["player1", "observation"], sample_action)

    return qval_0, qval_1, sample_action

def gather_samples(base_env, 
                   policy_0, policy_1, 
                   n_samples, n_sample_steps):
    reset_td = TensorDict({
        "player0": TensorDict({},
                              batch_size=(n_samples,)),
        "player1": TensorDict({},
                              batch_size=(n_samples,))

    },
                          batch_size=(n_samples,),
                          device=base_env.device)
    td_policy = base_env.reset(reset_td)

    policy_0_action, policy_0_energy = sample_policy(policy_0, td_policy["player0", "observation"], steps=n_sample_steps)
    policy_1_action, policy_1_energy = sample_policy(policy_1, td_policy["player1", "observation"], steps=n_sample_steps)
    return policy_0_action, policy_1_action, policy_0_energy, policy_1_energy


def plot_samples(action_samples_policy_0, action_samples_policy_1, 
                 policy_0_energy, policy_1_energy, 
                 qval_0, qval_1, sample_action,
                 iteration, min_energy, max_energy):
    #tbd
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

    def to_barycentric(point):
        # The point should sum to 1, or be normalized
        return point @ vertices

#    action_samples_policy_0[action_samples_policy_0.sum(-1) == 0, :] = 1.0
#    action_samples_policy_1[action_samples_policy_1.sum(-1) == 0, :] = 1.0
#    action_samples_policy_0 = action_samples_policy_0 / action_samples_policy_0.sum(-1, keepdims=True)
#    action_samples_policy_1 = action_samples_policy_1 / action_samples_policy_1.sum(-1, keepdims=True)
    points_2d_qval = np.array([to_barycentric(p) for p in sample_action]).squeeze()

    points_2d_policy_0 = np.array([to_barycentric(p) for p in action_samples_policy_0]).squeeze()
    points_2d_policy_1 = np.array([to_barycentric(p) for p in action_samples_policy_1]).squeeze()
    norm_policy_0_energy = (policy_0_energy.squeeze() - min_energy) / (max_energy - min_energy)
    norm_policy_1_energy = (policy_1_energy.squeeze() - min_energy) / (max_energy - min_energy)
    qval_0 = qval_0.squeeze()
    qval_1 = qval_1.squeeze()
    # Plot the simplex (triangle)
    fig, ax = plt.subplots(3, 2, figsize=(6, 6))
    for i in range(3):
        for j in range(2):
            ax[i, j].plot([vertices[0, 0], vertices[1, 0]], [vertices[0, 1], vertices[1, 1]], 'k-', lw=2)
            ax[i, j].plot([vertices[1, 0], vertices[2, 0]], [vertices[1, 1], vertices[2, 1]], 'k-', lw=2)
            ax[i, j].plot([vertices[2, 0], vertices[0, 0]], [vertices[2, 1], vertices[0, 1]], 'k-', lw=2)    

            # Annotate the vertices to show actions corresponding to each corner
            ax[i, j].text(vertices[0, 0], vertices[0, 1] - 0.05, 'Action 1', fontsize=12, ha='center')
            ax[i, j].text(vertices[1, 0], vertices[1, 1] - 0.05, 'Action 2', fontsize=12, ha='center')
            ax[i, j].text(vertices[2, 0], vertices[2, 1] + 0.05, 'Action 3', fontsize=12, ha='center')

            ax[i, j].set_xlim(-0.1, 1.1)
            ax[i, j].set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
            ax[i, j].set_aspect('equal', adjustable='box')

    # Scatter plot the points inside the triangle
    ax[0, 0].scatter(points_2d_policy_0[:, 0], points_2d_policy_0[:, 1], c=norm_policy_0_energy, cmap='coolwarm', alpha=0.8)
    ax[0, 1].scatter(points_2d_policy_1[:, 0], points_2d_policy_1[:, 1], c=norm_policy_1_energy, cmap='coolwarm', alpha=0.8)
    ax[1, 0].scatter(points_2d_qval[:, 0], points_2d_qval[:, 1], c=qval_0, cmap='coolwarm', alpha=0.8, vmin=-1.5, vmax=1.5)
    ax[1, 1].scatter(points_2d_qval[:, 0], points_2d_qval[:, 1], c=qval_1, cmap='coolwarm', alpha=0.8, vmin=-1.5, vmax=1.5)
    ax[2, 0].scatter(points_2d_qval[:, 0], points_2d_qval[:, 1], c=qval_0, cmap='coolwarm', alpha=0.8)
    ax[2, 1].scatter(points_2d_qval[:, 0], points_2d_qval[:, 1], c=qval_1, cmap='coolwarm', alpha=0.8)
    # ax[0, 0].legend()
    # ax[0, 1].legend()
    # ax[1, 0].legend()
    # ax[1, 1].legend()
    
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
    #device = torch.device("cpu")
    action_dim = 3
    num_envs_rollout = 10000
    min_energy = -100.0
    max_energy = 100.0
    num_iters = 30
    train_iters = 10000
    lr_steps = num_iters * train_iters
    policy_lr = 3e-4
    critic_lr = 3e-3
    policy_sample_steps = 100
    n_samples_per_observation = 10

    ref_env = ColonelBlottoParallelEnv(num_players=2, num_battlefields=3, device=device)
    env = ref_env

    #check_env_specs(env)

    
    policy_modules = {}
    policy_reg_modules = {}
    policy_optimizers = {}
    for group, agents in ref_env.group_map.items():
        policy_group = PolicyModule(1, action_dim, min_energy, max_energy).to(device)
        policy_modules[group] = policy_group
        policy_reg_group = PolicyModule(1, action_dim, min_energy, max_energy).to(device)
        copy_weights(policy_group, policy_reg_group)
        policy_reg_modules[group] = policy_reg_group
        policy_optimizers[group] = optim.Adam(policy_group.parameters(), weight_decay=0.0, lr=policy_lr)

    critics = {}
    critic_optimizers = {}
    for group, agents in ref_env.group_map.items():
        critic_group = CriticModule(1).to(device)
        critics[group] = critic_group
        critic_optimizers[group] = optim.Adam(critic_group.parameters(), weight_decay=0.0, lr=critic_lr)

    qvals = {}
    qval_optimizers = {}
    for group, agents in ref_env.group_map.items(): 
        critic_group = QCriticModule(1, action_dim).to(device)
        qvals[group] = critic_group
        qval_optimizers[group] = optim.Adam(critic_group.parameters(), weight_decay=0.0, lr=critic_lr)
     
    groups = list(ref_env.group_map.keys())
    group0 = groups[0]
    group1 = groups[1]

    action_samples_policy_0, action_samples_policy_1, policy_0_energy, policy_1_energy = \
        gather_samples(ref_env, policy_modules[group0], policy_modules[group1], 
                       1000, n_sample_steps=policy_sample_steps)
    qval_0, qval_1, sample_action = gather_qval_samples(ref_env, qvals[group0], qvals[group1], action_dim, 1000)
    qval_0 = qval_0.detach().cpu().numpy()
    qval_1 = qval_1.detach().cpu().numpy()
    sample_action = sample_action.detach().cpu().numpy()

    action_samples_policy_0 = action_samples_policy_0.detach().cpu().numpy()
    action_samples_policy_1 = action_samples_policy_1.detach().cpu().numpy()
    policy_0_energy = policy_0_energy.detach().cpu().numpy()
    policy_1_energy = policy_1_energy.detach().cpu().numpy()
    plot_samples(action_samples_policy_0, action_samples_policy_1, policy_0_energy, policy_1_energy, \
                 qval_0, qval_1, sample_action,
                 0, min_energy, max_energy)

    lr_step = 0
    for fix_iter in range(num_iters):
        for group in groups:
            copy_weights(policy_modules[group], policy_reg_modules[group])
            
        for train_iter in range(train_iters):
            reset_td = TensorDict({
                "player0": TensorDict({},
                              batch_size=(num_envs_rollout,)),
                "player1": TensorDict({},
                              batch_size=(num_envs_rollout,))

                },
                batch_size=(num_envs_rollout,),
                device=env.device)
            td = env.reset(reset_td)
            action0_sampled, action0_energy = sample_policy(policy_modules[group0], td[group0]['observation'], steps=policy_sample_steps)
            action1_sampled, action1_energy = sample_policy(policy_modules[group1], td[group1]['observation'], steps=policy_sample_steps)
            action0_z_sampled, action0_z_energy = sample_policy(policy_modules[group0], td[group0]['observation'], steps=policy_sample_steps)
            action1_z_sampled, action1_z_energy = sample_policy(policy_modules[group1], td[group1]['observation'], steps=policy_sample_steps)
            td[group0]["action"] = action0_sampled
            #td[group0]["action_unconstrained"] = action0_unconstrained
            td[group0]["action_energy"] = action0_energy
            td[group0]["z_action"] = action0_z_sampled
            td[group0]["z_action_energy"] = action0_z_energy
            #td[group0]["z_action_unconstrained_energy"] = action0_z_unconstrained_energy
            
            td[group1]["action"] = action1_sampled
            td[group1]["action_energy"] = action1_energy
            #td[group1]["action_unconstrained_energy"] = action1_unconstrained_energy
            td[group1]["z_action"] = action1_z_sampled
            #td[group1]["z_action_unconstrained"] = action1_z_unconstrained
            td[group1]["z_action_energy"] = action1_z_energy

            td = env.step(td)           
            
            #perform an update  
            observations0 = td[group0, 'observation']
            observations1 = td[group1, 'observation']
            #this does not include reward regularization
            reward0_game = td['next'][group0, 'reward']
            reward1_game = td['next'][group1, 'reward']
            action0 = td[group0, 'action']
            action1 = td[group1, 'action']
            action0_z = td[group0, 'z_action']
            action1_z = td[group1, 'z_action']
            
            #action0_unconstrained = td[group0, 'action_unconstrained']
            #action1_unconstrained = td[group1, 'action_unconstrained']
            action0_energy = td[group0, 'action_energy']
            action1_energy = td[group1, 'action_energy']

            value0 = critics[group0](observations0)
            value1 = critics[group1](observations1)
            qvalue0 = qvals[group0](observations0, action0)
            qvalue1 = qvals[group1](observations1, action1)
        
            action0_energy_max = torch.max(action0_energy, dim = 0)[0].squeeze()
            action1_energy_max = torch.max(action1_energy, dim = 0)[0].squeeze()
            action0_energy_min = torch.min(action0_energy, dim = 0)[0].squeeze()
            action1_energy_min = torch.min(action1_energy, dim = 0)[0].squeeze()

            with torch.no_grad():
                energy0 = policy_modules[group0](observations0, action0)
                reg_energy0 = policy_reg_modules[group0](observations0, action0)
                energy0_z = policy_modules[group0](observations0, action0_z)
                reg_energy0_z = policy_reg_modules[group0](observations0, action0_z)

                energy1 = policy_modules[group1](observations1, action1)                 
                reg_energy1 = policy_reg_modules[group1](observations1, action1)
                energy1_z = policy_modules[group1](observations1, action1_z)
                reg_energy1_z = policy_reg_modules[group1](observations1, action1_z)

                reward0 = regularize_reward_energy(reward0_game, energy0, reg_energy0, energy1, reg_energy1, energy0_z, reg_energy0_z, energy1_z, reg_energy1_z)
                reward1 = regularize_reward_energy(reward1_game, energy1, reg_energy1, energy0, reg_energy0, energy1_z, reg_energy1_z, energy0_z, reg_energy0_z)

            #Train critics
            critic0_loss = F.mse_loss(value0, reward0)
            critic_optimizers[group0].zero_grad()
            critic0_loss.backward()
            critic_optimizers[group0].step()
            
            critic1_loss = F.mse_loss(value1, reward1)
            critic_optimizers[group1].zero_grad()
            critic1_loss.backward()
            critic_optimizers[group1].step()

            qvalue0_loss = F.mse_loss(qvalue0, reward0)
            qval_optimizers[group0].zero_grad()
            qvalue0_loss.backward()
            qval_optimizers[group0].step()

            qvalue1_loss = F.mse_loss(qvalue1, reward1)
            qval_optimizers[group1].zero_grad()
            qvalue1_loss.backward()
            qval_optimizers[group1].step()

            train_action0 = torch.distributions.Dirichlet(torch.ones(action_dim)).sample((num_envs_rollout,1,)).to(observations0.device)

            with torch.no_grad():
                train_qvals0 = qvals[group0](observations0, train_action0)
                train_qvals1 = qvals[group1](observations1, train_action0)
                #train_qvals0 = qvals[group0](observations0, action0)
                #train_qvals1 = qvals[group1](observations1, action1)                
                train_vals0 = critics[group0](observations0)
                train_vals1 = critics[group1](observations1)

                advantages0 = train_qvals0# - train_vals0
                advantages1 = train_qvals1# - train_vals1
                
                #expectation_qvals0 = qvals[group0](observations0, expectation_action0)
                #expectation_qvals1 = qvals[group1](observations1, expectation_action1)
                #expectation_advantages0 = expectation_qvals0 - train_vals0
                #expectation_advantages1 = expectation_qvals1 - train_vals1
                

            train_energy0 = policy_modules[group0](observations0, train_action0)
            train_energy1 = policy_modules[group1](observations1, train_action0)   
#            train_energy0 = policy_modules[group0](observations0, action0)
#            train_energy1 = policy_modules[group1](observations1, action1)        
            #expectation_energy0 = policy_modules[group0](observations0, expectation_action0)
            #expectation_energy1 = policy_modules[group1](observations1, expectation_action1)          

            # for g in policy_optimizers[group0].param_groups:
            #     g['lr'] = policy_lr * (1.0 - lr_step / lr_steps)
            # for g in policy_optimizers[group1].param_groups:
            #     g['lr'] = policy_lr * (1.0 - lr_step / lr_steps)
            # for g in critic_optimizers[group0].param_groups:
            #     g['lr'] = critic_lr * (1.0 - lr_step / lr_steps)    
            # for g in critic_optimizers[group1].param_groups:
            #     g['lr'] = critic_lr * (1.0 - lr_step / lr_steps)      

            lr_step += 1
            with torch.no_grad():
                indicator_decrease0 = (train_energy0 > min_energy)
                indicator_increase0 = (train_energy0 < max_energy)
                negative_advantages0 = torch.clip(advantages0, max=0.0)
                positive_advantages0 = torch.clip(advantages0, min=0.0)
                indicator_advantages0 = indicator_decrease0 * positive_advantages0 + indicator_increase0 * negative_advantages0
                #indicator_expectation_advantages0 = (indicator_decrease0 | indicator_increase0) * expectation_advantages0
                
                indicator_decrease1 = (train_energy1 > min_energy)
                indicator_increase1 = (train_energy1 < max_energy)
                negative_advantages1 = torch.clip(advantages1, max=0.0)
                positive_advantages1 = torch.clip(advantages1, min=0.0)
                indicator_advantages1 = indicator_decrease1 * positive_advantages1 + indicator_increase1 * negative_advantages1
                #indicator_expectation_advantages1 = (indicator_decrease1 | indicator_increase1) * expectation_advantages1

            #0.5 is due to importance weight of uniform sampling from 3-d simplex
            policy0_loss = 0.5 * torch.mean(train_energy0 * indicator_advantages0) + 0.01 * torch.square(torch.mean(train_energy0))#how do we construct the policy loss?
            policy_optimizers[group0].zero_grad()
            policy0_loss.backward()
            policy_optimizers[group0].step()
            #0.5 is due to importance weight of uniform sampling from 3-d simplex

            policy1_loss = 0.5 * torch.mean(train_energy1 * indicator_advantages1) + 0.01 * torch.square(torch.mean(train_energy1))
            #policy1_loss = 0.5 * torch.mean(train_energy1 * indicator_advantages1 - expectation_energy1 * indicator_expectation_advantages1)
            #policy1_loss = torch.mean(torch.exp(action1_logprob) * advantages1)
            policy_optimizers[group1].zero_grad()
            policy1_loss.backward()
            policy_optimizers[group1].step()

            if train_iter % 10 == 0:
                print(f"Fix iter: {fix_iter}, train iter: {train_iter}, policy_0_loss: {policy0_loss.item()}, policy_1_loss: {policy1_loss.item()}")
                print(f"Policy 0 energy min: {action0_energy_min}, policy 0 energy max: {action0_energy_max}")
                print(f"Policy 1 energy min: {action1_energy_min}, policy 1 energy max: {action1_energy_max}")                
                print(f"Mean critic prediction policy 0: {value0.mean()} mean reward: {reward0_game.mean()}, mean regularized reward: {reward0.mean()}")
                print(f"Mean critic prediction policy 1: {value1.mean()} mean reward: {reward1_game.mean()}, mean regularized reward: {reward1.mean()}")

            if train_iter % 500 == 0:
                action_samples_policy_0, action_samples_policy_1, policy_0_energy, policy_1_energy = gather_samples(ref_env, policy_modules[group0], policy_modules[group1], 1000, n_sample_steps=policy_sample_steps)
                action_samples_policy_0 = action_samples_policy_0.detach().cpu().numpy()
                action_samples_policy_1 = action_samples_policy_1.detach().cpu().numpy()
                policy_0_energy = policy_0_energy.detach().cpu().numpy()
                policy_1_energy = policy_1_energy.detach().cpu().numpy()
                qval_0, qval_1, sample_action = gather_qval_samples(ref_env, qvals[group0], qvals[group1], action_dim, 1000)
                qval_0 = qval_0.detach().cpu().numpy()
                qval_1 = qval_1.detach().cpu().numpy()
                sample_action = sample_action.detach().cpu().numpy()

                plot_samples(action_samples_policy_0, action_samples_policy_1, policy_0_energy, policy_1_energy, 
                             qval_0, qval_1, sample_action,
                             fix_iter * train_iters + train_iter + 1, min_energy, max_energy)
            if train_iter % 1000 == 0:
                store_state = {}
                for group in groups:
                    store_state.update({"policy_" + group: policy_modules[group].state_dict()})
                    store_state.update({"value_" + group: critics[group].state_dict()})
                    store_state.update({"qvalue_" + group: qvals[group].state_dict()})
                torch.save(store_state, "savestate_" + str(fix_iter * train_iters + train_iter + 1))
        action_samples_policy_0, action_samples_policy_1, policy_0_energy, policy_1_energy = gather_samples(ref_env, policy_modules[group0], policy_modules[group1], 1000, n_sample_steps=policy_sample_steps)
        action_samples_policy_0 = action_samples_policy_0.detach().cpu().numpy()
        action_samples_policy_1 = action_samples_policy_1.detach().cpu().numpy()
        policy_0_energy = policy_0_energy.detach().cpu().numpy()
        policy_1_energy = policy_1_energy.detach().cpu().numpy()

        qval_0, qval_1, sample_action = gather_qval_samples(ref_env, qvals[group0], qvals[group1], action_dim, 1000)
        qval_0 = qval_0.detach().cpu().cpu().numpy()
        qval_1 = qval_1.detach().cpu().cpu().numpy()
        sample_action = sample_action.detach().cpu().cpu().numpy()
        plot_samples(action_samples_policy_0, action_samples_policy_1, policy_0_energy, policy_1_energy, 
                        qval_0, qval_1, sample_action,
                        100000 * (fix_iter + 1), min_energy, max_energy)

        print('Ok')

if __name__ == "__main__":
    main()
