import torch
import hydra
from torch import multiprocessing
from torchrl.envs.libs import PettingZooWrapper
from envs.colonel_blotto_env import ColonelBlottoParallelEnv
from torchrl.envs.utils import MarlGroupMapType
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tensordict import TensorDict
from cont_actions.cont_actions_policy import ActorDefaultNet

def vectorized_perturb_network(network, epsilon, noise_dim, device):
    # Convert all parameters into a flat vector
    params_vector = parameters_to_vector(network.parameters())
    
    # Sample noise from normal distribution in a vectorized manner
    noise = torch.randn_like(params_vector)
    
    # Create perturbed vectors (add and subtract noise)
    perturbation = epsilon * noise
    params_add_vector = params_vector + perturbation
    params_sub_vector = params_vector - perturbation
    
    # Reassign perturbed weights back to the networks in a vectorized way
    perturbed_add = ActorDefaultNet(noise_dim).to(device)
    perturbed_sub = ActorDefaultNet(noise_dim).to(device)
    vector_to_parameters(params_add_vector, perturbed_add.parameters())
    vector_to_parameters(params_sub_vector, perturbed_sub.parameters())
    
    return perturbed_add, perturbed_sub, noise

def step_network(network, pseudo_grad, optimizer):
    optimizer.zero_grad()
   # Flatten the network parameters and pseudo_grad to vectors
    #params_vector = torch.cat([p.view(-1) for p in network.parameters()])
    pseudo_grad_vector = pseudo_grad.view(-1)

    # Reassign the pseudo_grad as gradients to each parameter
    start = 0
    for param in network.parameters():
        param_length = param.numel()  # Number of elements in the parameter
        # Assign the gradient manually (detach the grad tensor to avoid autograd tracking)
        grad = pseudo_grad_vector[start:start + param_length].view_as(param).detach()
        param.grad = grad  # Set the gradient
        start += param_length

    # Perform the optimization step (SGD step)
    optimizer.step()
    
def calculate_node_0_grad(base_env, policy_1, perturbation_policy_0, perturbed_policy_0_add, perturbed_policy_0_sub, epsilon):
    td_policy_0_add = base_env.reset()
    perturbed_policy_0_add_action = perturbed_policy_0_add(td_policy_0_add['player_0', 'observation'])
    policy_1_action = policy_1(td_policy_0_add['player_1', 'observation'])

    node_0_actions_add = TensorDict(
        {'player_0':
         TensorDict({
             'action': perturbed_policy_0_add_action
         }),
         'player_1':
                TensorDict({
             'action': policy_1_action
         }),
         },
         device=td_policy_0_add.device
    )
   
    td_policy_0_add = td_policy_0_add.update(node_0_actions_add)
    node_0_add_next_td = base_env.step(td_policy_0_add)
    #node 0, sub
    td_policy_0_sub = base_env.reset()
    perturbed_policy_0_sub_action = perturbed_policy_0_sub(td_policy_0_sub['player_0', 'observation'])
    policy_1_action = policy_1(td_policy_0_sub['player_0', 'observation'])
    
    node_0_actions_sub = TensorDict(
        {'player_0':
         TensorDict({
             'action': perturbed_policy_0_sub_action
         }),
         'player_1':
                TensorDict({
             'action': policy_1_action
         }),
         },
         device=td_policy_0_sub.device
    )

    td_policy_0_sub = td_policy_0_sub.update(node_0_actions_sub)
    node_0_sub_next_td = base_env.step(td_policy_0_sub)
    du_policy_0 = (node_0_add_next_td['next', 'player_0', 'reward'] - node_0_sub_next_td['next', 'player_0', 'reward']) / (2 * epsilon)
    du_policy_0 = du_policy_0.item()
    
    pseudo_grad_policy_0 = du_policy_0 * perturbation_policy_0
    return pseudo_grad_policy_0, policy_1_action, du_policy_0

def calculate_node_1_grad(base_env, policy_0, perturbation_policy_1, perturbed_policy_1_add, perturbed_policy_1_sub, epsilon):
    td_policy_1_add = base_env.reset()
    policy_0_action = policy_0(td_policy_1_add['player_0', 'observation'])
    perturbed_policy_1_add_action = perturbed_policy_1_add(td_policy_1_add['player_1', 'observation'])

    node_1_actions_add = TensorDict(
        {'player_0':
         TensorDict({
             'action': policy_0_action
         }),
         'player_1':
                TensorDict({
             'action': perturbed_policy_1_add_action
         }),
         },
         device=td_policy_1_add.device
    )

    td_policy_1_add = td_policy_1_add.update(node_1_actions_add)
    node_1_add_next_td = base_env.step(td_policy_1_add)

    #node 0, sub
    td_policy_1_sub = base_env.reset()
    perturbed_policy_1_sub_action = perturbed_policy_1_sub(td_policy_1_sub['player_1', 'observation'])
    policy_0_action = policy_0(td_policy_1_sub['player_0', 'observation'])
    
    node_1_actions_sub = TensorDict(
        {'player_0':
         TensorDict({
             'action': policy_0_action
         }),
         'player_1':
                TensorDict({
             'action': perturbed_policy_1_sub_action
         }),
         },
         device=td_policy_1_sub.device
    )

    td_policy_1_sub = td_policy_1_sub.update(node_1_actions_sub)
    node_1_sub_next_td = base_env.step(td_policy_1_sub)
    du_policy_1 = (node_1_add_next_td['next', 'player_1', 'reward'] - node_1_sub_next_td['next', 'player_1', 'reward']) / (2 * epsilon)
    du_policy_1 = du_policy_1.item()
    
    pseudo_grad_policy_1 = du_policy_1 * perturbation_policy_1
    return pseudo_grad_policy_1, policy_0_action, du_policy_1

def gather_samples(base_env, policy_0, policy_1, n_samples):
    td_policy = base_env.reset()
    sample_actions_policy_0 = torch.zeros((n_samples, 3))
    sample_actions_policy_1 = torch.zeros((n_samples, 3))
    for i in range(n_samples):
        policy_0_action = policy_0(td_policy['player_0', 'observation'])
        sample_actions_policy_0[i, :] = policy_0_action
        policy_1_action = policy_1(td_policy['player_1', 'observation'])
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
    plt.figure(figsize=(6, 6))
    plt.plot([vertices[0, 0], vertices[1, 0]], [vertices[0, 1], vertices[1, 1]], 'k-', lw=2)
    plt.plot([vertices[1, 0], vertices[2, 0]], [vertices[1, 1], vertices[2, 1]], 'k-', lw=2)
    plt.plot([vertices[2, 0], vertices[0, 0]], [vertices[2, 1], vertices[0, 1]], 'k-', lw=2)
    
    # Scatter plot the points inside the triangle
    plt.scatter(points_2d_policy_0[:, 0], points_2d_policy_0[:, 1], color='blue', alpha=0.6)

    # Annotate the vertices to show actions corresponding to each corner
    plt.text(vertices[0, 0], vertices[0, 1] - 0.05, 'Action 1', fontsize=12, ha='center')
    plt.text(vertices[1, 0], vertices[1, 1] - 0.05, 'Action 2', fontsize=12, ha='center')
    plt.text(vertices[2, 0], vertices[2, 1] + 0.05, 'Action 3', fontsize=12, ha='center')

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, np.sqrt(3)/2 + 0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show() 
    plt.savefig(f'policy_distribution_iter_{iteration}.png')
    plt.close()
    
@hydra.main(config_path=".", config_name="cont_actions_blotto", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    torch.manual_seed(cfg.seed)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    device = "cpu"
    # Time to add noise to the input
    noise_dim = 2
    policy_0 = ActorDefaultNet(noise_dim).to(device)
    lr = 1e-6
    optimizer_0 = optim.SGD(policy_0.parameters(), lr=lr)

    policy_1 = ActorDefaultNet(noise_dim).to(device)
    optimizer_1 = optim.SGD(policy_1.parameters(), lr=lr)

    base_env = ColonelBlottoParallelEnv(num_players=2, num_battlefields=3)
    base_env = PettingZooWrapper(base_env, group_map=MarlGroupMapType.ONE_GROUP_PER_AGENT, device=device)

    epsilon = 1e-2
    du_policy_0_avg = 0.0
    du_policy_1_avg = 0.0

    for i in range(50_000_001):
        with torch.no_grad():
            perturbed_policy_0_add, perturbed_policy_0_sub, perturbation_policy_0 = vectorized_perturb_network(policy_0, epsilon, noise_dim, device)
            perturbed_policy_1_add, perturbed_policy_1_sub, perturbation_policy_1 = vectorized_perturb_network(policy_1, epsilon, noise_dim, device)
    

            pseudo_grad_policy_0, policy_1_action, du_policy_0 = calculate_node_0_grad(base_env, policy_1, perturbation_policy_0, perturbed_policy_0_add, perturbed_policy_0_sub, epsilon)
            pseudo_grad_policy_1, policy_0_action, du_policy_1 = calculate_node_1_grad(base_env, policy_0, perturbation_policy_1, perturbed_policy_1_add, perturbed_policy_1_sub, epsilon)
            du_policy_0_avg += abs(du_policy_0)
            du_policy_1_avg += abs(du_policy_1)

        step_network(policy_0, pseudo_grad_policy_0, optimizer_0)
        step_network(policy_1, pseudo_grad_policy_1, optimizer_1)
        #todo: evaluate (?)
        #todo: distribution over actions
        if i % 1000 == 0:
            print(f"Iter {i}, Policy 0: {policy_0_action}")
            print(f"Iter {i}, Policy 1: {policy_1_action}")
            print(f"Iter {i}, Policy 0 grad: {du_policy_0_avg}")
            du_policy_0_avg = 0.0
            du_policy_1_avg = 0.0

        if i % 10000 == 0:
            with torch.no_grad():
                action_samples_policy_0, action_samples_policy_1 = gather_samples(base_env, policy_0, policy_1, 10000)
                action_samples_policy_0 = action_samples_policy_0.detach().numpy()
                action_samples_policy_1 = action_samples_policy_1.detach().numpy()
                plot_samples(action_samples_policy_0, action_samples_policy_1, i)
    print('ok')

if __name__ == "__main__":
    main()