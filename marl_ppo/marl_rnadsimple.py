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
 
def transform_reward(reward, policy_i, policy_reg_i, policy_adv, policy_reg_adv, a_i, a_adv):
    eta = 0.2
    return reward - eta * (torch.log(policy_i[a_i]) - torch.log(policy_reg_i[a_i])) + \
            eta * (torch.log(policy_adv[a_adv]) - torch.log(policy_reg_adv[a_adv]))


def evaluate_policies(critics, policies, reg_policies, env):
    groups = list(env.group_map.keys())
    group1 = groups[0]
    group2 = groups[1]
    #q(0) = p(g2=0)*r(0, 0) + p(g2=1)*r(0, 1)
    td00 = env.reset()
    td00[group1]["action"] = torch.tensor(0).reshape((1,))
    td00[group2]["action"] = torch.tensor(0).reshape((1,))
    td00 = env.step(td00)
    td01 = env.reset()
    td01[group1]["action"] = torch.tensor(0.).reshape((1,))
    td01[group2]["action"] = torch.tensor(1).reshape((1,))
    td01 = env.step(td01)
    td10 = env.reset()
    td10[group1]["action"] = torch.tensor(1).reshape((1,))
    td10[group2]["action"] = torch.tensor(0).reshape((1,))
    td10 = env.step(td10)
    td11 = env.reset()
    td11[group1]["action"] = torch.tensor(1).reshape((1,))
    td11[group2]["action"] = torch.tensor(1).reshape((1,))
    td11 = env.step(td11)
    q10 = policies[group2][0] * transform_reward(td00.get(("next", group1, "reward")), policies[group1], reg_policies[group1], policies[group2], reg_policies[group2], 0, 0) + \
          policies[group2][1] * transform_reward(td01.get(("next", group1, "reward")), policies[group1], reg_policies[group1], policies[group2], reg_policies[group2], 0, 1)
    q11 = policies[group2][0] * transform_reward(td10.get(("next", group1, "reward")), policies[group1], reg_policies[group1], policies[group2], reg_policies[group2], 1, 0) + \
          policies[group2][1] * transform_reward(td11.get(("next", group1, "reward")), policies[group1], reg_policies[group1], policies[group2], reg_policies[group2], 1, 1)
    q20 = policies[group1][0] * transform_reward(td00.get(("next", group2, "reward")), policies[group2], reg_policies[group2], policies[group1], reg_policies[group1], 0, 0) + \
          policies[group1][1] * transform_reward(td10.get(("next", group2, "reward")), policies[group2], reg_policies[group2], policies[group1], reg_policies[group1], 0, 1) 
    q21 = policies[group1][0] * transform_reward(td01.get(("next", group2, "reward")), policies[group2], reg_policies[group2], policies[group1], reg_policies[group1], 1, 0) + \
          policies[group1][1] * transform_reward(td11.get(("next", group2, "reward")), policies[group2], reg_policies[group2], policies[group1], reg_policies[group1], 1, 1)     
    
    critics[group1] = torch.tensor([q10.item(), q11.item()])
    critics[group2] = torch.tensor([q20.item(), q21.item()])

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
        critic_group = torch.tensor([0.0, 0.0])
        critics[group] = critic_group

    num_iters = 100

    for i in range(num_iters):
        groups = list(env.group_map.keys())

        for fix_iter in range(5):
            policy_reg_modules = {
                group: policy_modules[group].clone()
             for group in env.group_map.keys()}
            
            for iter in range(3500):
                evaluate_policies(critics, policy_modules, policy_reg_modules, env)
                adv_1 = policy_modules[groups[0]][0] * critics[groups[0]][0] + policy_modules[groups[0]][1] * critics[groups[0]][1]
                adv_2 = policy_modules[groups[1]][0] * critics[groups[1]][0] + policy_modules[groups[1]][1] * critics[groups[1]][1]
                policy_1_gradient_1 = policy_modules[groups[0]][0] * (critics[groups[0]][0] - adv_1)
                policy_1_gradient_2 = policy_modules[groups[0]][1] * (critics[groups[0]][1] - adv_1)
                policy_2_gradient_1 = policy_modules[groups[1]][0] * (critics[groups[1]][0] - adv_2)
                policy_2_gradient_2 = policy_modules[groups[1]][1] * (critics[groups[1]][1] - adv_2)
                gradient_norms = torch.norm(policy_1_gradient_1) + torch.norm(policy_1_gradient_2) + torch.norm(policy_2_gradient_1) + torch.norm(policy_2_gradient_2)
                if iter % 100 == 0:
                    print(f"fix_iter: {fix_iter}, iter: {iter}, gradient norm: {gradient_norms}")
                if gradient_norms < torch.tensor(0.001):
                    break
                policy_modules[groups[0]][0] = policy_modules[groups[0]][0] + 0.01 * policy_1_gradient_1
                policy_modules[groups[0]][1] = policy_modules[groups[0]][1] + 0.01 * policy_1_gradient_2
                policy_modules[groups[1]][0] = policy_modules[groups[1]][0] + 0.01 * policy_2_gradient_1
                policy_modules[groups[1]][1] = policy_modules[groups[1]][1] + 0.01 * policy_2_gradient_2
                policy_modules[groups[0]] = torch.clamp(policy_modules[groups[0]], 0.0, 1.0)
                policy_modules[groups[1]] = torch.clamp(policy_modules[groups[1]], 0.0, 1.0)
                policy_modules[groups[0]] = policy_modules[groups[0]] / policy_modules[groups[0]].sum()
                policy_modules[groups[1]] = policy_modules[groups[1]] / policy_modules[groups[1]].sum()
            print(f"Fix iter: {fix_iter}, policy_modules: {policy_modules}")
        print('Ok')
       
if __name__ == "__main__":
    main()
