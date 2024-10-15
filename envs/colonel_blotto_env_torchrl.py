from __future__ import annotations

from typing import Optional, List, Dict

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    Composite,
    Unbounded,
    Bounded,
    Categorical
)
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import MarlGroupMapType
from tensordict.utils import NestedKey

class ColonelBlottoParallelEnv(EnvBase):
    batch_locked: bool = False

    def __init__(self, *, num_players=2, num_battlefields=5, 
                 batch_size=None,
                 budgets=None, values=None,
                 device=None):
        super().__init__(device=device, batch_size=batch_size)
        self.num_players = num_players
        self.num_battlefields = num_battlefields
        self.agent_names: List[str] = ["player0", "player1"]
        self.agent_names_to_indices_map: Dict[str, int] = {
            "player0": 0,
            "player1": 1
        }
        self.group_map: Dict[str, List[str]] = MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(self.agent_names)

        # Assign budgets for each player if not specified
        if budgets is None:
            self.budgets = torch.tensor([1.0] * num_players)
        else:
            self.budgets = budgets
        
        action_spec = Composite(device=self.device, shape=self.batch_size)
        observation_spec = Composite(device=self.device, shape=self.batch_size)
        reward_spec = Composite(device=self.device, shape=self.batch_size)
        done_spec = Composite(device=self.device, shape=self.batch_size)
        for group in self.group_map.keys():
            group_action_spec = torch.stack([Composite({"action": Bounded(low=0.0, high=1.0, shape=(*self.batch_size, num_battlefields,), device=device)})], dim=0)
            group_observation_spec = torch.stack([Composite({"observation": Unbounded(shape=(*self.batch_size, 1, ), dtype=torch.float32, device=device)})], dim=0)
            group_reward_spec = torch.stack([Composite({"reward": Unbounded(shape=(*self.batch_size, 1, ), device=device)})], dim=0)
            action_spec[group] = group_action_spec
            observation_spec[group] = group_observation_spec
            reward_spec[group] = group_reward_spec
        
            done_spec[group] = Composite(
                {
                    "done": Categorical(
                        n=2,
                        shape=torch.Size((*self.batch_size, 1, )),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "terminated": Categorical(
                        n=2,
                        shape=torch.Size((*self.batch_size, 1, )),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "truncated": Categorical(
                        n=2,
                        shape=torch.Size((*self.batch_size, 1, )),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                },)
        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec
        self.state_spec = self.observation_spec.clone()

        # Assign values for each battlefield if not specified
        if values is None:
            self.values = torch.ones((*self.batch_size, num_players, num_battlefields))
        else:
            self.values = values
    # Reward spec
    @property
    def reward_keys(self) -> List[NestedKey]:
        return [("player0", "reward"), ("player1", "reward")]
    
    @property
    def action_keys(self) -> List[NestedKey]:
        return [("player0", "action"), ("player1", "action")]
    
    @property
    def done_keys(self) -> List[NestedKey]:
        return [("player0", "done"), ("player1", "done"),
                ("player0", "terminated"), ("player1", "terminated"),
                ("player0", "truncated"), ("player1", "truncated"),]
    
    def _reset(self, reset_td: TensorDict) -> TensorDict:
        shape = reset_td.shape if reset_td is not None else ()
        state = self.state_spec.zero(shape)
        reset_result = state.update(self.full_done_spec.zero(shape))
        return reset_result
    
    def _step(self, state: TensorDict) -> TensorDict:
        result_td = TensorDict({"player0": TensorDict({}),
                                "player1": TensorDict({}),},
                                batch_size=state.batch_size)
        
        action0 = state['player0', 'action']
        action1 = state['player1', 'action']
        allocations = torch.cat((action0, action1), dim=-2)

        rewards = self._calculate_rewards(allocations)
        result_td['player0', 'reward'] = rewards[:, 0].unsqueeze(-1)
        result_td['player1', 'reward'] = rewards[:, 1].unsqueeze(-1)
        result_td['player0', 'terminated'] = torch.ones_like(state['player0', 'terminated']).bool()
        result_td['player1', 'terminated'] = torch.ones_like(state['player1', 'terminated']).bool()
        result_td['player0', 'done'] = torch.ones_like(state['player0', 'done']).bool()
        result_td['player1', 'done'] = torch.ones_like(state['player1', 'done']).bool()
        result_td['player0', 'truncated'] = torch.zeros_like(state['player0', 'truncated']).bool()
        result_td['player1', 'truncated'] = torch.zeros_like(state['player1', 'truncated']).bool()
        result_td['player0', 'observation'] = state['player0', 'observation'].clone()
        result_td['player1', 'observation'] = state['player1', 'observation'].clone()
        return result_td
    
    def _set_seed(self, seed: int | None):
        ...

    # def reset(
    #     self,
    #     seed: int | None = None,
    #     options: dict | None = None,
    # ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
    #     allocations = np.zeros((self.num_players, self.batch_size, self.num_battlefields))
    #     return {agent: allocations[self.agent_name_mapping(agent), i] for agent in self.agents for i in range(self.batch_size)}, self.infos
    
    # def softmax_stable(self, x):
    #     return(np.exp(x - np.max(x, axis=-1, keepdims=True)) / np.exp(x - np.max(x, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True))

#     def step(
#         self, actions: dict[AgentID, ActionType]
#     ) -> tuple[
#         dict[AgentID, ObsType],
#         dict[AgentID, float],
#         dict[AgentID, bool],
#         dict[AgentID, bool],
#         dict[AgentID, dict],
#     ]:
#         allocations = np.zeros((self.num_players, self.batch_size, self.num_battlefields))
#         # Store the allocations from all players
#         for agent, action in actions.items():
#             agent_index = self.agent_name_mapping(agent)
#             # action_sum = np.sum(action, ).item()
#             # if not np.isclose(action_sum, 1.0):
#             #     action = self.softmax_stable(action)
#             #     print("Warning, normalizing input")
# #                action = np.ones_like(action)
# #            action = action / np.sum(action) * self.budgets[agent_index]
#             allocations[agent_index, :] = action

#         # Calculate rewards based on the allocations
#         rewards = self._calculate_rewards(allocations)

#         return {agent: allocations[self.agent_name_mapping(agent)] for agent in self.agents}, rewards, self.terminateds, self.truncateds, self.infos

    # def _calculate_rewards(self):
    #     stability_epsilon = 1e-6
    #     for j in range(self.num_battlefields):
    #         allocations = self.allocations[:, j] + stability_epsilon
    #         win_probabilities = allocations / np.sum(allocations) #todo: numeric stability.
            
    #         for i in range(self.num_players):
    #             self.rewards[f"player_{i}"] += self.values[i, j] * win_probabilities[i]

    def _calculate_rewards(self, allocations):
        batch_size = allocations.shape[0]
        #allocations are batch_size x agent_id x battlefield
        rewards = torch.zeros((batch_size, self.num_players, 1), device=allocations.device)

        for j in range(self.num_battlefields):
            battlefield_allocations = allocations[:, :, j]

            max_allocations = torch.max(battlefield_allocations, dim=-1)[0]
            winners = battlefield_allocations == max_allocations.unsqueeze(1) #shape: (batch_size, num_players)
            unique_winners = torch.argmax(winners.float(), dim=1)  # Shape: (batch_size,), index of the winner

            num_winners_per_batch = torch.sum(winners, dim=1)  # Count number of winners per batch
            unique_winner_mask = num_winners_per_batch == 1  # Shape: (batch_size,), True where unique winner
            #only works for 2-player games
            tie_winners = torch.multinomial(torch.ones((batch_size, 2), device=allocations.device), 1).squeeze(-1)
            final_winners = torch.where(unique_winner_mask, unique_winners, tie_winners)

            for i in range(self.num_players):
                won_mask = final_winners == i
                rewards[:, i, 0] += won_mask * self.values[i, j]
                rewards[:, i, 0] -= 0.5
        return rewards
    

    # def render(self) -> None | np.ndarray | str | list:
    #     print(f"Allocations:\n {self.allocations}")
    #     print(f"Rewards: {self.rewards}")

    # def close(self):
    #     pass

    # def agent_name_mapping(self, agent):
    #     return int(agent.split("_")[1])

