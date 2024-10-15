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

class ColonelBlottoParallelEnv(EnvBase):
    batch_locked: bool = False

    def __init__(self, *, num_players=2, num_battlefields=5, 
                 budgets=None, values=None,
                 device=None):
        super().__init__(device=device)
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
        
        self.action_spec: Bounded = Bounded(low=0.0, high=1.0, shape=(num_battlefields,), device=device)

        self.full_observation_spec: Composite = Composite(
            observation=Unbounded(shape=(1,), dtype=torch.float32, device=device)
        )
        
        self.state_spec: Unbounded = self.observation_spec.clone()

        self.reward_spec: Unbounded = Composite(
            {
                ("player0", "reward"): Unbounded(shape=(1,), device=device),
                ("player1", "reward"): Unbounded(shape=(1,), device=device)
            },
            device=device
        )

        self.full_done_spec: Categorical = Composite(
            done=Categorical(2, shape=(1,), dtype=torch.bool, device=device),
            device=device,
        )
        self.full_done_spec["terminated"] = self.full_done_spec["done"].clone()
        self.full_done_spec["truncated"] = self.full_done_spec["done"].clone()
        # Assign values for each battlefield if not specified
        if values is None:
            self.values = torch.ones((num_players, num_battlefields))
        else:
            self.values = values

    def _reset(self, reset_td: TensorDict) -> TensorDict:
        shape = reset_td.shape if reset_td is not None else ()
        state = self.state_spec.zero(shape)
        return state.update(self.full_done_spec.zero(shape))
    
    def _step(self, state: TensorDict) -> TensorDict:
        action = state["action"]
        the_state = state["state"]
        reward0 = torch.zeros_like(action)
        reward1 = torch.zeros_like(action)
        state = TensorDict(
            {
                ("player0", "reward"): reward0.float(),
                ("player1", "reward"): reward1.float(),
            },
            batch_size = the_state.batch_size
        )
        return state
    
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
        rewards = {agent: np.zeros(self.batch_size) for agent in self.agents}

        for j in range(self.num_battlefields):
            battlefield_allocations = allocations[:, :, j]

            max_allocations = np.max(battlefield_allocations, axis=0)
            winners = battlefield_allocations == max_allocations
            tie_breakers = np.array([
                np.random.choice(np.flatnonzero(winners[:, b])) if np.sum(winners[:, b]) > 1 else np.argmax(winners[:, b])
                for b in range(self.batch_size)
            ])

            for i in range(self.num_players):
                won_mask = tie_breakers == i
                rewards[f"player_{i}"] += won_mask * self.values[i, j]
                rewards[f"player_{i}"] -= 0.5 #makes it zero-sum.
        return rewards
    

    # def render(self) -> None | np.ndarray | str | list:
    #     print(f"Allocations:\n {self.allocations}")
    #     print(f"Rewards: {self.rewards}")

    # def close(self):
    #     pass

    # def agent_name_mapping(self, agent):
    #     return int(agent.split("_")[1])

