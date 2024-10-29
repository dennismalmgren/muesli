import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from typing import Any, Dict, Generic, Iterable, Iterator, TypeVar

AgentID = TypeVar("AgentID")
ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")

class ColonelBlottoParallelEnv(ParallelEnv):
    def __init__(self, num_players=2, num_battlefields=5, budgets=None, values=None):
        super().__init__()
        self.num_players = num_players
        self.num_battlefields = num_battlefields
        
        # Assign budgets for each player if not specified
        if budgets is None:
            self.budgets = np.asarray([1.0] * num_players)
        else:
            self.budgets = np.asarray(budgets)
        
        # Assign values for each battlefield if not specified
        if values is None:
            self.values = np.ones((num_players, num_battlefields))
        else:
            self.values = values

        # Define action and observation spaces
        self.action_spaces = {f"player_{i}": spaces.Box(0, 1.0, shape=(self.num_battlefields,), dtype=np.float32) for i in range(self.num_players)}
        self.observation_spaces = {f"player_{i}": spaces.Box(0, np.inf, shape=(self.num_battlefields,), dtype=np.float32) for i in range(self.num_players)}
        
        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        # Track allocations and outcomes
        self.allocations = np.zeros((self.num_players, self.num_battlefields))
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observation_space(self, agent: AgentID) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self.allocations = np.zeros((self.num_players, self.num_battlefields))
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        return {agent: self.allocations[self.agent_name_mapping(agent)] for agent in self.agents}, self.infos
    
    def softmax_stable(self, x):
        return(np.exp(x - np.max(x, axis=-1, keepdims=True)) / np.exp(x - np.max(x, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True))

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        # Store the allocations from all players
        for agent, action in actions.items():
            agent_index = self.agent_name_mapping(agent)
            action_sum = np.sum(action).item()
            if not np.isclose(action_sum, 1.0):
                action = self.softmax_stable(action)
                print("Warning, normalizing input")
#                action = np.ones_like(action)
#            action = action / np.sum(action) * self.budgets[agent_index]
            self.allocations[agent_index] = action

        # Calculate rewards based on the allocations
        self._calculate_rewards()

        # All agents are done after one step in this game
        self.terminateds = {agent: True for agent in self.agents}
        self.truncateds = {agent: False for agent in self.agents}

        return {agent: self.allocations[self.agent_name_mapping(agent)] for agent in self.agents}, self.rewards, self.terminateds, self.truncateds, self.infos

    def _calculate_rewards(self):
        for j in range(self.num_battlefields):
            allocations = self.allocations[:, j]
            max_allocation = np.max(allocations)
            winners = np.where(allocations == max_allocation)[0]

            if len(winners) > 1:
                winner = np.random.choice(winners)
            else:
                winner = winners[0]

            for i in range(self.num_players):
                if i == winner:
                    self.rewards[f"player_{i}"] += self.values[i, j]
                else:
                    self.rewards[f"player_{i}"] += 0
                self.rewards[f"player_{i}"] -= 0.5 #makes it zero-sum.
                

    def render(self) -> None | np.ndarray | str | list:
        print(f"Allocations:\n {self.allocations}")
        print(f"Rewards: {self.rewards}")

    def close(self):
        pass

    def agent_name_mapping(self, agent):
        return int(agent.split("_")[1])

