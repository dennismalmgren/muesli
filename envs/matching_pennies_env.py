from typing import TypeVar

from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np


ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

class MatchingPenniesEnv(ParallelEnv):
    agents = ["1", "2"]
    possible_agents=["1","2"]

    observation_spaces = {
        "1": spaces.Box(-1, 1),
        "2": spaces.Box(-1, 1)
    }

    action_spaces = {
        "1": spaces.Discrete(2),
        "2": spaces.Discrete(2)
    }

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self.agents = ["1", "2"]
        return {
            "1": np.array([0.0]),
            "2": np.array([0.0])
        }, {
            "1": dict(),
            "2": dict()
        }

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        reward_dict = {
            (0, 0): (1, -1),
            (0, 1): (-1, 1),
            (1, 0): (-1, 1),
            (1, 1): (1, -1)
        }
        step_reward = reward_dict[(actions["1"], actions["2"])]
        obs = {
            "1": np.array([0.0]),
            "2": np.array([0.0])
        }
        reward = {
            "1": step_reward[0],
            "2": step_reward[1]
        }
        terminated = {
            "1": True,
            "2": True
        }
        truncated = {
            "1": False,
            "2": False
        }
        info = {
            "1": dict(),
            "2": dict()
        }
        """Receives a dictionary of actions keyed by the agent name.

        Returns the observation dictionary, reward dictionary, terminated dictionary, truncated dictionary
        and info dictionary, where each dictionary is keyed by the agent.
        """
        self.agents = []
        return obs, reward, terminated, truncated, info
    
    def render(self) -> None | np.ndarray | str | list:
        """Displays a rendered frame from the environment, if supported.

        Alternate render modes in the default environments are 'rgb_array'
        which returns a numpy array and is supported by all environments outside
        of classic, and 'ansi' which returns the strings printed
        (specific to classic environments).
        """
        pass
    
    def close(self):
        """Closes the rendering window."""
        pass


    def state(self) -> np.ndarray:
        """Returns the state.

        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )
    
    def observation_space(self, agent: AgentID) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> spaces.Space:
        return self.action_spaces[agent]