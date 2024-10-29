from __future__ import annotations

from typing import Optional, List, Dict

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BoundedContinuousTensorSpec,
    DiscreteTensorSpec,
)
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import MarlGroupMapType
from tensordict.utils import NestedKey

class ContinuousTimeInspectionGameParallelEnv(EnvBase):
    batch_locked: bool = False

    def __init__(self, *, num_players=2, T=1.0, 
                 batch_size=None, device=None):
        super().__init__(device=device, batch_size=batch_size)
        self.num_players = num_players
        self.T = T  # Upper bound of the time interval
        self.agent_names: List[str] = ["Inspector", "Worker"]
        self.agent_names_to_indices_map: Dict[str, int] = {
            "Inspector": 0,
            "Worker": 1
        }
        self.group_map: Dict[str, List[str]] = MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(self.agent_names)

        action_spec = CompositeSpec(device=self.device, shape=self.batch_size)
        observation_spec = CompositeSpec(device=self.device, shape=self.batch_size)
        reward_spec = CompositeSpec(device=self.device, shape=self.batch_size)
        done_spec = CompositeSpec(device=self.device, shape=self.batch_size)

        for agent in self.agent_names:
            group_action_spec = CompositeSpec({
                "action": BoundedContinuousTensorSpec(
                    low=0.0, high=self.T, shape=(*self.batch_size, 1), device=device)
            })
            group_observation_spec = CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, 1), dtype=torch.float32, device=device)
            })
            group_reward_spec = CompositeSpec({
                "reward": UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, 1), device=device)
            })
            action_spec[agent] = group_action_spec
            observation_spec[agent] = group_observation_spec
            reward_spec[agent] = group_reward_spec

            done_spec[agent] = CompositeSpec(
                {
                    "done": DiscreteTensorSpec(
                        n=2,
                        shape=torch.Size((*self.batch_size, 1)),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "terminated": DiscreteTensorSpec(
                        n=2,
                        shape=torch.Size((*self.batch_size, 1)),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "truncated": DiscreteTensorSpec(
                        n=2,
                        shape=torch.Size((*self.batch_size, 1)),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                },)

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec
        self.done_spec = done_spec
        self.state_spec = self.observation_spec.clone()

    # Reward keys
    @property
    def reward_keys(self) -> List[NestedKey]:
        return [("Inspector", "reward"), ("Worker", "reward")]

    @property
    def action_keys(self) -> List[NestedKey]:
        return [("Inspector", "action"), ("Worker", "action")]

    @property
    def done_keys(self) -> List[NestedKey]:
        return [
            ("Inspector", "done"), ("Worker", "done"),
            ("Inspector", "terminated"), ("Worker", "terminated"),
            ("Inspector", "truncated"), ("Worker", "truncated"),
        ]

    def _reset(self, reset_td: TensorDict) -> TensorDict:
        shape = reset_td.shape if reset_td is not None else ()
        state = self.state_spec.zero(shape)
        reset_result = state.update(self.full_done_spec.zero(shape))
        return reset_result

    def _step(self, state: TensorDict) -> TensorDict:
        result_td = TensorDict(
            {agent: TensorDict({}) for agent in self.agent_names},
            batch_size=state.batch_size
        )

        inspector_action = state['Inspector', 'action'].squeeze(-1)
        worker_action = state['Worker', 'action'].squeeze(-1)

        # Compute rewards based on the game's payoff structure
        inspector_wins = (inspector_action <= worker_action).float()
        inspector_reward = inspector_wins * 1.0 + (1 - inspector_wins) * (-1.0)
        worker_reward = -inspector_reward  # Zero-sum game

        result_td['Inspector', 'reward'] = inspector_reward.unsqueeze(-1)
        result_td['Worker', 'reward'] = worker_reward.unsqueeze(-1)

        # Set 'done', 'terminated', and 'truncated' flags
        for agent in self.agent_names:
            result_td[agent, 'done'] = torch.ones_like(state[agent, 'done']).bool()
            result_td[agent, 'terminated'] = torch.ones_like(state[agent, 'terminated']).bool()
            result_td[agent, 'truncated'] = torch.zeros_like(state[agent, 'truncated']).bool()
            result_td[agent, 'observation'] = state[agent, 'observation'].clone()

        return result_td

    def _set_seed(self, seed: int | None):
        pass  # Implement seed setting if necessary
