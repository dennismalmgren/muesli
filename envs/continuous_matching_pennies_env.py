import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import Composite, Bounded, Categorical, Unbounded
from torchrl.envs.utils import MarlGroupMapType
from torchrl.envs.common import EnvBase
from typing import Optional, List, Dict
from tensordict.utils import NestedKey

class ContinuousMatchingPenniesEnv(EnvBase):
    batch_locked = False
    def __init__(self, num_players=2, batch_size=None, device=None):
        super().__init__(device=device, batch_size=batch_size)
        self.num_players = num_players
        self.agent_names = ["player0", "player1"]
        self.action_bounds = (0.0, 1.0)  # Bounded actions in [0, 1]
        self.group_map = MarlGroupMapType.ONE_GROUP_PER_AGENT.get_group_map(self.agent_names)
        # Define action and observation specs
        action_spec = Composite(device=self.device, shape=self.batch_size)
        observation_spec = Composite(device=self.device, shape=self.batch_size)
        reward_spec = Composite(device=self.device, shape=self.batch_size)
        done_spec = Composite(device=self.device, shape=self.batch_size)

        for i in range(self.num_players):
            action_spec[f"player{i}"] = torch.stack([Composite({"action": Bounded(low=self.action_bounds[0], high=self.action_bounds[1], shape=(*self.batch_size, 1,), device=device)})], dim=0)
            observation_spec[f"player{i}"] = torch.stack([Composite({"observation": Unbounded(shape=(*self.batch_size, 1, ), dtype=torch.float32, device=device)})], dim=0)
            reward_spec[f"player{i}"] = torch.stack([Composite({"reward": Unbounded(shape=(*self.batch_size, 1, ), device=device)})], dim=0)
            done_spec[f"player{i}"] = Composite(
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
    
    def _reset(self, reset_td: TensorDict):
        shape = reset_td.shape if reset_td is not None else ()
        state = self.observation_spec.zero(shape)
        return state.update(self.done_spec.zero(shape))

    def _step(self, state: TensorDict) -> TensorDict:
        actions_player0 = state['player0', 'action']
        actions_player1 = state['player1', 'action']
        
        # Matching pennies payoff structure
        rewards = self._calculate_rewards(actions_player0, actions_player1)
        
        result_td = TensorDict({
            "player0": TensorDict({"reward": rewards[0], 
                                   "done": torch.ones((*state.batch_size, 1, 1), dtype=torch.bool),
                                    "terminated": torch.ones((*state.batch_size, 1, 1), dtype=torch.bool),
                                   "truncated": torch.zeros((*state.batch_size, 1, 1), dtype=torch.bool),
                                   "observation": state["player0", "observation"].clone()
                                   }),
                                
                                   
            "player1": TensorDict({"reward": rewards[1], 
                                   "done": torch.ones((*state.batch_size, 1, 1), dtype=torch.bool),
                                   "terminated": torch.ones((*state.batch_size, 1, 1), dtype=torch.bool),
                                   "truncated": torch.zeros((*state.batch_size, 1, 1), dtype=torch.bool),
                                   "observation": state["player1", "observation"].clone()
                                   })
        }, batch_size=state.batch_size)
        
        return result_td

    def _calculate_rewards(self, actions0, actions1):
        """
        Reward structure for continuous matching pennies:
        Player 0's reward: r0 = sin(2 * pi * (a0 - a1))
        Player 1's reward: r1 = -r0
        """
        a0 = actions0.squeeze(-1)  # Player 0's actions
        a1 = actions1.squeeze(-1)  # Player 1's actions

        # Payoff for player 0
        reward0 = torch.sin(2 * torch.pi * (a0 - a1))  # Reward for player 0
        reward1 = -reward0  # Player 1's reward is the negative of Player 0's reward

        return reward0.unsqueeze(-1), reward1.unsqueeze(-1)

    def _set_seed(self, seed: int | None):
        torch.manual_seed(seed)



# # Example usage
# env = ContinuousMatchingPenniesEnv(device='cpu')

# reset_state = TensorDict({
#     "player0": {},
#     "player1": {}
# },
# batch_size=(4,))
# out_reset_state = env.reset(reset_state)
# out_reset_state["player0", "action"] = torch.rand(4, 1, 1)
# out_reset_state["player1", "action"] = torch.rand(4, 1, 1)

# # Perform a step in the environment
# step_result = env.step(out_reset_state)
# print("Step result:", step_result)