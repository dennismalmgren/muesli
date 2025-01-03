from torchrl.envs import GymEnv
import torch
from tensordict import TensorDict, LazyStackedTensorDict
from torchrl.data import TensorDictMap, ListStorage
from torchrl.data.map.tree import MCTSForest

from torchrl.envs import PendulumEnv, CatTensors, UnsqueezeTransform, StepCounter
# Create the MCTS Forest
forest = MCTSForest()
# Create an environment. We're using a stateless env to be able to query it at any given state (like an oracle)
env = PendulumEnv()
obs_keys = list(env.observation_spec.keys(True, True))
state_keys = set(env.full_state_spec.keys(True, True)) - set(obs_keys)
# Appending transforms to get an "observation" key that concatenates the observations together
env = env.append_transform(
    UnsqueezeTransform(
        in_keys=obs_keys,
        out_keys=[("unsqueeze", key) for key in obs_keys],
        dim=-1
    )
)
env = env.append_transform(
    CatTensors([("unsqueeze", key) for key in obs_keys], "observation")
)
env = env.append_transform(StepCounter())
env.set_seed(0)
# Get a reset state, then make a rollout out of it
reset_state = env.reset()
rollout0 = env.rollout(6, auto_reset=False, tensordict=reset_state.clone())
# Append the rollout to the forest. We're removing the state entries for clarity
rollout0 = rollout0.copy()
rollout0.exclude(*state_keys, inplace=True).get("next").exclude(*state_keys, inplace=True)
forest.extend(rollout0)
# The forest should have 6 elements (the length of the rollout)
assert len(forest) == 6
# Let's make another rollout from the same reset state
rollout1 = env.rollout(6, auto_reset=False, tensordict=reset_state.clone())
rollout1.exclude(*state_keys, inplace=True).get("next").exclude(*state_keys, inplace=True)
forest.extend(rollout1)
assert len(forest) == 12
# Let's make another final rollout from an intermediate step in the second rollout
rollout1b = env.rollout(6, auto_reset=False, tensordict=rollout1[3].exclude("next"))
rollout1b.exclude(*state_keys, inplace=True)
rollout1b.get("next").exclude(*state_keys, inplace=True)
forest.extend(rollout1b)
assert len(forest) == 18
# Since we have 2 rollouts starting at the same state, our tree should have two
#  branches if we produce it from the reset entry. Take the state, and call `get_tree`:
r = rollout0[0]
# Let's get the compact tree that follows the initial reset. A compact tree is
#  a tree where nodes that have a single child are collapsed.
tree = forest.get_tree(r)
print(tree.max_length())
#2
print(list(tree.valid_paths()))
#[(0,), (1, 0), (1, 1)]
from tensordict import assert_close
# We can manually rebuild the tree
assert_close(
    rollout1,
    torch.cat([tree.subtree[1].rollout, tree.subtree[1].subtree[0].rollout]),
    intersection=True,
)
#True
# Or we can rebuild it using the dedicated method
assert_close(
    rollout1,
    tree.rollout_from_path((1, 0)),
    intersection=True,
)
#True
tree.plot()
tree = forest.get_tree(r, compact=False)
print(tree.max_length())
#9
print(list(tree.valid_paths()))
#[(0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0, 0, 0, 0)]
assert_close(
    rollout1,
    tree.rollout_from_path((1, 0, 0, 0, 0, 0)),
    intersection=True,
)
#True
