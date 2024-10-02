from torchrl.envs.libs import PettingZooWrapper
from envs.matching_pennies_env import MatchingPenniesEnv

base_env = MatchingPenniesEnv()
env = PettingZooWrapper(base_env)

rollout = env.rollout(1)
print(rollout)