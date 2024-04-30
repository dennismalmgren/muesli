from pettingzoo.test import parallel_api_test
from envs.matching_pennies_env import MatchingPenniesEnv

env = MatchingPenniesEnv()
print(env.agents)

parallel_api_test(env, num_cycles=100)