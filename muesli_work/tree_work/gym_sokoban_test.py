import gym
import gym_sokoban
import time
from torchrl.envs import GymWrapper

# Before you can make a Sokoban Environment you need to call:
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package
env_name = 'Sokoban-small-v0'
env = gym.make(env_name)
env = GymWrapper(env)

#ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))
env.rollout(120)

#time.sleep(10)