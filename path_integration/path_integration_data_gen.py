import time

import torch.optim
import tqdm

from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives.value.advantages import GAE
from torchrl.envs import (
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.utils import RandomPolicy
from torchrl.envs.libs.gym import GymEnv
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer
import tempfile
from tempfile import tempdir

def make_env(env_name="FrozenLake-v5", device="cpu"):
    env = GymEnv(env_name, device=device)
    env = TransformedEnv(env)
    #env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    #env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env

env = make_env("Swimmer-v5")
actor = RandomPolicy(env.action_spec)

def eval_model(actor, test_env, num_episodes=3):
    trajectory_length = 1000
    trajectory_count = trajectory_length * num_episodes
    num_slices = 2 #per sample, how many trajectories.
    batch_size = num_slices * trajectory_length
    storage_size = trajectory_count * trajectory_length
    final_sampler = SliceSampler(num_slices=num_slices)
    final_replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(storage_size),
        sampler=final_sampler,
    )
    #final_replay_buffer.loads("test_buffer")
    #dat = final_replay_buffer.sample(batch_size) #should be 2 full trajectories
    #trajs = split_trajectories(dat, trajectory_key="episode")
    print("Generating episodes")
    test_rewards = []
    for episode_id in range(1, num_episodes + 1):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
        td_test['episode'] = torch.ones(len(td_test)) * episode_id
        final_replay_buffer.extend(td_test)
        del td_test
        if episode_id % 10 == 0:
             print('Episode %d' % episode_id)
    final_replay_buffer.dumps("trainbuffer")
    return torch.cat(test_rewards, 0)

# Get test rewards
with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        #actor.eval()
        eval_start = time.time()
        test_rewards = eval_model(
            actor, env, num_episodes=100
        )
        eval_time = time.time() - eval_start
        print(f"Test reward: {test_rewards.mean().item():.2f} ± {test_rewards.std():.2f} (time: {eval_time:.2f}s)")
        #actor.train()