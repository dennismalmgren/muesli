import time
import sys

import torch.optim
import tqdm

from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives.value.advantages import GAE
from torchrl.envs import (
    DoubleToFloat,
    ExplorationType,
    RenameTransform,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
    TensorDictPrimer,
    ParallelEnv,
    EnvCreator,
    CatFrames
)
from torchrl.envs.utils import RandomPolicy
from torchrl.envs.libs.gym import GymEnv
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer
from tensordict import TensorDict, TensorDictBase, NestedKey
import hydra
from torchrl.data import TensorSpec, CompositeSpec

def get_project_root_path_vscode():
    trace_object = getattr(sys, 'gettrace', lambda: None)() #vscode debugging
    if trace_object is not None:
        return "../../../"
    else:
        return "../../../../"
    
def make_env(env_name="FrozenLake-v5", device="cpu"):
    env = GymEnv(env_name, device=device)
    env = TransformedEnv(env)
    #env.append_transform(CatFrames(2, in_keys=["observation"], dim=-1, padding="constant"))
    #env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    #env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    # env.append_transform(RenameTransform(in_keys=["action"], out_keys=["prev_action"], 
    #                                      in_keys_inv=["action"], out_keys_inv=["prev_action"],
    #                                      create_copy=True))
    #env.append_transform(CopyKeysTransform(in_keys_inv=["action"], out_keys_inv=[("next", "prev_action")], create_copy=True))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    action_spec = env.action_spec
    env.append_transform(TensorDictPrimer(prev_action=action_spec, default_value=0.0))
    return env

def make_vec_env(env_name, num_envs, device="cpu"):
    env = ParallelEnv(num_workers=num_envs, create_env_fn=EnvCreator(lambda: make_env(env_name, device)))
    return env

class MyRandomPolicy:
    """A random policy for data collectors.

    This is a wrapper around the action_spec.rand method.

    Args:
        action_spec: TensorSpec object describing the action specs

    Examples:
        >>> from tensordict import TensorDict
        >>> from torchrl.data.tensor_specs import BoundedTensorSpec
        >>> action_spec = BoundedTensorSpec(-torch.ones(3), torch.ones(3))
        >>> actor = RandomPolicy(action_spec=action_spec)
        >>> td = actor(TensorDict({}, batch_size=[])) # selects a random action in the cube [-1; 1]
    """

    def __init__(self, action_spec: TensorSpec, action_key: NestedKey = "action"):
        super().__init__()
        self.action_spec = action_spec.clone()
        self.action_key = action_key

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        if isinstance(self.action_spec, CompositeSpec):
            return td.update(self.action_spec.rand())
        else:
            action = self.action_spec.rand()
            td.set(("next", "prev_action"), action)
            return td.set(self.action_key, action)
        
def generate_episodes(env_name, 
                      num_envs = 10,
                      num_episodes = 100,
                      max_trajectory_length=1000):
    
    storage_size = max_trajectory_length * num_episodes
#    num_slices = 2 #per sample, how many trajectories.
#    batch_size = num_slices * trajectory_length
    sampler = SliceSampler(num_slices=1) #isn't used.
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(storage_size, ndim=2),
        sampler=sampler
    )
    env = make_vec_env(env_name, num_envs)
    actor = RandomPolicy(env.action_spec)
    collector = SyncDataCollector(
        env,
        policy=actor,
        frames_per_batch=num_envs * max_trajectory_length,
        max_frames_per_traj=-1,
        total_frames=storage_size,
    )

    print("Generating batches")
    gathered_rewards = []
    for i, data in enumerate(collector):
        #hack
        del data[('next', 'prev_action')] 
        replay_buffer.extend(data)
        print('Generated batch ', (i * num_envs))
        rewards = data["next", "episode_reward"][data["next", "done"]]
        gathered_rewards.append(rewards)
    return torch.cat(gathered_rewards, 0), replay_buffer

@hydra.main(config_path=".", config_name="generate_random_trajectories", version_base="1.1")
def main(cfg: "DictConfig"):
    # Get test rewards
    with torch.no_grad():
        eval_start = time.time()
        rewards, replay_buffer = generate_episodes(
            cfg.env.env_name, 
            num_envs=cfg.collector.num_envs,
            num_episodes=cfg.collector.num_episodes,
            max_trajectory_length = cfg.env.episode_max_len
        )
        eval_time = time.time() - eval_start
        print(f"Generated episodes rewards: {rewards.mean().item():.2f} ± {rewards.std():.2f} (time: {eval_time:.2f}s)")
    project_root_path = get_project_root_path_vscode()
    rb_dir = project_root_path + f"{cfg.rb.save_base_path}/{cfg.rb.save_dir}"

    replay_buffer.dumps(rb_dir)

if __name__ == "__main__":
    main()  