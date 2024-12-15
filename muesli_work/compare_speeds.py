
import torch.nn as nn
from torch.nn import functional as F
import time
import torch.optim
from tensordict.nn import TensorDictModule

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    RoundRobinWriter,
    TensorDictReplayBuffer,
    TensorStorage,
)
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import RandomSampler
from torchrl.envs import (
    CatFrames,
    Compose,
    DoubleToFloat,
    EnvCreator,
    ExcludeTransform,
    ObservationNorm,
    ParallelEnv,
    RandomCropTensorDict,
    RenameTransform,
    Reward2GoTransform,
    RewardScaling,
    RewardSum,
    TargetReturn,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    DTActor,
    OnlineDTActor,
    ProbabilisticActor,
    TanhDelta,
    TanhNormal,
)

from torchrl.objectives import DTLoss, OnlineDTLoss
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.envs import LIBS
from omegaconf import OmegaConf
from train_hoops_gru_no_reset import MinGRU

def make_offline_replay_buffer(rb_cfg, reward_scaling):
    r2g = Reward2GoTransform(
        gamma=1.0,
        in_keys=[("next", "reward"), "reward"],
        out_keys=[("next", "return_to_go"), "return_to_go"],
    )
    reward_scale = RewardScaling(
        loc=0,
        scale=reward_scaling,
        in_keys=[("next", "return_to_go"), "return_to_go"],
        standard_normal=False,
    )
    crop_seq = RandomCropTensorDict(sub_seq_len=rb_cfg.stacked_frames, sample_dim=-1)
    d2f = DoubleToFloat()
    rename = RenameTransform(
        in_keys=[
            "action",
            "observation",
            "return_to_go",
            ("next", "return_to_go"),
            ("next", "observation"),
        ],
        out_keys=[
            "action_cat",
            "observation_cat",
            "return_to_go_cat",
            ("next", "return_to_go_cat"),
            ("next", "observation_cat"),
        ],
    )
    exclude = ExcludeTransform(
        "terminal",
        "info",
        ("next", "timeout"),
        ("next", "terminal"),
        ("next", "observation"),
        ("next", "info"),
    )

    transforms = Compose(
        r2g,
        crop_seq,
        reward_scale,
        d2f,
        rename,
        exclude,
    )
    data = D4RLExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=True,
        batch_size=rb_cfg.batch_size,
        sampler=RandomSampler(),  # SamplerWithoutReplacement(drop_last=False),
        transform=None,
        use_truncated_as_done=True,
        direct_download=True,
        prefetch=4,
        writer=RoundRobinWriter(),
    )

    # since we're not extending the data, adding keys can only be done via
    # the creation of a new storage
    data_memmap = data[:]
    with data_memmap.unlock_():
        data_memmap = r2g.inv(data_memmap)
        data._storage = TensorStorage(data_memmap)

    loc = data[:]["observation"].flatten(0, -2).mean(axis=0).float()
    std = data[:]["observation"].flatten(0, -2).std(axis=0).float()

    obsnorm = ObservationNorm(
        loc=loc,
        scale=std,
        in_keys=["observation_cat", ("next", "observation_cat")],
        standard_normal=True,
    )
    for t in transforms:
        data.append_transform(t)
    data.append_transform(obsnorm)
    return data, loc, std


class Embedder(nn.Module):
    def __init__(self, input_dim_observation, input_dim_action, embed_dim):
        super().__init__()
        self.input_dim_action = input_dim_action
        self.input_dim_observation = input_dim_observation
        self.output_dim = embed_dim
        self.linear_action = nn.Linear(self.input_dim_action, embed_dim)
        self.linear_observation = nn.Linear(self.input_dim_observation, embed_dim)
        self.action_out = nn.Linear(embed_dim, embed_dim)
        self.observation_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, observation: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        if action is not None:
            action_embd = F.gelu(self.linear_action(action))
            action_out = self.action_out(action_embd)
        observation_embd = F.gelu(self.linear_observation(observation))
        observation_out = self.observation_out(observation_embd)
        if action is not None:
            return torch.cat((action_out, observation_out), dim=-1) #B x T x (2xembed_dim)
        else:
            return observation_out

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
  

    def forward(self, x: torch.Tensor) -> torch.Tensor:      
        hidden = self.linear(x)
        out = self.out(F.gelu(hidden))
        return out

def main():
    rb_cfg = {
        "dataset": "halfcheetah-medium-v2",
        "batch_size": 64,
        "prb": 0,
        "stacked_frames": 120,
        "buffer_prefetch": 64,
        "capacity": 1_000_000,
        "scratch_dir": None,
        "device": "cpu",
        "prefetch": 3
    }
    rb_cfg = OmegaConf.create(rb_cfg)
    #action is 6-dim
    #observation is 17-dim
    action_dim = 6
    observation_dim = 17
    embed_dim = 24
    offline_buffer, obs_loc, obs_std = make_offline_replay_buffer(
        rb_cfg, 1.0
    )
    embedder = Embedder(observation_dim, action_dim, embed_dim)
    projector = Projector(embed_dim, observation_dim, embed_dim)
    predictor_gru = nn.GRU(2 * embed_dim, embed_dim, batch_first=True)
    predictor_gru = MinGRU(2 * embed_dim, embed_dim)
    optimizer = torch.optim.Adam(list(embedder.parameters()) + list(predictor_gru.parameters()) + list(projector.parameters()), lr=1e-3)
    start_time = time.time()
    for i in range(1100):
        data = offline_buffer.sample(64)
        embedded_inputs = embedder(data["observation_cat"], data["action_cat"])
        embedded_targets = data["next", "observation_cat"]
        #now lets do 
        outputs = predictor_gru(embedded_inputs)
        predictions = projector(outputs[0])
        loss = F.mse_loss(predictions, embedded_targets.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            current_time = time.time()
            dt = current_time - start_time
            print(f"Iter {i}, time {dt}: ", loss.item())

    print('ok')

if __name__=="__main__":
    main()