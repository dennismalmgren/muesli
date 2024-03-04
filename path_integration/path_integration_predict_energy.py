import time
import sys

import torch.optim
import tqdm
import hydra
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, SliceSampler, RandomSampler
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
    
)
from torchrl.collectors.utils import split_trajectories
from torchrl.envs.utils import RandomPolicy
from torchrl.envs.libs.gym import GymEnv
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.modules import MLP
from torch import nn


class PlaceCellActivation(nn.Module):
    def __init__(self, device):
        super().__init__()
#        self.place_cell_centers = torch.load("/home/dennismalmgren/repos/muesli/path_integration/place_cell_centers.pt")
        self.place_cell_centers = torch.load("/mnt/f/repos/muesli/path_integration/place_cell_centers.pt").to(device)
        self.place_cell_centers = self.place_cell_centers.unsqueeze(0)
        self.place_cell_scale = 3

    def forward(self, loc: torch.Tensor):
        loc = loc.unsqueeze(-2)
        place_cell_scores = torch.linalg.vector_norm(self.place_cell_centers - loc, dim=-1)**2
        place_cell_scores = -place_cell_scores / (2 * self.place_cell_scale ** 2)
        all_exponents = torch.logsumexp(place_cell_scores, dim=-1, keepdim=True)
        normalized_scores = place_cell_scores - all_exponents
        place_cell_activations = torch.exp(normalized_scores)
        return place_cell_activations
    
def create_sample(dat, device):
    t0_state = dat['observation'][:, 0] #t0
    t1_state = dat['observation'][:, 1] #t1
    u = t0_state / torch.linalg.vector_norm(t0_state, dim=1).unsqueeze(1)
    v = t1_state / torch.linalg.vector_norm(t1_state, dim=1).unsqueeze(1)
    I = torch.eye(u.shape[-1], device=device).unsqueeze(0)
    u_plus_v = u + v
    u_plus_v = u_plus_v.unsqueeze(-1)
    uv = torch.linalg.vecdot(u, v)
    uv = uv.unsqueeze(-1).unsqueeze(-1)
    u_extended = u.unsqueeze(-1)
    v_extended = v.unsqueeze(-1)
    uvtranspose = torch.transpose(u_plus_v, 1, 2)
    vut = 2 * v_extended * torch.transpose(u_extended, 1, 2)
    R = I - u_plus_v / (1 + uv) * uvtranspose + vut
    indices = torch.triu_indices(R.shape[-2], R.shape[-1], offset=1)
    R_input = R[:, indices[0], indices[1]] #Bx28
    T_input = torch.linalg.vector_norm(dat['observation'][:, 1] - dat['observation'][:, 0], dim=1)
    T_input = T_input.unsqueeze(-1)
    R_input = R_input
    T_input = T_input
    t0_state = t0_state
    # State + Vel + Rot
    #source = torch.cat((t0_state, T_input, R_input), dim=1)
    # State + Vel
    #source_diff = torch.cat((t0_state, T_input), dim=1)
    # State + Rot
    #source_diff = torch.cat((t0_state, R_input), dim=1)
    # State
    source = t0_state
    target_state = t1_state
#    target_trans = t1_state
#    target_rot = (t1_state - t0_state) / torch.norm(t1_state - t0_state, dim=1).unsqueeze(1)
#    target = torch.cat((target_trans, target_rot), dim=-1)
    return source, target_state

@hydra.main(config_path=".", config_name="path_integration_predict", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    num_episodes = 100
    trajectory_length = 100
    trajectory_count = trajectory_length * num_episodes
    slice_len = 100 #use first two to predict the third
    slice_count_in_batch = 100
    batch_size = slice_len * slice_count_in_batch
#    num_slices = 64 #per sample, how many trajectories.
#    batch_size = num_slices * trajectory_length
    storage_size = trajectory_count * trajectory_length
    train_sampler = SliceSampler(slice_len=slice_len)
    train_replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(storage_size, device=device),
        sampler=train_sampler,
    #    batch_size=batch_size,
    )
    trace_object = getattr(sys, 'gettrace', lambda: None)() #vscode debugging
    if trace_object is not None:
        project_root_path = "../../../"
    else:
        project_root_path = "../../../../"
    train_replay_buffer.loads(project_root_path + "train_buffer")
    

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
                "mode": cfg.logger.mode
            },
        )

    dat = train_replay_buffer.sample(100) #should be 2 full trajectories
    test_trajs = dat.reshape((1, -1))
    test_input, test_target = create_sample(test_trajs, device)
    energy_scorer = PlaceCellActivation(device)
    test_energy = energy_scorer(test_target)
    model = MLP(in_features = test_input.shape[-1], out_features = test_energy.shape[-1], num_cells = [64, 64], dropout=0.5)

    model = model.to(device)
    params = TensorDict.from_module(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_module = torch.nn.CrossEntropyLoss()

    pretrain_gradient_steps = 100000
 #   dat = train_replay_buffer.sample(batch_size)
 #   dat = dat.reshape((slice_count_in_batch, -1))
    for step in range(pretrain_gradient_steps):
        log_info = {}

        dat = train_replay_buffer.sample(batch_size)
        dat = dat.reshape((slice_count_in_batch, -1))
        input, target = create_sample(dat, device)
        target = energy_scorer(target)
        input = input.to(device)
        target = target.to(device)
#        source_diff = t0_state
        with params.to_module(model):
            predict = model(input)
        loss = loss_module(predict, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_info.update(
            {
                "loss": loss.item(),
            }
        )
        for key, value in log_info.items():
                logger.log_scalar(key, value, step)
    print("Training complete, evaluating")
    # Evaluate model
    model.eval()
    eval_steps = 10
    eval_losses = []
    for i in range(eval_steps):
        dat = train_replay_buffer.sample(batch_size)
        dat = dat.reshape((slice_count_in_batch, -1))
        input, target = create_sample(dat)
        target = energy_scorer(target)
        input = input.to(device)
        target = target.to(device)
        predict = model(input)
        loss = loss_module(predict, target)
        eval_losses.append(loss)
    print("Evaluation loss: ", sum(eval_losses) / len(eval_losses))
    #start with just predicting the next state.

    logger.experiment.summary["eval_loss"] = (sum(eval_losses) / len(eval_losses)).item()
    params.memmap("model_state")
    #start with working with just the first 100 time steps.
    #these seem to range from -10 to 10.
    print('ok')

if __name__ == "__main__":
    main()