import time
import sys
from typing import Tuple

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
    #    self.place_cell_centers = torch.load("/home/dennismalmgren/repos/muesli/path_integration/place_cell_centers.pt").to(device)
        self.place_cell_centers = torch.load("/mnt/f/repos/muesli/path_integration/place_cell_centers.pt").to(device)
        self.place_cell_centers = self.place_cell_centers.unsqueeze(0)
        self.place_cell_scale = 3

        self._place_cell_activation_dim = self.place_cell_centers.shape[1]

    @property
    def place_cell_activation_dim(self):
        return self._place_cell_activation_dim
    
    def forward(self, loc: torch.Tensor):
        loc = loc.unsqueeze(-2)
        place_cell_scores = torch.linalg.vector_norm(self.place_cell_centers - loc, dim=-1)**2
        place_cell_scores = -place_cell_scores / (2 * self.place_cell_scale ** 2)
        all_exponents = torch.logsumexp(place_cell_scores, dim=-1, keepdim=True)
        normalized_scores = place_cell_scores - all_exponents
        place_cell_activations = torch.exp(normalized_scores)
        return place_cell_activations
    
class HeadCellActivation(nn.Module):
    def __init__(self, device):
        super().__init__()
        #self.head_cell_centers = torch.load("/home/dennismalmgren/repos/muesli/path_integration/head_cell_centers.pt").to(device)
        self.head_cell_centers = torch.load("/mnt/f/repos/muesli/path_integration/head_cell_centers.pt").to(device)
        self.head_cell_centers = self.head_cell_centers / torch.linalg.norm(self.head_cell_centers, dim=-1, keepdim=True)
        self.head_cell_centers = self.head_cell_centers.unsqueeze(0)

        self.head_cell_concentration = 15  # 20 degrees in radians
        self._head_cell_activation_dim = self.head_cell_centers.shape[1]

    @property
    def head_cell_activation_dim(self):
        return self._head_cell_activation_dim
    
    def forward(self, heading: torch.Tensor):
        heading = heading.unsqueeze(-2)
        heading = heading / torch.norm(heading)
        head_cell_scores = torch.linalg.vecdot(heading, self.head_cell_centers)
        head_cell_scores = self.head_cell_concentration * head_cell_scores
        all_exponents = torch.logsumexp(head_cell_scores, dim=-1, keepdim=True)
        normalized_scores = head_cell_scores - all_exponents
        head_cell_activations = torch.exp(normalized_scores)
        return head_cell_activations

class PlaceHeadPredictionLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.place_cell_activation = PlaceCellActivation(device)
        self.head_cell_activation = HeadCellActivation(device)
        self.device = device

    def forward(self, input: torch.Tensor, targets: Tuple[torch.Tensor]):
        head_predictions = input[:, :self.head_cell_activation.head_cell_activation_dim]
        place_predictions = input[:, self.head_cell_activation.head_cell_activation_dim:]
        place, heading = targets
        place = place.to(self.device)
        heading = heading.to(self.device)
        place_activations = self.place_cell_activation(place)
        head_activations = self.head_cell_activation(heading)
        place_loss = self.ce_loss(place_predictions, place_activations)
        head_loss = self.ce_loss(head_predictions, head_activations)
        loss = place_loss + head_loss
        losses = {
            "place_loss": place_loss,
            "head_loss": head_loss,
            "loss": loss
        }
        return losses

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
    source = torch.cat((t0_state, T_input, R_input), dim=1)
    # State + Vel
    #source_diff = torch.cat((t0_state, T_input), dim=1)
    # State + Rot
    #source_diff = torch.cat((t0_state, R_input), dim=1)
    # State
    #source = t0_state
    target_state = t1_state, t1_state - t0_state
#    target_trans = t1_state
#    target_rot = (t1_state - t0_state) / torch.norm(t1_state - t0_state, dim=1).unsqueeze(1)
#    target = torch.cat((target_trans, target_rot), dim=-1)
    return source, target_state

@hydra.main(config_path=".", config_name="path_integration_predict", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    num_episodes = 1000
    trajectory_length = 1000
    trajectory_count = trajectory_length * num_episodes
    slice_len = 2 #use first to predict the second
    slice_count_in_batch = 1000
    batch_size = slice_len * slice_count_in_batch
#    num_slices = 64 #per sample, how many trajectories.
#    batch_size = num_slices * trajectory_length
    storage_size = trajectory_count * trajectory_length
    train_sampler = SliceSampler(slice_len=slice_len)
    train_replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(storage_size),
        sampler=train_sampler,
    #    batch_size=batch_size,
    )
    trace_object = getattr(sys, 'gettrace', lambda: None)() #vscode debugging
    if trace_object is not None:
        project_root_path = "../../../"
    else:
        project_root_path = "../../../../"
    train_replay_buffer.loads(project_root_path + "trainbuffer")
    train_replay_buffer.sample(batch_size=batch_size) #1000 slices

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

    dat = train_replay_buffer.sample(batch_size) #should be 2 full trajectories
    dat = dat.to(device)
    test_trajs = dat.reshape((1, -1))
    test_input, (test_target, test_heading) = create_sample(test_trajs, device)
    test_place_cell_activation = PlaceCellActivation(device)
    test_head_cell_activation = HeadCellActivation(device)

    test_place_cell_activation = test_place_cell_activation(test_target)
    test_head_cell_activation = test_head_cell_activation(test_heading)

    model = MLP(in_features = test_input.shape[-1], 
                out_features = test_place_cell_activation.shape[-1] + test_head_cell_activation.shape[-1], 
                num_cells = [256, 256])

    model = model.to(device)
    params = TensorDict.from_module(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_module = PlaceHeadPredictionLoss(device)

    pretrain_gradient_steps = 100000
    for step in range(pretrain_gradient_steps):
        log_info = {}
        dat = train_replay_buffer.sample(batch_size)
        dat = dat.reshape((slice_count_in_batch, -1))
        dat = dat.to(device)

        input, target = create_sample(dat, device)
        input = input.to(device)
        with params.to_module(model):
            predict = model(input)
        loss_dict = loss_module(predict, target)
        head_loss = loss_dict["head_loss"]
        place_loss = loss_dict["place_loss"]
        loss = loss_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_info.update(
            {
                "loss": loss.item(),
                "heading_loss": head_loss.item(),
                "place_loss": place_loss.item(),
            }
        )
        for key, value in log_info.items():
                logger.log_scalar(key, value, step)
    print("Training complete, evaluating")
    # Evaluate model
    model.eval()
    eval_steps = 10
    eval_losses = []
    eval_head_losses = []
    eval_place_losses = []
    for i in range(eval_steps):
        dat = train_replay_buffer.sample(batch_size)
        dat = dat.reshape((slice_count_in_batch, -1))
        dat = dat.to(device)
        input, target = create_sample(dat, device)
        input = input.to(device)
        with params.to_module(model):
            predict = model(input)
        loss_dict = loss_module(predict, target)
        loss = loss_dict["loss"]
        head_loss = loss_dict["head_loss"]
        place_loss = loss_dict["place_loss"]
        eval_losses.append(loss)
        eval_head_losses.append(head_loss)
        eval_place_losses.append(place_loss)

    print("Evaluation loss: ", sum(eval_losses) / len(eval_losses))
    print("Evaluation head loss: ", sum(eval_head_losses) / len(eval_head_losses))
    print("Evaluation place loss: ", sum(eval_place_losses) / len(eval_place_losses))

    logger.experiment.summary["eval_loss"] = (sum(eval_losses) / len(eval_losses)).item()
    logger.experiment.summary["eval_head_loss"] = (sum(eval_head_losses) / len(eval_head_losses)).item()
    logger.experiment.summary["eval_place_loss"] = (sum(eval_place_losses) / len(eval_losses)).item()
    params.memmap("model_state")
    #start with working with just the first 100 time steps.
    #these seem to range from -10 to 10.
    print('ok')

if __name__ == "__main__":
    main()