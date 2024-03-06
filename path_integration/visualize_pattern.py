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
        self.place_cell_centers = torch.load("/home/dennismalmgren/repos/muesli/path_integration/place_cell_centers.pt").to(device)
    #    self.place_cell_centers = torch.load("/mnt/f/repos/muesli/path_integration/place_cell_centers.pt").to(device)
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
        self.head_cell_centers = torch.load("/home/dennismalmgren/repos/muesli/path_integration/head_cell_centers.pt").to(device)
        #self.head_cell_centers = torch.load("/mnt/f/repos/muesli/path_integration/head_cell_centers.pt").to(device)
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

def get_project_root_path_vscode():
    trace_object = getattr(sys, 'gettrace', lambda: None)() #vscode debugging
    if trace_object is not None:
        return "../../../"
    else:
        return "../../../../"
    
@hydra.main(config_path=".", config_name="path_integration_predict", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    device = "cpu"# if not torch.cuda.device_count() else "cuda"
  
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

    model = MLP(in_features = 37, 
                out_features = 256 + 12, 
                num_cells = [256, 256], dropout=0.5)
    model = model.to(device)
    params = TensorDict.from_module(model)
    params.load_memmap(get_project_root_path_vscode() + "model_state_dropout")
    #create test data.
    max_per_dim = torch.tensor([2.3198, 0.3236, 0.8711, 1.3061, 1.8617, 2.7261, 3.5292, 4.7789])
    min_per_dim = torch.tensor([-0.1643, -1.7658, -1.7496, -0.4820, -2.6647, -1.8809, -5.4557, -3.9380])
    #lets have a look at gridding the first two dimensions and keeping the latter ones fixed.
    mean_vector = torch.zeros(8,)
    mean_vector[2:] = (max_per_dim[2:] + min_per_dim[2:]) / 2
    #lets do 200 points
    x = torch.linspace(min_per_dim[0], max_per_dim[0], 200)
    y = torch.linspace(min_per_dim[1], max_per_dim[1], 200)

    grid = torch.meshgrid(x, y, indexing="ij")
    zero_grid = torch.zeros_like(grid[0]).unsqueeze(-1).expand(-1, -1, 6)
    
    grid = torch.stack(grid, dim=-1)
    grid = torch.cat((grid, zero_grid), dim=-1)
    #now lets generate the vectors
    vectors = grid + mean_vector
    
    #lets generate the activations
    for i in range(200):
        for j in range(200):
            input = vectors[i, j].unsqueeze(-1).expand(-1, 3)
            print('ok')

    print("Training complete, evaluating")
    # Evaluate model
    model.eval()
    eval_steps = 10
    eval_losses = []
    eval_head_losses = []
    eval_place_losses = []

    #start with working with just the first 100 time steps.
    #these seem to range from -10 to 10.
    print('ok')

if __name__ == "__main__":
    main()