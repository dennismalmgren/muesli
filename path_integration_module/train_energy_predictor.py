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
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
import tqdm
from train_utils import get_project_root_path_vscode
from energy_predictor_module import EnergyPredictor

class PlaceCellActivationCalculator:
    def __init__(self, place_cell_centers):
        self.place_cell_centers = place_cell_centers
        self.place_cell_centers = self.place_cell_centers.unsqueeze(0)
        self.place_cell_scale = 3

        self._place_cell_activation_dim = self.place_cell_centers.shape[1]

    @property
    def place_cell_activation_dim(self):
        return self._place_cell_activation_dim
    
    def __call__(self, loc: torch.Tensor):
        loc = loc.unsqueeze(-2)
        place_cell_scores = torch.linalg.vector_norm(self.place_cell_centers - loc, dim=-1)**2
        place_cell_scores = -place_cell_scores / (2 * self.place_cell_scale ** 2)
        all_exponents = torch.logsumexp(place_cell_scores, dim=-1, keepdim=True)
        normalized_scores = place_cell_scores - all_exponents
        place_cell_activations = torch.exp(normalized_scores)
        return place_cell_activations
    
class HeadCellActivationCalculator:
    def __init__(self, head_cell_centers):
        super().__init__()
        self.head_cell_centers = head_cell_centers
        self.head_cell_centers = self.head_cell_centers.unsqueeze(0)

        self.head_cell_concentration = 15 
        self._head_cell_activation_dim = self.head_cell_centers.shape[1]

    @property
    def head_cell_activation_dim(self):
        return self._head_cell_activation_dim
    
    def __call__(self, heading: torch.Tensor):
        heading = heading.unsqueeze(-2)
        heading = heading / torch.norm(heading)
        head_cell_scores = torch.linalg.vecdot(heading, self.head_cell_centers)
        head_cell_scores = self.head_cell_concentration * head_cell_scores
        all_exponents = torch.logsumexp(head_cell_scores, dim=-1, keepdim=True)
        normalized_scores = head_cell_scores - all_exponents
        head_cell_activations = torch.exp(normalized_scores)
        return head_cell_activations

class PlaceHeadPredictionLoss(nn.Module):
    def __init__(self, 
                 place_cell_activation_calculator,
                 head_cell_activation_calculator):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.place_cell_activation_calculator = place_cell_activation_calculator
        self.head_cell_activation_calculator = head_cell_activation_calculator
        
    def calculate_heading(self, t0_state: torch.Tensor, t1_state: torch.Tensor):        
        heading = (t1_state - t0_state) / torch.linalg.vector_norm(t1_state - t0_state, dim=-1, keepdim=True)
        return heading
    
    def forward(self, tensordict: TensorDictBase):
        head_prediction_key = "head_energy_prediction"
        place_prediction_key = "place_energy_prediction"
        observation_key = "observation"
        observation = tensordict[observation_key]
        obs_dim = observation.shape[-1]
        t0_state = observation[..., :obs_dim//2]
        t1_state = observation[..., obs_dim//2:]
        heading = self.calculate_heading(t0_state, t1_state)
        place_activations = self.place_cell_activation_calculator(t1_state)
        head_activations = self.head_cell_activation_calculator(heading)
        place_loss = self.ce_loss(tensordict[place_prediction_key], place_activations)
        head_loss = self.ce_loss(tensordict[head_prediction_key], head_activations)
        loss = place_loss + head_loss
        loss_td = TensorDict(
           {
            "place_loss": place_loss,
            "head_loss": head_loss,
            "loss": loss
        }
        )
        return loss_td
    
def create_cell_centers(replay_buffer, num_place_cells, num_head_cells, cell_seed):
    obs = replay_buffer["observation"]
    obs = torch.cat((obs[..., :obs.shape[-1] // 2], obs[..., obs.shape[-1] // 2:]), dim=-2)
    min_per_dim = torch.min(obs, dim = 0)[0] #ts x num_envs x 8
    min_per_dim = torch.min(min_per_dim, dim = 0)[0]
    max_per_dim = torch.max(obs, dim = 0)[0]
    max_per_dim = torch.max(max_per_dim, dim = 0)[0]
    generator=torch.Generator(device='cpu').manual_seed(cell_seed)
    place_cell_centers = torch.rand((num_place_cells, min_per_dim.shape[-1]), generator=generator) * (max_per_dim - min_per_dim) + min_per_dim
    head_cell_centers = torch.rand((num_head_cells, min_per_dim.shape[-1]), generator=generator) * 2 - 1 # from -1 to 1
    head_cell_centers = head_cell_centers / torch.linalg.norm(head_cell_centers, dim=-1, keepdim=True)
    return place_cell_centers, head_cell_centers

@hydra.main(config_path=".", config_name="train_energy_predictor", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    cfg_num_trajectories = cfg.rb.num_trajectories
    cfg_trajectory_length = cfg.rb.trajectory_length
    cfg_slice_len = cfg.rb.slice_len #use first to predict the second
    cfg_batch_size = cfg.rb.batch_size
    cfg_saved_rb_name = cfg.rb.saved_rb_name
    cfg_num_place_cells = cfg.cell_placement.num_place_cells
    cfg_num_head_cells = cfg.cell_placement.num_head_cells
    cfg_cell_seed = cfg.cell_placement.seed
    cfg_model_num_cells = cfg.model.num_cells
    
    transition_count = cfg_trajectory_length * cfg_num_trajectories
    storage_size = transition_count
    train_sampler = SliceSampler(slice_len=cfg_slice_len)
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(storage_size, ndim=2),
        sampler=train_sampler,
        batch_size=cfg_batch_size,
    )

    project_root_path = get_project_root_path_vscode()
    replay_buffer.loads(project_root_path + f"data/{cfg_saved_rb_name}")
    
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

    place_cell_centers, head_cell_centers = create_cell_centers(replay_buffer, cfg_num_place_cells, cfg_num_head_cells, cfg_cell_seed)
    place_cell_centers = place_cell_centers.to(device)
    head_cell_centers = head_cell_centers.to(device)
    test_place_cell_activation_calculator = PlaceCellActivationCalculator(place_cell_centers)
    test_head_cell_activation_calculator = HeadCellActivationCalculator(head_cell_centers)

    loss_module = PlaceHeadPredictionLoss(test_place_cell_activation_calculator, 
                                          test_head_cell_activation_calculator)
    test_data = replay_buffer.sample(cfg_batch_size) 
    test_observation = test_data["observation"]
    model = EnergyPredictor(test_observation.shape[-1]// 2, cfg_num_place_cells, cfg_num_head_cells, cfg_model_num_cells)
    energy_prediction_module = TensorDictModule(
        model,
        in_keys=["observation"],
        out_keys=["integration_prediction", "place_energy_prediction", "head_energy_prediction"],
    )
    energy_prediction_module = energy_prediction_module.to(device)
    #params = TensorDict.from_module(energy_prediction_module)
    optimizer = torch.optim.AdamW(energy_prediction_module.parameters(), lr=1e-3) #maybe config?
    cfg_pretrain_gradient_steps = cfg.optim.pretrain_gradient_steps #maybe config
    pbar = tqdm.tqdm(total=cfg_pretrain_gradient_steps)
    collected_steps = 0
    for step in range(cfg_pretrain_gradient_steps):
        if (step % 100 == 0):
            additional_steps = step - collected_steps
            pbar.update(additional_steps)
            collected_steps = step
        log_info = {}
        data = replay_buffer.sample(cfg_batch_size)
        data = data.to(device)
        #with params.to_module(energy_prediction_module):
        predict = energy_prediction_module(data)
        loss_td = loss_module(predict)
        head_loss = loss_td["head_loss"]
        place_loss = loss_td["place_loss"]
        loss = loss_td["loss"]
        optimizer.zero_grad()        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(energy_prediction_module.parameters(), 0.5)
        optimizer.step()
        log_info.update(
            {
                "loss": loss.item(),
                "head_loss": head_loss.item(),
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
        data = replay_buffer.sample(cfg_batch_size)
        data = data.to(device)
        #with params.to_module(energy_prediction_module):
        predict = energy_prediction_module(data)
        loss_dict = loss_module(predict)
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
    params = TensorDict.from_module(energy_prediction_module)
    params.memmap("model_state")
    #start with working with just the first 100 time steps.
    #these seem to range from -10 to 10.
    print('ok')

if __name__ == "__main__":
    main()