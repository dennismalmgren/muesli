import time
import sys
from typing import Tuple
import os
import shutil

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
        self.place_cell_scale = 3 #todo: investigate impact

        self._place_cell_activation_dim = self.place_cell_centers.shape[1]

    @property
    def is_active(self):
        return len(self.place_cell_centers) > 0
    
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

        self.head_cell_concentration = 15  #todo: investigate impact
        self._head_cell_activation_dim = self.head_cell_centers.shape[1]

    @property
    def is_active(self):
        return len(self.head_cell_centers) > 0
    
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
        self.mse_loss = nn.MSELoss()
        self.place_cell_activation_calculator = place_cell_activation_calculator
        self.head_cell_activation_calculator = head_cell_activation_calculator
        self.predict_heading = head_cell_activation_calculator is not None
        self.predict_place = place_cell_activation_calculator is not None
        self.predict_state = not (self.predict_heading or self.predict_place)
        
    def calculate_heading(self, t0_state: torch.Tensor, t1_state: torch.Tensor):        
        heading = (t1_state - t0_state) / torch.linalg.vector_norm(t1_state - t0_state, dim=-1, keepdim=True)
        return heading
    
    def forward(self, tensordict: TensorDictBase):
        observation_key = "observation"
        observation = tensordict[observation_key]
        obs_dim = observation.shape[-1]
        t0_state = observation[..., :obs_dim//2]
        t1_state = observation[..., obs_dim//2:]
        
        loss_dict = {}
        losses = []
        if self.predict_place:
            place_prediction_key = "place_energy_prediction"
            place_activations = self.place_cell_activation_calculator(t1_state)
            place_loss = self.ce_loss(tensordict[place_prediction_key], place_activations)
            loss_dict.update({
                "place_loss": place_loss,
            })
            losses.append(place_loss)

        if self.predict_heading:
            head_prediction_key = "head_energy_prediction"
            heading = self.calculate_heading(t0_state, t1_state)
            head_activations = self.head_cell_activation_calculator(heading)
            head_loss = self.ce_loss(tensordict[head_prediction_key], head_activations)
            loss_dict.update({
                "head_loss": head_loss,
            })
            losses.append(head_loss)

        if self.predict_state:
            state_prediction_key = "state_prediction"
            state_loss = self.mse_loss(tensordict[state_prediction_key], t1_state)
            loss_dict.update({
                "state_loss": state_loss,
            })
            losses.append(state_loss)

        loss = sum(losses)
        loss_dict.update({
            "loss": loss,   
        })

        loss_td = TensorDict(loss_dict)

        return loss_td
    
def create_place_cell_centers(replay_buffer, num_place_cells, cell_seed):
    obs = replay_buffer["observation"]
    obs = torch.cat((obs[..., :obs.shape[-1] // 2], obs[..., obs.shape[-1] // 2:]), dim=-2)
    min_per_dim = torch.min(obs, dim = 0)[0] #ts x num_envs x 8
    min_per_dim = torch.min(min_per_dim, dim = 0)[0]
    max_per_dim = torch.max(obs, dim = 0)[0]
    max_per_dim = torch.max(max_per_dim, dim = 0)[0]
    generator=torch.Generator(device='cpu').manual_seed(cell_seed)
    place_cell_centers = torch.rand((num_place_cells, min_per_dim.shape[-1]), generator=generator) * (max_per_dim - min_per_dim) + min_per_dim
    return place_cell_centers

def create_head_cell_centers(replay_buffer, num_head_cells, cell_seed):
    obs = replay_buffer["observation"]
    obs = torch.cat((obs[..., :obs.shape[-1] // 2], obs[..., obs.shape[-1] // 2:]), dim=-2)
    min_per_dim = torch.min(obs, dim = 0)[0] #ts x num_envs x 8
    min_per_dim = torch.min(min_per_dim, dim = 0)[0]
    max_per_dim = torch.max(obs, dim = 0)[0]
    max_per_dim = torch.max(max_per_dim, dim = 0)[0]
    generator=torch.Generator(device='cpu').manual_seed(cell_seed)
    head_cell_centers = torch.rand((num_head_cells, min_per_dim.shape[-1]), generator=generator) * 2 - 1 # from -1 to 1
    head_cell_centers = head_cell_centers / torch.linalg.norm(head_cell_centers, dim=-1, keepdim=True)
    return head_cell_centers

@hydra.main(config_path=".", config_name="train_energy_predictor", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    cfg_num_trajectories = cfg.rb.num_trajectories
    cfg_trajectory_length = cfg.rb.trajectory_length
    cfg_slice_len = cfg.rb.slice_len #use first to predict the second
    cfg_batch_size = cfg.rb.batch_size
    cfg_saved_rb_base_path = cfg.rb.saved_base_path
    cfg_saved_rb_name = cfg.rb.saved_dir
    cfg_num_place_cells = cfg.energy_prediction.num_place_cells
    cfg_num_head_cells = cfg.energy_prediction.num_head_cells
    cfg_num_energy_heads = cfg.energy_prediction.num_energy_heads
    cfg_cell_seed = cfg.energy_prediction.seed
    cfg_model_num_cells = cfg.energy_prediction.num_cells
    cfg_num_cat_frames = cfg.rb.num_cat_frames
    cfg_predict_heading = cfg.energy_prediction.num_head_cells > 0
    cfg_predict_place = cfg.energy_prediction.num_place_cells > 0
    cfg_predict_state = not (cfg_predict_heading or cfg_predict_place)
    cfg_from_source = cfg.energy_prediction.from_source
    cfg_use_dropout = cfg.energy_prediction.use_dropout
    cfg_include_action = cfg.energy_prediction.include_action
    transition_count = cfg_trajectory_length * cfg_num_trajectories
    storage_size = transition_count
    train_sampler = SliceSampler(slice_len=cfg_slice_len)
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(storage_size, ndim=2),
        sampler=train_sampler,
        batch_size=cfg_batch_size,
    )

    project_root_path = get_project_root_path_vscode()
    replay_buffer.loads(project_root_path + f"{cfg_saved_rb_base_path}/{cfg_saved_rb_name}")
    
    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("EModel", f"{cfg.logger.exp_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="energy_model",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
                "mode": cfg.logger.mode
            },
        )
    place_cell_activation_calculator = None
    if cfg_predict_place:
        place_cell_centers = create_place_cell_centers(replay_buffer, cfg_num_place_cells, cfg_cell_seed)
        place_cell_centers = place_cell_centers.to(device)
        place_cell_activation_calculator = PlaceCellActivationCalculator(place_cell_centers)
    head_cell_activation_calculator = None
    if cfg_predict_heading:
        head_cell_centers = create_head_cell_centers(replay_buffer, cfg_num_head_cells, cfg_cell_seed)
        head_cell_centers = head_cell_centers.to(device)
        head_cell_activation_calculator = HeadCellActivationCalculator(head_cell_centers)

    
    loss_module = PlaceHeadPredictionLoss(place_cell_activation_calculator, 
                                          head_cell_activation_calculator)
    
    test_data = replay_buffer.sample(cfg_batch_size) 
    test_observation = test_data["observation"]
    test_prev_action = test_data["prev_action"]
    model = EnergyPredictor(test_observation.shape[-1], 
                            test_prev_action.shape[-1],
                            cfg_num_cat_frames, 
                            cfg_num_place_cells, 
                            cfg_num_head_cells, 
                            cfg_num_energy_heads,
                            cfg_model_num_cells,
                            cfg_from_source,
                            cfg_use_dropout,
                            cfg_include_action)
    out_keys =["integration_prediction"]
    if cfg_predict_heading:
        out_keys.append("head_energy_prediction") 
    if cfg_predict_place:
        out_keys.append("place_energy_prediction")
    if cfg_predict_state:
        out_keys.append("state_prediction")

    energy_prediction_module = TensorDictModule(
        model,
        in_keys=["observation"],
        out_keys=out_keys,
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

        predict = energy_prediction_module(data)
        loss_td = loss_module(predict)
        
        if cfg_predict_heading:
            head_loss = loss_td["head_loss"]
            log_info.update({
                "head_loss": head_loss.detach().cpu().item(),
            })
        if cfg_predict_place:
            place_loss = loss_td["place_loss"]
            log_info.update({
                "place_loss": place_loss.detach().cpu().item(),
            })
        if cfg_predict_state:
            state_loss = loss_td["state_loss"]
            log_info.update({
                "state_loss": state_loss.detach().cpu().item(),
            })

        loss = loss_td["loss"]
        optimizer.zero_grad()        
        loss.backward()

        optimizer.step()

     
        for key, value in log_info.items():
                logger.log_scalar(key, value, step)
    print("Training complete, evaluating")
    # Evaluate model
    model.eval()
    eval_steps = 10
    eval_losses = []
    eval_head_losses = []
    eval_place_losses = []
    eval_state_losses = []
    for i in range(eval_steps):
        data = replay_buffer.sample(cfg_batch_size)
        data = data.to(device)
        #with params.to_module(energy_prediction_module):
        predict = energy_prediction_module(data)
        loss_td = loss_module(predict)
        
        if cfg_predict_heading:
            head_loss = loss_td["head_loss"]
            eval_head_losses.append(head_loss.detach().cpu().item())

        if cfg_predict_place:
            place_loss = loss_td["place_loss"]
            eval_place_losses.append(place_loss.detach().cpu().item())

        if cfg_predict_state:
            state_loss = loss_td["state_loss"]
            eval_state_losses.append(state_loss.detach().cpu().item())

        loss = loss_td["loss"]
        eval_losses.append(loss.detach().cpu().item())
      

    print("Evaluation loss: ", sum(eval_losses) / len(eval_losses))
    logger.experiment.summary["eval_loss"] = (sum(eval_losses) / len(eval_losses))
    if cfg_predict_place:
        print("Evaluation place loss: ", sum(eval_place_losses) / len(eval_place_losses))
        logger.experiment.summary["eval_place_loss"] = (sum(eval_place_losses) / len(eval_losses))
    if cfg_predict_heading:
        print("Evaluation head loss: ", sum(eval_head_losses) / len(eval_head_losses))
        logger.experiment.summary["eval_head_loss"] = (sum(eval_head_losses) / len(eval_head_losses))
    if cfg_predict_state:
        print("Evaluation state loss: ", sum(eval_state_losses) / len(eval_state_losses))
        logger.experiment.summary["eval_state_loss"] = (sum(eval_state_losses) / len(eval_state_losses))

    metadata = TensorDict({})
    metadata["num_place_cells"] = torch.tensor(cfg_num_place_cells)
    metadata["num_head_cells"] = torch.tensor(cfg_num_head_cells)
    if cfg_predict_heading:
        metadata["head_cell_centers"] = head_cell_centers.cpu()
        metadata["head_cell_concentration"] = torch.tensor(head_cell_activation_calculator.head_cell_concentration)
    if cfg_predict_place:
        metadata["place_cell_centers"] = place_cell_centers.cpu()
        metadata["place_cell_scale"] = torch.tensor(place_cell_activation_calculator.place_cell_scale)
    metadata["num_energy_heads"] = torch.tensor(cfg_num_energy_heads)
    metadata["num_cells"] = torch.tensor(cfg_model_num_cells)
    metadata["num_cat_frames"] = torch.tensor(cfg_num_cat_frames)
    metadata["from_source"] = cfg_from_source
    metadata["use_dropout"] = torch.tensor(cfg_use_dropout)
    metadata["include_action"] = torch.tensor(cfg_include_action)
    params = TensorDict.from_module(energy_prediction_module)
    model_dir = project_root_path + f"{cfg.artefacts.model_save_base_path}/{cfg.artefacts.model_save_dir}"
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    params.memmap(project_root_path + f"{cfg.artefacts.model_save_base_path}/{cfg.artefacts.model_save_dir}/model_params")
    metadata.memmap(project_root_path + f"{cfg.artefacts.model_save_base_path}/{cfg.artefacts.model_save_dir}/model_metadata")

    #start with working with just the first 100 time steps.
    #these seem to range from -10 to 10.
    print('Model saved')

if __name__ == "__main__":
    main()