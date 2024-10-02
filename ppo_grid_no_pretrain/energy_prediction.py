import torch
from torch import nn
from tensordict import TensorDictBase

def create_place_cell_centers(obs_dim, num_place_cells, cell_seed):
    place_center_origo = torch.zeros(obs_dim)
    place_center_others = torch.eye(obs_dim) * 5
    place_cell_centers = torch.cat((place_center_origo, place_center_others), dim=0)
    return place_cell_centers

def create_head_cell_centers(obs_dim, num_head_cells, cell_seed):
    head_cell_center_origo = torch.zeros(obs_dim)
    head_cell_center_others = torch.eye(obs_dim) * 1
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
                 head_cell_activation_calculator,
                 predict_state_mse: bool):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.place_cell_activation_calculator = place_cell_activation_calculator
        self.head_cell_activation_calculator = head_cell_activation_calculator
        self.predict_heading = head_cell_activation_calculator is not None
        self.predict_place = place_cell_activation_calculator is not None
        self.predict_state = not (self.predict_heading or self.predict_place)
        self.predict_state_mse = predict_state_mse

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
            if self.predict_state_mse:
                state_prediction_key = "state_prediction"
                state_loss = self.mse_loss(tensordict[state_prediction_key], t1_state)
                loss_dict.update({
                    "state_loss_mse": state_loss,
                })
                losses.append(state_loss)
            else:
                state_prediction_key = "state_prediction"
                state_loss = self.cosine_loss(tensordict[state_prediction_key], t1_state, torch.ones(t1_state.shape[0], device=t1_state.device))
                loss_dict.update({
                    "state_loss_cosine": state_loss,
                })
                losses.append(state_loss)

        loss = sum(losses)
        loss_dict.update({
            "loss": loss,   
        })

        loss_td = TensorDict(loss_dict)

        return loss_td