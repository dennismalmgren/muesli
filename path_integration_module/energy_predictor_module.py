import torch
from torch import nn
from torchrl.modules import MLP

class EnergyPredictor(nn.Module):
    def __init__(self, 
                 in_features_obs: int,
                 in_features_act: int,
                 num_cat_frames: int,
                 num_place_cells: int, 
                 num_head_cells: int, 
                 num_energy_heads: int,
                 num_cells: int,
                 from_source: str,
                 use_dropout: bool,
                 include_action: bool):
        super().__init__()
        self.predict_heading = num_head_cells > 0
        self.predict_place = num_place_cells > 0
        self.predict_state = not (self.predict_heading or self.predict_place)
        self.from_source = from_source
        self.include_action = include_action
        #from_source == path_integration, current_state, delta

        #first MLP should take in_features + 1 (vel) + n (n - 1) / 2 (rot)
        self.obs_dim = in_features_obs // num_cat_frames
        self.act_dim = in_features_act
        if self.from_source == "path_integration":
            input_dim = self.calculate_path_integration_input_dim()
        elif self.from_source == "current_state":
            input_dim = self.obs_dim
        elif self.from_source == "old_state":
            input_dim = self.obs_dim
        elif self.from_source == "delta_state":
            input_dim = 2*self.obs_dim
        dropout = None
        if use_dropout:
            dropout = 0.5
        #this is more like the grid cell model, but too late to change.
        self.path_integration_model = MLP(in_features = input_dim,
                                out_features = num_energy_heads, #should probably be another layer here.
                                num_cells = [num_cells],
                                activate_last_layer=True)
        
        if self.predict_heading:
            self.head_energy_output_model = MLP(in_features = num_energy_heads,
                                    out_features = num_head_cells,
                                    num_cells = [num_cells],
                                    activate_last_layer=False,
                                    dropout=dropout)
        if self.predict_place:
            self.place_energy_output_model = MLP(in_features = num_energy_heads,
                            out_features = num_place_cells,
                            num_cells = [num_cells],
                            activate_last_layer=False,
                            dropout=dropout)
        if self.predict_state:
            self.state_output_model = MLP(in_features = num_energy_heads,
                            out_features = self.obs_dim,
                            num_cells = [num_cells],
                            activate_last_layer=False,
                            dropout=dropout)
                    
    def calculate_path_integration_input_dim(self):
        if not self.include_action:
            return self.obs_dim + 1 + self.obs_dim * (self.obs_dim - 1) // 2
        else:
            return self.obs_dim + 1 + self.obs_dim * (self.obs_dim - 1) // 2 + self.act_dim
        
    def calculate_current_state_input_dim(self):
        if not self.include_action:            
            return self.obs_dim
        else:
            return self.obs_dim + self.act_dim
        
    def calculate_old_state_input_dim(self, obs_dim):
        if not self.include_action:
            return obs_dim
        else:
            return obs_dim + self.act_dim
        
    def create_current_state_input(self, t0_state, t1_state):
        return t1_state
    
    def create_old_state_input(self, t0_state, t1_state):
        return t0_state
    
    def create_delta_state_input(self, t0_state, t1_state):
        return torch.cat((t0_state, (t1_state - t0_state)), dim=-1)

    def create_path_integration_input(self, t0_state, t1_state):
        u = t0_state / torch.linalg.vector_norm(t0_state, dim=-1).unsqueeze(-1)
        v = t1_state / torch.linalg.vector_norm(t1_state, dim=-1).unsqueeze(-1)

        u_plus_v = u + v
        u_plus_v = u_plus_v.unsqueeze(-1)
        uv = torch.linalg.vecdot(u, v)
        uv = uv.unsqueeze(-1).unsqueeze(-1)
        u_extended = u.unsqueeze(-1)
        v_extended = v.unsqueeze(-1)
        uvtranspose = torch.transpose(u_plus_v, -2, -1)
        vut = 2 * v_extended * torch.transpose(u_extended, -2, -1)

        I = torch.eye(u.shape[-1], device=t0_state.device)
        I = I.expand_as(vut)
        R = I - u_plus_v / (1 + uv) * uvtranspose + vut
        indices = torch.triu_indices(R.shape[-2], R.shape[-1], offset=1)
        R_input = R[..., indices[0], indices[1]] #Bx28
        T_input = torch.linalg.vector_norm(t1_state - t0_state, dim=-1)
        T_input = T_input.unsqueeze(-1)
        # State + Vel + Rot
        source = torch.cat((t0_state, T_input, R_input), dim=-1)
        return source

    def forward(self, observation, prev_action):
        t0_state = observation[..., :observation.shape[-1]//2] #t0
        t1_state = observation[..., observation.shape[-1]//2:] #t1
        if self.from_source == "path_integration":
            integration_input = self.create_path_integration_input(t0_state, t1_state)
            integration_input = torch.nan_to_num(integration_input, nan=0.0, posinf=0.0, neginf=0.0)
        elif self.from_source == "current_state":
            integration_input = self.create_current_state_input(t0_state, t1_state)
        elif self.from_source == "old_state":
            integration_input = self.create_old_state_input(t0_state, t1_state)
        elif self.from_source == "delta_state":
            integration_input = self.create_delta_state_input(t0_state, t1_state)
        if self.include_action:
            integration_input = torch.cat((integration_input, prev_action), dim=-1)

        integration_prediction = self.path_integration_model(integration_input) 
        if self.predict_state:
            state_prediction = self.state_output_model(integration_prediction)
            return integration_prediction, state_prediction
        
        if self.predict_place and not self.predict_heading:
            place_energy_prediction = self.place_energy_output_model(integration_prediction)
            return integration_prediction, place_energy_prediction

        if self.predict_heading and not self.predict_place:
            head_energy_prediction = self.head_energy_output_model(integration_prediction)
            return integration_prediction, head_energy_prediction
        
        #default:
        return integration_prediction, place_energy_prediction, head_energy_prediction