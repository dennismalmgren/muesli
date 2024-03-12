import torch
from torch import nn
from torchrl.modules import MLP

class EnergyPredictor(nn.Module):
    def __init__(self, in_features: int, num_place_cells: int, num_head_cells: int, num_cells: int):
        super().__init__()

        #first MLP should take in_features + 1 (vel) + n (n - 1) / 2 (rot)
        self.path_integration_model = MLP(in_features = in_features + 1 + in_features * (in_features - 1) // 2,
                                out_features = 256, #should probably be another layer here.
                                num_cells = [num_cells],
                                activate_last_layer=True)

        self.head_energy_output_model = MLP(in_features = 256,
                                out_features = num_head_cells,
                                num_cells = [num_cells],
                                activate_last_layer=False,
                                dropout=0.5)
        
        self.place_energy_output_model = MLP(in_features = 256,
                        out_features = num_place_cells,
                        num_cells = [num_cells],
                        activate_last_layer=False,
                        dropout=0.5)

    def create_path_integration_input(self, t0_state, t1_state):
#        batch_dims = t0_state.shape[:-1]
        u = t0_state / torch.linalg.vector_norm(t0_state, dim=-1).unsqueeze(-1)
        v = t1_state / torch.linalg.vector_norm(t1_state, dim=-1).unsqueeze(-1)
        I = torch.eye(u.shape[-1], device=t0_state.device)
#        if len(batch_dims) > 0:
#            I = I.unsqueeze(0)
        u_plus_v = u + v
        u_plus_v = u_plus_v.unsqueeze(-1)
        uv = torch.linalg.vecdot(u, v)
        uv = uv.unsqueeze(-1).unsqueeze(-1)
        u_extended = u.unsqueeze(-1)
        v_extended = v.unsqueeze(-1)
        uvtranspose = torch.transpose(u_plus_v, -2, -1)
        vut = 2 * v_extended * torch.transpose(u_extended, -2, -1)
        I = I.expand_as(vut)
        R = I - u_plus_v / (1 + uv) * uvtranspose + vut
        indices = torch.triu_indices(R.shape[-2], R.shape[-1], offset=1)
        R_input = R[..., indices[0], indices[1]] #Bx28
        T_input = torch.linalg.vector_norm(t1_state - t0_state, dim=-1)
        T_input = T_input.unsqueeze(-1)
        # State + Vel + Rot
        source = torch.cat((t0_state, T_input, R_input), dim=-1)
        return source

    def forward(self, observation):
        t0_state = observation[..., :observation.shape[-1]//2] #t0
        t1_state = observation[..., observation.shape[-1]//2:] #t1
        integration_input = self.create_path_integration_input(t0_state, t1_state)
        integration_input = torch.nan_to_num(integration_input, nan=0.0, posinf=0.0, neginf=0.0)
        integration_prediction = self.path_integration_model(integration_input) #256
        #integration_prediction = torch.nan_to_num(integration_prediction, nan=0.0, posinf=0.0, neginf=0.0)
        head_energy_prediction = self.head_energy_output_model(integration_prediction)
        place_energy_prediction = self.place_energy_output_model(integration_prediction)
        return integration_prediction, place_energy_prediction, head_energy_prediction
#        policy_input = torch.cat((integration_prediction, t1_state), dim=-1) #+8
#        return policy_input