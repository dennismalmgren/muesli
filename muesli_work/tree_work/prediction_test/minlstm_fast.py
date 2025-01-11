import torch
from torch import nn
from torch.nn import functional as F

from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.utils import (
    _set_compilation_env
)
from torchrl.modules import recurrent_mode


with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
    associative_scan_fn = torch.compile(associative_scan)

def combine_pairs_no_log(p1, p2):
    A1, B1, b1 = p1[..., 0], p1[..., 1], p1[..., 2]
    A2, B2, b2 = p2[..., 0], p2[..., 1], p2[..., 2]
    # A_merged = A2 * A1 
    # B_merged = A2 * B1 + B2
    A_merged = torch.where(b2.bool(), A2, A2 * A1)
    B_merged = torch.where(b2.bool(), B2, A2 * B1 + B2) 
    b_out = torch.clamp(b1 + b2, max=1)
    return torch.stack((A_merged, B_merged, b_out), dim=-1)

def associative_scan_no_log_gru(coeffs, values, is_init, h_0):
    B, T, D = coeffs.shape
    assert coeffs.is_cuda and values.is_cuda and h_0.is_cuda, \
    "Tensors need to be stored on cuda device"

    gate_hidden = coeffs * values
    m_coeffs = 1 - coeffs
    transforms = torch.stack((m_coeffs, gate_hidden, is_init.expand_as(values)), dim=-1)
    identity = torch.tensor([1.0, 0.0, 0.0], device='cuda').reshape(1, 1, 3)
    identity = identity.expand(B, 1, D, 3)
    transforms = torch.cat([identity, transforms], dim=1)
    output = associative_scan_fn(combine_pairs_no_log, transforms, dim=1, combine_mode="generic")

    A_out, B_out = output[..., 0], output[..., 1]
    A_out = A_out[:, -T:]
    B_out = B_out[:, -T:]
    h_t = A_out * h_0 + B_out
    return h_t

def associative_scan_no_log_lstm(f_t, i_t, h_t, is_init, h_0):
    B, T, D = f_t.shape
    assert f_t.is_cuda and i_t.is_cuda and h_t.is_cuda and h_0.is_cuda, \
    "Tensors need to be stored on cuda device"

    gate_hidden = i_t * h_t
    transforms = torch.stack((f_t, gate_hidden, is_init.expand_as(f_t)), dim=-1)
    identity = torch.tensor([1.0, 0.0, 0.0], device='cuda').reshape(1, 1, 3)
    identity = identity.expand(B, 1, D, 3)
    transforms = torch.cat([identity, transforms], dim=1)
    output = associative_scan_fn(combine_pairs_no_log, transforms, dim=1, combine_mode="generic")

    A_out, B_out = output[..., 0], output[..., 1]
    h_t = A_out * h_0 + B_out
    return h_t

def default(v, d):
    return v if v is not None else d

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

class MinGRU(nn.Module):
    """
        Usage:
            predictor_gru = MinGRU(input_dim, hidden_dim, output_dim (Optional))

            Training mode:
            sequence = torch.tensor([[[1., 2, 3, 4]]], device="cuda") # B x T X input_dim
            targets = torch.tensor([[[, 2., 3, 4, 5]]], device="cuda") # B x T x output_dim
            output, hidden = predictor_gru(sequence) #DIMS?
            loss = torch.nn.functional.mse_loss(output, targets)

            Inference mode:
            init = torch.zeros(1, 1, 6) # B x T x dim
            inputs = torch.tensor([[[1., 2, 3, 4]]]) # B x T x input_dim, T == 1
            output, hidden = predictor_gru(inputs, init) # B x T x output_dim
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.allow_training = torch.cuda.is_available()
        self.to_hidden_and_gate = nn.Linear(input_dim, hidden_dim * 2, bias = False, device=device)
        self.to_out = nn.Linear(hidden_dim, output_dim, bias=False, device=device) if output_dim else nn.Identity(device=device)

    def forward(self, input: torch.Tensor, is_init: torch.Tensor = None, h_0: torch.Tensor = None, traj_ids: torch.Tensor = None):
        r_mode = recurrent_mode()
        # We want inputs to be B x T x dim
        # But h_0 is B x 1 x dim
        missing_batch = False
        missing_time = False
        time_dim = -2 #assuming last dim is flat, for now

        if len(input.shape) < 2: #single element
            missing_batch = True
            input = input.unsqueeze(0)
            if h_0 is not None:
                h_0 = h_0.unsqueeze(0)
            if is_init is not None:
                is_init = is_init.unsqueeze(0)

        if len(input.shape) < 3: #single element
            missing_time = True
            if r_mode:
                input = input.unsqueeze(0) # assume what we got is a sequence
                if h_0 is not None:
                    h_0 = h_0.unsqueeze(0)
                if is_init is not None:
                    is_init = is_init.unsqueeze(0)
            else:
                input = input.unsqueeze(time_dim)
                if h_0 is not None:
                    h_0 = h_0.unsqueeze(time_dim)
                if is_init is not None:
                    is_init = is_init.unsqueeze(time_dim)


        batch = input.shape[0]
        seq_len = input.shape[1]

        if h_0 is None:
            h_0 = torch.zeros((batch, 1, self.hidden_dim), dtype=torch.float32, device=input.device)
        elif r_mode:
            #h_0 = torch.zeros((batch, 1, self.hidden_dim), dtype=torch.float32, device=input.device)
            B, T = input.shape[:2]
            boundaries = torch.where(traj_ids[1:] != traj_ids[:-1])[0] + 1
            episode_start_indices = torch.cat((torch.tensor([0], device=boundaries.device), boundaries))
            selected_h0 = h_0[:, episode_start_indices, :]  # Shape: [1, num_episodes, dim]
            lengths = torch.diff(torch.cat((episode_start_indices, torch.tensor([T], device=boundaries.device))))  # Shape: [num_episodes]
            repeated_h_0 = torch.cat([selected_h0[:, i:i+1, :].repeat(1, lengths[i], 1) for i in range(len(lengths))], dim=1)  # Shape: [1, T, dim]

            #expect B x T x dim.
            h_0 = repeated_h_0
        hidden, gate = self.to_hidden_and_gate(input).chunk(2, dim=-1)

        if seq_len == 1: #lets run it in inference mode
            #hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(h_0, hidden, gate)
        else:
            if not self.allow_training:
                raise Exception("Cuda not available, model requires cuda for training")
            gate = gate.sigmoid()
            
            out = associative_scan_no_log_gru(gate, hidden, is_init, h_0)
            

        if r_mode:
            h_n = out #save all
        else:
            h_n = out[:, -1:] #save the last.

        out = self.to_out(out)

        if missing_time:
            if r_mode:
                out = out.squeeze(0)
                h_n = h_n.squeeze(0)
            else:
                out = out.squeeze(time_dim)
                h_n = h_n.squeeze(time_dim)
            
        if missing_batch:
            out = out.squeeze(0)
            h_n = h_n.squeeze(0)
    
        return out, h_n
    


class MinLSTM(nn.Module):
    """
        Usage:
            predictor_gru = MinGRU(input_dim, hidden_dim, output_dim (Optional))

            Training mode:
            sequence = torch.tensor([[[1., 2, 3, 4]]], device="cuda") # B x T X input_dim
            targets = torch.tensor([[[, 2., 3, 4, 5]]], device="cuda") # B x T x output_dim
            output, hidden = predictor_gru(sequence) #DIMS?
            loss = torch.nn.functional.mse_loss(output, targets)

            Inference mode:
            init = torch.zeros(1, 1, 6) # B x T x dim
            inputs = torch.tensor([[[1., 2, 3, 4]]]) # B x T x input_dim, T == 1
            output, hidden = predictor_gru(inputs, init) # B x T x output_dim
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.allow_training = torch.cuda.is_available()
        self.to_hidden_and_gate = nn.Linear(input_dim, hidden_dim * 3, bias = False, device=device)
        self.to_out = nn.Linear(hidden_dim, output_dim, bias=False, device=device) if output_dim else nn.Identity(device=device)

    def forward(self, input: torch.Tensor, is_init: torch.Tensor = None, h_0: torch.Tensor = None):
        r_mode = recurrent_mode()
        # We want inputs to be B x T x dim
        # But h_0 is B x 1 x dim
        missing_batch = False
        missing_time = False
        time_dim = -2 #assuming last dim is flat, for now

        if len(input.shape) < 2: #single element
            missing_batch = True
            input = input.unsqueeze(0)
            if h_0 is not None:
                h_0 = h_0.unsqueeze(0)
            if is_init is not None:
                is_init = is_init.unsqueeze(0)

        if len(input.shape) < 3: #single element
            missing_time = True
            if r_mode:
                input = input.unsqueeze(0) # assume what we got is a sequence
                if h_0 is not None:
                    h_0 = h_0[0].unsqueeze(0)
                    h_0 = h_0.unsqueeze(0)
                if is_init is not None:
                    is_init = is_init.unsqueeze(0)
            else:
                input = input.unsqueeze(time_dim)
                if h_0 is not None:
                    h_0 = h_0.unsqueeze(time_dim)
                if is_init is not None:
                    is_init = is_init.unsqueeze(time_dim)

        batch = input.shape[0]
        seq_len = input.shape[1]

        if h_0 is None:
            h_0 = torch.zeros((batch, 1, self.hidden_dim), dtype=torch.float32, device=input.device)
        
        hiddenf, hidden_i, hidden_t = self.to_hidden_and_gate(input).chunk(3, dim=-1)

        if seq_len == 1: 
            f_t = hiddenf.sigmoid()
            i_t = hidden_i.sigmoid()
            out = f_t * h_0 + i_t * hidden_t
        else:
            if not self.allow_training:
                raise Exception("Cuda not available, model requires cuda for training")
            f_t = hiddenf.sigmoid()
            i_t = hidden_i.sigmoid()
            out = associative_scan_no_log_lstm(f_t, i_t, hidden_t, is_init, h_0)
            out = out[:, -seq_len:]

        if r_mode:
            h_n = out #save all
        else:
            h_n = out[:, -1:] #save the last.

        out = self.to_out(out)

        if missing_time:
            if r_mode:
                out = out.squeeze(0)
                h_n = h_n.squeeze(0)
            else:
                out = out.squeeze(time_dim)
                h_n = h_n.squeeze(time_dim)
            
        if missing_batch:
            out = out.squeeze(0)
            h_n = h_n.squeeze(0)
    
        return out, h_n