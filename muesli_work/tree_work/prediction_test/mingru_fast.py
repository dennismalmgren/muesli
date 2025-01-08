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


def combine_pairs(p1, p2):
    logA1, logB1, b1 = p1[..., 0], p1[..., 1], p1[..., 2]
    logA2, logB2, b2 = p2[..., 0], p2[..., 1], p2[..., 2]

    logA_out = logA2 * (1 - b2) + logA1 * b2
    logB_out = torch.logaddexp(logA2 + logB1, logB2) * (1 - b2) + logB2 * b2
    b_out = torch.clamp(b1 + b2, max=1)
#    b_out = b1 | b2
    return torch.stack((logA_out, logB_out, b_out), dim=-1)

def associative_scan_log(log_coeffs, log_values, is_init, log_h_0):
    B, T, D = log_coeffs.shape
    assert log_coeffs.is_cuda and log_values.is_cuda and log_h_0.is_cuda, \
    "Tensors need to be stored on cuda device"

    transforms = torch.stack((log_coeffs, log_values, is_init.expand_as(log_values)), dim=-1)
    identity = torch.tensor([0.0, float('-inf'), 0.0], device='cuda').reshape(1, 1, 3)
    identity = identity.expand(B, 1, D, 3)
    transforms = torch.cat([identity, transforms], dim=1)
    output = associative_scan_fn(combine_pairs, transforms, dim=1, combine_mode="generic")

    logA_out, logB_out = output[..., 0], output[..., 1]
    h_t = torch.logaddexp(logA_out + log_h_0, logB_out)
    return h_t.exp()

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
        
        hidden, gate = self.to_hidden_and_gate(input).chunk(2, dim=-1)

        if seq_len == 1: #lets run it in inference mode
            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(h_0, hidden, gate) if h_0 is not None else (hidden * gate)
        else:
            if not self.allow_training:
                raise Exception("Cuda not available, model requires cuda for training")
            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h
            out = associative_scan_log(log_coeffs, log_values, is_init, h_0.log())
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
    