import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import triton
import triton.language as tl
import math
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.utils import (
    _set_compilation_env,
)


def expand_as_right(
    tensor: torch.Tensor,
    dest: torch.Tensor,
) -> torch.Tensor:
    """Expand a tensor on the right to match another tensor shape.
        Proudly stolen from github.com/pytorch/tensordict -Dennis Malmgren

    Args:
        tensor: tensor to be expanded
        dest: tensor providing the target shape

    Returns:
         a tensor with shape matching the dest input tensor shape.

    Examples:
        >>> tensor = torch.zeros(3,4)
        >>> dest = torch.zeros(3,4,5)
        >>> print(expand_as_right(tensor, dest).shape)
        torch.Size([3,4,5])

    """
    if dest.ndimension() < tensor.ndimension():
        raise RuntimeError(
            "expand_as_right requires the destination tensor to have less "
            f"dimensions than the input tensor, got"
            f" tensor.ndimension()={tensor.ndimension()} and "
            f"dest.ndimension()={dest.ndimension()}"
        )
    if any(
        tensor.shape[i] != dest.shape[i] and tensor.shape[i] != 1
        for i in range(tensor.ndimension())
    ):
        raise RuntimeError(
            f"tensor shape is incompatible with dest shape, "
            f"got: tensor.shape={tensor.shape}, dest={dest.shape}"
        )
    for _ in range(dest.ndimension() - tensor.ndimension()):
        tensor = tensor.unsqueeze(-1)
    return tensor.expand(dest.shape)

def combine_pairs(p1, p2):
    logA1, logB1 = p1[..., 0], p1[..., 1]
    logA2, logB2 = p2[..., 0], p2[..., 1]
    
    logA_out = logA2 + logA1
    logB_out = torch.logaddexp(logA2 + logB1, logB2)

    return torch.stack((logA_out, logB_out), dim=-1)

with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
    associative_scan_fn = torch.compile(associative_scan)

def associative_scan_log(log_coeffs, log_values, log_h_0):
    B, T, D = log_coeffs.shape

    log_coeffs = log_coeffs.to("cuda")
    log_values = log_values.to("cuda")
    log_h_0 = log_h_0.to("cuda")
    transforms = torch.stack((log_coeffs, log_values), dim=-1)
    identity = torch.tensor([0.0, float('-inf')], device='cuda').reshape(1, 1, 2)
    identity = identity.expand(B, 1, D, 2)
    transforms = torch.cat([identity, transforms], dim=1)
    output = associative_scan_fn(combine_pairs, transforms, dim=1, combine_mode="generic")
    h_t = torch.logaddexp(output[..., 0] + log_h_0, output[..., 1])
    return h_t.exp().to("cpu")

def default(v, d):
    return v if v is not None else d

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

class MinGRU(nn.Module):
    """ Usage:
       predictor_gru = MinGRU(input_dim, hidden_dim, output_dim (Optional))

       Training mode:
       sequence = torch.tensor([[[1., 2, 3, 4]]], device="cuda") # B x T x dim
       targets = torch.tensor([[[2., 3, 4, 5]]], device="cuda) # B x T x dim
       output, hidden = predictor_gru(sequence)
       loss = torch.nn.functional.mse_loss(output, targets)

       Inference mode:
       init = torch.zeros(1, 1, 6) # B x T x dim
        inputs = torch.tensor([[[1., 2, 3, 4]]]) # B x T x dim, T = 1!
        output, hidden = predictor_gru(inputs, init)
        
    """

    def __init__(self, input_dim, hidden_dim, output_dim = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.to_hidden_and_gate = nn.Linear(input_dim, hidden_dim * 2, bias = False)
        self.to_out = nn.Linear(hidden_dim, output_dim, bias = False) if output_dim else nn.Identity()

    def forward(self, input: torch.Tensor, h_0 = None):
        missing_batch = False
        #if we don't send in a 'starting value, we assume we're in a training phase and should start with zeros.
        #that's not a great assumption so this should probably be changed.
        training = h_0 is None 
        time_dim = 1 #Very strong assumption
        if not training and len(input.shape) < 2:
            missing_batch = True
            input = input.unsqueeze(0)
            h_0 = h_0.unsqueeze(0)
            is_init = is_init.unsqueeze(0)
        
        if not training and len(input.shape) < 3:
            input = input.unsqueeze(time_dim) #add time dim
        
        batch = input.shape[0]
        seq_len = input.shape[1]

        if h_0 is None:
            h_0 = torch.zeros((batch, 1, self.hidden_dim), dtype=torch.float32, device=input.device)
        else:
            h_0 = torch.where(expand_as_right(is_init, h_0), 0, h_0)
            
        if not training and len(h_0.shape) < 3:
            h_0 = h_0.unsqueeze(time_dim) #add time dim

        hidden, gate = self.to_hidden_and_gate(input).chunk(2, dim = -1)
        
        if seq_len == 1:
            # handle sequential. Can be run on CPU
            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(h_0, hidden, gate) if h_0 is not None else (hidden * gate)
        else:
            # parallel. Only on GPU
            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h
            out = associative_scan_log(log_coeffs, log_values, h_0.log())
            out = out[:, -seq_len:]

        h_n = out[:, -1:]

        out = self.to_out(out)

        if not training:
            out = out.squeeze(1)
            h_n = h_n.squeeze(1)

        if missing_batch:
            out = out.squeeze(0)
            h_n = h_n.squeeze(0)

        return out, h_n