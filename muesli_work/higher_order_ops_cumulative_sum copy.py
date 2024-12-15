import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.utils import (
    _set_compilation_env,
)

def cumsum_fn(x, y):
    return x + y

def log_cumsum_exp_fn(x, y):
    return torch.logaddexp(x, y)
    
a = torch.arange(start=1, end=19, dtype=torch.float32, device='cuda').reshape(2, 9)
running_sum = torch.zeros_like(a)

a_out_example = torch.logcumsumexp(a, dim=1)
print("Expected result: ", a_out_example)


with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
    fn = torch.compile(associative_scan)

out = fn(log_cumsum_exp_fn, a, dim=1, combine_mode="generic")

print("Actual result 1: ", out)

