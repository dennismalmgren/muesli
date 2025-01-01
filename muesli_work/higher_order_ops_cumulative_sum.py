import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.utils import (
    _set_compilation_env,
)

def cumsum_fn(t1, t2):
    v1, s1 = t1
    v2, s2 = t2
    return v1 + v2, s1 + v2

a = torch.arange(18, dtype=torch.float32, device='cuda').reshape(2, 9)
running_sum = torch.zeros_like(a)

a_out_example = torch.cumsum(a, dim=1)
print("Expected result: ", a_out_example)


with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
    fn = torch.compile(associative_scan)

out = fn(cumsum_fn, (a, running_sum), dim=1, combine_mode="generic")

print("Actual result: ", out[0])
# print(torch.allclose(out, expected_out))

