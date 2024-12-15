import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.utils import (
    _set_compilation_env,
)

a = torch.arange(18, dtype=torch.float32, device='cuda').reshape(2, 9)

expected_out = torch.tensor([
    [ 36.,  36.,  35.,  33.,  30.,  26.,  21.,  15.,   8.],
    [117., 108.,  98.,  87.,  75.,  62.,  48.,  33.,  17.],
], dtype=torch.float32, device='cuda')


with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
    fn = torch.compile(associative_scan)

out = fn(lambda x, y: x + y, a, dim=1, combine_mode="generic", reverse=True)
print(out)
print(torch.allclose(out, expected_out))

