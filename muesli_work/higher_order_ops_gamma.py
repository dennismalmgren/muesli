import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.utils import (
    _set_compilation_env,
)

def combine_fn(x_a, x_b):
    value_a, gamma_a = x_a
    value_b, gamma_b = x_b
    # `a` and `b` are tuples: (value, gamma_product)
    combined_value = value_a + gamma_a * value_b
    combined_gamma = gamma_a * gamma_b
    return combined_value, combined_gamma

# Example usage
values_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda")
gammas_tensor = torch.tensor([[0.9, 0.9, 0.9, 0.9]], device="cuda")

with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
    fn = torch.compile(associative_scan)

output_tensor = fn(combine_fn, (values_tensor, gammas_tensor), dim=1)

# Print result
print("Values:", values_tensor)
print("Gammas:", gammas_tensor)
print("Output:", output_tensor)