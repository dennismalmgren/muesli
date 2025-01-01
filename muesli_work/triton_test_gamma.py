import torch
import triton
import triton.language as tl
import math

# Initialize the tuple for the scan
@triton.jit
def combine_fn(value_a, gamma_a, value_b, gamma_b):
    # `a` and `b` are tuples: (value, gamma_product)
    combined_value = value_a + gamma_a * value_b
    combined_gamma = gamma_a * gamma_b
    return combined_value, combined_gamma

@triton.jit
def inclusive_gamma_weighted_sum_scan(
    input_ptr, gamma_ptr, output_ptr, n_elements, stride, BLOCK_SIZE: tl.constexpr
):
    # Define program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + offsets * stride
    gamma_ptrs = gamma_ptr + offsets * stride

    # Load input data
    mask = offsets < n_elements
    values = tl.load(input_ptrs, mask=mask, other=0)  # Load values
    gammas = tl.load(gamma_ptrs, mask=mask, other=1)  # Load gammas (default is 1 for unused offsets)

    # Perform the scan with tuples
    values, gamma_products = tl.associative_scan((values, gammas), 0, combine_fn)

    # Store the results
    tl.store(output_ptr + offsets * stride, values, mask=mask)  # Only store the values (not gamma products)

# Function to launch the kernel
def run_gamma_weighted_sum_scan(values_tensor, gammas_tensor):
    assert values_tensor.is_cuda and gammas_tensor.is_cuda, "Input tensors must be on the GPU"
    assert values_tensor.shape == gammas_tensor.shape, "Values and gammas must have the same shape"
    time_dim = 1

    seq_len = values_tensor.shape[time_dim]
    BLOCK_SIZE = 2**math.ceil(math.log2(seq_len))  # Define block size
    #assert seq_len <= BLOCK_SIZE, ""
    # Allocate output tensor

    output_tensor = torch.empty_like(values_tensor)
    stride = values_tensor.stride()[time_dim]

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(seq_len, meta['BLOCK_SIZE']),)
    inclusive_gamma_weighted_sum_scan[grid](
        values_tensor, gammas_tensor, output_tensor, seq_len, stride, BLOCK_SIZE
    )

    return output_tensor

# Example usage
values_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda")
gammas_tensor = torch.tensor([[0.9, 0.9, 0.9, 0.9]], device="cuda")

output_tensor = run_gamma_weighted_sum_scan(values_tensor, gammas_tensor)

# Print result
print("Values:", values_tensor)
print("Gammas:", gammas_tensor)
print("Output:", output_tensor)
