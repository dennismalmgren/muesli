import torch
import matplotlib.pyplot as plt

def true_bump_function(x, sigma=1.0):
    """
    A compactly supported smooth bump function.
    Args:
        x (torch.Tensor): Input tensor.
        sigma (float): Width of the bump.
    Returns:
        torch.Tensor: Output of the bump function.
    """
    out = torch.zeros_like(x)
    mask = torch.abs(x) < sigma
    out[mask] = torch.exp(-1.0 / (1.0 - (x[mask] / sigma)**2))
    return out

def jacobian_true_bump_function(x, sigma=1.0):
    """
    Computes the analytical derivative (Jacobian) of the bump function.
    Args:
        x (torch.Tensor): Input tensor.
        sigma (float): Width of the bump.
    Returns:
        torch.Tensor: Jacobian of the bump function.
    """
    out = torch.zeros_like(x)
    mask = torch.abs(x) < sigma
    out[mask] = (2 * x[mask] / sigma**2) * torch.exp(-1.0 / (1.0 - (x[mask] / sigma)**2)) / (1.0 - (x[mask] / sigma)**2)**2
    return out

# Generate a range of input values
x_values = torch.linspace(-2, 30, 5, requires_grad=True)
sigma = 1.0  # Support of the bump function

# Compute the true outputs of the bump function
y_values = true_bump_function(x_values, sigma)

# Compute the Jacobian norm as an estimate of the Lipschitz constant
jacobian_norm = torch.abs(jacobian_true_bump_function(x_values, sigma))

# Compute the state difference-based Lipschitz constant
x_diffs = x_values[1:] - x_values[:-1]
y_diffs = y_values[1:] - y_values[:-1]
state_diff_lipschitz = torch.abs(y_diffs / (x_diffs + 1e-6))  # Add epsilon for stability

# Compute the true Lipschitz constant (maximum of the Jacobian norm)
true_lipschitz_constant = jacobian_norm.max().item()

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x_values.detach(), jacobian_norm.detach(), label="Jacobian Norm (Local)")
plt.plot(x_values[:-1].detach(), state_diff_lipschitz.detach(), label="State Diff (Empirical)", linestyle="dashed")
plt.axhline(true_lipschitz_constant, color="red", linestyle=":", label="True Lipschitz Constant")
plt.title("Comparison of Lipschitz Constant Estimates (True Bump Function)")
plt.xlabel("x")
plt.ylabel("Lipschitz Constant Estimate")
plt.legend()
plt.grid(True)
plt.show()
