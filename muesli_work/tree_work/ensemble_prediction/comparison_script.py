import torch
import matplotlib.pyplot as plt

# Define a nonlinear function with a controllable Lipschitz constant
def nonlinear_function(x, alpha=1.0):
    """
    A nonlinear function with a tunable Lipschitz constant.
    Args:
        x (torch.Tensor): Input tensor.
        alpha (float): Controls the nonlinearity of the function.
    Returns:
        torch.Tensor: Output of the nonlinear function.
    """
    return torch.sin(alpha * x) + 0.5 * x

# Analytical derivative (Jacobian) of the nonlinear function
def jacobian_nonlinear_function(x, alpha=1.0):
    """
    Computes the analytical Jacobian (derivative) of the nonlinear function.
    Args:
        x (torch.Tensor): Input tensor.
        alpha (float): Controls the nonlinearity of the function.
    Returns:
        torch.Tensor: Jacobian (derivative) of the nonlinear function.
    """
    return alpha * torch.cos(alpha * x) + 0.5

# Generate a range of input values
x_values = torch.linspace(-3, 3, 5, requires_grad=True)  # 30 points for a small experiment
alpha = 2.0  # Control the nonlinearity

# Compute the true outputs of the nonlinear function
y_values = nonlinear_function(x_values, alpha)

# Compute the Jacobian norm as an estimate of the Lipschitz constant
jacobian_norm = torch.abs(jacobian_nonlinear_function(x_values, alpha))

# Compute the state difference-based Lipschitz constant
x_diffs = x_values[1:] - x_values[:-1]
y_diffs = y_values[1:] - y_values[:-1]
state_diff_lipschitz = torch.abs(y_diffs / (x_diffs + 1e-6))  # Add epsilon for numerical stability

# Compute the true Lipschitz constant (maximum of the Jacobian norm)
true_lipschitz_constant = jacobian_norm.max().item()

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(x_values.detach(), jacobian_norm.detach(), label="Jacobian Norm (Local)")
plt.plot(x_values[:-1].detach(), state_diff_lipschitz.detach(), label="State Diff (Empirical)", linestyle="dashed")
plt.axhline(true_lipschitz_constant, color="red", linestyle=":", label="True Lipschitz Constant")
plt.title("Comparison of Lipschitz Constant Estimates")
plt.xlabel("x")
plt.ylabel("Lipschitz Constant Estimate")
plt.legend()
plt.grid(True)
plt.show()
