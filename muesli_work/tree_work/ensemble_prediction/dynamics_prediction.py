import torch
import torch.nn as nn
import torch.autograd

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        A simple feedforward neural network to represent the dynamics model.
        Args:
            state_dim (int): Dimension of the state.
            action_dim (int): Dimension of the action.
            hidden_dim (int): Number of hidden units in the intermediate layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        """
        Forward pass of the dynamics model.
        Args:
            state (torch.Tensor): Current state, shape (batch_size, state_dim).
            action (torch.Tensor): Current action, shape (batch_size, action_dim).
        Returns:
            torch.Tensor: Predicted next state, shape (batch_size, state_dim).
        """
        x = torch.cat((state, action), dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def compute_ensemble_disagreement(predictions):
    """
    Computes epistemic uncertainty as the variance of predictions across an ensemble of models.
    
    Args:
        ensemble (list of nn.Module): List of dynamics models.
        state (torch.Tensor): Current state, shape (batch_size, state_dim).
        action (torch.Tensor): Current action, shape (batch_size, action_dim).
        
    Returns:
        torch.Tensor: Mean disagreement across the ensemble for each input pair, shape (batch_size,).
    """
    disagreement = torch.var(predictions, dim=-1).mean(-1, keepdim=True)
    return disagreement

def compute_jacobian_norm_ensemble(ensemble, tensordict):
    """
    Computes the average Frobenius norm of the Jacobian across an ensemble of models.
    
    Args:
        ensemble (list of nn.Module): List of dynamics models.
        state (torch.Tensor): Current state, shape (batch_size, state_dim).
        action (torch.Tensor): Current action, shape (batch_size, action_dim).
        
    Returns:
        torch.Tensor: Mean Jacobian norm across the ensemble for each input pair, shape (batch_size,).
    """
    jacobian_norms = []
    input_td = tensordict.select("observation", "action").clone()

    input_td["observation"].requires_grad_(True)
    input_td["action"].requires_grad_(True)
    output_td = ensemble(input_td)
    jacobian_state = torch.autograd.grad(
            outputs=output_td["predicted_state"],
            inputs=input_td["observation"],
            grad_outputs=torch.ones_like(output_td["predicted_state"]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
    
    jacobian_action = torch.autograd.grad(
            outputs=output_td["predicted_state"],
            inputs=input_td["action"],
            grad_outputs=torch.ones_like(output_td["predicted_state"]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
    
    norm_state = torch.norm(jacobian_state, p='fro', dim=-1) if jacobian_state is not None else torch.zeros(state.size(0))
    norm_action = torch.norm(jacobian_action, p='fro', dim=-1) if jacobian_action is not None else torch.zeros(action.size(0))
    jacobian_norms = norm_state + norm_action

    # Return mean Jacobian norm across the ensemble
    return jacobian_norms

# def compute_jacobian_norm_ensemble(ensemble, state, action):
#     """
#     Computes the average Frobenius norm of the Jacobian across an ensemble of models.
    
#     Args:
#         ensemble (list of nn.Module): List of dynamics models.
#         state (torch.Tensor): Current state, shape (batch_size, state_dim).
#         action (torch.Tensor): Current action, shape (batch_size, action_dim).
        
#     Returns:
#         torch.Tensor: Mean Jacobian norm across the ensemble for each input pair, shape (batch_size,).
#     """
#     jacobian_norms = []

#     for model in ensemble:
#         state.requires_grad_(True)
#         action.requires_grad_(True)

#         output = model(state, action)

#         # Compute gradients of output w.r.t state and action
#         jacobian_state = torch.autograd.grad(
#             outputs=output,
#             inputs=state,
#             grad_outputs=torch.ones_like(output),
#             create_graph=True,
#             retain_graph=True,
#             allow_unused=True
#         )[0]

#         jacobian_action = torch.autograd.grad(
#             outputs=output,
#             inputs=action,
#             grad_outputs=torch.ones_like(output),
#             create_graph=True,
#             retain_graph=True,
#             allow_unused=True
#         )[0]

#         # Compute Frobenius norms
#         norm_state = torch.norm(jacobian_state, p='fro', dim=-1) if jacobian_state is not None else torch.zeros(state.size(0))
#         norm_action = torch.norm(jacobian_action, p='fro', dim=-1) if jacobian_action is not None else torch.zeros(action.size(0))

#         jacobian_norms.append(norm_state + norm_action)

#     # Return mean Jacobian norm across the ensemble
#     return torch.mean(torch.stack(jacobian_norms), dim=-1)


def hybrid_nonlin_metric(ensemble, state, action, alpha=0.5):
    """
    Computes a hybrid metric combining epistemic uncertainty and nonlinearity.
    
    Args:
        ensemble (list of nn.Module): List of dynamics models.
        state (torch.Tensor): Current state, shape (batch_size, state_dim).
        action (torch.Tensor): Current action, shape (batch_size, action_dim).
        alpha (float): Weighting factor for combining Jacobian norm and ensemble disagreement.
        
    Returns:
        torch.Tensor: Hybrid metric for each input pair, shape (batch_size,).
    """
    jacobian_norm = compute_jacobian_norm_ensemble(ensemble, state, action)
    disagreement = compute_ensemble_disagreement(ensemble, state, action)
    
    return alpha * jacobian_norm + (1 - alpha) * disagreement


# Example usage
if __name__ == "__main__":
    # Parameters
    state_dim = 4
    action_dim = 2
    hidden_dim = 64
    ensemble_size = 5
    batch_size = 10

    # Create an ensemble of dynamics models
    ensemble = [DynamicsModel(state_dim, action_dim, hidden_dim) for _ in range(ensemble_size)]

    # Example inputs
    state = torch.rand(batch_size, state_dim)
    action = torch.rand(batch_size, action_dim)

    # Compute metrics
    with torch.no_grad():
        disagreement = compute_ensemble_disagreement(ensemble, state, action)
        jacobian_norm = compute_jacobian_norm_ensemble(ensemble, state, action)
        hybrid_metric = hybrid_nonlin_metric(ensemble, state, action, alpha=0.7)

    print("Ensemble Disagreement (Epistemic Uncertainty):", disagreement)
    print("Jacobian Norm (Nonlinearity):", jacobian_norm)
    print("Hybrid Metric:", hybrid_metric)
