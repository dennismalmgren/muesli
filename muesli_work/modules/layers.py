import torch

class SoftmaxLayer(torch.nn.Module):
    def __init__(self,  internal_dim: int):
        super().__init__()
        self.internal_dim = internal_dim

    def forward(self, x):
        x_shape = x.shape
        x = x.view(*x_shape[:-1], -1, self.internal_dim)
        x = torch.softmax(x, dim=-1)
        return x.view(x_shape)    
    
class ClampOperator(torch.nn.Module):
    def __init__(self, vmin, vmax):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, x):
        return torch.clamp(x, self.vmin, self.vmax) 
    
class SupportOperator(torch.nn.Module):
    def __init__(self, support):
        super().__init__()
        self.register_buffer("support", support)

    def forward(self, x):
        return (x.softmax(-1) * self.support).sum(-1, keepdim=True)
    
class ReshapeModule(torch.nn.Module):
    def __init__(self, n_action_dims = 1):
        super().__init__()
        self.n_action_dims = n_action_dims

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(*x_shape[:-1], self.internal_dim, -1)
        return x