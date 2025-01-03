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
    
class OrdinalLogitsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sj = torch.sigmoid(x)
        log_sj = torch.log(sj)
        log_1_m_sj = torch.log(1 - sj)

        logits_sum = torch.cumsum(log_sj, dim=-1)
        
        logits_1_m_sj_flipped = torch.flip(log_1_m_sj, dims=(-1,))
        #these logits are -1, -2, ... 0

        logits_m1_sum = torch.cumsum(logits_1_m_sj_flipped, dim=-1)
        #these are -1, -1 + -2, -1 + -2 + -3, ...
        logits_inv_sum = torch.flip(logits_m1_sum, dims=(-1,))+0
        #these are 0 + 1 + .., 1 + 2 + ..
        #these are 
        logits = logits_sum
        logits[..., :-1] += logits_inv_sum[..., 1:]
        return logits
    
