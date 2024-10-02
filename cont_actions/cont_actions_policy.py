import torch

from torch import nn

class ActorDefaultNet(nn.Module):
    def __init__(self, noise_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(3 + noise_dim, 10)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        self.linear2 = nn.Linear(10, 10)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)        
        self.output = nn.Linear(10, 3)
        torch.nn.init.kaiming_normal_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

        self.activation_fun = nn.ELU()
        self.noise_dim = noise_dim

    def forward(self, x):
        batch_dim = x.shape[:-1]
        noise_size = (*batch_dim, self.noise_dim)
        noise = torch.randn(noise_size, device=x.device)
        input = torch.cat((x, noise), dim=-1)
        hidden = self.linear1(input)
        hidden = self.activation_fun(hidden)
        hidden = self.linear2(hidden)
        hidden = self.activation_fun(hidden)
        out = self.output(hidden)
        out = nn.functional.softmax(out, dim=-1)
        return out
    