import math

import torch
import matplotlib.pyplot as plt

import torch.random


R = 100
dim = 3
R_tensor = torch.tensor(R)
sqrt_two = torch.sqrt(torch.tensor(2))

#Sample bias
b = torch.rand((R, 1)) * 2 * torch.pi
wave_length_stddev = 1.0

wavelengths = torch.randn((R, dim)) * wave_length_stddev **2 + torch.zeros((R, dim)) #gaussian with mean zero and designated std dev 
if dim == 1:
    x = torch.tensor([[5.5]])
    y = torch.tensor([[6.0]])
elif dim == 2:
    x = torch.tensor([[5.5, 5.5]])
    y = torch.tensor([[6.0, 6.0]])
elif dim == 3:
    x = torch.tensor([[5.5, 5.5, 1.0]])
    y = torch.tensor([[6.0, 6.0, 2.0]])
args_x = (wavelengths * x).sum(-1, keepdim=True) + b
args_y = (wavelengths * y).sum(-1, keepdim=True) + b
#z_w_x = 1 / torch.sqrt(R_tensor) * torch.cos(args_x)
#z_w_y = 1 / torch.sqrt(R_tensor) * torch.cos(args_y)
z_w_x = torch.cos(args_x)
z_w_y = torch.cos(args_y)
kernel_approx = 2 * torch.mean((z_w_x * z_w_y).sum(dim=-1)).item()

delta = x - y
kernel = torch.exp(-0.5 * (delta * delta).sum(dim=-1)).item()

error = kernel - kernel_approx

print(f"Kernel: {kernel}, approximation: {kernel_approx}, error: {error}")
print('ok')