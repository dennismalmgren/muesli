import torch

batch_size=torch.Size([10])
p_change_directions=0.35

change_directions = torch.rand(size=batch_size) < p_change_directions
print(change_directions.shape)
print(change_directions)