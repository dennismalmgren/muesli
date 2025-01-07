from gru import GRU, Gru, GruConfig
import torch

model = GRU(input_size = 3, hidden_size=256, n_layers=1, dropout=0.0, bias=False, device="cpu")


input_tensor = torch.rand(size=(1, 15, 3))
is_init = torch.zeros(size=(1, 15, 1), dtype=torch.bool)
is_init[0, 0] = True
hidden = torch.zeros(size=(1, 15, 256)) #one per layer?

output, h_n = model(input_tensor, is_init, hidden)

print(output.shape)
print(output)