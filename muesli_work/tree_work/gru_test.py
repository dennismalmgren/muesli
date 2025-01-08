from gru import GRU, Gru, GruConfig
import torch
from prediction_test.mingru_fast import MinGRU
from torchrl.modules import set_recurrent_mode


model = GRU(input_size = 3, hidden_size=256, n_layers=1, dropout=0.0, bias=False, device="cpu")


input_tensor = torch.rand(size=(1, 15, 3))
is_init = torch.zeros(size=(1, 15, 1), dtype=torch.bool)
is_init[0, 0] = True
hidden = torch.zeros(size=(1, 15, 1, 256)) #one per layer?

with set_recurrent_mode("recurrent"):
    output_1, h_n_1 = model(input_tensor, is_init, hidden)

print(output_1.shape)
print(h_n_1.shape)

model2 = MinGRU(input_dim = 3, hidden_dim=256).to("cuda")

with set_recurrent_mode("recurrent"):
    output, h_n = model2(input_tensor.to("cuda"), is_init.to("cuda"), hidden.to("cuda"))

print(output.shape)
print(output)