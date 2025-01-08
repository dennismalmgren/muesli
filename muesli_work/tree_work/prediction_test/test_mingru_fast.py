import torch

from mingru_fast import MinGRU
from torchrl.modules import set_recurrent_mode

predictor_gru = MinGRU(1, 32, 1).to("cuda")
sequence = torch.tensor([[1.], [2], [3], [4]], device="cuda") # B x T X dim
targets = torch.tensor([[2.], [3], [4], [5]], device="cuda") # B x T x dim
h_0 = torch.exp(torch.randn(size=(4, 32), device="cuda"))
init = torch.tensor([[False], [False], [True], [False]], device="cuda") # B x T x dim
with set_recurrent_mode("recurrent"):
    output, hidden = predictor_gru(sequence, init, h_0) #DIMS?
    loss = torch.nn.functional.mse_loss(output, targets)

#Inference mode:
h_0 = torch.randn(size=(4, 32), device="cuda")
with set_recurrent_mode("sequential"):
    output, hidden = predictor_gru(sequence, init, h_0)
print('ok')