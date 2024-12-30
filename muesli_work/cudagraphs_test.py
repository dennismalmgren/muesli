import torch
from tensordict.nn import CudaGraphModule
from torch import nn
from tensordict import TensorDict
import torch._dynamo
torch._dynamo.config.suppress_errors = True

n_warmup = 2
device = torch.device("cuda:0")

def eval_fun(td: TensorDict):
    td["output"] = torch.zeros_like(td["input"])
    return td

td = TensorDict(device=device)
td["input"] = torch.zeros(1, device=device)

eval_fun = torch.compile(eval_fun)
eval_fun_cgm = CudaGraphModule(eval_fun, warmup=n_warmup)

#stepping one step longer than the pre warmup test
for i in range(n_warmup):
    td_out2 = eval_fun_cgm(td.clone())