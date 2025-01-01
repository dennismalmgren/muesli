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
        log_sj = torch.nn.functional.logsigmoid(x)
        log_1_m_sj = torch.nn.functional.logsigmoid(-x)

        logits_sum = torch.cumsum(log_sj, dim=-1)
        
        logits_1_m_sj_flipped = torch.flip(log_1_m_sj, dims=(-1,))
        #these logits are -1, -2, ... 0

        logits_m1_sum = torch.cumsum(logits_1_m_sj_flipped, dim=-1)
        #these are -1, -1 + -2, -1 + -2 + -3, ...
        logits_inv_sum = torch.flip(logits_m1_sum, dims=(-1,))
        #these are 0 + 1 + .., 1 + 2 + ..
        #these are 
        logits = logits_sum
        logits[..., :-1] += logits_inv_sum[..., 1:]
        return logits
    
    import torch
import torch.nn.functional as F

class OrdinalLogitsKernelModule(torch.nn.Module): 
    def __init__(self, window_size=3):
        """
        Args:
            window_size (int): L in the [i-L..i] and [i+1..i+L] definition.
        """
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        """
        x shape: [batch_size, num_classes]
        We want, for each index i:
            sum_{k=i-L..i} log(sigmoid(x_k))
          + sum_{k=i+1..i+L} log(1 - sigmoid(x_k))
        """
        #sj = torch.sigmoid(x)
        log_sj = torch.nn.functional.logsigmoid(x)            # log(sigmoid(x))
        log_1_m_sj = torch.nn.functional.logsigmoid(-x)    # log(1 - sigmoid(x))

        # ------------------------------------------------
        # 1) LEFT WINDOW: sum_{k=i-L.. i} of log(sigmoid(x_k))
        # ------------------------------------------------
        # Kernel size is (L + 1)
        left_kernel_size = self.window_size + 1
        left_kernel = torch.ones(1, 1, left_kernel_size, device=x.device)

        # Pad on the left side so index i has access to i-L..i
        # With stride=1, output_length = input_length if
        #   left_pad + right_pad = kernel_size - 1.
        left_padding = (left_kernel_size - 1, 0)  # (pad_left, pad_right)
        left_padded = F.pad(log_sj.unsqueeze(1), left_padding, mode="constant", value=0.0)
        # Now convolve
        left_windowed = F.conv1d(left_padded, left_kernel, stride=1).squeeze(1)
        # left_windowed.shape == [batch_size, num_classes]
        # left_windowed[:, i] = sum_{k = i-L.. i} log_sj[:, k], properly handling boundary zeros

        # ------------------------------------------------
        # 2) RIGHT WINDOW: sum_{k=i+1.. i+L} of log(1 - sigmoid(x_k))
        # ------------------------------------------------
        right_kernel_size = self.window_size
        if right_kernel_size > 0:
            right_kernel = torch.ones(1, 1, right_kernel_size, device=x.device)
            # For sum_{k=i+1.. i+L}, we first gather [i.. i+L-1],
            # then shift by 1 to exclude i and start at (i+1).

            # We need total pad = kernel_size - 1 to preserve length.
            # If kernel_size=L, then pad=(L-1). We'll pad on the right
            # so we can "reach forward."
            right_padding = (0, right_kernel_size - 1)  # (left_pad, right_pad)
            right_padded = F.pad(log_1_m_sj.unsqueeze(1), right_padding, mode="constant", value=0.0)

            right_windowed = F.conv1d(right_padded, right_kernel, stride=1).squeeze(1)
            # Now right_windowed[:, i] = sum_{k=i.. i+L-1} log_1_m_sj[:, k]

            # We want sum_{k=i+1.. i+L}, so shift each position by -1.
            # That effectively discards the i-th term and includes (i+L)-th term.
            right_windowed = torch.roll(right_windowed, shifts=-1, dims=-1)
            # Now right_windowed[:, i] = sum_{k=i+1.. i+L} log_1_m_sj[:, k]

        else:
            # If window_size=0, no right summation
            right_windowed = torch.zeros_like(log_sj)

        # ------------------------------------------------
        # 3) Combine
        # ------------------------------------------------
        logits = left_windowed + right_windowed
        return logits