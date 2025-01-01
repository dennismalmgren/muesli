import math

import torch
import matplotlib.pyplot as plt

num_wave_angles = 3
wave_length_stddev = 0.5 #single "scale"
R = 2000
dim = 2

wavelengths = torch.randn((R, dim)) * wave_length_stddev **2 + torch.zeros((R, dim))
wavelengths = wavelengths.sort(dim=0)[0]
wavelengths = wavelengths.unsqueeze(-2).repeat(1, num_wave_angles, 1)
b_initial = torch.rand((R, 1, 1)) * 2 * torch.pi
b_initial = b_initial.sort(dim=0)[0]
b = b_initial + (torch.arange(num_wave_angles) / num_wave_angles).unsqueeze(-1).unsqueeze(0) * 2 * torch.pi  
b = b % (2 * torch.pi)
b_elem_sorted = b.sort(dim=1)[0]
_, b_elem_indices = torch.sort(b_elem_sorted[:, 0], dim=0)
b = b_elem_sorted[b_elem_indices.squeeze()]

#wavelengths are R x num_wave_angles x dim
#b are R x num_wave_angles x 1

# Define the 2D grid
num_grid_points = 40000
num_grid_points_1d = 200
dx = 4 / num_grid_points_1d
dy = 4 / num_grid_points_1d
x = torch.linspace(-2, 2, num_grid_points_1d)  # x-coordinates
y = torch.linspace(-2, 2, num_grid_points_1d)  # y-coordinates
X, Y = torch.meshgrid(x, y, indexing="xy")  # Create the 2D grid
X = X.reshape(-1)
Y = Y.reshape(-1)
XY  = torch.stack((X, Y), dim=-1)
g = wavelengths.reshape((-1, dim))
b = b.reshape((-1, 1))
gxy = torch.einsum("ij, kj->ki", g, XY).unsqueeze(-1)
z = torch.cos(gxy + b)
z = z.squeeze(-1)
kernel_approx = (2 / R) * (z * z).sum(-1)


g = g.reshape(num_grid_points_1d, num_grid_points_1d, N, num_wave_angles)

def distance_fun(x1, y1, x2, y2):
    distance_squared = torch.sum(g_summed[x1, y1] * g_summed[x1, y1].conj()) + torch.sum(g_summed[x2, y2] * g_summed[x2, y2].conj()) - 2 * torch.sum(g_summed[x1, y1] * g_summed[x2, y2].conj())
    distance = torch.sqrt(distance_squared)
    return distance

def k_y_x_approx(x1, y1, x2, y2):
    return torch.sum((g[x2, y2].conj() * g[x1, y1])).real

def distance_fun_2(x1, y1, x2, y2): 
    distance_squared = (
    torch.sum((g[x1, y1] * g[x2, y2].conj())) +
    torch.sum((g[x2, y2] * g[x2, y2].conj())) -
    2 * torch.sum((g[x1, y1] * g[x2, y2].conj()))
)   
    return torch.sqrt(distance_squared)

def kernel(x1, y1, x2, y2, sigma):
    the_dx = dx * (x1 - x2)
    the_dy = dy * (y1 - y2)
    return math.exp(-(the_dx**2 + the_dy**2) / (4 * sigma**2))

def euclid_dist(x1, y1, x2, y2):
    distance = math.sqrt((dx * (x1 -  x2))**2 + (dy * (y1 - y2))**2)
    return distance

g = g.reshape(num_grid_points_1d, num_grid_points_1d, -1) * math.sqrt(2 / N)

x1_test = 0
y1_test = 0
x2_test = 50
y2_test = 50

kernel_val = kernel(0, 0, 50, 50, wave_length_stddev)
kernel_approx_val = k_y_x_approx(0, 0, 50, 50)
kernel_approx_val_2 = k_y_x_approx(50, 50, 0, 0)
print("Kernel value: ", kernel_val)
print("Kernel approximation: ", kernel_approx_val)
print("Kernel approximation 2: ", kernel_approx_val_2)


kernel_dist_squared = kernel(x1_test, y1_test, x1_test, y1_test, wave_length_stddev) + kernel(x2_test, y2_test, x2_test, y2_test, wave_length_stddev) - 2 * kernel(x1_test, y1_test, x2_test, y2_test, wave_length_stddev)
kernel_dist = math.sqrt(kernel_dist_squared)
print("Kernel distance: ", kernel_dist)
e_dist = euclid_dist(x1_test, y1_test, x2_test, y2_test)
print("Euclidean distance: ", e_dist)

approx_dist = distance_fun_2(x1_test, y1_test, x2_test, y2_test)

print("Approximated distance: ", approx_dist)

print('Ok')
exit(0)
print(k_y_x_1)
print(k_y_x_2.item())
#lets do an inner product then.

print(i_p.item())
i_p_2 = euclid_dist(0, 0, 50, 50)
#euclidean distance:
print(i_p_2)
g_summed = g_summed.squeeze(-1).real
# Plot the real part of the wave
plt.figure(figsize=(8, 6))
plt.contourf(x.numpy(), y.numpy(), g_summed, levels=50, cmap="viridis")
plt.title(r"Real Part of Planar Waves $e^{i (\omega \cdot [x, y])}$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Amplitude")
plt.show()
