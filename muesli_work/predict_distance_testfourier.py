import math

import torch
import matplotlib.pyplot as plt

def givens_rotation(dim, p, q, theta):
    G = torch.eye(dim, dtype=torch.complex64)
    c = torch.cos(theta)
    s = torch.sin(theta)
    G[p, p] = c
    G[p, q] = -s
    G[q, p] = s
    G[q, q] = c
    return G

def diagonal_phase(dim, phases):
    D = torch.eye(dim, dtype=torch.complex64)
    for idx, ph in enumerate(phases):
        D[idx, idx] = torch.exp(1j * ph)
    return D

def generate_one_unitary_matrix(dim=3, N=1):
    """
    Generate a single unitary (3x3) matrix deterministically.
    By default N=1, but we add a constant offset to the angles so that 
    even when i=0 and N=1, we don't end up with the identity matrix.
    """
    i = 0  # Only one matrix
    # Adding a small offset (like pi/4) ensures a non-trivial matrix.
    theta = torch.tensor(2 * math.pi * i / N + math.pi/4)
    phi   = torch.tensor(math.pi * i / N + math.pi/4)
    psi1  = torch.tensor(2 * math.pi * i / N + math.pi/4)
    psi2  = torch.tensor(math.pi * i / (2 * N) + math.pi/4)
    psi3  = torch.tensor(-math.pi * i / N + math.pi/4)

    G1 = givens_rotation(dim, 0, 1, theta)
    G2 = givens_rotation(dim, 1, 2, phi)
    P  = diagonal_phase(dim, [psi1, psi2, psi3])
    U  = P @ G2 @ G1
    return U

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
#shape: wavelength x angles x coordinates
num_wave_angles = 3
wave_length_stddev = 0.5 #single "scale"
N = 2000
#wavelengths = torch.abs(torch.randn((N, 1)) * wave_length_stddev)

#wavelengths_x = torch.randn((N, 1)) * wave_length_stddev **2 + torch.zeros((N, 1))
#wavelengths_y = torch.randn((N, 1)) * wave_length_stddev **2 + torch.zeros((N, 1))
#wavelengths = torch.sqrt(wavelengths_x ** 2 + wavelengths_y**2)
wavelengths = torch.abs(torch.randn((N, 1)) * wave_length_stddev)

wavelengths = wavelengths.sort(dim=0)[0]

theta_initial = torch.rand((N,1)) * 360
theta_degrees = theta_initial + torch.arange(num_wave_angles).repeat(N, 1) * 360 / num_wave_angles #for now, don't randomize the directions..?
theta_degrees = theta_degrees % 360
theta = theta_degrees * torch.pi / 180.0
k = 2 * torch.pi / wavelengths
k_x = k * torch.cos(theta)
k_y = k * torch.sin(theta) 
omega = torch.stack((k_x, k_y), dim=-1)
waves = torch.exp(1j * torch.einsum("kib, jb->jki", omega, XY))#num_grid_points, N, num_wave_angles
waves = waves.reshape(-1, num_wave_angles, 1)

C = generate_one_unitary_matrix().unsqueeze(0)
#C = torch.eye(3).unsqueeze(0) * 1j
g = torch.matmul(C, waves)
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
