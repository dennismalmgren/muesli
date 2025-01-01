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
x = torch.linspace(-2, 2, 200)  # x-coordinates
y = torch.linspace(-2, 2, 200)  # y-coordinates
X, Y = torch.meshgrid(x, y, indexing="xy")  # Create the 2D grid
X = X.reshape(-1)
Y = Y.reshape(-1)
XY  = torch.stack((X, Y), dim=-1)
#shape: wavelength x angles x coordinates
num_wave_angles= 3
wavelengths = torch.tensor([1.0]).unsqueeze(-1)
num_wavelengths = wavelengths.numel()
theta_degrees = torch.arange(num_wave_angles) * 360 / num_wave_angles
theta = theta_degrees * torch.pi / 180.0
k = 2 * torch.pi / wavelengths
k_x = k * torch.cos(theta)
k_y = k * torch.sin(theta) 
omega = torch.stack((k_x, k_y), dim=-1)
waves = torch.exp(1j * torch.einsum("kib, jb->kij", omega, XY))#.reshape(num_wavelengths, num_wave_angles, 200, 200) 
waves = waves.sum(0)
waves = waves.permute(1, 0).unsqueeze(-1)

#waves_real = waves.real
#wave_patterns = waves_real.sum(dim=1)
C = generate_one_unitary_matrix().unsqueeze(0)
#coefficients_pt_1 = wave_patterns[:, 10, 10]
#coefficients_pt_2 = wave_patterns[:, 100, 100]
g = torch.matmul(C, waves)
g_summed = g[:, 2, 0].reshape(200, 200)
#euclidean distance:
#print("Euclidan distance: ", math.sqrt(100**2 - 10**2))
#wave_pattern = waves_real.sum(dim=1)[0]

# Plot the real part of the wave
plt.figure(figsize=(8, 6))
plt.contourf(x.numpy(), y.numpy(), g_summed, levels=50, cmap="viridis")
plt.title(r"Real Part of Planar Waves $e^{i (\omega \cdot [x, y])}$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Amplitude")
plt.show()
