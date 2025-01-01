import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the 2D grid
x = torch.linspace(-2, 2, 200)  # x-coordinates
y = torch.linspace(-2, 2, 200)  # y-coordinates
X, Y = torch.meshgrid(x, y, indexing="ij")  # Create the 2D grid

# Define the planar wave
k_y = 2 * np.pi  # Wavenumber in the y-direction
wave = torch.exp(1j * k_y * Y)  # Planar wave e^(i * k_y * y), vertical propagation

# Separate the real and imaginary parts for plotting
wave_real = wave.real.numpy()
wave_imag = wave.imag.numpy()

# Plot the real part of the wave
plt.figure(figsize=(8, 6))
plt.contourf(x.numpy(), y.numpy(), wave_real, levels=50, cmap="viridis")
plt.title(r"Real Part of Planar Wave $e^{i k_y y}$ (Upwards)")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Amplitude")
plt.show()

# Plot the imaginary part of the wave
plt.figure(figsize=(8, 6))
plt.contourf(x.numpy(), y.numpy(), wave_imag, levels=50, cmap="plasma")
plt.title(r"Imaginary Part of Planar Wave $e^{i k_y y}$ (Upwards)")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Amplitude")
plt.show()
