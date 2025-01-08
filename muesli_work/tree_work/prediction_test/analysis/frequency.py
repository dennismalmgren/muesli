import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a time series with frequent -1 -> 1 jumps
time_step = 0.2  # 0.2 seconds between decisions
total_time = 5  # Simulate 5 seconds
time = np.arange(0, total_time, time_step)

# Create a signal alternating between -1 and 1
signal = np.array([(-1) ** i for i in range(len(time))])

# Step 2: Perform the FFT (Fourier Transform)
fft_result = np.fft.fft(signal)
amplitudes = np.abs(fft_result)  # Magnitude of the FFT
frequencies = np.fft.fftfreq(len(signal), time_step)  # Frequency corresponding to each FFT bin

# Step 3: Filter positive frequencies (since FFT is symmetric)
positive_frequencies = frequencies[frequencies >= 0]
positive_amplitudes = amplitudes[frequencies >= 0]

# Step 4: Calculate the smoothness measure (Sm)
n = len(positive_frequencies)
sampling_frequency = 1 / time_step  # Sampling frequency = 5 Hz

# Calculate the smoothness measure using the formula
Sm = (2 / (n * sampling_frequency)) * np.sum(positive_amplitudes * positive_frequencies)

# Print the result
print(f"Smoothness measure (Sm): {Sm}")

# Optional: Plot the frequency spectrum for visualization
plt.figure(figsize=(10, 4))
plt.plot(positive_frequencies, positive_amplitudes)
plt.title('Fourier Frequency Spectrum of Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()