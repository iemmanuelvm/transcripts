import numpy as np
import matplotlib.pyplot as plt

# %% Normally distributed noise

# Simulation details
srate = 100  # sampling rate in Hz
time = np.linspace(-1, 2, int(3 * srate + 1))  # corrected time vector
pnts = len(time)  # number of points

# Frequencies for the power spectrum
hz = np.linspace(0, srate/2, int(np.floor(len(time)/2)+1))

# Noise parameters
stretch = 3
shift = 0

# (Optional: fix the random-number generator state)
np.random.seed(3)

# Generate random data
noise = stretch * np.random.randn(len(time)) + shift

# Plotting
plt.figure(4, figsize=(12, 8))
plt.clf()

# Time domain
plt.subplot(211)
plt.plot(time, noise, 'k')
plt.title('Normally distributed: Time domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.gca().tick_params(axis='both', labelsize=15)

# Signal histogram (distribution)
plt.subplot(223)
y, x = np.histogram(noise, bins=100)
plt.plot(x[:-1], y, 'k', linewidth=2)
plt.xlabel('Values')
plt.ylabel('N per bin')
plt.title('Signal histogram (distribution)')
plt.xlim([min(x), max(x)])
plt.gca().tick_params(axis='both', labelsize=15)

# Frequency domain
plt.subplot(224)
amp = np.abs(np.fft.fft(noise)/pnts)
amp[1:] = 2*amp[1:]  # adjust for one-sided spectrum
plt.plot(hz, amp[:len(hz)], 'k')
plt.title('Frequency domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.gca().tick_params(axis='both', labelsize=15)

plt.tight_layout()
plt.show()

# %% Pink noise (aka 1/f aka fractal)

# Simulation details for this video
srate = 500  # sampling rate in Hz
time = np.linspace(-1, 2, int(3 * srate + 1))  # corrected time vector
pnts = len(time)  # number of points
hz = np.linspace(0, srate/2, int(np.floor(len(time)/2)+1))

# Generate 1/f amplitude spectrum
ed = 50  # exponential decay parameter
as_ = np.random.rand(int(np.floor(pnts/2))-1) * np.exp(-(np.arange(1, int(np.floor(pnts/2))))/ed)
as_ = np.concatenate([[as_[0]], as_, [0], [0], as_[::-1]])  # Create symmetric spectrum

# Fourier coefficients
fc = as_ * np.exp(1j*2*np.pi*np.random.rand(len(as_)))

# Inverse Fourier transform to create the noise
noise = np.real(np.fft.ifft(fc)) * pnts

# Plotting
plt.figure(5, figsize=(12, 8))
plt.clf()

# Time domain
plt.subplot(211)
plt.plot(time, noise, 'k')
plt.title('Pink noise: Time domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.gca().tick_params(axis='both', labelsize=15)

# Signal histogram (distribution)
plt.subplot(223)
y, x = np.histogram(noise, bins=100)
plt.plot(x[:-1], y, 'k', linewidth=2)
plt.xlabel('Values')
plt.ylabel('N per bin')
plt.title('Signal histogram (distribution)')
plt.gca().tick_params(axis='both', labelsize=15)

# Frequency domain
plt.subplot(224)
amp = np.abs(np.fft.fft(noise)/pnts)
amp[1:] = 2*amp[1:]  # adjust for one-sided spectrum
plt.plot(hz, amp[:len(hz)], 'k')
plt.title('Frequency domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.gca().tick_params(axis='both', labelsize=15)

plt.tight_layout()
plt.show()
