#     COURSE: Solved challenges in neural time series analysis
#    SECTION: Simulating EEG data
#      VIDEO: Non-stationary narrowband activity via filtered noise
# Instructor: sincxpress.com
#

import numpy as np
import matplotlib.pyplot as plt

# %% Simulation details
pnts = 4567  # number of points
srate = 987  # sampling rate in Hz

# Signal parameters in Hz
peakfreq = 14
fwhm = 5

# Frequencies
hz = np.linspace(0, srate, pnts)

# %% Create frequency-domain Gaussian
s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # normalized width
x = hz - peakfreq  # shifted frequencies
fg = np.exp(-0.5 * (x/s)**2)  # Gaussian function

# %% Fourier coefficients of random spectrum
fc = np.random.rand(pnts) * np.exp(1j * 2 * np.pi * np.random.rand(pnts))

# Taper with Gaussian
fc = fc * fg

# Go back to time domain to get EEG data
signal = 2 * np.real(np.fft.ifft(fc))

# %% Plotting
plt.figure(1, figsize=(12, 6))
plt.clf()

# Frequency domain plot
plt.subplot(211)
plt.plot(hz, np.abs(fc), 'k')
plt.xlim([0, peakfreq * 3])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Frequency domain')
plt.grid()

# Time domain plot
plt.subplot(212)
time = np.arange(0, pnts) / srate
plt.plot(time, signal, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time domain')
plt.grid()

plt.tight_layout()
plt.show()
