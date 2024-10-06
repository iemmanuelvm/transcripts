# COURSE: Solved challenges in neural time series analysis
# SECTION: Simulating EEG data
# VIDEO: Generating transient oscillations
# Instructor: sincxpress.com

import numpy as np
import matplotlib.pyplot as plt

# Simulation details

pnts = 4000
srate = 1000
time = np.arange(0, pnts) / srate - 1

# Gaussian parameters
peaktime = 1  # seconds
fwhm = 0.4

# Sine wave parameters
sinefreq = 7  # for sine wave

# Create signal

# Create Gaussian taper
gaus = np.exp(-(4 * np.log(2) * (time - peaktime)**2) / fwhm**2)

# Sine wave with random phase value ("non-phase-locked")
cosw = np.cos(2 * np.pi * sinefreq * time + 2 * np.pi * np.random.rand())

# Signal
signal = cosw * gaus

# And plot
plt.figure(1)
plt.plot(time, signal, 'k', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Generated Transient Oscillations')
plt.show()
