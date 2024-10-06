#     COURSE: Solved challenges in neural time series analysis
#    SECTION: Simulating EEG data
#      VIDEO: Generating "chirps" (frequency-modulated signals)
# Instructor: sincxpress.com
#

import numpy as np
import matplotlib.pyplot as plt

# %% simulation details

pnts = 10000  # number of points
srate = 1024  # sampling rate
time = np.arange(0, pnts) / srate  # time vector

# %% chirps

# "bipolar" chirp
freqmod = np.linspace(5, 15, pnts)  # frequency modulation linearly from 5 to 15 Hz

# Uncomment the following lines for multipolar chirp
# k = 10  # poles for frequencies
# freqmod = 20 * np.interp(np.linspace(1, k, pnts), np.arange(1, k+1), np.random.rand(k))

# signal time series
signal = np.sin(2 * np.pi * ((time + np.cumsum(freqmod)) / srate))
# Note: the code in the video has a bug in the previous line,
# due to incorrect parentheses: srate should scale both time
# and freqmod, as above.

# %% plotting

plt.figure(1, figsize=(10, 6))
plt.clf()

# Plot instantaneous frequency
plt.subplot(211)
plt.plot(time, freqmod, 'r', linewidth=3)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Instantaneous frequency')

# Plot signal (chirp)
plt.subplot(212)
plt.plot(time, signal, 'k')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Signal (chirp)')

plt.tight_layout()
plt.show()
