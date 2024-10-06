# COURSE: Solved challenges in neural time series analysis
# SECTION: Simulating EEG data
# VIDEO: The three important equations (sine, Gaussian, Euler's)
# Instructor: sincxpress.com

import numpy as np
import matplotlib.pyplot as plt

# Intuition about sine waves

# Define some variables
freq = 2  # frequency in Hz
srate = 1000  # sampling rate in Hz
time = np.arange(-1, 1, 1/srate)  # time vector in seconds
ampl = 2
phas = np.pi/3

# Now create the sinewave
sinewave = ampl * np.sin(2 * np.pi * freq * time + phas)

# Now plot it!
plt.figure(1)
plt.plot(time, sinewave)

# Optional prettification of the plot
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.xlabel('Time (s)')
plt.title('My first sine wave plot! Mom will be so proud!')
plt.show()


# The sum of sine waves can appear like a complicated time series

# Define a sampling rate
srate = 1000

# List some frequencies
frex = [3, 10, 5, 15, 35]

# List some random amplitudes... make sure there are the same number of amplitudes as there are frequencies!
amplit = [5, 15, 10, 5, 7]

# Phases... list some random numbers between -pi and pi
phases = [np.pi/7, np.pi/8, np.pi, np.pi/2, -np.pi/4]

# Define time...
time = np.arange(-1, 1, 1/srate)

# Now we loop through frequencies and create sine waves
sine_waves = np.zeros((len(frex), len(time)))
for fi in range(len(frex)):
    sine_waves[fi, :] = amplit[fi] * np.sin(2 * np.pi * time * frex[fi] + phases[fi])

# Now plot the result
plt.figure(2)
plt.plot(time, np.sum(sine_waves, axis=0))
plt.title('Sum of sine waves')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (arb. units)')
plt.show()

# Now plot each wave separately
plt.figure(3)
for fi in range(len(frex)):
    plt.subplot(len(frex), 1, fi+1)
    plt.plot(time, sine_waves[fi, :])
    plt.axis([time[0], time[-1], -max(amplit), max(amplit)])
plt.show()


# Gaussian

# Simulation parameters
time = np.arange(-2, 2, 1/1000)
ptime = 1  # peak time
ampl = 45  # amplitude
fwhm = 0.9

# Gaussian
gwin = np.exp(-(4 * np.log(2) * (time - ptime)**2) / fwhm**2)

# Empirical FWHM
gwinN = gwin / np.max(gwin)
midp = np.argmin(np.abs(time))  # Closest index to zero
pst5 = midp-1 + np.argmax(gwinN[midp:] >= 0.5)
pre5 = np.argmax(gwinN[:midp] >= 0.5)
empfwhm = time[pst5] - time[pre5]

plt.figure(4)
plt.plot(time, gwin, 'k', linewidth=2)
plt.plot(time[[pre5, pst5]], gwin[[pre5, pst5]], 'ro--', markerfacecolor='k')
plt.plot([time[pre5], time[pre5]], [0, gwin[pre5]], 'r:')
plt.plot([time[pst5], time[pst5]], [0, gwin[pst5]], 'r:')
plt.title(f'Requested FWHM: {fwhm}s, empirical FWHM: {empfwhm}s')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


# Euler's formula

M = 2.4
k = 3 * np.pi / 4

meik = M * np.exp(1j * k)

plt.figure(5)

# Polar plane
ax1 = plt.subplot(121, projection='polar')
ax1.plot([0, k], [0, M], 'r')
ax1.plot([k], [M], 'ro')
ax1.set_title('Polar plane')

# Cartesian (rectangular) plane
plt.subplot(122)
plt.plot([0, np.real(meik)], [0, np.imag(meik)], 'ro')
plt.plot(np.real(meik), np.imag(meik), 'gs')
plt.axis(np.array([-1, 1, -1, 1]) * abs(meik))  # Corrected axis limits
plt.axis('square')
plt.xlabel('Real')
plt.ylabel('Imag')
plt.grid(True)
plt.title('Cartesian (rectangular) plane')

plt.show()
