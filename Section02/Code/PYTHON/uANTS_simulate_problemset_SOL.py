import numpy as np
import matplotlib.pyplot as plt
from plot_simEEG import plot_simEEG

# plot_simEEG - plot function for MXC's course on neural time series analysis
# 
# INPUTS:  
#     EEG : eeglab structure (a custom dictionary in this implementation)
#     chan : channel to plot (default = 1)
#   fignum : figure to plot into (default = 1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def plot_simEEG(EEG, chan=0, fignum=1):
    if EEG is None:
        raise ValueError("No inputs! Provide the EEG structure to the function.")

    # Create figure and clear it
    plt.figure(fignum, figsize=(10, 8))
    plt.clf()

    ## ERP
    plt.subplot(211)
    plt.title(f'ERP from channel {chan}')
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')

    # Plot all trials
    trials = np.squeeze(EEG.data[chan, :, :])
    h = plt.plot(EEG.times, trials, linewidth=0.5, color=[0.75, 0.75, 0.75])

    # Plot average ERP
    plt.plot(EEG.times, np.mean(trials, axis=1), 'k', linewidth=3)

    ## Static power spectrum
    hz = np.linspace(0, EEG.srate, EEG.pnts)
    if len(EEG.data.shape) == 3:
        pw = np.mean((2 * np.abs(fft(np.squeeze(EEG.data[chan, :, :]), axis=0) / EEG.pnts) ** 2), axis=1)
    else:
        pw = (2 * np.abs(fft(EEG.data[chan, :]) / EEG.pnts)) ** 2

    plt.subplot(223)
    plt.plot(hz, pw, linewidth=2)
    plt.xlim([0, 40])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Static power spectrum')

    ## Time-frequency analysis
    # Frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    frex = np.linspace(2, 30, 40)

    # Number of wavelet cycles (hard-coded to 3 to 10)
    waves = 2 * (np.linspace(3, 10, len(frex)) / (2 * np.pi * frex)) ** 2

    # Setup wavelet and convolution parameters
    wavet = np.arange(-2, 2, 1 / EEG.srate)
    halfw = len(wavet) // 2
    nConv = EEG.pnts * EEG.trials + len(wavet) - 1

    # Initialize time-frequency matrix
    tf = np.zeros((len(frex), EEG.pnts))

    # Spectrum of data
    dataX = fft(EEG.data[chan, :, :].reshape(1, -1), nConv)

    # Loop over frequencies
    for fi in range(len(frex)):

        # Create wavelet
        waveX = fft(np.exp(2 * 1j * np.pi * frex[fi] * wavet) * np.exp(-wavet ** 2 / waves[fi]), nConv)
        waveX = waveX / np.max(waveX)  # Normalize

        # Convolve
        asig = ifft(waveX * dataX)

        # Trim and reshape
        asig = asig[0, halfw:-halfw + 1].reshape((EEG.pnts, EEG.trials))

        # Power
        tf[fi, :] = np.mean(np.abs(asig), axis=1)

    # Show a map of the time-frequency power
    plt.subplot(224)
    plt.contourf(EEG.times, frex, tf, 40, cmap='jet')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title('Time-frequency plot')

    plt.tight_layout()
    plt.show()


# Corrected EEG class definition
class EEG:
    def __init__(self):
        self.srate = 500  # sampling rate in Hz
        self.pnts = 1500  # number of time points
        self.trials = 30  # number of trials
        self.nbchan = 23  # number of channels
        self.times = np.arange(0, 1500) / 500  # time vector
        self.data = None  # placeholder for data

# Create an instance of EEG
EEG = EEG()

# 1) White noise
EEG.data = np.random.randn(EEG.nbchan, EEG.pnts, EEG.trials)
plot_simEEG(EEG, 2, 3)

# 2) Pink noise
EEG.nbchan = 4
ed = 50  # exponential decay parameter
EEG.data = np.zeros((EEG.nbchan, EEG.pnts, EEG.trials))

for chani in range(EEG.nbchan):
    for triali in range(EEG.trials):
        aspectrum = np.random.rand(EEG.pnts) * np.exp(-(np.arange(EEG.pnts)) / ed)
        fc = aspectrum * np.exp(1j * 2 * np.pi * np.random.rand(EEG.pnts))
        EEG.data[chani, :, triali] = np.real(np.fft.ifft(fc))

plot_simEEG(EEG)

# 3) Ongoing stationary
frequencies = [3, 5, 16]  # in Hz
amplitudes = [3, 4, 5]  # in arbitrary units

for chani in range(EEG.nbchan):
    for triali in range(EEG.trials):
        sinewave = np.zeros(EEG.pnts)
        for si in range(len(frequencies)):
            sinewave += amplitudes[si] * np.sin(2 * np.pi * frequencies[si] * EEG.times)
        EEG.data[chani, :, triali] = sinewave + np.random.randn(EEG.pnts)

plot_simEEG(EEG, 1, 2)

# 4) Ongoing nonstationary
peakfreq = 14  # in Hz
fwhm = 5  # Full Width at Half Maximum

hz = np.linspace(0, EEG.srate, EEG.pnts)
s = fwhm * (2 * np.pi - 1) / (4 * np.pi)
x = hz - peakfreq
fg = np.exp(-0.5 * (x / s) ** 2)

for chani in range(EEG.nbchan):
    for triali in range(EEG.trials):
        fc = np.random.rand(EEG.pnts) * np.exp(1j * 2 * np.pi * np.random.rand(EEG.pnts))
        fc *= fg
        EEG.data[chani, :, triali] = np.real(np.fft.ifft(fc))

plot_simEEG(EEG)

# 5) Transients #1: Gaussian
peaktime = 1
width = 0.1

EEG.data = np.zeros((EEG.nbchan, EEG.pnts, EEG.trials))

for chani in range(EEG.nbchan):
    for triali in range(EEG.trials):
        trialpeak = peaktime + np.random.randn() * 0.25
        gaus = np.exp(-(EEG.times - trialpeak) ** 2 / (2 * width ** 2))
        EEG.data[chani, :, triali] = gaus

plot_simEEG(EEG, 1, 1)

# 6) Transients #2: Oscillations with Gaussian
sfreq = 8  # sine wave frequency
peaktime = 1
width = 0.2

EEG.data = np.zeros((EEG.nbchan, EEG.pnts, EEG.trials))

for chani in range(EEG.nbchan):
    for triali in range(EEG.trials):
        trialpeak = peaktime + np.random.randn() / 5
        gaus = np.exp(-(EEG.times - trialpeak) ** 2 / (2 * width ** 2))
        sw = np.cos(2 * np.pi * sfreq * EEG.times)
        EEG.data[chani, :, triali] = sw * gaus

plot_simEEG(EEG, 1, 1)