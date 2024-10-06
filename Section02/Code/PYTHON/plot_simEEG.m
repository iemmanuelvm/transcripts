import numpy as np
import matplotlib.pyplot as plt

def plot_simEEG(EEG, chan=1, fignum=1):
    """
    plot_simEEG - plot function for MXC's course on neural time series analysis

    INPUTS:
      EEG : eeglab structure (assumed to be a dictionary in Python)
      chan : channel to plot (default = 1)
      fignum : figure to plot into (default = 1)
    """

    if EEG is None:
        raise ValueError("No inputs!")
    else:
        # Set up channel and figure number based on the inputs
        if len([EEG, chan, fignum]) == 1:
            chan, fignum = 1, 1
        elif len([EEG, chan, fignum]) == 2:
            fignum = 1

    plt.figure(fignum)
    plt.clf()

    # ERP
    plt.subplot(211)
    plt.title(f"ERP from channel {chan}")
    plt.plot(EEG['times'], np.squeeze(EEG['data'][chan - 1, :, :]), linewidth=0.5, color=[0.75, 0.75, 0.75])
    plt.plot(EEG['times'], np.squeeze(np.mean(EEG['data'][chan - 1, :, :], axis=1)), 'k', linewidth=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')

    # Static power spectrum
    hz = np.linspace(0, EEG['srate'], EEG['pnts'])
    if len(EEG['data'].shape) == 3:
        pw = np.mean((2 * np.abs(np.fft.fft(np.squeeze(EEG['data'][chan - 1, :, :]), axis=0) / EEG['pnts']) ** 2), axis=1)
    else:
        pw = (2 * np.abs(np.fft.fft(EEG['data'][chan - 1, :]) / EEG['pnts']) ** 2)

    plt.subplot(223)
    plt.plot(hz, pw, linewidth=2)
    plt.xlim([0, 40])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Static power spectrum')

    # Time-frequency analysis

    # frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    frex = np.linspace(2, 30, 40)

    # number of wavelet cycles (hard-coded to 3 to 10)
    waves = 2 * (np.linspace(3, 10, len(frex)) / (2 * np.pi * frex)) ** 2

    # setup wavelet and convolution parameters
    wavet = np.arange(-2, 2, 1 / EEG['srate'])
    halfw = int(np.floor(len(wavet) / 2)) + 1
    nConv = EEG['pnts'] * EEG['trials'] + len(wavet) - 1

    # initialize time-frequency matrix
    tf = np.zeros((len(frex), EEG['pnts']))

    # spectrum of data
    dataX = np.fft.fft(EEG['data'][chan - 1, :, :].reshape(1, -1), n=nConv)

    # loop over frequencies
    for fi in range(len(frex)):
        
        # create wavelet
        waveX = np.fft.fft(np.exp(2 * 1j * np.pi * frex[fi] * wavet) * np.exp(-wavet ** 2 / waves[fi]), nConv)
        waveX = waveX / np.max(waveX)  # normalize

        # convolve
        as_ = np.fft.ifft(waveX * dataX)

        # trim and reshape
        as_ = as_[:, halfw - 1:-halfw]
        as_ = np.reshape(as_, (EEG['pnts'], EEG['trials']))

        # power
        tf[fi, :] = np.mean(np.abs(as_), axis=1)

    # show a map of the time-frequency power
    plt.subplot(224)
    plt.contourf(EEG['times'], frex, tf, 40, cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title('Time-frequency plot')

    # Show the final plot
    plt.show()
