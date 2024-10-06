from topoplotIndie import topoplot_indie
from plot_simEEG import plot_simEEG
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Placeholder functions
# def plot_simEEG(EEG, channel, figure_num):
#     pass

# def topoplotIndie(data, chanlocs, numcontour=0, electrodes='numbers', shading='interp'):
#     pass

# Load the MAT file containing EEG, leadfield, and channel locations
data = loadmat('emptyEEG.mat')
EEG = data['EEG'][0][0]
lf = data['lf'][0][0]

# Select dipole location (more-or-less random)
diploc = 109

# Plot brain dipoles
fig = plt.figure(1, figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(lf['GridLoc'][:, 0], lf['GridLoc'][:, 1], lf['GridLoc'][:, 2], c='y', marker='o')
ax1.scatter(lf['GridLoc'][diploc, 0], lf['GridLoc'][diploc, 1], lf['GridLoc'][diploc, 2], c='r', marker='s', s=100)
ax1.set_title('Brain dipole locations')
plt.axis('square')

# Each dipole can be projected onto the scalp using the forward model.
# The code below shows this projection from one dipole.
ax2 = fig.add_subplot(122)
topoplot_indie(-lf['Gain'][:, 0, diploc], EEG['chanlocs'][0])
plt.title('Signal dipole projection')
plt.show()

# Reduce data size a bit
EEG['pnts'] = 2000
EEG['times'] = np.arange(0, EEG['pnts']) / EEG['srate'][0][0]

# Initialize all dipole data
dipole_data = np.zeros((lf['Gain'].shape[2], EEG['pnts']))

# Add signal to one dipole
dipole_data[diploc, :] = np.sin(2 * np.pi * 10 * EEG['times'])

# Project dipole data to scalp electrodes
EEG['data'] = np.dot(lf['Gain'][:, 0, :], dipole_data)

# Plot the data
plot_simEEG(EEG, 31, 2)

# 1) Pure sine wave with amplitude explorations
EEG['trials'] = 40
ampl = 1e-30

# Initialize all dipole data
dipole_data = np.zeros((lf['Gain'].shape[2], EEG['pnts']))
dipole_data[diploc, :] = ampl * np.sin(2 * np.pi * 10 * EEG['times'])

# Compute one trial
signal = np.dot(lf['Gain'][:, 0, :], dipole_data)

# Repeat that for N trials
EEG['data'] = np.tile(signal[:, :, np.newaxis], (1, 1, EEG['trials']))

# Plot the data
plot_simEEG(EEG, 31, 2)

# 2) Sine wave with noise
noiseSD = 5

for triali in range(EEG['trials']):
    dipole_data = noiseSD * np.random.randn(lf['Gain'].shape[2], EEG['pnts'])
    dipole_data[diploc, :] = np.sin(2 * np.pi * 10 * EEG['times'])
    EEG['data'][:, :, triali] = np.dot(lf['Gain'][:, 0, :], dipole_data)

# Plot the data
plot_simEEG(EEG, 31, 2)

# 3) Non-oscillatory transient in one dipole, noise in all other dipoles
peaktime = 1  # seconds
width = 0.12
ampl = 70

# Create Gaussian taper
gaus = ampl * np.exp(-(EEG['times'] - peaktime) ** 2 / (2 * width ** 2))

for triali in range(EEG['trials']):
    dipole_data = np.random.randn(lf['Gain'].shape[2], EEG['pnts'])
    dipole_data[diploc, :] = gaus
    EEG['data'][:, :, triali] = np.dot(lf['Gain'][:, 0, :], dipole_data)

plot_simEEG(EEG, 31, 2)

# 4) Non-stationary oscillation in one dipole, transient oscillation in another dipole, noise in all dipoles
diploc1 = 109
diploc2 = 510

# Plot brain dipoles
fig = plt.figure(1, figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(lf['GridLoc'][:, 0], lf['GridLoc'][:, 1], lf['GridLoc'][:, 2], c='y', marker='o')
ax1.scatter(lf['GridLoc'][diploc1, 0], lf['GridLoc'][diploc1, 1], lf['GridLoc'][diploc1, 2], c='k', marker='s', s=100)
ax1.scatter(lf['GridLoc'][diploc2, 0], lf['GridLoc'][diploc2, 1], lf['GridLoc'][diploc2, 2], c='r', marker='s', s=100)
ax1.set_title('Brain dipole locations')
plt.axis('square')

ax2 = fig.add_subplot(132)
# topoplotIndie(-lf['Gain'][:, 0, diploc1], EEG['chanlocs'][0])
plt.title('Signal dipole projection')

ax3 = fig.add_subplot(133)
# topoplotIndie(-lf['Gain'][:, 0, diploc2], EEG['chanlocs'][0])
plt.title('Signal dipole projection')
plt.show()

# Gaussian and sine parameters
peaktime = 1  # seconds
width = 0.12
sinefreq = 7  # for sine wave

# Create Gaussian taper
gaus = np.exp(-(EEG['times'] - peaktime) ** 2 / (2 * width ** 2))

# Initialize EEG.data
EEG['data'] = np.zeros((EEG['nbchan'][0][0], EEG['pnts'], EEG['trials']))

for triali in range(EEG['trials']):
    dipole_data = np.random.randn(lf['Gain'].shape[2], EEG['pnts']) / 5

    # Non-stationary oscillation in dipole1 (range of 5-10 Hz)
    freqmod = 5 + 5 * np.interp(np.arange(EEG['pnts']), np.linspace(0, EEG['pnts'], 10), np.random.rand(10))
    dipole_data[diploc1, :] = np.sin(2 * np.pi * ((EEG['times'] + np.cumsum(freqmod)) / EEG['srate'][0][0]))

    # Transient oscillation in dipole2
    dipole_data[diploc2, :] = np.sin(2 * np.pi * sinefreq * EEG['times'] + np.random.rand() * np.pi) * gaus

    # Compute one trial of data
    EEG['data'][:, :, triali] = np.dot(lf['Gain'][:, 0, :], dipole_data)

# Plot the data
plot_simEEG(EEG, 56, 3)
plot_simEEG(EEG, 31, 2)
