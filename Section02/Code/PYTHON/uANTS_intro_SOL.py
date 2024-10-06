import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def topoplotIndie(Values, chanlocs, **kwargs):
    """
    Function to plot a topographical map of EEG data.
    
    Parameters:
    Values (np.array): Values for each channel.
    chanlocs (list): List of channel locations containing X, Y, and labels.
    kwargs: Additional parameters for customization.
    """

    # Set defaults
    headrad = 0.5           # actual head radius - Don't change this!
    GRID_SCALE = 67         # plot map on a 67x67 grid
    CIRCGRID = 201          # number of angles to use in drawing circles
    HEADCOLOR = [0, 0, 0]   # default head color (black)
    HLINEWIDTH = 1.7        # default linewidth for head, nose, ears
    BLANKINGRINGWIDTH = .035  # width of the blanking ring
    HEADRINGWIDTH = .007    # width of the cartoon head ring
    plotrad = .6
    SHADING = 'interp'
    CONTOURNUM = 6
    ELECTRODES = 'on'

    # Parse additional input arguments
    for key, value in kwargs.items():
        param = key.lower()
        if param == 'numcontour':
            CONTOURNUM = value
        elif param == 'electrodes':
            ELECTRODES = value.lower()
        elif param == 'plotrad':
            plotrad = value
        elif param == 'shading':
            SHADING = value.lower()
            if SHADING not in ['flat', 'interp']:
                raise ValueError('Invalid shading parameter')

    # Make Values a column vector
    Values = Values.flatten()

    # Read channel location
    labels = [chan['labels'][0][0] for chan in chanlocs[0]]  # Channel labels
    Th = np.array([chan['theta'][0][0] for chan in chanlocs[0]]) * np.pi / 180  # Convert degrees to radians
    Rd = np.array([chan['radius'][0][0] for chan in chanlocs[0]])

    # Remove NaN and Inf values
    valid_idx = np.where(~np.isnan(Values) & ~np.isinf(Values))[0]

    # Transform electrode locations from polar to cartesian coordinates
    x, y = Rd * np.cos(Th), Rd * np.sin(Th)
    x, y, Values = x[valid_idx], y[valid_idx], Values[valid_idx]
    labels = [labels[i] for i in valid_idx]

    # Create grid for interpolation
    xmin, xmax = min(-headrad, min(x)), max(headrad, max(x))
    ymin, ymax = min(-headrad, min(y)), max(headrad, max(y))
    xi = np.linspace(xmin, xmax, GRID_SCALE)
    yi = np.linspace(ymin, ymax, GRID_SCALE)

    # Interpolate data
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), Values, (Xi, Yi), method='cubic')

    # Mask out data outside the head
    mask = np.sqrt(Xi**2 + Yi**2) <= headrad
    Zi[~mask] = np.nan

    # Scale the axes and make the plot
    plt.clf()
    # Use numpy arrays for xlim and ylim operations to prevent TypeError
    headrad_limits = np.array([-headrad, headrad]) * 1.05
    plt.gca().set_xlim(headrad_limits)
    plt.gca().set_ylim(headrad_limits)

    # Plot the interpolated data
    if SHADING == 'interp':
        plt.imshow(Zi, extent=(xi[0], xi[-1], yi[0], yi[-1]), origin='lower', aspect='auto')
    else:
        plt.pcolormesh(Xi, Yi, Zi, shading='flat')

    # Plot contour lines
    plt.contour(Xi, Yi, Zi, CONTOURNUM, colors='k')

    # Plot electrode locations
    if ELECTRODES == 'on':
        plt.plot(x, y, 'o', color='k', markersize=5)
    elif ELECTRODES == 'labels':
        for i, label in enumerate(labels):
            plt.text(y[i], x[i], label, ha='center', va='middle', color='k')
    elif ELECTRODES == 'numbers':
        for i, ind in enumerate(valid_idx):
            plt.text(y[i], x[i], str(ind), ha='center', va='middle', color='k')

    # Draw head outline
    circ = np.linspace(0, 2 * np.pi, CIRCGRID)
    plt.plot(np.cos(circ) * headrad, np.sin(circ) * headrad, color=HEADCOLOR, linewidth=HLINEWIDTH)

    # Plot nose and ears (simplified version)
    plt.plot([0, 0.1 * headrad], [headrad, headrad * 1.1], color=HEADCOLOR, linewidth=HLINEWIDTH)
    plt.plot([-0.1 * headrad, 0], [headrad, headrad * 1.1], color=HEADCOLOR, linewidth=HLINEWIDTH)

    plt.axis('off')
    plt.show()
#   COURSE: Neural signal processing and analysis: Zero to hero
#  SESSION: Introduction
#  TEACHER: Mike X Cohen, sincxpress.com
#

# %% become familiar with the sample data used in class
#

# %% start with the EEG data
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from topoplotIndie import topoplotIndie

# Load the sample EEG data
data = sio.loadmat('sampleEEGdata.mat')
EEG = data['EEG']

# explore a bit... what are the different fields? What is the size of the data?
# How many channels/time points/trials?
# What is the earliest and last time point?
# Where is time = 0?
# What is the sampling rate?
print(EEG)

# %% plot ERPs and topographical maps

# compute the ERP of each channel 
# (remember that the ERP is the time-domain average across all trials at each time point)
erp = np.mean(EEG['data'][0, 0], axis=2)

# pick a channel and plot ERP
chan2plot = 'FCz'
channel_labels = [EEG['chanlocs'][0, 0][0, i]['labels'][0] for i in range(len(EEG['chanlocs'][0, 0][0]))]
chan_idx = channel_labels.index(chan2plot)

plt.figure(1)
plt.clf()
plt.plot(EEG['times'][0, 0][0], erp[chan_idx, :], linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Activity (µV)')
plt.xlim([-400, 1200])
plt.show()

# %% plot topographical maps

time2plot = 300  # in ms

# convert time in ms to time in indices
tidx = np.argmin(np.abs(EEG['times'][0, 0][0] - time2plot))


plt.figure(2)
plt.clf()
# Call the topoplotIndie function (this is just a placeholder)
topoplotIndie(erp[:, tidx], EEG['chanlocs'][0, 0])
plt.title(f'ERP from {time2plot} ms')
plt.colorbar()
plt.show()

# %% now for sample CSD V1 data

data_csd = sio.loadmat('v1_laminar.mat')
csd = data_csd['csd']
timevec = data_csd['timevec'][0]

# check out the variables in this mat file, using the function whos
# If you don't know what variables are in this file vs. already in the workspace,
# you can clear the workspace and then load the file in again.

# %% plot ERP from channel 7 in one line of code!
plt.figure(3)
plt.clf()
plt.plot(timevec, np.squeeze(np.mean(csd[6, :, :], axis=2)))
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.axvline(0.5, color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Activity (µV)')
plt.xlim([-0.1, 1.4])
plt.show()

# plot depth-by-time image of ERP
plt.figure(4)
plt.clf()
plt.contourf(timevec, np.arange(1, 17), np.squeeze(np.mean(csd, axis=2)), 40, cmap='viridis')
plt.xlim([0, 1.3])
plt.xlabel('Time (sec.)')
plt.ylabel('Cortical depth')
plt.colorbar()
plt.show()

# %% done.
