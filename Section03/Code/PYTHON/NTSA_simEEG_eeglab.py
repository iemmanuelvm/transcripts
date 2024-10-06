import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata


def topoplotIndieOctave(Values, chanlocs, **kwargs):
    """
    Function to plot a topographical map of EEG data, adapted from Octave/MATLAB.
    
    Parameters:
    Values (np.array): Values for each channel.
    chanlocs (list): List of channel locations containing X, Y, and labels.
    kwargs: Additional parameters for customization.
    """
    
    # Set defaults
    headrad = 0.5           # actual head radius - Don't change this!
    GRID_SCALE = 67         # plot map on a 67X67 grid
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

    # Ensure Values is a column vector
    Values = Values.flatten()

    # Read channel location
    labels = [chan['labels'][0][0] for chan in chanlocs[0]]  # Accessing labels correctly
    Th = np.array([chan['theta'][0][0] for chan in chanlocs[0]]) * np.pi / 180  # convert degrees to radians
    Rd = np.array([chan['radius'][0][0] for chan in chanlocs[0]])

    # Remove infinite and NaN values
    valid_idx = np.where(~np.isnan(Values) & ~np.isinf(Values))[0]

    # Transform electrode locations from polar to Cartesian coordinates
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


# Cargar los datos de EEG (asegúrate de tener el archivo 'sampleEEGdata.mat' en tu directorio de trabajo)
data = loadmat('sampleEEGdata.mat')
EEG = data['EEG'][0, 0]

# Explorar la estructura de EEG (opcional)
print(EEG)

# Calcular el ERP para cada canal
erp = np.mean(EEG['data'], axis=2)

# Seleccionar un canal para graficar ERP
chan2plot = 'FCz'
channel_labels = [EEG['chanlocs'][0, i]['labels'][0] for i in range(len(EEG['chanlocs'][0]))]
chan_idx = channel_labels.index(chan2plot)

# Graficar ERP para el canal seleccionado
plt.figure(1)
plt.clf()
plt.plot(EEG['times'][0], erp[chan_idx, :], linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (µV)')
plt.title(f'ERP for channel {chan2plot}')
plt.show()

# Graficar mapas topográficos
time2plot = 300  # Tiempo en ms
tidx = np.argmin(np.abs(EEG['times'][0] - time2plot))  # Convertir tiempo en índices

# Graficar el mapa topográfico para el tiempo especificado
plt.figure(2)
plt.clf()
topoplotIndieOctave(erp[:, tidx], EEG['chanlocs'])
