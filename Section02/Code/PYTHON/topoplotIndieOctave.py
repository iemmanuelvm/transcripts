import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def topoplotIndieOctave(Values, chanlocs, **kwargs):
    """
    Set defaults
    """
    headrad = 0.5          # actual head radius - Don't change this!
    GRID_SCALE = 67        # plot map on a 67x67 grid
    CIRCGRID = 201         # number of angles to use in drawing circles
    HEADCOLOR = [0, 0, 0]  # default head color (black)
    HLINEWIDTH = 1.7       # default linewidth for head, nose, ears
    BLANKINGRINGWIDTH = 0.035  # width of the blanking ring
    HEADRINGWIDTH = 0.007  # width of the cartoon head ring
    plotrad = 0.6
    Values = np.array(Values, dtype=float)
    SHADING = 'interp'
    CONTOURNUM = 6
    ELECTRODES = 'on'

    for key, value in kwargs.items():
        if key == 'numcontour':
            CONTOURNUM = value
        elif key == 'electrodes':
            ELECTRODES = value.lower()
        elif key == 'plotrad':
            plotrad = value
        elif key == 'shading':
            SHADING = value.lower()
            if SHADING not in ['flat', 'interp']:
                raise ValueError('Invalid shading parameter')

    Values = Values.flatten()  # make Values a column vector

    """
    Read channel location
    """
    labels = [ch['labels'] for ch in chanlocs]
    Th = np.array([ch['theta'] for ch in chanlocs]) * np.pi / 180  # convert degrees to radians
    Rd = np.array([ch['radius'] for ch in chanlocs])

    allchansind = np.arange(len(Th))
    plotchans = np.arange(len(chanlocs))

    """
    Remove infinite and NaN values
    """
    inds = np.union1d(np.where(np.isnan(Values))[0], np.where(np.isinf(Values))[0])  # NaN and Inf values
    for chani, ch in enumerate(chanlocs):
        if 'X' not in ch or ch['X'] is None:
            inds = np.append(inds, chani)

    plotchans = np.setdiff1d(plotchans, inds)

    x, y = Rd * np.cos(Th), Rd * np.sin(Th)  # transform electrode locations from polar to cartesian coordinates
    plotchans = np.abs(plotchans)  # reverse indicated channel polarities
    allchansind = allchansind[plotchans]
    Th = Th[plotchans]
    Rd = Rd[plotchans]
    x = x[plotchans]
    y = y[plotchans]
    labels = [labels[i] for i in plotchans]
    Values = Values[plotchans]
    intrad = min(1.0, max(Rd) * 1.02)  # default: just outside the outermost electrode location

    """
    Find plotting channels
    """
    pltchans = np.where(Rd <= plotrad)[0]  # plot channels inside plotting circle
    intchans = np.where((x <= intrad) & (y <= intrad))[0]  # interpolate and plot channels inside interpolation square

    """
    Eliminate channels not plotted
    """
    allx = x
    ally = y
    allchansind = allchansind[pltchans]
    intTh = Th[intchans]  # eliminate channels outside the interpolation area
    intRd = Rd[intchans]
    intx = x[intchans]
    inty = y[intchans]
    Th = Th[pltchans]  # eliminate channels outside the plotting area
    Rd = Rd[pltchans]
    x = x[pltchans]
    y = y[pltchans]

    intValues = Values[intchans]
    Values = Values[pltchans]

    """
    Squeeze channel locations to <= headrad
    """
    squeezefac = headrad / plotrad
    intRd = intRd * squeezefac  # squeeze electrode arc_lengths towards the vertex
    Rd = Rd * squeezefac        # squeeze electrode arc_lengths towards the vertex
    intx = intx * squeezefac
    inty = inty * squeezefac
    x = x * squeezefac
    y = y * squeezefac
    allx = allx * squeezefac
    ally = ally * squeezefac

    """
    Create grid
    """
    xmin = min(-headrad, min(intx))
    xmax = max(headrad, max(intx))
    ymin = min(-headrad, min(inty))
    ymax = max(headrad, max(inty))
    xi = np.linspace(xmin, xmax, GRID_SCALE)  # x-axis description (row vector)
    yi = np.linspace(ymin, ymax, GRID_SCALE)  # y-axis description (row vector)

    xx, yy = np.meshgrid(xi, yi)
    Zi = griddata((inty, intx), intValues, (yy, xx), method='cubic')  # interpolate data

    """
    Mask out data outside the head
    """
    mask = np.sqrt(xx ** 2 + yy ** 2) <= headrad  # mask outside the plotting circle
    Zi[~mask] = np.nan  # mask non-plotting voxels with NaNs

    """
    Scale the axes and make the plot
    """
    plt.cla()  # clear current axis
    plt.gca().set_xlim([-headrad, headrad])
    plt.gca().set_ylim([-headrad, headrad])

    if SHADING == 'interp':
        plt.contourf(xx, yy, Zi, CONTOURNUM, cmap='viridis')

    """
    Final plot settings
    """
    plt.show()
