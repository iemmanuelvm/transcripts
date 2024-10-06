import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Función para la interpolación y visualización de un topoplot
def topoplot_indie(values, chanlocs, numcontour=6, electrodes='on', plotrad=0.6, shading='interp'):
    # Set defaults
    headrad = 0.5
    GRID_SCALE = 67
    CIRCGRID = 201
    HEADCOLOR = [0, 0, 0]
    HLINEWIDTH = 1.7
    BLANKINGRINGWIDTH = 0.035
    HEADRINGWIDTH = 0.007
    values = np.array(values, dtype=float)
    
    # Convert degrees to radians and filter channels
    labels = [ch['labels'] for ch in chanlocs]
    Th = np.array([ch['theta'] for ch in chanlocs]) * np.pi / 180
    Rd = np.array([ch['radius'] for ch in chanlocs])

    # Cartesian coordinates
    x, y = Rd * np.cos(Th), Rd * np.sin(Th)

    # Filter valid channels
    plotchans = np.arange(len(chanlocs))
    values = values[plotchans]
    x = x[plotchans]
    y = y[plotchans]
    Rd = Rd[plotchans]
    Th = Th[plotchans]
    labels = [labels[i] for i in plotchans]

    # Eliminate channels not plotted
    pltchans = Rd <= plotrad
    x, y, values = x[pltchans], y[pltchans], values[pltchans]

    # Squeeze channel locations to <= headrad
    squeezefac = headrad / plotrad
    x, y = x * squeezefac, y * squeezefac

    # Create grid
    xi = np.linspace(min(-headrad, min(x)), max(headrad, max(x)), GRID_SCALE)
    yi = np.linspace(min(-headrad, min(y)), max(headrad, max(y)), GRID_SCALE)
    xx, yy = np.meshgrid(xi, yi)

    # Interpolate data
    Zi = griddata((x, y), values, (xx, yy), method='cubic')

    # Mask out data outside the head
    mask = np.sqrt(xx**2 + yy**2) <= headrad
    Zi[~mask] = np.nan

    # Create figure
    fig, ax = plt.subplots()
    ax.set_xlim(-headrad, headrad)
    ax.set_ylim(-headrad, headrad)
    ax.set_aspect('equal')

    # Plot data
    if shading == 'interp':
        contour = ax.contourf(xx, yy, Zi, numcontour, cmap='viridis')
    else:
        contour = ax.contour(xx, yy, Zi, numcontour, cmap='viridis')

    # Plot the head
    circ = np.linspace(0, 2 * np.pi, CIRCGRID)
    headx, heady = headrad * np.cos(circ), headrad * np.sin(circ)
    ax.plot(headx, heady, color=HEADCOLOR, linewidth=HLINEWIDTH)

    # Plot ears and nose
    base = headrad - 0.0046
    basex = 0.18 * headrad
    tip = 1.15 * headrad
    tiphw = 0.04 * headrad
    ear_x = np.array([0.497, 0.510, 0.518, 0.5299, 0.5419, 0.54, 0.547, 0.532, 0.510, 0.489]) - 0.005
    ear_y = np.array([0.0555, 0.0775, 0.0783, 0.0746, 0.0555, -0.0055, -0.0932, -0.1313, -0.1384, -0.1199]) + 0.04
    ax.plot([basex, tiphw, 0, -tiphw, -basex], [base, tip, tip, tip, base], color=HEADCOLOR, linewidth=HLINEWIDTH)
    ax.plot(ear_x * headrad, ear_y * headrad, color=HEADCOLOR, linewidth=HLINEWIDTH)
    ax.plot(-ear_x * headrad, ear_y * headrad, color=HEADCOLOR, linewidth=HLINEWIDTH)

    # Plot electrode locations
    if electrodes == 'on':
        ax.scatter(y, x, c='k', s=30)
    elif electrodes == 'labels':
        for i, label in enumerate(labels):
            ax.text(y[i], x[i], label, fontsize=9, ha='center', va='center')

    plt.colorbar(contour, ax=ax)
    plt.title("Topoplot Example")
    plt.show()