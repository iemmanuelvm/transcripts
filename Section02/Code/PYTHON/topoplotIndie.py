import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def topoplotIndie(values, chanlocs, numcontour=6, electrodes='on', plotrad=0.6, shading='interp'):
    """
    Function to plot EEG topographic data similar to MATLAB's topoplot function.
    Parameters:
    - values: numpy array of values to plot
    - chanlocs: dictionary with 'labels', 'theta', 'radius', 'X', 'Y' fields for channel positions
    - numcontour: number of contour levels (default = 6)
    - electrodes: show electrode locations ('on', 'labels', 'numbers') (default = 'on')
    - plotrad: radius for plotting (default = 0.6)
    - shading: shading method ('interp' or 'flat') (default = 'interp')
    """
    # Set defaults
    headrad = 0.5
    GRID_SCALE = 67
    CIRCGRID = 201
    HEADCOLOR = [0, 0, 0]
    HLINEWIDTH = 1.7
    BLANKINGRINGWIDTH = 0.035
    HEADRINGWIDTH = 0.007

    # Convert input values to numpy array
    values = np.array(values).astype(float)

    # Read channel locations
    labels = [chan['labels'] for chan in chanlocs]
    Th = np.array([chan['theta'] for chan in chanlocs]) * np.pi / 180  # Convert degrees to radians
    Rd = np.array([chan['radius'] for chan in chanlocs])

    # Remove channels without location data
    plotchans = np.arange(len(chanlocs))
    for i, chan in enumerate(chanlocs):
        if not chan.get('X') or not chan.get('Y'):
            plotchans = np.setdiff1d(plotchans, [i])

    # Handle case where no valid channels are found
    if len(plotchans) == 0:
        raise ValueError("No valid channels to plot after filtering. Check 'chanlocs' structure.")

    # Convert polar coordinates to cartesian
    x, y = Rd * np.cos(Th), Rd * np.sin(Th)
    labels = np.array(labels)[plotchans]  # Actualizar etiquetas basadas en los canales filtrados
    values = values[plotchans]

    # Find plotting channels
    pltchans = np.where(Rd[plotchans] <= plotrad)[0]
    intchans = np.where((x[plotchans] <= plotrad) & (y[plotchans] <= plotrad))[0]

    # Handle empty channels in plotting area
    if len(pltchans) == 0 or len(intchans) == 0:
        raise ValueError("No channels within the specified plot radius. Adjust 'plotrad' or check data.")

    # Scale electrode locations
    squeezefac = headrad / plotrad
    x, y = x[plotchans] * squeezefac, y[plotchans] * squeezefac

    # Create grid for interpolation
    xmin, xmax = min(-headrad, np.min(x)), max(headrad, np.max(x))
    ymin, ymax = min(-headrad, np.min(y)), max(headrad, np.max(y))
    xi = np.linspace(xmin, xmax, GRID_SCALE)
    yi = np.linspace(ymin, ymax, GRID_SCALE)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x[intchans], y[intchans]), values[intchans], (Xi, Yi), method='cubic')

    # Mask out data outside the head
    mask = np.sqrt(Xi ** 2 + Yi ** 2) <= headrad
    Zi[~mask] = np.nan

    # Plotting the head and data
    fig, ax = plt.subplots()
    if shading == 'interp':
        contour = ax.contourf(Xi, Yi, Zi, numcontour, cmap='viridis', levels=numcontour)
    else:
        contour = ax.pcolor(Xi, Yi, Zi, shading='flat')

    # Add contour lines
    ax.contour(Xi, Yi, Zi, numcontour, colors='k')

    # Draw head outline
    circ = np.linspace(0, 2 * np.pi, CIRCGRID)
    head_x = headrad * np.sin(circ)
    head_y = headrad * np.cos(circ)
    ax.plot(head_x, head_y, color=HEADCOLOR, linewidth=HLINEWIDTH)

    # Plot ears and nose
    base = headrad - 0.0046
    basex, tip, tiphw, tipr = 0.18 * headrad, 1.15 * headrad, 0.04 * headrad, 0.01 * headrad
    earX = np.array([0.497, 0.510, 0.518, 0.530, 0.542, 0.540, 0.547, 0.532, 0.510, 0.489])
    earY = np.array([0.0555, 0.0775, 0.0783, 0.0746, 0.0555, -0.0055, -0.0932, -0.1313, -0.1384, -0.1199]) + 0.04
    sf = headrad / plotrad

    # Plot nose and ears (modificado)
    ax.plot(np.array([basex, tiphw, 0, -tiphw, -basex]) * sf, 
            np.array([base, tip - tipr, tip, tip - tipr, base]) * sf, color=HEADCOLOR, linewidth=HLINEWIDTH)
    ax.plot(np.array(earX) * sf, np.array(earY) * sf, color=HEADCOLOR, linewidth=HLINEWIDTH)
    ax.plot(-np.array(earX) * sf, np.array(earY) * sf, color=HEADCOLOR, linewidth=HLINEWIDTH)

    # Mark electrode locations
    if electrodes == 'on':
        ax.plot(y, x, 'k.', markersize=5)
    elif electrodes == 'labels':
        for i in range(len(labels)):  # Asegurar que los índices estén dentro del rango de `labels`, `x`, `y`
            if i < len(x) and i < len(y):
                ax.text(y[i], x[i], labels[i], ha='center', va='center')
    elif electrodes == 'numbers':
        for i in range(len(labels)):
            if i < len(x) and i < len(y):
                ax.text(y[i], x[i], str(plotchans[i]), ha='center', va='center')

    ax.axis('off')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax)
    plt.show()

    return fig, pltchans, np.array([x, y])




import numpy as np

# Definir posiciones de los electrodos simulados
chanlocs = [
    {'labels': 'Fz', 'theta': -90, 'radius': 0.5, 'X': 0, 'Y': 0.5},
    {'labels': 'Cz', 'theta': 0, 'radius': 0.0, 'X': 0, 'Y': 0},
    {'labels': 'Pz', 'theta': 90, 'radius': 0.5, 'X': 0, 'Y': -0.5},
    {'labels': 'Oz', 'theta': 180, 'radius': 0.6, 'X': -0.6, 'Y': 0},
    {'labels': 'Fp1', 'theta': -45, 'radius': 0.6, 'X': 0.6, 'Y': 0.6},
    {'labels': 'Fp2', 'theta': -135, 'radius': 0.6, 'X': 0.6, 'Y': -0.6},
    {'labels': 'O1', 'theta': 135, 'radius': 0.7, 'X': -0.7, 'Y': 0.7},
    {'labels': 'O2', 'theta': -135, 'radius': 0.7, 'X': -0.7, 'Y': -0.7}
]

# Simular valores de amplitud en los canales
np.random.seed(42)  # Para resultados consistentes
values = np.random.rand(len(chanlocs))  # Valores aleatorios para cada canal

# Definir estructura de EEG con parámetros adicionales
EEG = {
    'data': np.random.randn(len(chanlocs), 1000, 10),  # Datos simulados (8 canales, 1000 puntos, 10 ensayos)
    'times': np.linspace(-1, 1, 1000),  # Vector de tiempo
    'srate': 1000,  # Frecuencia de muestreo
    'pnts': 1000,
    'trials': 10
}

# Llamar a la función para visualizar el topoplot
fig, pltchans, epos = topoplotIndie(values, chanlocs, numcontour=8, electrodes='labels', plotrad=0.7)
