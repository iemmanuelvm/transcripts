import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata

def topoplotIndie(Values, chanlocs, **kwargs):
    """
    TopoplotIndie - Plot function for EEG topographic maps.

    INPUTS:
    Values : Array of values to plot
    chanlocs : List of channel locations with 'labels', 'theta', and 'radius' attributes
    """

    # Set defaults
    headrad = 0.5          # Actual head radius - Don't change this!
    GRID_SCALE = 67        # Plot map on a 67x67 grid
    CIRCGRID = 201         # Number of angles to use in drawing circles
    HEADCOLOR = [0, 0, 0]  # Default head color (black)
    HLINEWIDTH = 1.7       # Default linewidth for head, nose, ears
    BLANKINGRINGWIDTH = 0.035  # Width of the blanking ring
    HEADRINGWIDTH = 0.007      # Width of the cartoon head ring
    plotrad = 0.6
    Values = np.array(Values, dtype=float)  # Convert to float
    SHADING = 'interp'
    CONTOURNUM = 6
    ELECTRODES = 'on'

    # Parse optional parameters
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
                raise ValueError("Invalid shading parameter")

    # Ensure Values is a column vector
    Values = Values.flatten()

    # Read channel location
    labels = [chan['labels'] for chan in chanlocs]
    Th = np.array([chan['theta'] for chan in chanlocs]) * np.pi / 180  # Convert degrees to radians
    Rd = np.array([chan['radius'] for chan in chanlocs])

    allchansind = np.arange(len(Th))
    plotchans = allchansind.copy()

    # Remove infinite and NaN values
    inds = np.union1d(np.where(np.isnan(Values))[0], np.where(np.isinf(Values))[0])
    plotchans = np.setdiff1d(plotchans, inds)

    # Transform electrode locations from polar to Cartesian coordinates
    x, y = Rd * np.cos(Th), Rd * np.sin(Th)
    plotchans = np.abs(plotchans)
    allchansind = allchansind[plotchans]
    Th, Rd, x, y, labels = Th[plotchans], Rd[plotchans], x[plotchans], y[plotchans], np.array(labels)[plotchans]
    Values = Values[plotchans]
    intrad = min(1.0, max(Rd) * 1.02)  # Default: just outside the outermost electrode location

    # Find plotting channels
    pltchans = np.where(Rd <= plotrad)[0]  # Plot channels inside plotting circle
    intchans = np.where((x <= intrad) & (y <= intrad))[0]  # Interpolate and plot channels inside interpolation square

    # Eliminate channels not plotted
    allx, ally = x, y
    allchansind = allchansind[pltchans]
    intTh, intRd, intx, inty = Th[intchans], Rd[intchans], x[intchans], y[intchans]
    Th, Rd, x, y = Th[pltchans], Rd[pltchans], x[pltchans], y[pltchans]
    intValues = Values[intchans]
    Values = Values[pltchans]
    labels = labels[pltchans]

    # Squeeze channel locations to <= headrad
    squeezefac = headrad / plotrad
    intRd, Rd = intRd * squeezefac, Rd * squeezefac
    intx, inty = intx * squeezefac, inty * squeezefac
    x, y = x * squeezefac, y * squeezefac
    allx, ally = allx * squeezefac, ally * squeezefac

    # Create grid
    xmin, xmax = min(-headrad, min(intx)), max(headrad, max(intx))
    ymin, ymax = min(-headrad, min(inty)), max(headrad, max(inty))
    xi, yi = np.linspace(xmin, xmax, GRID_SCALE), np.linspace(ymin, ymax, GRID_SCALE)

    Xi, Yi = np.meshgrid(xi, yi)

    # Debugging: Print shapes and verify dimensions
    print("Shapes before griddata:")
    print(f"inty shape: {inty.shape}")
    print(f"intx shape: {intx.shape}")
    print(f"intValues shape: {intValues.shape}")
    print(f"Yi shape: {Yi.shape}")
    print(f"Xi shape: {Xi.shape}")

    # Ensure intx and inty are 1D
    intx_flat = np.asarray(intx).flatten()
    inty_flat = np.asarray(inty).flatten()
    intValues_flat = np.asarray(intValues).flatten()

    print("After flattening:")
    print(f"inty_flat shape: {inty_flat.shape}")
    print(f"intx_flat shape: {intx_flat.shape}")
    print(f"intValues_flat shape: {intValues_flat.shape}")

    # Stack the coordinates correctly
    points = np.column_stack((intx_flat, inty_flat))  # Shape should be (N, 2)

    print(f"points shape: {points.shape}")

    # Check if there are enough points for the interpolation method
    if points.shape[0] < 3:
        raise ValueError("Not enough points for interpolation. Need at least 3 points.")

    # Perform interpolation
    try:
        Zi = griddata(points, intValues_flat, (Xi, Yi), method='linear')
    except ValueError as e:
        print(f"Interpolation failed: {e}")
        print("Attempting with 'nearest' interpolation.")
        Zi = griddata(points, intValues_flat, (Xi, Yi), method='nearest')

    # Mask out data outside the head
    mask = np.sqrt(Xi**2 + Yi**2) <= headrad  # Mask outside the plotting circle
    Zi[~mask] = np.nan  # Mask non-plotting voxels with NaNs

    # Scale the axes and make the plot
    plt.clf()
    plt.axis('equal')
    plt.xlim([-headrad, headrad])
    plt.ylim([-headrad, headrad])

    if SHADING == 'interp':
        plt.pcolormesh(Xi, Yi, Zi, shading='auto', cmap='RdBu_r')  # Added colormap for better visualization
    else:
        plt.pcolormesh(Xi, Yi, Zi, shading='flat', cmap='RdBu_r')  # Added colormap for better visualization

    plt.contour(Xi, Yi, Zi, levels=CONTOURNUM, colors='k')

    # Mark electrode locations
    if ELECTRODES == 'on':  # Plot electrodes as spots
        plt.scatter(y, x, c='k', s=20)
    elif ELECTRODES == 'labels':  # Print electrode names (labels)
        for i in range(len(labels)):
            plt.text(y[i], x[i], labels[i], ha='center', va='middle', color='k')
    elif ELECTRODES == 'numbers':
        for i in range(len(labels)):
            plt.text(y[i], x[i], str(allchansind[i]), ha='center', va='middle', color='k')

    plt.title('Topographic Plot')
    plt.colorbar(label='µV')  # Added label for colorbar
    plt.show()

    return plt, pltchans, np.vstack((x, y))