import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata
from topoplotIndie import topoplotIndie

# %% Start with the EEG data
data = loadmat('sampleEEGdata.mat')
EEG = data['EEG'][0, 0]  # Acceder al primer elemento, debido a la estructura cargada

# Verificar la estructura del array
print("Dimensiones de EEG['data']:", EEG['data'].shape)

# Dependiendo de la estructura, ajustamos el índice del promedio.
# Si el array tiene solo 2 dimensiones (canales x puntos de tiempo), no hay ensayos
if EEG['data'].ndim == 2:
    erp = EEG['data']  # No se necesita calcular el promedio, ya que no hay ensayos
else:
    # Si el array tiene 3 dimensiones (canales x puntos de tiempo x ensayos), calculamos el ERP
    erp = np.mean(EEG['data'], axis=2)

# %% Pick a channel and plot ERP
chan2plot = 'FCz'

plt.figure(1)
plt.clf()

# Obtener el índice del canal a graficar
# Modificación para acceder correctamente a las etiquetas dentro de `EEG['chanlocs']`
chanlocs = EEG['chanlocs'][0]  # Acceder al primer elemento de chanlocs
labels = [str(chanlocs[i]['labels'][0]) for i in range(len(chanlocs))]

# Encontrar el índice del canal
chan_idx = labels.index(chan2plot)

# Graficar el ERP del canal seleccionado
plt.plot(EEG['times'][0], erp[chan_idx, :], linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Activity (µV)')
plt.xlim([-400, 1200])

# %% Plot topographical maps

time2plot = 300  # in ms

# Convertir tiempo en ms a índices
tidx = np.argmin(np.abs(EEG['times'][0] - time2plot))

plt.figure(2)
plt.clf()


topoplotIndie(erp[:, tidx], EEG['chanlocs'][0])
# plt.title(f'ERP from {time2plot} ms')
# plt.colorbar()

# %% Now for sample CSD V1 data

data = loadmat('v1_laminar.mat')
timevec = data['timevec'].flatten()
csd = data['csd']

# %% Plot ERP from channel 7 in one line of code!
plt.figure(3)
plt.clf()
plt.plot(timevec, np.mean(csd[6, :, :], axis=1))  # Channel 7 (index 6 en Python)
plt.axhline(0, linestyle='--', color='k')
plt.axvline(0, linestyle='--', color='k')
plt.axvline(0.5, linestyle='--', color='k')
plt.xlabel('Time (s)')
plt.ylabel('Activity (µV)')
plt.xlim([-0.1, 1.4])

# %% Plot depth-by-time image of ERP
plt.figure(4)
plt.clf()
plt.contourf(timevec, np.arange(1, 17), np.mean(csd, axis=2), 40, cmap='viridis')
plt.xlim([0, 1.3])
plt.xlabel('Time (sec.)')
plt.ylabel('Cortical depth')

# %% Done.
plt.show()
