from plot_simEEG import plot_simEEG
import numpy as np

# Define la estructura EEG como un diccionario
EEG = {}
EEG['srate']  = 500  # sampling rate in Hz
EEG['pnts']   = 1500
EEG['trials'] = 30
EEG['nbchan'] = 23

sinefreq = 6.75  # en Hz

# Vector de tiempo
EEG['times'] = np.arange(0, EEG['pnts']) / EEG['srate']

# Inicializar la estructura de datos
EEG['data'] = np.zeros((EEG['nbchan'], EEG['pnts'], EEG['trials']))

# 1) Onda sinusoidal pura con fase bloqueada
for chani in range(EEG['nbchan']):
    for triali in range(EEG['trials']):
        EEG['data'][chani, :, triali] = np.sin(2 * np.pi * sinefreq * EEG['times'])



plot_simEEG(EEG, 2, 1)


# 2) Onda sinusoidal pura sin fase bloqueada
for chani in range(EEG['nbchan']):
    for triali in range(EEG['trials']):
        EEG['data'][chani, :, triali] = np.sin(2 * np.pi * sinefreq * EEG['times'] + 2 * np.pi * np.random.rand())

plot_simEEG(EEG, 2, 2)


# 3) Onda sinusoidal con múltiples frecuencias
frex = [3, 5, 16]
amps = [2, 4, 5]

for chani in range(EEG['nbchan']):
    for triali in range(EEG['trials']):
        sinewave = np.zeros(EEG['pnts'])
        for si in range(len(frex)):
            sinewave += amps[si] * np.sin(2 * np.pi * frex[si] * EEG['times'])
        EEG['data'][chani, :, triali] = sinewave

plot_simEEG(EEG, 2, 3)


# 4) Onda sinusoidal no estacionaria
for chani in range(EEG['nbchan']):
    for triali in range(EEG['trials']):
        freqmod = 20 * np.interp(np.linspace(1, 10, EEG['pnts']), np.arange(1, 11), np.random.rand(10))
        signal = np.sin(2 * np.pi * ((EEG['times'] + np.cumsum(freqmod)) / EEG['srate']))
        EEG['data'][chani, :, triali] = signal

plot_simEEG(EEG, 2, 4)


# 5) Oscilaciones transitorias con Gaussian
peaktime = 1  # segundos
width = 0.12
sinefreq = 7  # frecuencia del seno

# Crear la envolvente Gaussian
gaus = np.exp(-(EEG['times'] - peaktime) ** 2 / (2 * width ** 2))

for chani in range(EEG['nbchan']):
    for triali in range(EEG['trials']):
        cosw = np.cos(2 * np.pi * sinefreq * EEG['times'])
        EEG['data'][chani, :, triali] = cosw * gaus

plot_simEEG(EEG, 2, 5)


# 6) Repetir #3 con ruido blanco
for chani in range(EEG['nbchan']):
    for triali in range(EEG['trials']):
        sinewave = np.zeros(EEG['pnts'])
        for si in range(len(frex)):
            sinewave += amps[si] * np.sin(2 * np.pi * frex[si] * EEG['times'])
        EEG['data'][chani, :, triali] = sinewave + 5 * np.random.randn(EEG['pnts'])

plot_simEEG(EEG, 2, 6)


# 7) Repetir #5 con ruido 1/f
noiseamp = 0.3

peaktime = 1  # segundos
width = 0.12
sinefreq = 7  # frecuencia del seno

# Crear la envolvente Gaussian
gaus = np.exp(-(EEG['times'] - peaktime) ** 2 / (2 * width ** 2))

for chani in range(EEG['nbchan']):
    for triali in range(EEG['trials']):
        cosw = np.cos(2 * np.pi * sinefreq * EEG['times'] + 2 * np.pi * np.random.rand())

        # Ruido 1/f
        ed = 50
        as_vals = np.random.rand(EEG['pnts'] // 2 - 1) * np.exp(-(np.arange(1, EEG['pnts'] // 2)) / ed)
        as_vals = np.concatenate(([as_vals[0]], as_vals, [0], as_vals[::-1]))

        # Coeficientes de Fourier
        fc = as_vals * np.exp(1j * 2 * np.pi * np.random.rand(len(as_vals)))

        # Transformada inversa de Fourier para crear el ruido
        noise = np.real(np.fft.ifft(fc)) * EEG['pnts']

        # Señal + ruido
        EEG['data'][chani, :, triali] = cosw * gaus + noiseamp * noise

plot_simEEG(EEG, 2, 7)
