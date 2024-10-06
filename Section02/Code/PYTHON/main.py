from scipy.io import loadmat

# Cargar el archivo .mat
data = loadmat('sampleEEGdata.mat')

# Mostrar las claves del diccionario
print("Claves del archivo .mat:", data.keys())

# Explorar el contenido de una clave específica (por ejemplo, 'EEG')
if 'EEG' in data:
    eeg_data = data['EEG']
    print("Tipo de dato de EEG:", type(eeg_data))
    print("Estructura de EEG:", eeg_data.dtype)

# Visualizar la estructura general del archivo
for key in data:
    if not key.startswith("__"):
        print(f"\nClave: {key}")
        print(f"Tipo de dato: {type(data[key])}")
        print(f"Tamaño: {data[key].shape}")
