import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import firwin2
from scipy.interpolate import CubicSpline


# Cargar la señal ECG
fs = 1000  # Hz
f_Nyquist = fs / 2
mat_struct = sio.loadmat('./ECG_TP4.mat')

# Extraer las señales
ecg = mat_struct['ecg_lead'].flatten()


# Normalización tipo z-score: (x - media) / std
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# Aplicar normalización
ecg = normalize(ecg)

# Crear un vector de tiempo para la señal ECG completa
t_ecg = np.arange(len(ecg)) / fs




#############################
########## DETREND ##########
#############################


# Parámetros de ventana
window_duration = 1  # segundos
window_size = int(fs * window_duration)  # muestras por ventana
step_size = window_size  # sin superposición, podés reducirlo para ventanas solapadas

# Crear listas para almacenar tiempos y medianas
median_times = []
median_values = []

# Recorrer la señal en bloques
for start in range(0, len(ecg) - window_size + 1, step_size):
    end = start + window_size
    window = ecg[start:end]
    median = np.median(window)
    time = (start + end) / 2 / fs  # tiempo central de la ventana
    median_times.append(time)
    median_values.append(median)

# Convertir a arrays
median_times = np.array(median_times)
median_values = np.array(median_values)

# Interpolación cúbica
cs = CubicSpline(median_times, median_values)

# Tiempo continuo para la spline (toda la duración de la señal)
t = np.arange(len(ecg)) / fs
trend = cs(t)

ecg=ecg-trend

##############################
########## VENTANAS ##########
##############################

M= 10

# Parámetros del filtro pasabajos
fc = 1/M  # Frecuencia de corte en Hz 
numtaps = 101  # Orden del filtro 

# Frecuencias y ganancias normalizadas (0 a fs/2)
freq = [0, fc * f_Nyquist * 0.98, fc * f_Nyquist, f_Nyquist]
gain = [1, 1, 0, 0]

# Normalizar frecuencias para firwin2 (de 0 a fs/2)
freq = np.array(freq) / f_Nyquist

b = firwin2(numtaps, freq, gain, fs=fs)

# Filtrado de la señal ECG
ecg_filtrada = sig.lfilter(b, 1, ecg)
# Graficar señal original y filtrada
plt.figure(figsize=(12, 5))
plt.plot(t_ecg, ecg, label='ECG original', alpha=0.5)
plt.plot(t_ecg, ecg_filtrada, label='ECG filtrada (pasabajos)', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.title('Filtrado Pasabajos FIR (Ventana de Hamming)')

# Graficar la densidad espectral de potencia (PSD) de la señal ECG inter-diezmada
f, Pxx = sig.welch(ecg_filtrada, fs=fs/M, nperseg=1024)
plt.figure(figsize=(10, 5))
plt.semilogy(f, Pxx)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [V**2/Hz]')
plt.title('Densidad espectral de potencia (Welch) - filtrado')
plt.grid()



#####################################
######## CUCHILLAZO #################
#####################################

ecg_diez=ecg_filtrada[::M]  # Inter-diezmo de la señal
t_ecg = t_ecg[::M]  # Ajustar el vector de tiempo
# Graficar señal inter-diezmada
plt.figure(figsize=(12, 5))
plt.plot(t_ecg, ecg_diez, label='ECG inter-diezmado', color='orange')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.title('ECG Inter-Diezmado')


# Graficar la densidad espectral de potencia (PSD) de la señal ECG inter-diezmada
f, Pxx = sig.welch(ecg_diez, fs=fs/M, nperseg=1024)
plt.figure(figsize=(10, 5))
plt.semilogy(f, Pxx)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [V**2/Hz]')
plt.title('Densidad espectral de potencia (Welch) - ECG inter-diezmado')
plt.grid()
plt.show()