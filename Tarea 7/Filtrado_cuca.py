import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import wavfile
from scipy.signal import firls
from pytc2.sistemas_lineales import plot_plantilla
import os

# Leer archivo de audio
ruta_audio = os.path.join(os.path.dirname(__file__), 'la cucaracha.wav')
fs, audio = wavfile.read(ruta_audio)

# Si es estéreo, usar un solo canal
if audio.ndim > 1:
    audio = audio[:, 0]

# Normalización tipo z-score
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

audio = normalize(audio)

# Crear vector de tiempo
t_audio = np.arange(len(audio)) / fs

#########################################
########## PLANTILLA DE DISEÑO ##########
#########################################

# Banda pasante para voz: 300 Hz a 3400 Hz
fpass = np.array([300, 3400])
ripple = 1          # dB de ondulación en la banda pasante
fstop = np.array([100, 4000])
attenuation = 40    # dB de atenuación fuera de banda

# Mostrar plantilla
plt.figure(figsize=(10, 6))
plt.title('Plantilla de diseño (Audio)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plt.xlim(0, 5000)
plt.ylim(-100, 20)
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple,
               fstop=fstop, attenuation=attenuation, fs=fs)

# Frecuencias para evaluar la respuesta en frecuencia
w_rad = np.linspace(0, fs / 2, 600, endpoint=True)

###########################################
############### FILTROS IIR ###############
###########################################

# Filtro Butterworth
sos_filter_butter = sig.iirdesign(fpass, fstop, ripple, attenuation,
                                  ftype='butter', output='sos', fs=fs)
w_b, hh_b = sig.sosfreqz(sos_filter_butter, worN=w_rad, fs=fs)

# Filtro Chebyshev tipo I
sos_filter_cheby = sig.iirdesign(fpass, fstop, ripple, attenuation,
                                 ftype='cheby1', output='sos', fs=fs)
w_c, hh_c = sig.sosfreqz(sos_filter_cheby, worN=w_rad, fs=fs)

# Graficar respuestas en frecuencia
plt.figure(figsize=(10, 5))
plt.plot(w_b, 20 * np.log10(np.abs(hh_b) + 1e-15), label='Butterworth')
plt.plot(w_c, 20 * np.log10(np.abs(hh_c) + 1e-15), label='Chebyshev I')
plt.title('Respuesta en módulo de los filtros IIR (Audio)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid(which='both', axis='both')
plt.xlim(0, 5000)
plt.ylim(-100, 20)
plt.legend()
plt.tight_layout()
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple,
               fstop=fstop, attenuation=attenuation, fs=fs)


###########################################
############### FILTROS FIR ###############
###########################################

######### MÉTODO DE VENTANAS #########

# --- Pasaaltos FIR (ventanas) ---
freq_hp = [0, fpass[0]*0.98, fpass[0], fs/2]
gain_hp = [0, 0, 1, 1]
freq_hp = np.array(freq_hp)

numtaps_hp = 1501
b_hp = sig.firwin2(numtaps_hp, freq_hp, gain_hp, fs=fs)

# --- Pasabajos FIR (ventanas) ---
freq_lp = [0, fpass[1], fpass[1]*1.02, fs/2]
gain_lp = [1, 1, 0, 0]
freq_lp = np.array(freq_lp)

numtaps_lp = 201
b_lp = sig.firwin2(numtaps_lp, freq_lp, gain_lp, fs=fs)

# --- Filtro pasabanda por convolución ---
b_vent = np.convolve(b_hp, b_lp)

# --- Respuesta en frecuencia ---
w_fir, h_fir = sig.freqz(b_vent, worN=w_rad, fs=fs)

######### MÉTODO DE CUADRADOS MÍNIMOS #########

# --- Pasaaltos FIR (firls) ---
numtaps = 1501
bands_hp = [0, fpass[0]*0.98, fpass[0], fs/2]
desired_hp = [0, 0, 1, 1]
bands_hp = np.array(bands_hp) / (fs / 2)
b_firls_hp = firls(numtaps, bands_hp, desired_hp)

# --- Pasabajos FIR (firls) ---
numtaps = 501
bands_lp = [0, fpass[1], fpass[1]*1.02, fs/2]
desired_lp = [1, 1, 0, 0]
bands_lp = np.array(bands_lp) / (fs / 2)
b_firls_lp = firls(numtaps, bands_lp, desired_lp)

# --- Filtro pasabanda por convolución ---
b_firls = np.convolve(b_firls_hp, b_firls_lp)

# --- Respuesta en frecuencia ---
w_firls, h_firls = sig.freqz(b_firls, worN=w_rad, fs=fs)

######### GRAFICAR RESPUESTAS #########

plt.figure(figsize=(10, 5))
plt.plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-15), label='Ventanas', color='green')
plt.plot(w_firls, 20 * np.log10(np.abs(h_firls) + 1e-15), label='Cuadrados Mínimos', color='blue')
plt.title('Respuesta en módulo de los filtros FIR (Audio)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid(which='both', axis='both')
plt.xlim(0, 5000)
plt.ylim(-100, 20)
plt.legend()
plt.tight_layout()
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple,
               fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()


#########################################
######### APLICANDO LOS FILTROS #########
#########################################

# Aplicar filtros IIR (sin desfase)
audio_butter = sig.sosfiltfilt(sos_filter_butter, audio)
audio_cheby = sig.sosfiltfilt(sos_filter_cheby, audio)

# Aplicar filtros FIR (sin desfase)
audio_vent = sig.filtfilt(b_vent, 1.0, audio)
audio_cuad = sig.filtfilt(b_firls, 1.0, audio)


#########################################
############# GRAFICANDO ################
#########################################

plt.figure(figsize=(12, 6))

# Señal original
plt.plot(t_audio, audio, label='Audio Original', color='lightblue', linewidth=2)

# Señales filtradas
plt.plot(t_audio, audio_butter, label='Butterworth', linewidth=0.7)
plt.plot(t_audio, audio_cheby, label='Chebyshev I', linewidth=0.7)
plt.plot(t_audio, audio_vent, label='Ventanas', linewidth=0.7)
plt.plot(t_audio, audio_cuad, label='Cuadrados Mínimos', linewidth=0.7)

plt.title('Filtrado de Audio - Comparación de Métodos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud (normalizada)')
plt.legend()
plt.grid(True)
plt.xlim(0, 3)  # Mostrar solo los primeros 5 segundos
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))

# Señal original
plt.plot(t_audio, audio, label='Audio Original', color='lightblue', linewidth=2)

# Señales filtradas
plt.plot(t_audio, audio_butter, label='Butterworth', linewidth=0.7)
plt.plot(t_audio, audio_cheby, label='Chebyshev I', linewidth=0.7)
plt.plot(t_audio, audio_vent, label='Ventanas', linewidth=0.7)
plt.plot(t_audio, audio_cuad, label='Cuadrados Mínimos', linewidth=0.7)

plt.title('Filtrado de Audio - Comparación de Métodos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud (normalizada)')
plt.legend()
plt.grid(True)
plt.xlim(0.5, 1)  # Mostrar solo los primeros 5 segundos
plt.tight_layout()
plt.show()
