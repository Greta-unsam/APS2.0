

from pytc2.sistemas_lineales import plot_plantilla
import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import firls



# Normalización tipo z-score: (x - media) / std
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# Cargar la señal de audio
fs, ecg = sio.wavfile.read('la cucaracha.wav')


# Aplicar normalización
ecg = normalize(ecg)

# Crear un vector de tiempo para la señal ECG completa
t_ecg = np.arange(len(ecg)) / fs


plt.figure(figsize=(10, 6))
plt.plot(t_ecg, ecg, label='ECG Original')


#########################################
########## PLANTILLA DE DISEÑO ##########
#########################################



# Configuración de la plantilla de diseño del filtro
fpass = np.array([300, 3400])
ripple = 1
fstop = np.array([100,4000])
attenuation = 40

plt.figure(figsize=(10, 6))
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.xlim(0, 75)
plt.ylim(-100,20)
plt.grid(which='both', axis='both')
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)



# Grilla logarítmica y lineal en Hz
w_rad = np.linspace(0, fs/2, 600, endpoint=True)  # lineal de 0 a fs/2


###########################################
############### FILTROS IIR ###############
###########################################

############# BUTTER ###############
sos_filter_vent = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype='butter', output='sos',fs=fs)
w_b, hh_b = sig.sosfreqz(sos_filter_vent, worN=w_rad, fs=fs)

############# CHEBVY 1 ###############
sos_filter_ch = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype='cheby1', output='sos',fs=fs)
w_c, hh_c = sig.sosfreqz(sos_filter_ch, worN=w_rad, fs=fs)

# Graficar la respuesta en módulo (dB)
plt.figure(figsize=(10, 5))
plt.plot(w_b, 20 * np.log10(np.abs(hh_b) + 1e-15), label='Butterworth')
plt.plot(w_c, 20 * np.log10(np.abs(hh_c) + 1e-15), label='Chebyshev I')
plt.title('Respuesta en módulo de los filtros IIR')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid(which='both', axis='both')
plt.xlim(0, 75)
plt.ylim(-100,20)
plt.legend()
plt.tight_layout()
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)


###########################################
############### FILTROS FIR ###############
###########################################


######### METODO VENTANAS #########
# --- Filtro pasaaltos FIR ---
freq_hp = [0, fpass[0]*0.98, fpass[0], fs/2]
gain_hp = [0, 0, 1, 1]
freq_hp = np.array(freq_hp)

numtaps = 1501  # Orden 
b_hp = sig.firwin2(numtaps, freq_hp, gain_hp, fs=fs)

# --- Filtro pasabajos FIR ---
freq_lp = [0, fpass[1], fpass[1]*1.02, fs/2]
gain_lp = [1, 1, 0, 0]
freq_lp = np.array(freq_lp)

numtaps = 201  # Orden 
b_lp = sig.firwin2(numtaps, freq_lp, gain_lp, fs=fs)

# --- Convolucionamos ambos filtros para obtener el pasabanda ---
b_vent = np.convolve(b_hp, b_lp)

# --- Respuesta en frecuencia del filtro FIR pasabanda ---
w_fir, h_fir = sig.freqz(b_vent, worN=w_rad, fs=fs)

####### METODO CUADRADOS MINIMOS #########

# --- Filtro pasaaltos FIR (firls) ---
numtaps = 1501  # Orden 
bands_hp = [0, fpass[0]*0.98, fpass[0], fs/2]
desired_hp = [0, 0, 1, 1]
bands_hp = np.array(bands_hp)
bands_hp_norm = bands_hp / (fs/2)
b_firls_hp = firls(numtaps, bands_hp_norm, desired_hp)

# --- Filtro pasabajos FIR (firls) ---
numtaps = 501  # Orden 
bands_lp = [0, fpass[1], fpass[1]*1.02, fs/2]
desired_lp = [1, 1, 0, 0]
bands_lp = np.array(bands_lp)
bands_lp_norm = bands_lp / (fs/2)
b_firls_lp = firls(numtaps, bands_lp_norm, desired_lp)

# --- Convolucionamos ambos filtros para obtener el pasabanda ---
b_firls = np.convolve(b_firls_hp, b_firls_lp)

# --- Respuesta en frecuencia del filtro FIR pasabanda (firls) ---
w_firls, h_firls = sig.freqz(b_firls, worN=w_rad, fs=fs)


# --- Graficar junto a los FIR ---
plt.figure(figsize=(10, 5))
plt.plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-15), label='Ventanas', color='green')
plt.plot(w_firls, 20 * np.log10(np.abs(h_firls) + 1e-15), label='Cuadrados', color='blue')
plt.title('Respuesta en módulo de los filtros FIR')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Módulo [dB]')
plt.grid(which='both', axis='both')
plt.xlim(0, 75)
plt.ylim(-100, 20)
plt.legend()
plt.tight_layout()
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()


#########################################
######### APLICANDO LOS FILTROS #########
#########################################


# Aplicar el filtro Butterworth (sin desfase)
ecg_butter = sig.sosfiltfilt(sos_filter_vent, ecg)
# Aplicar el filtro Chebyshev I (sin desfase)
ecg_cheby = sig.sosfiltfilt(sos_filter_ch, ecg)
# Aplicar el filtro FIR por Ventanas (sin desfase)
ecg_vent = sig.filtfilt(b_vent, 1.0, ecg)
# Aplicar el filtro FIR por Cuadrados Mínimos (sin desfase)
ecg_cuad = sig.filtfilt(b_firls, 1.0, ecg)

#########################################
############# GRAFICANDO ################
#########################################

plt.figure(figsize=(10, 6))
plt.plot(t_ecg, ecg, label='ECG Original', color='lightblue')
plt.plot(t_ecg, ecg_butter, label='Butter', linewidth=0.5)
plt.plot(t_ecg, ecg_cheby, label='Chebyshev I', linewidth=0.5)
plt.plot(t_ecg, ecg_vent, label='Ventanas', linewidth=0.5)
plt.plot(t_ecg, ecg_cuad, label='Cuadrados Mínimos', linewidth=0.5)

plt.title('ECG filtering example')
plt.ylabel('Adimensional')
plt.xlabel('Tiempo [s]')
axes_hdl = plt.gca()
axes_hdl.legend()
axes_hdl.set_yticks(())
plt.xlim(0, 10)  # Limitar el eje x a los primeros 10 segundos
