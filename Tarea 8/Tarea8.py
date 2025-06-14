import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
from scipy.signal import convolve
from scipy.signal import find_peaks

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

# Cargar la señal ECG
fs = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')

# Extraer las señales
ecg = mat_struct['ecg_lead'].flatten()
hb1 = mat_struct['heartbeat_pattern1'].flatten()
hb2 = mat_struct['heartbeat_pattern2'].flatten()
qrs = mat_struct['qrs_pattern1'].flatten()
qrs_det = mat_struct['qrs_detections'].flatten()

# Normalización tipo z-score: (x - media) / std
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# Aplicar normalización
ecg = normalize(ecg)
hb1 = normalize(hb1)
hb2 = normalize(hb2)
qrs = normalize(qrs)

# Crear un vector de tiempo para la señal ECG completa
t_ecg = np.arange(len(ecg)) / fs

# Crear vectores de tiempo para los patrones (usualmente de menor duración)
t_hb1 = np.arange(len(hb1)) / fs
t_hb2 = np.arange(len(hb2)) / fs
t_qrs = np.arange(len(qrs)) / fs

#########################################
########## FILTRO DE MEDIANA ##########
#########################################

# Aplica filtro de mediana con ventana de 200 muestras
trend= medfilt(ecg, kernel_size=201)

# Luego aplica filtro de mediana con ventana de 600 muestras
trend = medfilt(trend, kernel_size=601)

ecg_filtrada = ecg - trend


cant_muestras = ecg.shape[0]
fig_sz_x = 10
fig_sz_y = 5
fig_dpi = 100

regs_interes = ( 
        #np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=0.5, color='lightblue')
    plt.plot(zoom_region, trend[zoom_region], label='Base',linewidth=1,color='blue')
    plt.plot(zoom_region, ecg_filtrada[zoom_region], label='ECG filtrado',linewidth=1,color='magenta')
    plt.title('Filtro de mediana' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()


#########################################
## INTERPOLACIÓN CON SPLINES CÚBICOS ####
#########################################

# Definir el retraso
delay = 0.06  # segundos

puntos_inter = (qrs_det - delay * fs).astype(int)


plt.figure(figsize=(10, 6))
plt.plot(t_ecg, ecg, label='ECG Original')
plt.plot(t_ecg[qrs_det], ecg[qrs_det], 'ro', label='Picos')
plt.plot(t_ecg[puntos_inter], ecg[puntos_inter], 'go', label='Puntos a interpolar')
plt.xlim(0, 25)  # Limitar el eje x para ver mejor
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Puntos a Interpolar')
plt.grid()
plt.show()

cs = CubicSpline(t_ecg[puntos_inter], ecg[puntos_inter])
t_cs = np.arange(len(ecg)) / fs
trend = cs(t_cs)


ecg_filtrada = ecg - trend


cant_muestras = ecg.shape[0]
fig_sz_x = 10
fig_sz_y = 5
fig_dpi = 100

regs_interes = ( 
        #np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg[zoom_region], label='ECG', linewidth=0.5, color='lightblue')
    plt.plot(zoom_region, trend[zoom_region], label='Base',linewidth=1,color='blue')
    plt.plot(zoom_region, ecg_filtrada[zoom_region], label='ECG filtrado',linewidth=1,color='magenta')
    plt.title('Filtro por interpolacion' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()

##########################################
######### FILTRO ADAPTADO ################
##########################################


######## Crear filtro adaptado ########


matched_filter = hb1[::-1]
filtered_ecg = convolve(ecg, matched_filter, mode='same')

########### Detectar latidos #########

peaks, _ = find_peaks(filtered_ecg, height=np.max(filtered_ecg) * 0.5, distance=fs*0.4)

plt.figure(figsize=(15, 4))
plt.plot(t_ecg, filtered_ecg, label='Salida del filtro adaptado')
plt.plot(t_ecg[peaks], filtered_ecg[peaks], 'ro', label='Picos detectados')
plt.legend()
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Detección de latidos con filtro adaptado")
plt.grid(True)
plt.tight_layout()
plt.show()





delay= 0.165  # segundos
peaks= (peaks - delay * fs).astype(int)
peaks = peaks[peaks > 0]


tolerance = int(0.05 * fs)  # 50 ms de tolerancia
TP = 0
FP = 0
FN = 0

qrs_detected = np.zeros_like(qrs_det, dtype=bool)

for peak in peaks:
    if np.any(np.abs(qrs_det - peak) <= tolerance):
        TP += 1
        match_idx = np.argmin(np.abs(qrs_det - peak))
        qrs_detected[match_idx] = True
    else:
        FP += 1

FN = np.sum(~qrs_detected)

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

print(f"Sensibilidad (Recall): {sensitivity:.2f}")
print(f"Valor predictivo positivo (Precision): {precision:.2f}")


plt.figure(figsize=(15, 4))
plt.plot(t_ecg, ecg, label='Salida del filtro adaptado')
plt.plot(t_ecg[peaks], ecg[peaks], 'ro', label='Picos detectados')
plt.plot(t_ecg[qrs_det], ecg[qrs_det], 'kx', label='QRS verdaderos')
plt.legend()
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Detección de latidos con filtro adaptado")
plt.grid(True)
plt.tight_layout()
plt.show()

