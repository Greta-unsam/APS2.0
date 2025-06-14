import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import medfilt, convolve, find_peaks
from scipy.interpolate import CubicSpline

# Helper functions
def normalize(signal):
    """Normalize signal using z-score"""
    return (signal - np.mean(signal)) / np.std(signal)

def plot_region(data_dict, title, xlabel='Muestras (#)', ylabel='Adimensional'):
    """Helper function to plot ECG regions"""
    plt.figure(figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')
    for label, (values, style) in data_dict.items():
        plt.plot(zoom_region, values[zoom_region], label=label, **style)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.gca().set_yticks(())
    plt.show()

# Load and prepare data
fs = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg = normalize(mat_struct['ecg_lead'].flatten())
hb1 = normalize(mat_struct['heartbeat_pattern1'].flatten())
hb2 = normalize(mat_struct['heartbeat_pattern2'].flatten())
qrs = normalize(mat_struct['qrs_pattern1'].flatten())
qrs_det = mat_struct['qrs_detections'].flatten()

t_ecg = np.arange(len(ecg)) / fs
regions_of_interest = [
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs
]

# Median Filter Processing
trend = medfilt(medfilt(ecg, kernel_size=201), kernel_size=601)
ecg_filtrada = ecg - trend

# Plot median filter results
for region in regions_of_interest:
    zoom_region = np.arange(max(0, region[0]), min(len(ecg), region[1]), dtype='uint')
    plot_data = {
        'ECG': (ecg, {'linewidth': 0.5, 'color': 'lightblue'}),
        'Base': (trend, {'linewidth': 1, 'color': 'blue'}),
        'ECG filtrado': (ecg_filtrada, {'linewidth': 1, 'color': 'magenta'})
    }
    plot_region(plot_data, f'Filtro de mediana {region[0]} to {region[1]}')

# Cubic Spline Interpolation
delay = 0.06  # seconds
puntos_inter = (qrs_det - delay * fs).astype(int)

plt.figure(figsize=(10, 6))
plt.plot(t_ecg, ecg, label='ECG Original')
plt.plot(t_ecg[qrs_det], ecg[qrs_det], 'ro', label='Picos')
plt.plot(t_ecg[puntos_inter], ecg[puntos_inter], 'go', label='Puntos a interpolar')
plt.xlim(0, 25)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Puntos a Interpolar')
plt.grid()
plt.legend()
plt.show()

cs = CubicSpline(t_ecg[puntos_inter], ecg[puntos_inter])
trend = cs(t_ecg)
ecg_filtrada = ecg - trend

# Plot interpolation results
for region in regions_of_interest:
    zoom_region = np.arange(max(0, region[0]), min(len(ecg), region[1]), dtype='uint')
    plot_data = {
        'ECG': (ecg, {'linewidth': 0.5, 'color': 'lightblue'}),
        'Base': (trend, {'linewidth': 1, 'color': 'blue'}),
        'ECG filtrado': (ecg_filtrada, {'linewidth': 1, 'color': 'magenta'})
    }
    plot_region(plot_data, f'Filtro por interpolacion {region[0]} to {region[1]}')

# Matched Filter Processing
matched_filter = hb1[::-1]
filtered_ecg = convolve(ecg, matched_filter, mode='same')
peaks, _ = find_peaks(filtered_ecg, height=np.max(filtered_ecg) * 0.5, distance=fs*0.4)

# Plot matched filter results
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

# Performance evaluation
delay = 0.165  # seconds
peaks = (peaks - delay * fs).astype(int)
peaks = peaks[peaks > 0]

tolerance = int(0.05 * fs)
qrs_detected = np.zeros_like(qrs_det, dtype=bool)
TP, FP = 0, 0

for peak in peaks:
    matches = np.abs(qrs_det - peak) <= tolerance
    if np.any(matches):
        TP += 1
        qrs_detected[np.argmin(np.abs(qrs_det - peak))] = True
    else:
        FP += 1

FN = np.sum(~qrs_detected)
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

print(f"Sensibilidad (Recall): {sensitivity:.2f}")
print(f"Valor predictivo positivo (Precision): {precision:.2f}")

# Final comparison plot
plt.figure(figsize=(15, 4))
plt.plot(t_ecg, ecg, label='ECG')
plt.plot(t_ecg[peaks], ecg[peaks], 'ro', label='Picos detectados')
plt.plot(t_ecg[qrs_det], ecg[qrs_det], 'kx', label='QRS verdaderos')
plt.legend()
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Comparación de detecciones")
plt.grid(True)
plt.tight_layout()
plt.show()