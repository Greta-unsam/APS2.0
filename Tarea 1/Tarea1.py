"""
La primer tarea consistirá en programar una función que genere
 señales senoidales y que permita parametrizar:

Parámetros:
vmax : float -> Amplitud máxima de la senoidal (V)
dc   : float -> Valor medio (V)
f0   : float -> Frecuencia de la señal (Hz)
ph   : float -> Fase inicial (radianes)
nn   : int   -> Cantidad de muestras
fs   : float -> Frecuencia de muestreo (Hz)

Retorna:
tt : numpy array -> Vector de tiempos
yy : numpy array -> Valores de la señal en cada instante de tiempo
"""
import matplotlib.pyplot as plt
import numpy as np

#FUNCION

def crear_sin(vmax, dc, f0, ph, nn, fs):
    Ts = 1 / fs  # Periodo de muestreo
    Ttotal = nn * Ts  # Duración total de la señal

    # Crear el vector de tiempos / el false indica que no incluye el ultimo punto
    tt = np.linspace(0, Ttotal, nn, endpoint=False)

    # Generar la señal senoidal
    yy = dc + vmax * np.sin(2 * np.pi * f0 * tt + ph)
    return tt, yy

#IMPLEMENTACION

plt.figure(1)
# Prueba general
tt, xx = crear_sin(1, 0, 4, 0.5*np.pi, 50, 100)

# Graficar
plt.plot(tt, xx,'m', marker='*', linestyle='-')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal Muestreada")
plt.grid()
plt.show()

# Ejercicios de clase puse nn=100 para que se vea mejor

plt.figure(2)
tt, xx = crear_sin(1, 0, 500, 0, 100, 1000)

# Graficar
plt.plot(tt, xx,'m', marker='*', linestyle='-')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal Muestreada")
plt.grid()
plt.show()

plt.figure(3)
tt, xx = crear_sin(1, 0, 999, 0, 100, 1000)

# Graficar
plt.plot(tt, xx,'m', marker='*', linestyle='-')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal Muestreada")
plt.grid()
plt.show()

plt.figure(4)
tt, xx = crear_sin(1, 0, 1000, 0, 100, 1000)

# Graficar
plt.plot(tt, xx,'m', marker='*', linestyle='-')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal Muestreada")
plt.grid()
plt.show()

plt.figure(5)
tt, xx = crear_sin(1, 0, 1001, 0, 100, 1000)

# Graficar
plt.plot(tt, xx,'m', marker='*', linestyle='-')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal Muestreada")
plt.grid()
plt.show()




