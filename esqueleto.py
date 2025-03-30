#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:58:13 2025

@author: mariano
"""

#%% módulos y funciones a importar
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def generador_sen(vmax, dc, ff, ph, nn, fs):
    '''
    Esta funcion genera una señal senoidal.
    descripcion de los parametros:
    vmax:amplitud max de la senoidal [Volts]
    dc:valor medio [Volts]
    ff:frecuencia [Hz]
    ph:fase en [rad]
    nn:cantidad de muestras
    fs:frecuencia de muestreo [Hz]
    '''
    Ts= 1/fs
    tt=np.linspace(0,(nn-1)*Ts,nn)
    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc
    return tt, xx
    
N= 1000
fs = 1000
ff = 1
A_inicial = 13 #puse cualquier cosa, total luego voy a normalizar
tt, xx = generador_sen(A_inicial, 0, ff, 0, N, fs)
##############################################################################################################################################
print ('La potencia de mi señal es: ',np.var(xx)) #Ya que por ser una señal senoideal con media cero la potencia es la varianza
xn=xx/np.std(xx) #divido a mi señal por el desvio estandar para normalizarla
print ('Luego de normalizar, la potencia de mi señal es: ',np.var(xn)) #Ya que por ser una señal senoideal con media cero la potencia es la varianza



#%% Datos de la simulación

fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras

# Datos del ADC
B =  4# bits
Vf = 1.5# rango simétrico de +/- Vf Volts
q = 2*Vf/(2**B) # paso de cuantización de q Volts


# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = (q**2 ) / 12 # Watts 
kn = 1. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 

df = fs / N  # resolución espectral (Hz)
ts = 1 / fs  # tiempo de muestreo (segundos)


#%% Experimento: 
"""
   Se desea simular el efecto de la cuantización sobre una señal senoidal de 
   frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
   ruido gausiano e incorrelado.
   
   Se pide analizar el efecto del muestreo y cuantización sobre la señal 
   analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
   a construir para luego analizar los resultados.
   
"""

# np.random.normal
# np.random.uniform


# Señales
s = xn  # señal sin ruido
nn=np.random.normal(0,np.sqrt(pot_ruido_analog),N) #señal de ruido analogico
sr = xn + nn# señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # señal cuantizada, (señal divida la cantidad total de bits)
nq = srq-sr# señal de ruido de cuantización


#%% Visualización de resultados

# # cierro ventanas anteriores
plt.close('all')

# ##################
# # Señal temporal
# ##################

plt.figure(1)

plt.plot(tt, srq, lw=1, linestyle=':', color='blue', marker='o', markersize=2, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='Srq')
plt.plot(tt, sr, lw=1, linestyle=':', color='c', marker='o', markersize=2, markerfacecolor='c', markeredgecolor='c', fillstyle='none', label='Sr')
plt.plot(tt, xn, color='magenta',label='S') 


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

# #############
# # Histograma
# #############

plt.figure(2)

# Cálculo de la transformada de Fourier
ft_S = 1/N * np.fft.fft(s)
ft_SR = 1/N * np.fft.fft(sr)
ft_Srq = 1/N * np.fft.fft(srq)
ft_Nq = 1/N * np.fft.fft(nq)
ft_Nn = 1/N * np.fft.fft(nn)

# Grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2

# Cálculo de las medias
Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

# Crear la figura y los subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Gráfico original en el subplot izquierdo
ax1.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Srq[bfrec])**2), lw=1.5, color='purple', label='$ s_{R_Q}$')
ax1.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SR[bfrec])**2), lw=1, color='magenta', ls=':', label='$ s_R $')
ax1.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_S[bfrec])**2), lw=1, color='blue', ls=':', label='s')
ax1.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nn[bfrec])**2), lw=1, color='c', ls=':')
ax1.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nq[bfrec])**2), lw=1, color='orange', ls=':')
ax1.plot(np.array([ff[bfrec][0], ff[bfrec][-1]]), 10 * np.log10(2 * np.array([nNn_mean, nNn_mean])), color='c', ls='--', label='$ \overline{nn} = $' + '{:3.1f} dB'.format(10 * np.log10(2 * nNn_mean)))
ax1.plot(np.array([ff[bfrec][0], ff[bfrec][-1]]), 10 * np.log10(2 * np.array([Nnq_mean, Nnq_mean])), color='orange', ls='--', label='$ \overline{nq} = $' + '{:3.1f} dB'.format(10 * np.log10(2 * Nnq_mean)))

ax1.set_title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
ax1.set_ylabel('Densidad de Potencia [dB]')
ax1.set_xlabel('Frecuencia [Hz]')
ax1.legend()

# Zoom en el subplot derecho
ax2.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Srq[bfrec])**2), lw=1.5, color='purple', label='$ s_{R_Q}$')
ax2.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SR[bfrec])**2), lw=0.5, color='magenta', ls=':', label='$ s_R $')
ax2.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_S[bfrec])**2), lw=0.5, color='blue', ls=':', label='s')
ax2.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nn[bfrec])**2), lw=0.5, color='c', ls=':')
ax2.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nq[bfrec])**2), lw=0.5, color='orange', ls=':')
ax2.plot(np.array([ff[bfrec][0], ff[bfrec][-1]]), 10 * np.log10(2 * np.array([nNn_mean, nNn_mean])), color='c', ls='--', label='$ \overline{nn} = $' + '{:3.1f} dB'.format(10 * np.log10(2 * nNn_mean)))
ax2.plot(np.array([ff[bfrec][0], ff[bfrec][-1]]), 10 * np.log10(2 * np.array([Nnq_mean, Nnq_mean])), color='orange', ls='--', label='$ \overline{nq} = $' + '{:3.1f} dB'.format(10 * np.log10(2 * Nnq_mean)))

ax2.set_xlim(-10, 300)
ax2.set_ylim(-80, 0)
ax2.set_ylabel('Densidad de Potencia [dB]')
ax2.set_xlabel('Frecuencia [Hz]')
ax2.legend()

plt.tight_layout()
plt.show()




#############
# Histograma
#############


plt.figure(3)
bins = 20
plt.hist(nq.flatten()/(q), bins=bins, edgecolor='black')
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
plt.xlabel('Pasos de cuantización (q) [V]')


