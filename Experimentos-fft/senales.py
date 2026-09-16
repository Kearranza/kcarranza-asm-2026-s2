#Senales de prueba para los experimentos con FFT

import numpy as np

FS = 8000.0


def eje_tiempo(N, fs=FS):
    return np.arange(N) / fs


def senoidal(N, f0, fs=FS, amplitud=1.0, fase=0.0):
    return amplitud * np.sin(2 * np.pi * f0 * eje_tiempo(N, fs) + fase)


def senoidal_en_bin(N, k, fs=FS, amplitud=1.0):
    #Frecuencia que cae exactamente en el bin k, sin fuga espectral
    return senoidal(N, k * fs / N, fs, amplitud)


def suma_de_tonos(N, frecuencias, amplitudes=None, fs=FS):
    t = eje_tiempo(N, fs)
    if amplitudes is None:
        amplitudes = [1.0] * len(frecuencias)
    x = np.zeros(N)
    for f0, a in zip(frecuencias, amplitudes):
        x += a * np.sin(2 * np.pi * f0 * t)
    return x


def pulso_rectangular(N, ancho, retardo=0):
    x = np.zeros(N)
    x[retardo:retardo + ancho] = 1.0
    return x


def tren_de_onda(N, f0, largo, retardo=0, fs=FS):
    #Tono con envolvente de Hann: su espectro no tiene ceros exactos en el lobulo principal
    x = np.zeros(N)
    x[retardo:retardo + largo] = senoidal(largo, f0, fs) * ventana_hann(largo)
    return x


def chirp_lineal(N, f_inicial, f_final, fs=FS, amplitud=1.0):
    #La frecuencia instantanea barre de f_inicial a f_final de forma lineal
    t = eje_tiempo(N, fs)
    pendiente = (f_final - f_inicial) / (N / fs)
    return amplitud * np.sin(2 * np.pi * (f_inicial * t + 0.5 * pendiente * t ** 2))


def tono_con_ruido(N, f0, snr_db, fs=FS, semilla=12345):
    rng = np.random.default_rng(semilla)
    x = senoidal(N, f0, fs)
    potencia_ruido = np.mean(x ** 2) / (10 ** (snr_db / 10))
    return x + rng.normal(0.0, np.sqrt(potencia_ruido), N)


def ventana_hann(N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)
