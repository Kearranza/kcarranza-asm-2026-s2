#Estilo comun de las figuras y utilidades de espectro

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AZUL = "#1f4e79"
NARANJA = "#c55a11"
VERDE = "#2e6f40"
GRIS = "#7f7f7f"
COLORES = {"azul": AZUL, "naranja": NARANJA, "verde": VERDE, "gris": GRIS}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
})


def frecuencias(N, fs):
    return np.arange(N) * fs / N


def media_banda(X, fs):
    #Bins de 0 a fs/2: por simetria hermitica la otra mitad es redundante
    mitad = X.size // 2 + 1
    return frecuencias(X.size, fs)[:mitad], X[:mitad]


def magnitud_db(X, piso_db=-120.0):
    magnitud = np.abs(X)
    referencia = magnitud.max() if magnitud.max() > 0 else 1.0
    return 20 * np.log10(np.maximum(magnitud / referencia, 10 ** (piso_db / 20)))


def fase_limpia(X, umbral=1e-3):
    #Sin esta limpieza la fase de los bins que solo tienen ruido numerico es aleatoria
    magnitud = np.abs(X)
    return np.where(magnitud > umbral * magnitud.max(), np.angle(X), 0.0)
