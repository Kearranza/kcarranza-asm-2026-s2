#Punto c) - Representacion de magnitud y fase de diferentes senales

import numpy as np

from dft_fft import fft
from graficas import plt, AZUL, NARANJA, VERDE, frecuencias, magnitud_db, fase_limpia
import senales as sg

N = 512
FS = sg.FS


def casos():
    return [
        ("Suma de tonos (500, 1200, 2500 Hz)",
         sg.suma_de_tonos(N, [500, 1200, 2500], [1.0, 0.6, 0.3], FS)),
        ("Pulso rectangular (32 muestras, retardo 64)",
         sg.pulso_rectangular(N, 32, 64)),
        ("Chirp lineal 300 -> 3000 Hz",
         sg.chirp_lineal(N, 300, 3000, FS)),
        ("Tono de 1 kHz con ruido (SNR 5 dB)",
         sg.tono_con_ruido(N, 1000.0, snr_db=5, fs=FS)),
    ]


def figura(ruta):
    lista = casos()
    fig, ejes = plt.subplots(len(lista), 3, figsize=(10.5, 2.3 * len(lista)))
    t_ms = sg.eje_tiempo(N, FS) * 1e3

    for fila, (titulo, x) in enumerate(lista):
        X = fft(x)
        f = frecuencias(N, FS)[:N // 2 + 1]

        ejes[fila, 0].plot(t_ms, x, color=AZUL)
        ejes[fila, 0].set_ylabel("Amplitud")
        ejes[fila, 0].set_title(titulo, fontsize=9, loc="left")

        ejes[fila, 1].plot(f, magnitud_db(X)[:f.size], color=NARANJA)
        ejes[fila, 1].set_ylim(-90, 5)
        ejes[fila, 1].set_ylabel("|X(f)| [dB]")

        ejes[fila, 2].plot(f, fase_limpia(X)[:f.size], color=VERDE, linewidth=0.9)
        ejes[fila, 2].set_ylim(-np.pi - 0.3, np.pi + 0.3)
        ejes[fila, 2].set_yticks([-np.pi, 0, np.pi])
        ejes[fila, 2].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        ejes[fila, 2].set_ylabel("Fase [rad]")

    ejes[0, 1].set_title("Magnitud", fontsize=9)
    ejes[0, 2].set_title("Fase", fontsize=9)
    for columna, etiqueta in enumerate(["Tiempo [ms]", "Frecuencia [Hz]", "Frecuencia [Hz]"]):
        ejes[-1, columna].set_xlabel(etiqueta)

    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
