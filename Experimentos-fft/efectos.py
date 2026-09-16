#Punto d) - Imagenes de los efectos principales de la DFT

import numpy as np

from dft_fft import fft
from graficas import plt, AZUL, NARANJA, VERDE, GRIS, frecuencias, magnitud_db, fase_limpia
import senales as sg

FS = sg.FS


def fuga_espectral(ruta):
    #Un tono que no cae en un bin reparte su energia por todo el espectro
    N, k = 256, 20
    f = frecuencias(N, FS)[:N // 2 + 1]
    entre_bins = sg.senoidal(N, (k + 0.5) * FS / N, FS)

    fig, eje = plt.subplots(figsize=(7.0, 3.2))
    curvas = [(sg.senoidal_en_bin(N, k, FS), f"Tono en el bin k={k} (sin fuga)", AZUL, "-"),
              (entre_bins, f"Tono en k={k}.5 (con fuga)", NARANJA, "-"),
              (entre_bins * sg.ventana_hann(N), f"Tono en k={k}.5 con ventana de Hann", VERDE, "--")]
    for x, etiqueta, color, estilo in curvas:
        eje.plot(f, magnitud_db(fft(x))[:f.size], label=etiqueta, color=color, linestyle=estilo)

    eje.set_xlim(0, 1500)
    eje.set_ylim(-90, 5)
    eje.set_xlabel("Frecuencia [Hz]")
    eje.set_ylabel("|X(f)| [dB]")
    eje.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def resolucion_frecuencial(ruta):
    #Dos tonos cercanos solo se separan si el bloque es lo bastante largo
    f1, f2 = 1000.0, 1050.0
    fig, eje = plt.subplots(figsize=(7.0, 3.2))
    for N, color in [(128, GRIS), (512, NARANJA), (2048, AZUL)]:
        x = sg.suma_de_tonos(N, [f1, f2], fs=FS) * sg.ventana_hann(N)
        f = frecuencias(N, FS)[:N // 2 + 1]
        eje.plot(f, magnitud_db(fft(x))[:f.size], color=color,
                 label=f"N={N} ($\\Delta f$={FS/N:.1f} Hz)")

    for marca in (f1, f2):
        eje.axvline(marca, color="k", linewidth=0.5, linestyle=":")
    eje.set_xlim(850, 1200)
    eje.set_ylim(-70, 5)
    eje.set_xlabel("Frecuencia [Hz]")
    eje.set_ylabel("|X(f)| [dB]")
    eje.set_title(f"Dos tonos separados {f2 - f1:.0f} Hz", fontsize=9, loc="left")
    eje.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def desplazamiento(ruta):
    #Un retardo deja la magnitud intacta y agrega una fase lineal de pendiente -2*pi*n0/N
    N, retardo, largo = 256, 20, 48
    x0 = sg.tren_de_onda(N, 1000.0, largo, 0, FS)
    x1 = sg.tren_de_onda(N, 1000.0, largo, retardo, FS)
    X0, X1 = fft(x0), fft(x1)
    f = frecuencias(N, FS)[:N // 2 + 1]
    magnitud = np.abs(X0)[:f.size]

    fig, ejes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    ejes[0].plot(np.arange(N), x0, color=AZUL, label="sin retardo")
    ejes[0].plot(np.arange(N), x1, color=NARANJA, label=f"retardo $n_0$={retardo}")
    ejes[0].set_xlim(0, 110)
    ejes[0].set_xlabel("Muestra $n$")
    ejes[0].set_ylabel("Amplitud")
    ejes[0].set_title("Tren de onda de 1 kHz", fontsize=9)
    ejes[0].legend(fontsize=8)

    ejes[1].plot(f, magnitud, color=AZUL, label="sin retardo")
    ejes[1].plot(f, np.abs(X1)[:f.size], color=NARANJA, linestyle="--", label="con retardo")
    ejes[1].set_xlabel("Frecuencia [Hz]")
    ejes[1].set_ylabel("$|X(f)|$")
    ejes[1].set_title("La magnitud no cambia", fontsize=9)
    ejes[1].legend(fontsize=8)

    # Solo los bins con energia apreciable: donde la magnitud es ruido la fase no significa nada
    utiles = magnitud > 1e-2 * magnitud.max()
    k = np.arange(f.size)
    medida = np.unwrap(np.angle(X1[:f.size][utiles] / X0[:f.size][utiles]))
    teorica = -2 * np.pi * k[utiles] * retardo / N
    medida += np.round((teorica[0] - medida[0]) / (2 * np.pi)) * 2 * np.pi

    ejes[2].plot(f[utiles], medida, color=VERDE, label="medida")
    ejes[2].plot(f[utiles], teorica, color=GRIS, linestyle=":", label=r"$-2\pi k n_0/N$")
    ejes[2].set_xlabel("Frecuencia [Hz]")
    ejes[2].set_ylabel("Diferencia de fase [rad]")
    ejes[2].set_title("La fase se vuelve lineal", fontsize=9)
    ejes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return float(np.max(np.abs(medida - teorica)))


def simetria_hermitica(ruta):
    #En una senal real la mitad superior del espectro es el conjugado reflejado de la inferior
    N = 128
    x = sg.suma_de_tonos(N, [500, 1500], [1.0, 0.5], FS)
    X = fft(x)
    k = np.arange(N)

    fig, ejes = plt.subplots(1, 2, figsize=(9.0, 3.0))

    ejes[0].stem(k, np.abs(X), linefmt=AZUL, markerfmt=" ", basefmt=" ")
    ejes[0].axvline(N / 2, color=GRIS, linestyle=":", linewidth=0.8)
    ejes[0].set_xlabel("Bin $k$")
    ejes[0].set_ylabel("$|X[k]|$")
    ejes[0].set_title("Magnitud par respecto a $N/2$", fontsize=9)

    # Solo los bins con energia real: la fase de los bins vacios es ruido numerico
    ejes[1].stem(k, fase_limpia(X), linefmt=VERDE, markerfmt=" ", basefmt=" ")
    ejes[1].axvline(N / 2, color=GRIS, linestyle=":", linewidth=0.8)
    ejes[1].set_xlabel("Bin $k$")
    ejes[1].set_ylabel(r"$\angle X[k]$ [rad]")
    ejes[1].set_title("Fase impar respecto a $N/2$", fontsize=9)

    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return float(np.max(np.abs(X[1:N // 2] - np.conj(X[-1:-N // 2:-1]))))
