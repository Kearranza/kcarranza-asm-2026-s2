#Punto b) - Comparacion de tiempos de ejecucion entre DFT y FFT para distintos N

import csv
import time

import numpy as np

from dft_fft import dft, dft_matricial, fft_recursiva, fft
from graficas import plt, COLORES, GRIS
from senales import tono_con_ruido

# La DFT de dos lazos tarda segundos ya en N=512; la matricial guarda una matriz
# NxN de complejos, que en N=2048 son unos 67 MB.
N_MAXIMO_DFT_LAZOS = 256
N_MAXIMO_DFT_MATRICIAL = 2048

EXPONENTES = range(5, 17)


def cronometrar(funcion, x, repeticiones=3):
    #Se toma el minimo y no el promedio: es la medida menos contaminada por el resto del sistema
    mejor = float("inf")
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion(x)
        mejor = min(mejor, time.perf_counter() - inicio)
    return mejor


def correr(exponentes=EXPONENTES, repeticiones=3):
    filas = []
    for p in exponentes:
        N = 2 ** p
        x = tono_con_ruido(N, 1000.0, snr_db=20)

        t_lazos = cronometrar(dft, x, 1) if N <= N_MAXIMO_DFT_LAZOS else None
        t_matricial = (cronometrar(dft_matricial, x, repeticiones)
                       if N <= N_MAXIMO_DFT_MATRICIAL else None)
        t_recursiva = cronometrar(fft_recursiva, x, repeticiones)
        t_iterativa = cronometrar(fft, x, repeticiones)

        filas.append({
            "N": N,
            "dft_lazos_s": t_lazos,
            "dft_matricial_s": t_matricial,
            "fft_recursiva_s": t_recursiva,
            "fft_iterativa_s": t_iterativa,
            "aceleracion": t_matricial / t_iterativa if t_matricial else None,
        })
        print(f"N={N:6d}  DFT={_f(t_matricial)}  FFT={_f(t_iterativa)}  "
              f"aceleracion={_x(filas[-1]['aceleracion'])}")
    return filas


def _f(v):
    return "    -     " if v is None else f"{v:10.6f}"


def _x(v):
    return "   -  " if v is None else f"{v:6.1f}x"


COLUMNAS = [("N", "N", None),
            ("dft_lazos_s", "DFT lazos [ms]", 3),
            ("dft_matricial_s", "DFT matricial [ms]", 3),
            ("fft_recursiva_s", "FFT recursiva [ms]", 3),
            ("fft_iterativa_s", "FFT iterativa [ms]", 3),
            ("aceleracion", "Aceleracion", 1)]


def _celda(fila, clave, decimales):
    #Los tiempos se guardan en segundos y se reportan en milisegundos
    valor = fila[clave]
    if valor is None:
        return ""
    if decimales is None:
        return str(valor)
    if clave.endswith("_s"):
        valor *= 1e3
    return f"{valor:.{decimales}f}"


def guardar_csv(filas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow([titulo for _, titulo, _ in COLUMNAS])
        for fila in filas:
            escritor.writerow([_celda(fila, c, d) for c, _, d in COLUMNAS])


def guardar_tabla(filas, ruta):
    #Version alineada para leer de un vistazo o pegar en el documento
    encabezados = [titulo for _, titulo, _ in COLUMNAS]
    cuerpo = [[_celda(fila, c, d) for c, _, d in COLUMNAS] for fila in filas]
    anchos = [max(len(e), *(len(f[i]) for f in cuerpo))
              for i, e in enumerate(encabezados)]

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("  ".join(e.rjust(a) for e, a in zip(encabezados, anchos)) + "\n")
        f.write("  ".join("-" * a for a in anchos) + "\n")
        for fila in cuerpo:
            f.write("  ".join(v.rjust(a) for v, a in zip(fila, anchos)) + "\n")


def figura(filas, ruta):
    fig, eje = plt.subplots(figsize=(7.0, 4.2))

    def serie(clave):
        n = np.array([f["N"] for f in filas if f[clave] is not None])
        t = np.array([f[clave] for f in filas if f[clave] is not None])
        return n, t

    curvas = [("dft_lazos_s", "DFT por definicion (dos lazos)", GRIS, "^"),
              ("dft_matricial_s", "DFT matricial $O(N^2)$", COLORES["naranja"], "o"),
              ("fft_recursiva_s", "FFT radix-2 recursiva", COLORES["verde"], "d"),
              ("fft_iterativa_s", "FFT radix-2 iterativa $O(N\\log N)$", COLORES["azul"], "s")]
    for clave, etiqueta, color, marcador in curvas:
        n, t = serie(clave)
        if n.size:
            eje.loglog(n, t, marker=marcador, markersize=4, color=color, label=etiqueta)

    # Recta de pendiente 2 anclada al ultimo punto de la DFT, como referencia visual
    n, t = serie("dft_matricial_s")
    eje.loglog(n, t[-1] * (n / n[-1]) ** 2, color=GRIS,
               linestyle=":", linewidth=0.8, label="pendiente $N^2$")

    eje.set_xlabel("Tamano de la transformada $N$")
    eje.set_ylabel("Tiempo de ejecucion [s]")
    eje.grid(True, which="both", alpha=0.25, linewidth=0.5)
    eje.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
