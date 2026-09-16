#Corre los puntos b), c) y d) y deja todo en resultados/

import os

import benchmark
import efectos
import espectros

RAIZ = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(RAIZ, "resultados")
FIGS = os.path.join(SALIDA, "figs")


def main():
    os.makedirs(FIGS, exist_ok=True)

    print("== b) Tiempos de ejecucion DFT vs FFT ==")
    filas = benchmark.correr()
    benchmark.guardar_csv(filas, os.path.join(SALIDA, "tiempos.csv"))
    benchmark.guardar_tabla(filas, os.path.join(SALIDA, "tiempos.txt"))
    benchmark.figura(filas, os.path.join(FIGS, "tiempos.png"))

    print("\n== c) Magnitud y fase ==")
    espectros.figura(os.path.join(FIGS, "magnitud_fase.png"))
    print("   magnitud_fase.png")

    print("\n== d) Efectos principales ==")
    efectos.fuga_espectral(os.path.join(FIGS, "fuga_espectral.png"))
    efectos.resolucion_frecuencial(os.path.join(FIGS, "resolucion.png"))
    error_fase = efectos.desplazamiento(os.path.join(FIGS, "desplazamiento.png"))
    error_simetria = efectos.simetria_hermitica(os.path.join(FIGS, "simetria_hermitica.png"))
    print(f"   error de la fase respecto a la teorica: {error_fase:.2e} rad")
    print(f"   error de la simetria hermitica:         {error_simetria:.2e}")

    print(f"\nResultados en {SALIDA}")


if __name__ == "__main__":
    main()
