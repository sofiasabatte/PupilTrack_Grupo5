"""
Pipeline PupilTrack — entrega final (Módulos 0 a 4).

Uso:
    python run_pipeline.py [ruta/a/config.yaml]

TODO: implementar una vez que estén listos los módulos de pupiltrack/.
Debe:
  1. Cargar config.yaml
  2. Leer el video indicado en video.path
  3. Correr Módulos 0-2 (adquisición + segmentación) frame a frame
  4. Medir área/diámetro/centroide por frame (Módulo 3.2)
  5. Calcular parámetros fisiológicos: latencia, amplitud, velocidad (Módulo 4)
  6. Calcular textura GLCM sobre un frame representativo (Módulo 3.1)
  7. Guardar todo en resultados/*.csv
"""
import sys


def main():
    raise NotImplementedError("Pendiente: armar una vez integrados todos los módulos")


if __name__ == "__main__":
    main()
