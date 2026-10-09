# PupilTrack

Pipeline para el análisis del reflejo fotomotor pupilar — PSIB, ITBA
(2do cuatrimestre 2026). Grupo 5.

## Objetivo

Automatizar el procesamiento de un video del reflejo pupilar para obtener,
de punta a punta, la curva temporal del diámetro de la pupila y los
parámetros fisiológicos del reflejo (latencia, amplitud, velocidad de
constricción/redilatación).

## Estructura del repositorio

```
config.yaml                 # única config a editar para correr con un video nuevo
run_pipeline.py              # entrypoint: corre Módulos 0-4 de punta a punta
generar_video_sintetico.py   # crea un video de prueba si no hay uno real todavía
pupiltrack/
  config.py                  # carga config.yaml
  acquisition.py              # Módulo 0 — extracción de frames, ROI
  preprocessing.py            # Módulo 1.1/1.2 — color, cuantización, histograma
  filtering.py                 # Módulo 1.2 — ruido y filtros, PSNR/SSIM
  edges_frequency.py           # Módulo 1.3 — bordes y Fourier
  segmentation.py               # Módulo 2.1 — Otsu, K-means, Snakes, Watershed
  morphology.py                  # Módulo 2.2 — morfología, Dice/IoU, descriptores de forma
  texture.py                     # Módulo 3.1 — textura 1er orden + GLCM
  temporal.py                    # Módulos 3.2/4 — curva temporal + parámetros fisiológicos
notebooks/
  Modulo1_Fundamentos.ipynb       # desarrollo y respuestas a la guía, sección 1
  Modulo2_Segmentacion.ipynb      # desarrollo y respuestas a la guía, sección 2
  Modulo3_Textura.ipynb           # desarrollo y respuestas a la guía, sección 3
data/                        # videos de entrada
resultados/                  # CSV de salida (generados al correr el pipeline)
```

## División de trabajo (Grupo 5)

| Quién | Módulos | Archivos |
|---|---|---|
| Persona 1 | 0 y 1 — Fundamentos y mejoramiento | `acquisition.py`, `preprocessing.py`, `filtering.py`, `edges_frequency.py`, `Modulo1_Fundamentos.ipynb` |
| Persona 2 | 2 — Segmentación y morfología | `segmentation.py`, `morphology.py`, `Modulo2_Segmentacion.ipynb` |
| Persona 3 | 3 y 4 — Textura, dinámica temporal e integración final | `texture.py`, `temporal.py`, `run_pipeline.py`, `Modulo3_Textura.ipynb` |

Cada módulo está armado como esqueleto (firma de función + docstring +
`TODO: implementar`) para que cada persona complete su parte en su propia
rama y la integre con un Pull Request.

## Cómo correrlo (una vez completo)

```bash
pip install -r requirements.txt
python generar_video_sintetico.py   # si todavía no hay un video real
python run_pipeline.py
```

Genera en `resultados/`:
- `pupila_serie_temporal.csv` — frame a frame: área, diámetro, centroide
- `parametros_fisiologicos.csv` — latencia, amplitud, velocidad máxima
- `textura_glcm.csv` — textura de primer orden + GLCM por canal (R, G, B)

## Elegir el método de segmentación

En `config.yaml`, `segmentacion.metodo` acepta: `otsu`, `li`, `triangle`,
`adaptativo`, `kmeans`, `region_growing`, `chan_vese`, `snake`, `watershed`.
