# PupilTrack

Pipeline para el análisis del reflejo fotomotor pupilar — PSIB, ITBA
(2do cuatrimestre 2026). Grupo 5.

Automatiza el procesamiento de un video del reflejo pupilar para obtener,
de punta a punta, la curva temporal del diámetro de la pupila y los
parámetros fisiológicos del reflejo (latencia, amplitud, velocidad de
constricción/redilatación).

---

## Para evaluar con un video nuevo (cátedra)

No hace falta tocar ningún archivo de código. Solo:

1. Poner el video en la carpeta `data/`
2. Abrir `config.yaml` y cambiar una línea:
   ```yaml
   video:
     path: "data/SU_VIDEO.mp4"
   ```
3. Instalar dependencias (una sola vez) y correr:
   ```bash
   pip install -r requirements.txt
   python run_pipeline.py
   ```

Esto genera en `resultados/`:
- **`pupila_serie_temporal.csv`** — frame a frame: área, diámetro, centroide
- **`parametros_fisiologicos.csv`** — latencia, amplitud, velocidad máxima
- **`textura_glcm.csv`** — textura de primer orden + GLCM por canal (R, G, B)

Si todavía no hay un video real para probar, `python generar_video_sintetico.py`
crea uno de prueba en `data/registro_ejemplo.mp4`.

---

## Estructura del repositorio

```
config.yaml                  # única config a editar para correr con un video nuevo
run_pipeline.py               # entrypoint: corre todo el pipeline de punta a punta
generar_video_sintetico.py    # crea un video de prueba si no hay uno real

pupiltrack/                   # funciones del proyecto, organizadas por módulo de la guía
  config.py                     # carga config.yaml
  acquisition.py                 # Módulo 0 — extracción de frames, ROI
  preprocessing.py               # Módulo 1.1/1.2 — color, cuantización, histograma
  filtering.py                    # Módulo 1.2 — ruido y filtros, PSNR/SSIM
  edges_frequency.py              # Módulo 1.3 — bordes y Fourier
  segmentation.py                  # Módulo 2.1 — Otsu, K-means, Snakes, Watershed
  morphology.py                     # Módulo 2.2 — morfología, Dice/IoU, descriptores de forma
  texture.py                         # Módulo 3.1 — textura 1er orden + GLCM
  temporal.py                         # Módulos 3.2/4 — curva temporal + parámetros fisiológicos

notebooks/                    # desarrollo y respuestas a la guía, un notebook por módulo
  Modulo1_Fundamentos.ipynb
  Modulo2_Segmentacion.ipynb
  Modulo3_Textura.ipynb

data/                         # videos de entrada
resultados/                   # CSV de salida (se generan al correr el pipeline)
```

**Los notebooks no hace falta correrlos para obtener resultados** — son el
desarrollo paso a paso de cada módulo, con las respuestas a las preguntas
de la guía. El resultado final (los 3 CSV) sale solo de `run_pipeline.py`.

---

## Elegir el método de segmentación

En `config.yaml`, `segmentacion.metodo` acepta: `otsu`, `li`, `triangle`,
`adaptativo`, `kmeans`, `region_growing`, `chan_vese`, `snake`, `watershed`.

---

