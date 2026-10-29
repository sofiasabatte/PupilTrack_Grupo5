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
- **`compresion_metodos.csv`** — Módulo 4: tasa de compresión, PSNR y SSIM de Huffman, RLE (H/V) y JPEG (Q = 90/50/10/5)
- **`compresion_diametro.csv`** — Módulo 4: cuánto cambia el diámetro pupilar medido al comprimir con JPEG

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
  temporal.py                         # Módulo 3.2 — curva temporal + parámetros fisiológicos
  compression.py                      # Módulo 4 — Huffman, RLE, JPEG, PSNR/SSIM, efecto en el diámetro
  evaluacion.py                       # validación contra la referencia manual (Dice/IoU, MAE/RMSE)

analizar_registros.py         # procesa A1, A2, B1, B2 de una vez + tabla de variabilidad
anotaciones/                  # referencia manual del borde pupilar (8 frames) para validar

notebooks/                    # desarrollo y respuestas a la guía, un notebook por módulo
  Modulo1_Fundamentos.ipynb
  Modulo2_Segmentacion.ipynb
  Modulo3_Textura.ipynb         # 3.1 textura + 3.2 curvas, parámetros, variabilidad, artefactos, MAE/RMSE
  Modulo4_Compresion.ipynb

data/                         # videos de entrada
resultados/                   # CSV de salida (se generan al correr el pipeline)
```

**Los notebooks no hace falta correrlos para obtener resultados** — son el
desarrollo paso a paso de cada módulo, con las respuestas a las preguntas
de la guía. El resultado final (los CSV) sale solo de `run_pipeline.py`.

---

## Elegir el método de segmentación

En `config.yaml`, `segmentacion.metodo` acepta: `otsu`, `li`, `triangle`,
`adaptativo`, `kmeans`, `region_growing`, `chan_vese`, `snake`, `watershed`.

---


---

## Módulo 0 — Registros del grupo

Cuatro registros de unos 22–25 s, grabados con celular a ~60 fps (1 ojo en primer plano, cabeza apoyada):

| Registro | Tipo | Protocolo |
|---|---|---|
| A1, A2 | A — contracción | penumbra → se enciende la luz (flash de otro celular) |
| B1, B2 | B — redilatación | luz encendida → se apaga |

Los videos no se suben al repositorio por su tamaño. Para reproducir los análisis, copiarlos en `data/` como `A1.mov`, `A2.mov`, `B1.mov` y `B2.mov` y correr `python analizar_registros.py` o los notebooks.

## Resumen de resultados

| Registro | Basal → extremo (px) | Amplitud | Latencia | Frames válidos |
|---|---|---|---|---|
| A1 | 75,4 → 53,4 | 29,2 % | 0,23 s | 92 % |
| A2 | 68,4 → 59,5 | 13,0 % | 0,57 s* | 88 % |
| B1 | 56,6 → 67,5 | 19,4 % | 0,20 s | 95 % |
| B2 | 56,0 → 66,6 | 19,0 % | 0,53 s | 88 % |

\* Latencia de A2 afectada por un artefacto de iluminación en el instante del flash (ver Módulo 3).

- **Segmentación** (Módulo 2): watershed con marcadores sobre el gradiente, con un Dice de **0,94** contra la referencia manual (Otsu 0,25; K-means K = 5 0,78; Snake 0,76).
- **Validación del diámetro** (Módulo 3.2): **MAE = 2,4 px (4,1 %)**, **RMSE = 2,8 px** sobre 8 frames anotados a mano.
- **Compresión** (Módulo 4): JPEG Q = 50 (19,7:1) no altera el diámetro (0,16 px); Q = 10 introduce un error de 0,85 px en promedio.
