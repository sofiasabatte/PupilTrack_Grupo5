"""
Procesa Registro A y Registro B en una sola corrida, sin tocar config.yaml
(ese archivo se deja intacto para la evaluación del profe).

Uso:
    python analizar_registros.py

Edita las rutas VIDEO_A / VIDEO_B de abajo si tus archivos se llaman distinto.
Genera resultados/registro_A/ y resultados/registro_B/, cada una con los
3 CSV estándar (serie temporal, parámetros fisiológicos, textura GLCM).
"""
import os
import copy
import pandas as pd

from pupiltrack.config import cargar_config
from pupiltrack.temporal import procesar_video, calcular_parametros_fisiologicos
from pupiltrack import acquisition as acq
from pupiltrack import texture as tex

# --- Editar si hace falta ---
VIDEO_A = "data/video1registroA.mp4"
VIDEO_B = "data/video2registroB.mp4"


def procesar(video_path, tipo, carpeta_out):
    if not os.path.exists(video_path):
        print(f"[ERROR] No se encontró: {video_path}")
        return

    cfg = copy.deepcopy(cargar_config("config.yaml"))
    cfg["video"]["path"] = video_path
    cfg["video"]["tipo"] = tipo

    print(f"\n=== Procesando {video_path} (tipo {tipo}) ===")
    df, fps, roi = procesar_video(cfg)
    print(f"Frames: {len(df)} | fps: {fps:.2f} | ROI: {roi}")

    frames_validos = df["diametro_px"].notna().sum()
    print(f"Frames con pupila detectada: {frames_validos}/{len(df)}")

    parametros = calcular_parametros_fisiologicos(df, fps, tipo=tipo)

    os.makedirs(carpeta_out, exist_ok=True)
    df.to_csv(os.path.join(carpeta_out, "pupila_serie_temporal.csv"), index=False)
    pd.DataFrame([parametros]).to_csv(os.path.join(carpeta_out, "parametros_fisiologicos.csv"), index=False)

    try:
        x, y, w, h = roi
        frames, _ = acq.extraer_frames(video_path, cfg["video"].get("cada_n_frames", 1))
        frame_rep = acq.frame_representativo(frames)[y:y + h, x:x + w]
        niveles = cfg["textura"]["niveles_glcm"]
        glcm_rgb = tex.textura_glcm_rgb(frame_rep, niveles=niveles)
        primer_orden = tex.textura_primer_orden_rgb(frame_rep)
        filas = []
        for canal in ["R", "G", "B"]:
            fila = {"canal": canal}
            fila.update(primer_orden[canal])
            fila.update(glcm_rgb[canal])
            filas.append(fila)
        pd.DataFrame(filas).to_csv(os.path.join(carpeta_out, "textura_glcm.csv"), index=False)
    except Exception as e:
        print(f"[AVISO] No se pudo calcular textura: {e}")

    print(f"\n--- Resumen {tipo} ---")
    for k, v in parametros.items():
        print(f"  {k}: {v}")
    print(f"Guardado en: {carpeta_out}/")


if __name__ == "__main__":
    procesar(VIDEO_A, "A", "resultados/registro_A")
    procesar(VIDEO_B, "B", "resultados/registro_B")
    print("\nListo. Resultados en resultados/registro_A/ y resultados/registro_B/.")
