"""
Procesa todos los registros del grupo en una sola corrida, sin tocar
config.yaml (ese archivo queda intacto para la evaluación de la cátedra).

Uso:
    python analizar_registros.py

Por cada registro genera resultados/<nombre>/ con los CSV estándar
(serie temporal, parámetros fisiológicos, textura GLCM) y, al final,
resultados/resumen_variabilidad.csv con los parámetros de todos los
registros y la media, el desvío y el CV por tipo (Módulo 3.2).

Editar REGISTROS si los videos se llaman distinto.
"""
import copy
import os

import pandas as pd

from pupiltrack.config import cargar_config
from pupiltrack.temporal import procesar_video, calcular_parametros_fisiologicos
from pupiltrack import acquisition as acq
from pupiltrack import texture as tex

# --- Editar si hace falta: nombre -> (ruta del video, tipo de registro) ---
REGISTROS = {
    "A1": ("data/A1.mov", "A"),
    "A2": ("data/A2.mov", "A"),
    "B1": ("data/B1.mov", "B"),
    "B2": ("data/B2.mov", "B"),
}


def procesar(nombre, video_path, tipo, carpeta_out):
    if not os.path.exists(video_path):
        print(f"[AVISO] No se encontró {video_path}: se saltea {nombre}")
        return None

    cfg = copy.deepcopy(cargar_config("config.yaml"))
    cfg["video"]["path"] = video_path
    cfg["video"]["tipo"] = tipo

    print(f"\n=== {nombre}: {video_path} (tipo {tipo}) ===")
    df, fps, roi = procesar_video(cfg)
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
        filas = [{"canal": c, **primer_orden[c], **glcm_rgb[c]} for c in ["R", "G", "B"]]
        pd.DataFrame(filas).to_csv(os.path.join(carpeta_out, "textura_glcm.csv"), index=False)
    except Exception as e:
        print(f"[AVISO] No se pudo calcular textura: {e}")

    for k in ["diametro_basal", "diametro_extremo", "amplitud_pct", "latencia_s", "frames_validos_pct"]:
        print(f"  {k}: {parametros[k]}")
    return {"registro": nombre, **parametros}


if __name__ == "__main__":
    filas = []
    for nombre, (ruta, tipo) in REGISTROS.items():
        fila = procesar(nombre, ruta, tipo, os.path.join("resultados", nombre))
        if fila:
            filas.append(fila)

    if filas:
        t = pd.DataFrame(filas)
        num = ["diametro_basal", "diametro_extremo", "amplitud", "amplitud_pct", "latencia_s",
               "velocidad_constriccion", "velocidad_redilatacion", "frames_validos_pct"]
        resumen = [t]
        for tipo, g in t.groupby("tipo_inferido"):
            for est, f in [("media", g[num].mean()), ("desvio", g[num].std(ddof=1)),
                           ("CV %", 100 * g[num].std(ddof=1) / g[num].mean())]:
                resumen.append(pd.DataFrame([{"registro": f"{tipo} {est}", **f.round(3).to_dict()}]))
        out = pd.concat(resumen, ignore_index=True)
        out.to_csv(os.path.join("resultados", "resumen_variabilidad.csv"), index=False)
        print("\nResumen de variabilidad guardado en resultados/resumen_variabilidad.csv")
