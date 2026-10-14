"""
Procesamiento temporal completo (Módulos 3.2 y 4): recorre el video frame
a frame, segmenta la pupila con el método elegido en config.yaml, mide
área/diámetro/centroide, arma la curva temporal y deriva los parámetros
fisiológicos (latencia, amplitud, velocidad de constricción/redilatación).
"""
import cv2
import numpy as np
import pandas as pd

from . import acquisition as acq
from . import segmentation as seg
from . import morphology as morf


def roi_automatica(frame_bgr, margen=1.5):
    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    m, _ = seg.umbral_otsu(gris)
    mascara_oscura = cv2.bitwise_not(m)
    mascara_oscura = morf.mayor_componente(mascara_oscura)
    ys, xs = np.where(mascara_oscura > 0)
    if len(xs) == 0:
        return 0, 0, frame_bgr.shape[1], frame_bgr.shape[0]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lado = max(x1 - x0, y1 - y0) * margen
    x = max(0, int(cx - lado / 2))
    y = max(0, int(cy - lado / 2))
    w = min(frame_bgr.shape[1] - x, int(lado))
    h = min(frame_bgr.shape[0] - y, int(lado))
    return x, y, w, h


def segmentar_frame(gris, metodo, tolerancia=15, tam_apertura=3, tam_cierre=5):
    if metodo == "otsu":
        m, _ = seg.umbral_otsu(gris)
        mascara = cv2.bitwise_not(m)
    elif metodo == "li":
        m, _ = seg.umbral_li(gris)
        mascara = cv2.bitwise_not(m)
    elif metodo == "triangle":
        m, _ = seg.umbral_triangle(gris)
        mascara = cv2.bitwise_not(m)
    elif metodo == "adaptativo":
        m = seg.umbral_adaptativo(gris, tam_bloque=31, c=5)
        mascara = cv2.bitwise_not(m)
    elif metodo == "kmeans":
        etiquetas, centros = seg.segmentar_kmeans(gris, k=3)
        mascara = seg.mascara_desde_kmeans(etiquetas, centros, cluster_mas_oscuro=True)
    elif metodo == "region_growing":
        semilla = seg.elegir_semilla_automatica(gris, modo="minimo")
        mascara = seg.region_growing(gris, semilla, tolerancia=tolerancia)
    elif metodo == "chan_vese":
        mascara = seg.segmentar_chan_vese(gris, iteraciones=100)
    elif metodo == "snake":
        mascara = seg.segmentar_snake(gris)
    elif metodo == "watershed":
        mascara = seg.segmentar_watershed(gris)
    else:
        raise ValueError(f"Método de segmentación desconocido: {metodo}")

    limpia = morf.limpiar_mascara(mascara, tam_apertura, tam_cierre)
    limpia = morf.mayor_componente(limpia)
    return limpia


def medir_pupila(mascara):
    """Área, diámetro equivalente y centroide del blob principal."""
    desc = morf.descriptores_morfometricos(mascara)
    area = desc["area_px"]
    diametro = 2 * np.sqrt(area / np.pi) if area and not np.isnan(area) else np.nan
    return area, diametro, desc["centroide_x"], desc["centroide_y"]


def procesar_video(cfg):
    video_cfg = cfg["video"]
    seg_cfg = cfg["segmentacion"]

    frames, fps = acq.extraer_frames(video_cfg["path"], video_cfg.get("cada_n_frames", 1))
    if not frames:
        raise RuntimeError(f"No se pudieron leer frames de: {video_cfg['path']}")

    roi = video_cfg.get("roi")
    if roi is None:
        roi = roi_automatica(frames[0])
    x, y, w, h = roi

    filas = []
    for i, frame in enumerate(frames):
        recorte = frame[y:y + h, x:x + w]
        gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        mascara = segmentar_frame(
            gris, seg_cfg["metodo"],
            seg_cfg.get("tolerancia_region_growing", 15),
            seg_cfg.get("tam_apertura", 3),
            seg_cfg.get("tam_cierre", 5),
        )
        area, diametro, cx, cy = medir_pupila(mascara)
        tiempo_s = (i * video_cfg.get("cada_n_frames", 1)) / fps if fps > 0 else i
        filas.append({
            "frame": i, "tiempo_s": tiempo_s,
            "area_px": area, "diametro_px": diametro,
            "centroide_x": cx, "centroide_y": cy,
        })

    df = pd.DataFrame(filas)
    px_por_mm = cfg["calibracion"].get("px_por_mm")
    if px_por_mm:
        df["diametro_mm"] = df["diametro_px"] / px_por_mm
        df["area_mm2"] = df["area_px"] / (px_por_mm ** 2)

    return df, fps, roi


def _suavizar(serie, ventana=5):
    return serie.rolling(window=ventana, center=True, min_periods=1).median()


def inferir_tipo(df):
    col = "diametro_mm" if "diametro_mm" in df.columns else "diametro_px"
    inicio = df[col].iloc[: max(1, len(df) // 10)].median()
    fin = df[col].iloc[-max(1, len(df) // 10):].median()
    return "A" if fin < inicio else "B"


def calcular_parametros_fisiologicos(df, fps, tipo="auto"):
    col = "diametro_mm" if "diametro_mm" in df.columns else "diametro_px"
    unidad = "mm" if col == "diametro_mm" else "px"

    d = _suavizar(df[col])
    t = df["tiempo_s"]

    if tipo == "auto":
        tipo = inferir_tipo(df)

    d0 = d.iloc[: max(1, len(d) // 10)].median()

    if tipo == "A":
        objetivo = d.idxmin()
        d_extremo = d.loc[objetivo]
        umbral_inicio = d0 - 0.05 * abs(d0 - d.min())
    else:
        objetivo = d.idxmax()
        d_extremo = d.loc[objetivo]
        umbral_inicio = d0 + 0.05 * abs(d.max() - d0)

    if tipo == "A":
        idx_inicio = d[d < umbral_inicio].index.min()
    else:
        idx_inicio = d[d > umbral_inicio].index.min()
    idx_inicio = idx_inicio if pd.notna(idx_inicio) else 0

    latencia_s = t.loc[idx_inicio] - t.iloc[0]
    amplitud = abs(d0 - d_extremo)
    amplitud_pct = 100 * amplitud / d0 if d0 else np.nan

    tramo = d.loc[idx_inicio:objetivo]
    tramo_t = t.loc[idx_inicio:objetivo]
    if len(tramo) > 1:
        derivada = np.gradient(tramo.values, tramo_t.values)
        velocidad_max = np.max(np.abs(derivada))
    else:
        velocidad_max = np.nan

    return {
        "tipo_inferido": tipo,
        "diametro_basal": round(d0, 3),
        "diametro_extremo": round(d_extremo, 3),
        "amplitud": round(amplitud, 3),
        "amplitud_pct": round(amplitud_pct, 2),
        "latencia_s": round(latencia_s, 3),
        "velocidad_max": round(velocidad_max, 3),
        "unidad_longitud": unidad,
        "unidad_velocidad": f"{unidad}/s",
    }
