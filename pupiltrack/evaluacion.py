"""
Evaluación contra la referencia manual (Módulos 2.2 y 3.2).

La referencia está en `anotaciones/anotaciones_manuales.csv`: para cada
frame anotado se marcaron a mano, sobre el frame ampliado con una grilla de
5 px, los extremos izquierdo, derecho, superior e inferior del borde
pupilar. La pupila de referencia es la elipse que pasa por esos extremos.
"""
import os

import cv2
import numpy as np
import pandas as pd

from . import segmentation as seg
from . import morphology as morf
from . import temporal as tmp

RUTA_ANOTACIONES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "anotaciones", "anotaciones_manuales.csv")


def cargar_anotaciones(ruta=RUTA_ANOTACIONES):
    a = pd.read_csv(ruta)
    a["cx"] = (a["x_izq"] + a["x_der"]) / 2
    a["cy"] = (a["y_sup"] + a["y_inf"]) / 2
    a["eje_x"] = a["x_der"] - a["x_izq"]
    a["eje_y"] = a["y_inf"] - a["y_sup"]
    a["diametro_manual_px"] = np.sqrt(a["eje_x"] * a["eje_y"])        # diámetro equivalente
    a["area_manual_px"] = np.pi * a["eje_x"] * a["eje_y"] / 4
    return a


def mascara_referencia(fila, shape, offset=(0, 0)):
    """Elipse de referencia en coordenadas de una imagen de tamaño `shape`
    cuyo origen está en `offset` (x0, y0) del frame completo."""
    m = np.zeros(shape[:2], np.uint8)
    centro = (int(round(fila["cx"] - offset[0])), int(round(fila["cy"] - offset[1])))
    ejes = (int(round(fila["eje_x"] / 2)), int(round(fila["eje_y"] / 2)))
    cv2.ellipse(m, centro, ejes, 0, 0, 360, 255, -1)
    return m


def leer_frame(video, indice):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {indice} de {video}")
    return frame


def roi_automatica_pupila(frame_bgr, factor=4.0):
    """ROI cuadrada centrada en la pupila localizada AUTOMÁTICAMENTE (sin usar
    la anotación manual). Devuelve (gris_roi, (x0, y0), centro_roi, radio)."""
    (cx, cy), area = tmp.localizar_pupila_inicial(frame_bgr)
    r = np.sqrt(area / np.pi)
    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lado = int(2 * r * factor)
    x0, y0 = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
    roi = gris[y0:y0 + lado, x0:x0 + lado]
    return roi, (x0, y0), np.array([cx - x0, cy - y0]), r


def componente_central(mascara, centro):
    """Componente conexa que contiene el centro (o la más cercana a él)."""
    m = (mascara > 0).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m * 255
    x, y = int(np.clip(centro[0], 0, m.shape[1] - 1)), int(np.clip(centro[1], 0, m.shape[0] - 1))
    k = lab[y, x]
    if k == 0:
        k = 1 + int(np.argmin(np.hypot(*(cen[1:] - centro).T)))
    return (lab == k).astype(np.uint8) * 255


def segmentar_metodo(gris, metodo, centro, r, k=3):
    """Máscara cruda de la pupila con cada método de la guía."""
    if metodo == "Otsu":
        m, _ = seg.umbral_otsu(gris)
        return cv2.bitwise_not(m)
    if metodo.startswith("K-means"):
        cv2.setRNGSeed(0)                      # inicialización reproducible
        et, ce = seg.segmentar_kmeans(gris, k=k)
        return seg.mascara_desde_kmeans(et, ce, cluster_mas_oscuro=True)
    if metodo == "Snake":
        suave = cv2.GaussianBlur(gris, (0, 0), 2)
        ini = seg.contorno_inicial_circular(centro, 1.5 * r, 200)
        fin = seg.snake_activo(suave, ini, alpha=0.015, beta=10, gamma=0.001)
        return seg.contorno_a_mascara(fin, gris.shape)
    if metodo == "Watershed (distancia)":
        return seg.segmentar_watershed(gris)
    if metodo == "Watershed (marcadores)":
        m = tmp.refinar_borde_watershed(gris, centro, r)
        return m if m is not None else np.zeros_like(gris)
    if metodo == "Region growing":
        return tmp.segmentar_frame(gris, "region_growing", solo_pupila=False)
    raise ValueError(metodo)


def limpiar(mascara, centro, variante="apertura+cierre+relleno"):
    """Variantes de limpieza morfológica (Módulo 2.2)."""
    m = (mascara > 0).astype(np.uint8) * 255
    if "apertura" in variante:
        m = morf.apertura(m, 5)
    if "cierre" in variante:
        m = morf.cierre(m, 7)
    m = componente_central(m, centro)
    if "relleno" in variante:
        from scipy import ndimage as ndi
        m = ndi.binary_fill_holes(m > 0).astype(np.uint8) * 255
    if "convexa" in variante:
        m = tmp._envolvente(m)
    return m


def metricas(m, ref):
    return {"dice": morf.dice(m, ref), "iou": morf.jaccard_iou(m, ref)}


def errores_diametro(medido, manual):
    e = np.asarray(medido, float) - np.asarray(manual, float)
    return {"MAE_px": float(np.mean(np.abs(e))), "RMSE_px": float(np.sqrt(np.mean(e ** 2))),
            "sesgo_px": float(np.mean(e)), "MAE_%": float(100 * np.mean(np.abs(e) / np.asarray(manual, float)))}
