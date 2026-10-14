"""
Módulo 2.2 - Morfología matemática para limpieza de máscara, métricas de
comparación (Dice/IoU) y descriptores morfométricos de forma (Módulo 3.1).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label


def _kernel(tam=3, forma=cv2.MORPH_ELLIPSE):
    return cv2.getStructuringElement(forma, (tam, tam))


def erosionar(mascara, tam=3, iteraciones=1):
    return cv2.erode(mascara, _kernel(tam), iterations=iteraciones)


def dilatar(mascara, tam=3, iteraciones=1):
    return cv2.dilate(mascara, _kernel(tam), iterations=iteraciones)


def apertura(mascara, tam=3):
    """Erosión + dilatación: elimina destellos/puntos pequeños (reflejo
    corneal del flash) sin achicar demasiado el objeto principal."""
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, _kernel(tam))


def cierre(mascara, tam=3):
    """Dilatación + erosión: rellena huecos pequeños (oclusión por
    pestañas o sombras)."""
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _kernel(tam))


def limpiar_mascara(mascara, tam_apertura=3, tam_cierre=5):
    limpia = apertura(mascara, tam_apertura)
    limpia = cierre(limpia, tam_cierre)
    return limpia


def esqueletizar(mascara):
    binaria = (mascara > 0)
    esqueleto = skeletonize(binaria)
    return (esqueleto.astype(np.uint8)) * 255


def mayor_componente(mascara):
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, 8)
    if n <= 1:
        return mascara
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (etiquetas == idx).astype(np.uint8) * 255


# ---------- Métricas de comparación (máscara vs. referencia manual) ----------

def dice(mascara_a, mascara_b):
    a = mascara_a > 0
    b = mascara_b > 0
    interseccion = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return (2.0 * interseccion / total) if total > 0 else 1.0


def jaccard_iou(mascara_a, mascara_b):
    a = mascara_a > 0
    b = mascara_b > 0
    interseccion = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (interseccion / union) if union > 0 else 1.0


def comparar_metodos(mascaras_dict, referencia):
    resultados = {}
    for nombre, mascara in mascaras_dict.items():
        resultados[nombre] = {
            "dice": round(dice(mascara, referencia), 4),
            "iou": round(jaccard_iou(mascara, referencia), 4),
        }
    return resultados


# ---------- Módulo 3.1 — Descriptores morfométricos (forma) ----------

def descriptores_morfometricos(mascara):
    """Área, perímetro, centroide y circularidad/excentricidad del blob
    principal de la máscara, usando skimage.measure.regionprops.

    Circularidad = 4*pi*Area / Perimetro^2  (1.0 = círculo perfecto;
    se achica con la deformación/irregularidad del borde pupilar).
    """
    props = regionprops(label(mascara > 0))
    if not props:
        return {
            "area_px": np.nan, "perimetro_px": np.nan,
            "centroide_x": np.nan, "centroide_y": np.nan,
            "circularidad": np.nan, "excentricidad": np.nan,
        }
    p = max(props, key=lambda r: r.area)
    area = p.area
    perimetro = p.perimeter if p.perimeter > 0 else np.nan
    circularidad = (4 * np.pi * area / (perimetro ** 2)) if perimetro and perimetro > 0 else np.nan
    cy, cx = p.centroid
    return {
        "area_px": area,
        "perimetro_px": perimetro,
        "centroide_x": cx,
        "centroide_y": cy,
        "circularidad": round(circularidad, 4) if not np.isnan(circularidad) else np.nan,
        "excentricidad": round(p.eccentricity, 4),
    }
