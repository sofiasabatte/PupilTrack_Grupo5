"""
Módulo 2.2 - Morfología matemática para limpieza de máscara, métricas de
comparación (Dice/IoU) y descriptores morfométricos de forma (Módulo 3.1).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label

def _kernel(tam=3, forma=cv2.MORPH_ELLIPSE):
    """TODO: implementar."""
    raise NotImplementedError


def erosionar(mascara, tam=3, iteraciones=1):
    """TODO: implementar."""
    raise NotImplementedError


def dilatar(mascara, tam=3, iteraciones=1):
    """TODO: implementar."""
    raise NotImplementedError


def apertura(mascara, tam=3):
    """Erosión + dilatación: elimina destellos/puntos pequeños (reflejo
    TODO: implementar.
    """
    raise NotImplementedError


def cierre(mascara, tam=3):
    """Dilatación + erosión: rellena huecos pequeños (oclusión por
    TODO: implementar.
    """
    raise NotImplementedError


def limpiar_mascara(mascara, tam_apertura=3, tam_cierre=5):
    """TODO: implementar."""
    raise NotImplementedError


def esqueletizar(mascara):
    """TODO: implementar."""
    raise NotImplementedError


def mayor_componente(mascara):
    """TODO: implementar."""
    raise NotImplementedError


def dice(mascara_a, mascara_b):
    """TODO: implementar."""
    raise NotImplementedError


def jaccard_iou(mascara_a, mascara_b):
    """TODO: implementar."""
    raise NotImplementedError


def comparar_metodos(mascaras_dict, referencia):
    """TODO: implementar."""
    raise NotImplementedError


def descriptores_morfometricos(mascara):
    """Área, perímetro, centroide y circularidad/excentricidad del blob
    TODO: implementar.
    """
    raise NotImplementedError

