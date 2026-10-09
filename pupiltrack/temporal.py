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
    """TODO: implementar."""
    raise NotImplementedError


def segmentar_frame(gris, metodo, tolerancia=15, tam_apertura=3, tam_cierre=5):
    """TODO: implementar."""
    raise NotImplementedError


def medir_pupila(mascara):
    """Área, diámetro equivalente y centroide del blob principal.
    TODO: implementar.
    """
    raise NotImplementedError


def procesar_video(cfg):
    """TODO: implementar."""
    raise NotImplementedError


def _suavizar(serie, ventana=5):
    """TODO: implementar."""
    raise NotImplementedError


def inferir_tipo(df):
    """TODO: implementar."""
    raise NotImplementedError


def calcular_parametros_fisiologicos(df, fps, tipo='auto'):
    """TODO: implementar."""
    raise NotImplementedError

