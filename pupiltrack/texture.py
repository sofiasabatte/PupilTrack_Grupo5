"""
Módulo 3.1 - Textura de primer y segundo orden (GLCM).
Basado en la clase "Análisis de Texturas" (PSIB, 2do Q 2026).
"""
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

def entropia(canal):
    """TODO: implementar."""
    raise NotImplementedError


def textura_primer_orden(canal):
    """Media, Desvío Estándar, Entropía y Asimetría (Skewness) de un canal.
    TODO: implementar.
    """
    raise NotImplementedError


def textura_primer_orden_rgb(img_bgr):
    """Aplica textura_primer_orden a cada canal R, G, B por separado.
    TODO: implementar.
    """
    raise NotImplementedError


def calcular_glcm(canal_gris, niveles=32, distancias=(1,), simetrica=True, normada=True):
    """Construye la GLCM (Matriz de Co-ocurrencia de Niveles de Gris).
    TODO: implementar.
    """
    raise NotImplementedError


def atributos_glcm(glcm):
    """Contraste, Homogeneidad, Energía y Correlación — promediados sobre
    TODO: implementar.
    """
    raise NotImplementedError


def textura_glcm_rgb(img_bgr, niveles=32, distancias=(1,)):
    """GLCM + atributos para cada canal R, G, B por separado. Permite
    TODO: implementar.
    """
    raise NotImplementedError


def canal_mas_textural(resultado_glcm_rgb, atributo='contrast'):
    """Heurística simple para responder la pregunta de la guía: qué canal
    TODO: implementar.
    """
    raise NotImplementedError

