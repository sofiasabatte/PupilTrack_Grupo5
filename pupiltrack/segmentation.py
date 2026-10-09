"""
Módulo 2.1 - Algoritmos de segmentación de la pupila.

Basado en las clases "Segmentación de imágenes I y II" (PSIB, 2do Q 2026):
Otsu, K-Means, Contornos Activos (Snakes) y Watershed — los 4 métodos que
pide explícitamente la guía — más umbral adaptativo/Li/Triangle como
alternativas de umbralización global.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_li, threshold_triangle
from skimage.segmentation import active_contour, chan_vese, watershed as sk_watershed
from skimage.feature import peak_local_max

def umbral_manual(img_gris, t):
    """TODO: implementar."""
    raise NotImplementedError


def umbral_otsu(img_gris):
    """Otsu (1979): maximiza la varianza inter-clase. Umbral óptimo global.
    TODO: implementar.
    """
    raise NotImplementedError


def umbral_li(img_gris):
    """TODO: implementar."""
    raise NotImplementedError


def umbral_triangle(img_gris):
    """TODO: implementar."""
    raise NotImplementedError


def umbral_adaptativo(img_gris, tam_bloque=25, c=5, metodo='gaussiano'):
    """TODO: implementar."""
    raise NotImplementedError


def segmentar_kmeans(img_gris, k=3, criterios_iter=10):
    """TODO: implementar."""
    raise NotImplementedError


def mascara_desde_kmeans(etiquetas, centros, cluster_mas_oscuro=True):
    """TODO: implementar."""
    raise NotImplementedError


def region_growing(img_gris, semilla, tolerancia=10):
    """TODO: implementar."""
    raise NotImplementedError


def elegir_semilla_automatica(img_gris, modo='minimo'):
    """TODO: implementar."""
    raise NotImplementedError


def contorno_inicial_circular(centro, radio, n_puntos=100):
    """TODO: implementar."""
    raise NotImplementedError


def snake_activo(img_gris, contorno_init, alpha=0.015, beta=10, gamma=0.001):
    """Define un contorno inicial rodeando la pupila y ajusta las fuerzas
    TODO: implementar.
    """
    raise NotImplementedError


def contorno_a_mascara(contorno, shape):
    """TODO: implementar."""
    raise NotImplementedError


def segmentar_snake(img_gris, radio_inicial=None, alpha=0.015, beta=10, gamma=0.001):
    """Atajo: arma un contorno circular inicial centrado en el píxel más
    TODO: implementar.
    """
    raise NotImplementedError


def segmentar_chan_vese(img_gris, iteraciones=200):
    """TODO: implementar."""
    raise NotImplementedError


def segmentar_watershed(img_gris, invertir=True, distancia_min=10):
    """Segmentación por inundación (clase 'Segmentación II'): se binariza,
    TODO: implementar.
    """
    raise NotImplementedError

