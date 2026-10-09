"""
Módulo 1.2 - Análisis de ruido y filtrado espacial.
"""
import cv2
import numpy as np

def agregar_ruido_gaussiano(img_gris, media=0, sigma=15):
    """TODO: implementar."""
    raise NotImplementedError


def agregar_ruido_sal_pimienta(img_gris, prob=0.02):
    """TODO: implementar."""
    raise NotImplementedError


def agregar_ruido_speckle(img_gris, sigma=0.2):
    """TODO: implementar."""
    raise NotImplementedError


def filtro_media(img_gris, ksize=5):
    """TODO: implementar."""
    raise NotImplementedError


def filtro_mediana(img_gris, ksize=5):
    """TODO: implementar."""
    raise NotImplementedError


def filtro_gaussiano(img_gris, ksize=5, sigma=1.0):
    """TODO: implementar."""
    raise NotImplementedError


def psnr(img_original, img_procesada):
    """TODO: implementar."""
    raise NotImplementedError


def ssim(img_original, img_procesada):
    """TODO: implementar."""
    raise NotImplementedError

