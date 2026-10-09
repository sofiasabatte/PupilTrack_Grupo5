"""
Módulo 1.2 - Análisis de ruido y filtrado espacial.
"""
import cv2
import numpy as np


def agregar_ruido_gaussiano(img_gris, media=0, sigma=15):
    ruido = np.random.normal(media, sigma, img_gris.shape)
    return np.clip(img_gris.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def agregar_ruido_sal_pimienta(img_gris, prob=0.02):
    salida = img_gris.copy()
    mascara = np.random.random(img_gris.shape)
    salida[mascara < prob / 2] = 0
    salida[mascara > 1 - prob / 2] = 255
    return salida


def agregar_ruido_speckle(img_gris, sigma=0.2):
    ruido = np.random.normal(0, sigma, img_gris.shape)
    salida = img_gris.astype(np.float32) * (1 + ruido)
    return np.clip(salida, 0, 255).astype(np.uint8)


def filtro_media(img_gris, ksize=5):
    return cv2.blur(img_gris, (ksize, ksize))


def filtro_mediana(img_gris, ksize=5):
    return cv2.medianBlur(img_gris, ksize)


def filtro_gaussiano(img_gris, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(img_gris, (ksize, ksize), sigma)


def psnr(img_original, img_procesada):
    return cv2.PSNR(img_original, img_procesada)


def ssim(img_original, img_procesada):
    from skimage.metrics import structural_similarity
    return structural_similarity(img_original, img_procesada)
