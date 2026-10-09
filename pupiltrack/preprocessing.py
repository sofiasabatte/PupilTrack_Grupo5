"""
Módulo 1.1 y 1.2 - Fundamentos, espacios de color y mejoramiento digital.
"""
import cv2
import numpy as np


def info_imagen(img):
    alto, ancho = img.shape[:2]
    canales = 1 if img.ndim == 2 else img.shape[2]
    profundidad = img.dtype
    info = {"ancho": ancho, "alto": alto, "canales": canales, "dtype": str(profundidad)}
    print(f"Dimensiones: {ancho}x{alto} | Canales: {canales} | Profundidad: {profundidad}")
    return info


def submuestrear(img, porcentaje):
    h, w = img.shape[:2]
    nuevo_w, nuevo_h = max(1, int(w * porcentaje)), max(1, int(h * porcentaje))
    reducida = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(reducida, (w, h), interpolation=cv2.INTER_NEAREST)


def cuantizar(img_gris, n_bits):
    niveles = 2 ** n_bits
    paso = 256 // niveles
    return ((img_gris // paso) * paso).astype(np.uint8)


def separar_canales_rgb(img_bgr):
    b, g, r = cv2.split(img_bgr)
    return r, g, b


def rgb_a_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v


def histograma_gris(img_gris):
    hist = cv2.calcHist([img_gris], [0], None, [256], [0, 256])
    return hist.flatten()


def ajustar_brillo_contraste(img_gris, brillo=0, contraste=1.0):
    ajustada = img_gris.astype(np.float32) * contraste + brillo
    return np.clip(ajustada, 0, 255).astype(np.uint8)


def escalado_por_rango(img_gris, f1, f2):
    img_f = img_gris.astype(np.float32)
    escalada = (img_f - f1) * (255.0 / (f2 - f1))
    return np.clip(escalada, 0, 255).astype(np.uint8)


def ecualizar_histograma(img_gris, metodo="global", clip_limit=2.0, tile_size=(8, 8)):
    if metodo == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img_gris)
    return cv2.equalizeHist(img_gris)
