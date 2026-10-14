"""
Módulo 3.1 - Textura de primer y segundo orden (GLCM).
Basado en la clase "Análisis de Texturas" (PSIB, 2do Q 2026).
"""
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops


# ---------- Textura de primer orden (estadística global) ----------

def entropia(canal):
    hist, _ = np.histogram(canal, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def textura_primer_orden(canal):
    """Media, Desvío Estándar, Entropía y Asimetría (Skewness) de un canal."""
    datos = canal.astype(np.float64).ravel()
    return {
        "media": round(float(np.mean(datos)), 3),
        "desvio": round(float(np.std(datos)), 3),
        "entropia": round(entropia(canal), 3),
        "asimetria": round(float(skew(datos)), 3),
    }


def textura_primer_orden_rgb(img_bgr):
    """Aplica textura_primer_orden a cada canal R, G, B por separado."""
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    return {
        "R": textura_primer_orden(r),
        "G": textura_primer_orden(g),
        "B": textura_primer_orden(b),
    }


# ---------- Textura de segundo orden (GLCM) ----------

# Direcciones según la clase: 0°=[0,1], 45°=[1,-1], 90°=[-1,0], 135°=[-1,-1]
ANGULOS_GLCM = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
NOMBRES_ANGULOS = ["0°", "45°", "90°", "135°"]


def calcular_glcm(canal_gris, niveles=32, distancias=(1,), simetrica=True, normada=True):
    """Construye la GLCM (Matriz de Co-ocurrencia de Niveles de Gris).
    Cuantiza el canal a `niveles` (p. ej. 32 u 64, como pide la guía) antes
    de construir la matriz, y la normaliza para que sea una distribución
    de probabilidad (P_ij = C_ij / sum(C)), tal como muestra la clase.
    """
    paso = 256 // niveles
    cuantizada = (canal_gris // paso).astype(np.uint8)
    cuantizada = np.clip(cuantizada, 0, niveles - 1)
    return graycomatrix(
        cuantizada, distances=list(distancias), angles=ANGULOS_GLCM,
        levels=niveles, symmetric=simetrica, normed=normada,
    )


def atributos_glcm(glcm):
    """Contraste, Homogeneidad, Energía y Correlación — promediados sobre
    las 4 direcciones (0°, 45°, 90°, 135°) para un descriptor invariante
    a la orientación del patrón de iris."""
    atributos = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        valores = graycoprops(glcm, prop)  # shape (n_distancias, n_angulos)
        atributos[prop] = round(float(np.mean(valores)), 4)
    return atributos


def textura_glcm_rgb(img_bgr, niveles=32, distancias=(1,)):
    """GLCM + atributos para cada canal R, G, B por separado. Permite
    discutir qué canal aporta mayor riqueza textural del estroma iridiano."""
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    resultado = {}
    for nombre, canal in [("R", r), ("G", g), ("B", b)]:
        glcm = calcular_glcm(canal, niveles=niveles, distancias=distancias)
        resultado[nombre] = atributos_glcm(glcm)
    return resultado


def canal_mas_textural(resultado_glcm_rgb, atributo="contrast"):
    """Heurística simple para responder la pregunta de la guía: qué canal
    tiene mayor riqueza textural, usando el contraste de la GLCM como proxy
    (más contraste = más variación estructural = más textura)."""
    return max(resultado_glcm_rgb, key=lambda c: resultado_glcm_rgb[c][atributo])
