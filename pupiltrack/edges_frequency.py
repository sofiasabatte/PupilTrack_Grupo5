"""
Módulo 1.3 - Detección de bordes y dominio frecuencial.
"""
import cv2
import numpy as np


def bordes_sobel(img_gris):
    gx = cv2.Sobel(img_gris, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gris, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)


def bordes_laplaciano(img_gris):
    lap = cv2.Laplacian(img_gris, cv2.CV_64F)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


def bordes_canny(img_gris, umbral1=50, umbral2=150):
    return cv2.Canny(img_gris, umbral1, umbral2)


def fft2d(img_gris):
    return np.fft.fftshift(np.fft.fft2(img_gris))


def espectro_magnitud(fft_shift):
    mag = np.log1p(np.abs(fft_shift))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255
    return mag.astype(np.uint8)


def espectro_fase(fft_shift):
    return np.angle(fft_shift)


def reconstruir_desde_magnitud_fase(magnitud, fase):
    compleja = magnitud * np.exp(1j * fase)
    img = np.fft.ifft2(np.fft.ifftshift(compleja))
    img = np.abs(img)
    return np.clip(img / img.max() * 255, 0, 255).astype(np.uint8)


def filtro_frecuencial(img_gris, tipo="pasabajos", radio=30):
    filas, cols = img_gris.shape
    cy, cx = filas // 2, cols // 2
    y, x = np.ogrid[:filas, :cols]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if tipo == "pasabajos":
        mascara = (dist <= radio).astype(np.float32)
    else:
        mascara = (dist > radio).astype(np.float32)

    fft_shift = fft2d(img_gris)
    fft_filtrada = fft_shift * mascara
    reconstruida = np.fft.ifft2(np.fft.ifftshift(fft_filtrada))
    reconstruida = np.abs(reconstruida)
    return np.clip(reconstruida, 0, 255).astype(np.uint8)


def psd(img_gris):
    fft_shift = fft2d(img_gris)
    return np.abs(fft_shift) ** 2


def _magnitud_normalizada(gx, gy):
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)


def bordes_roberts(img_gris):
    """Operador de Roberts (cruz 2x2): gradiente en las diagonales."""
    f = img_gris.astype(np.float64)
    kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    return _magnitud_normalizada(cv2.filter2D(f, -1, kx), cv2.filter2D(f, -1, ky))


def bordes_prewitt(img_gris):
    """Operador de Prewitt 3x3 (como Sobel pero sin ponderar la fila central)."""
    f = img_gris.astype(np.float64)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    return _magnitud_normalizada(cv2.filter2D(f, -1, kx), cv2.filter2D(f, -1, kx.T))


def reconstruir_solo_magnitud(img_gris):
    """Transformada inversa conservando la magnitud y anulando la fase."""
    F = fft2d(img_gris)
    return reconstruir_desde_magnitud_fase(np.abs(F), np.zeros_like(np.angle(F)))


def reconstruir_solo_fase(img_gris):
    """Transformada inversa con magnitud unitaria y la fase original."""
    F = fft2d(img_gris)
    return reconstruir_desde_magnitud_fase(np.ones_like(np.abs(F)), np.angle(F))


def psd_radial(img_gris, ventana=True):
    """Perfil radial de la PSD (promedio de |F|^2 por anillo de frecuencia).
    Con ventana=True se aplica una ventana de Hann 2D antes de la FFT, para
    que los bordes del recorte no generen fuga espectral (que "llena" las
    frecuencias altas y enmascara el efecto de un filtro)."""
    f = img_gris.astype(np.float64)
    if ventana:
        f = (f - f.mean()) * np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    p = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    h, w = p.shape
    y, x = np.indices(p.shape)
    r = np.hypot(x - w // 2, y - h // 2).astype(int)
    suma = np.bincount(r.ravel(), p.ravel())
    cuenta = np.bincount(r.ravel())
    return suma / np.maximum(cuenta, 1)
