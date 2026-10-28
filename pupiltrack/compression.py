"""
Módulo 4 - Compresión de la información de imagen.

Esquemas evaluados (sobre el frame original en gris, su ROI del ojo y la
máscara binaria de la pupila segmentada):
  * Huffman (sin pérdida): códigos de longitud variable según la
    frecuencia de cada nivel de gris.
  * RLE (sin pérdida): pares (valor, longitud de corrida), recorriendo la
    imagen en dos direcciones (horizontal = por filas, vertical = por
    columnas).
  * JPEG (con pérdida): factor de calidad Q = 90, 50, 10, 5.

Métricas: tasa de compresión (bytes originales / bytes comprimidos),
PSNR y SSIM respecto del original, índice de "blocking" (artefactos de
bloque 8x8), error en el borde pupilar vs. en regiones uniformes, y cambio
del diámetro pupilar estimado (Módulo 3) al medir sobre frames comprimidos.
"""
import heapq
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from . import filtering as filt

CALIDADES_JPEG = (90, 50, 10, 5)


# ---------------------------------------------------------------------------
# Huffman
# ---------------------------------------------------------------------------

def codigos_huffman(datos):
    """Tabla {símbolo: código binario (str)} construida con un heap.
    `datos`: array de enteros (p. ej. niveles de gris 0..255)."""
    frec = Counter(np.asarray(datos).ravel().tolist())
    if len(frec) == 1:                       # imagen constante: 1 bit por símbolo
        return {next(iter(frec)): "0"}
    heap = [[f, i, [s, ""]] for i, (s, f) in enumerate(frec.items())]
    heapq.heapify(heap)
    contador = len(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        for par in a[2:]:
            par[1] = "0" + par[1]
        for par in b[2:]:
            par[1] = "1" + par[1]
        heapq.heappush(heap, [a[0] + b[0], contador] + a[2:] + b[2:])
        contador += 1
    return {s: c for s, c in heap[0][2:]}


def huffman_codificar(img):
    """Codifica la imagen. Devuelve (bits (str), tabla, shape)."""
    tabla = codigos_huffman(img)
    bits = "".join(tabla[v] for v in np.asarray(img).ravel().tolist())
    return bits, tabla, img.shape


def huffman_decodificar(bits, tabla, shape, dtype=np.uint8):
    inversa = {c: s for s, c in tabla.items()}
    salida, actual = [], ""
    for b in bits:
        actual += b
        if actual in inversa:
            salida.append(inversa[actual])
            actual = ""
    return np.array(salida, dtype=dtype).reshape(shape)


def bytes_huffman(img):
    """Tamaño comprimido en bytes: datos + tabla (1 byte de símbolo + 1 byte
    de longitud de código por entrada). No arma el string de bits: suma
    longitudes, así es rápido en imágenes grandes."""
    img = np.asarray(img)
    tabla = codigos_huffman(img)
    frec = Counter(img.ravel().tolist())
    bits = sum(len(tabla[s]) * f for s, f in frec.items())
    return int(np.ceil(bits / 8)) + 2 * len(tabla)


def entropia_bits(img):
    """Entropía de Shannon (bits/píxel): cota inferior para Huffman."""
    _, cuentas = np.unique(np.asarray(img), return_counts=True)
    p = cuentas / cuentas.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# RLE
# ---------------------------------------------------------------------------

def rle_codificar(img, direccion="horizontal"):
    """Pares (valor, corrida). horizontal = recorre por filas (row-major),
    vertical = por columnas (column-major)."""
    img = np.asarray(img)
    plano = img.ravel(order="C" if direccion == "horizontal" else "F")
    if plano.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    cambios = np.flatnonzero(np.diff(plano)) + 1
    inicios = np.r_[0, cambios]
    largos = np.diff(np.r_[inicios, plano.size])
    return np.c_[plano[inicios], largos]


def rle_decodificar(pares, shape, direccion="horizontal", dtype=np.uint8):
    plano = np.repeat(pares[:, 0], pares[:, 1]).astype(dtype)
    return plano.reshape(shape, order="C" if direccion == "horizontal" else "F")


def bytes_rle(img, direccion="horizontal"):
    """Tamaño en bytes con corridas de hasta 255 (1 byte de valor + 1 byte de
    longitud por par; una corrida más larga se parte en varias)."""
    pares = rle_codificar(img, direccion)
    return int(2 * np.ceil(pares[:, 1] / 255).sum())


# ---------------------------------------------------------------------------
# JPEG
# ---------------------------------------------------------------------------

def jpeg(img, calidad):
    """Comprime/descomprime con JPEG. Devuelve (imagen decodificada, bytes)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(calidad)])
    if not ok:
        raise RuntimeError("No se pudo codificar en JPEG")
    flag = cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR
    return cv2.imdecode(buf, flag), int(buf.size)


def indice_blocking(img, bloque=8):
    """Discontinuidad media en los bordes de bloque 8x8 dividida por la
    discontinuidad media dentro de los bloques. ~1 = sin blocking; > 1 =
    aparecen escalones en la grilla de 8x8 (artefacto de bloque)."""
    f = np.asarray(img, dtype=np.float64)
    dx = np.abs(np.diff(f, axis=1))          # entre columnas j y j+1
    dy = np.abs(np.diff(f, axis=0))
    en_borde_x = (np.arange(dx.shape[1]) % bloque) == bloque - 1
    en_borde_y = (np.arange(dy.shape[0]) % bloque) == bloque - 1
    borde = np.r_[dx[:, en_borde_x].ravel(), dy[en_borde_y, :].ravel()].mean()
    dentro = np.r_[dx[:, ~en_borde_x].ravel(), dy[~en_borde_y, :].ravel()].mean()
    return float(borde / dentro) if dentro > 0 else np.nan


# ---------------------------------------------------------------------------
# Regiones: borde pupilar vs. zonas uniformes
# ---------------------------------------------------------------------------

def regiones_borde_y_uniforme(gris, mascara_pupila, ancho_borde=3):
    """Máscaras booleanas de (a) la franja del borde pupilar (±ancho px) y
    (b) la región uniforme: píxeles fuera de la pupila y de su borde con
    gradiente bajo (iris/esclerótica lisos, percentil 30 del gradiente)."""
    m = (mascara_pupila > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ancho_borde + 1,) * 2)
    franja = (cv2.dilate(m, k) - cv2.erode(m, k)) > 0
    suave = cv2.GaussianBlur(gris, (0, 0), 1.5).astype(np.float32)
    grad = np.hypot(cv2.Sobel(suave, cv2.CV_32F, 1, 0), cv2.Sobel(suave, cv2.CV_32F, 0, 1))
    lejos = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) == 0
    uniforme = lejos & (grad <= np.percentile(grad[lejos], 30)) if lejos.any() else lejos
    return franja, uniforme


# ---------------------------------------------------------------------------
# Evaluación completa
# ---------------------------------------------------------------------------

def evaluar_compresion(gris, mascara_pupila=None, calidades=CALIDADES_JPEG):
    """Tabla con ratio, PSNR, SSIM y blocking para cada método aplicado a
    `gris` (uint8). Si se pasa la máscara de la pupila (mismo tamaño), suma
    los esquemas sin pérdida sobre la máscara binaria (ROI segmentada) y el
    error de JPEG en el borde pupilar vs. en la región uniforme."""
    gris = np.ascontiguousarray(gris, dtype=np.uint8)
    original = gris.size                     # bytes (1 byte por píxel)
    filas = []

    def fila(objeto, metodo, comprimido, psnr=np.inf, ssim=1.0, **extra):
        filas.append({"objeto": objeto, "metodo": metodo,
                      "bytes_original": original if objeto != "mascara" else mascara.size,
                      "bytes_comprimido": comprimido,
                      "ratio": round((original if objeto != "mascara" else mascara.size) / comprimido, 3),
                      "psnr_db": round(psnr, 2) if np.isfinite(psnr) else np.inf,
                      "ssim": round(ssim, 4), **extra})

    # Sin pérdida sobre el frame
    fila("frame", "Huffman", bytes_huffman(gris),
         entropia_bits=round(entropia_bits(gris), 3))
    for direc in ("horizontal", "vertical"):
        fila("frame", f"RLE {direc}", bytes_rle(gris, direc))

    # Sin pérdida sobre la ROI segmentada (máscara binaria)
    if mascara_pupila is not None:
        mascara = (np.asarray(mascara_pupila) > 0).astype(np.uint8)
        fila("mascara", "Huffman", bytes_huffman(mascara))
        for direc in ("horizontal", "vertical"):
            fila("mascara", f"RLE {direc}", bytes_rle(mascara, direc))
        franja, uniforme = regiones_borde_y_uniforme(gris, mascara)

    # Con pérdida: JPEG
    fila("frame", "sin comprimir", original, blocking=round(indice_blocking(gris), 3))
    for q in calidades:
        rec, nbytes = jpeg(gris, q)
        extra = {"blocking": round(indice_blocking(rec), 3)}
        if mascara_pupila is not None:
            err = np.abs(rec.astype(np.float64) - gris)
            extra["error_borde_pupila"] = round(float(err[franja].mean()), 3)
            extra["error_region_uniforme"] = round(float(err[uniforme].mean()), 3)
        fila("frame", f"JPEG Q={q}", nbytes, filt.psnr(gris, rec), filt.ssim(gris, rec), **extra)
    return pd.DataFrame(filas)


def efecto_en_diametro(frames, df_serie, medir, calidades=CALIDADES_JPEG, n=30):
    """¿La compresión cambia el diámetro estimado (Módulo 3)?

    Toma `n` frames válidos repartidos en el video, mide el diámetro sobre
    el frame original y sobre el mismo frame comprimido en JPEG con cada Q,
    usando la misma función de medición del pipeline y el mismo centro de
    partida. `medir(frame_bgr, fila)` devuelve el diámetro (px) o NaN.
    Devuelve una fila por Q con el error medio absoluto y relativo."""
    validos = df_serie.index[df_serie["valido"]] if "valido" in df_serie else df_serie.index
    if len(validos) == 0:
        return pd.DataFrame()
    elegidos = validos[np.linspace(0, len(validos) - 1, min(n, len(validos))).astype(int)]
    base = {i: medir(frames[i], df_serie.loc[i]) for i in elegidos}
    filas = []
    for q in calidades:
        dif = []
        for i in elegidos:
            rec, _ = jpeg(frames[i], q)
            d = medir(rec, df_serie.loc[i])
            if np.isfinite(d) and np.isfinite(base[i]):
                dif.append(d - base[i])
        dif = np.array(dif)
        ref = np.nanmean([base[i] for i in elegidos])
        filas.append({
            "calidad_jpeg": q, "frames": len(dif),
            "diametro_original_medio_px": round(float(ref), 2),
            "error_medio_px": round(float(dif.mean()), 3) if len(dif) else np.nan,
            "error_abs_medio_px": round(float(np.abs(dif).mean()), 3) if len(dif) else np.nan,
            "error_abs_medio_pct": round(float(100 * np.abs(dif).mean() / ref), 2) if len(dif) else np.nan,
            "error_abs_max_px": round(float(np.abs(dif).max()), 3) if len(dif) else np.nan,
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Conexión con la medición del Módulo 3 (misma función que el pipeline)
# ---------------------------------------------------------------------------

def mascara_pupila_en_frame(frame_bgr, cx, cy, diametro):
    """Máscara de la pupila (tamaño del frame) medida con la watershed del
    pipeline, partiendo del centro y diámetro de la serie temporal."""
    from . import temporal
    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    area = np.pi * (diametro / 2) ** 2
    centro = np.array([cx, cy], dtype=float)
    x0, y0, x1, y1 = temporal._ventana(centro, area, gris.shape)
    m = temporal.refinar_borde_watershed(gris[y0:y1, x0:x1], centro - [x0, y0], diametro / 2)
    if m is None:
        return None
    completa = np.zeros(gris.shape, np.uint8)
    completa[y0:y1, x0:x1] = m
    return completa


def medir_diametro(frame_bgr, fila_serie):
    """Diámetro (px) en un frame, con la misma watershed del pipeline."""
    m = mascara_pupila_en_frame(frame_bgr, fila_serie["centroide_x"],
                                fila_serie["centroide_y"], fila_serie["diametro_px"])
    return 2 * np.sqrt(np.count_nonzero(m) / np.pi) if m is not None else np.nan


def roi_ojo(gris, cx, cy, diametro, factor=4.0):
    """Recorte cuadrado alrededor de la pupila (ROI del ojo)."""
    lado = int(diametro * factor)
    x0 = int(max(0, cx - lado / 2)); y0 = int(max(0, cy - lado / 2))
    return gris[y0:y0 + lado, x0:x0 + lado], (x0, y0)
