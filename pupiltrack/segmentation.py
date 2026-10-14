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


# ---------- Umbralización global ----------

def umbral_manual(img_gris, t):
    return (img_gris > t).astype(np.uint8) * 255


def umbral_otsu(img_gris):
    """Otsu (1979): maximiza la varianza inter-clase. Umbral óptimo global."""
    t = threshold_otsu(img_gris)
    return (img_gris > t).astype(np.uint8) * 255, t


def umbral_li(img_gris):
    t = threshold_li(img_gris)
    return (img_gris > t).astype(np.uint8) * 255, t


def umbral_triangle(img_gris):
    t = threshold_triangle(img_gris)
    return (img_gris > t).astype(np.uint8) * 255, t


def umbral_adaptativo(img_gris, tam_bloque=25, c=5, metodo="gaussiano"):
    metodo_cv = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if metodo == "gaussiano"
                 else cv2.ADAPTIVE_THRESH_MEAN_C)
    return cv2.adaptiveThreshold(
        img_gris, 255, metodo_cv, cv2.THRESH_BINARY, tam_bloque, c
    )


# ---------- K-Means (agrupamiento en espacio croma/intensidad, K=2 a 5) ----------

def segmentar_kmeans(img_gris, k=3, criterios_iter=10):
    datos = img_gris.reshape(-1, 1).astype(np.float32)
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                 criterios_iter, 1.0)
    _, etiquetas, centros = cv2.kmeans(
        datos, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    etiquetas = etiquetas.reshape(img_gris.shape)
    return etiquetas, centros.flatten()


def mascara_desde_kmeans(etiquetas, centros, cluster_mas_oscuro=True):
    idx = np.argmin(centros) if cluster_mas_oscuro else np.argmax(centros)
    return (etiquetas == idx).astype(np.uint8) * 255


# ---------- Region Growing ----------

def region_growing(img_gris, semilla, tolerancia=10):
    alto, ancho = img_gris.shape
    visitado = np.zeros_like(img_gris, dtype=bool)
    mascara = np.zeros_like(img_gris, dtype=np.uint8)

    x0, y0 = semilla
    valor_semilla = int(img_gris[y0, x0])
    pila = [(x0, y0)]
    visitado[y0, x0] = True

    while pila:
        x, y = pila.pop()
        if abs(int(img_gris[y, x]) - valor_semilla) <= tolerancia:
            mascara[y, x] = 255
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < ancho and 0 <= ny < alto and not visitado[ny, nx]:
                    visitado[ny, nx] = True
                    pila.append((nx, ny))
    return mascara


def elegir_semilla_automatica(img_gris, modo="minimo"):
    idx = np.argmin(img_gris) if modo == "minimo" else np.argmax(img_gris)
    y, x = np.unravel_index(idx, img_gris.shape)
    return (x, y)


# ---------- Contornos activos (Snakes / Chan-Vese) ----------

def contorno_inicial_circular(centro, radio, n_puntos=100):
    s = np.linspace(0, 2 * np.pi, n_puntos)
    x = centro[0] + radio * np.cos(s)
    y = centro[1] + radio * np.sin(s)
    return np.array([y, x]).T


def snake_activo(img_gris, contorno_init, alpha=0.015, beta=10, gamma=0.001):
    """Define un contorno inicial rodeando la pupila y ajusta las fuerzas
    internas/externas hasta converger en la frontera del esfínter pupilar."""
    img_norm = img_gris.astype(np.float64) / 255.0
    return active_contour(img_norm, contorno_init, alpha=alpha, beta=beta,
                           gamma=gamma)


def contorno_a_mascara(contorno, shape):
    mascara = np.zeros(shape, dtype=np.uint8)
    puntos = contorno[:, ::-1].astype(np.int32)
    cv2.fillPoly(mascara, [puntos], 255)
    return mascara


def segmentar_snake(img_gris, radio_inicial=None, alpha=0.015, beta=10, gamma=0.001):
    """Atajo: arma un contorno circular inicial centrado en el píxel más
    oscuro (heurística de la pupila) y ajusta el snake sobre la imagen."""
    semilla = elegir_semilla_automatica(img_gris, modo="minimo")
    if radio_inicial is None:
        radio_inicial = min(img_gris.shape) // 4
    contorno_init = contorno_inicial_circular(semilla, radio_inicial)
    contorno_final = snake_activo(img_gris, contorno_init, alpha, beta, gamma)
    return contorno_a_mascara(contorno_final, img_gris.shape)


def segmentar_chan_vese(img_gris, iteraciones=200):
    img_norm = img_gris.astype(np.float64) / 255.0
    resultado = chan_vese(img_norm, max_num_iter=iteraciones)
    return (resultado.astype(np.uint8)) * 255


# ---------- Watershed (cuenca hidrográfica) ----------

def segmentar_watershed(img_gris, invertir=True, distancia_min=10):
    """Segmentación por inundación (clase 'Segmentación II'): se binariza,
    se calcula la Transformada de Distancia, se ubican los máximos locales
    como marcadores y se inunda desde ahí. Separa bien la pupila de
    reflejos corneales cercanos que quedarían unidos con Otsu simple.
    """
    # La pupila es oscura: invertimos para que sea la región "alta" en la
    # transformada de distancia (más alejada del fondo claro).
    m, _ = umbral_otsu(img_gris)
    binaria = cv2.bitwise_not(m) if invertir else m
    binaria_bool = binaria > 0

    distancia = ndi.distance_transform_edt(binaria_bool)
    coords = peak_local_max(distancia, min_distance=distancia_min, labels=binaria_bool)
    marcadores = np.zeros(distancia.shape, dtype=int)
    for i, (y, x) in enumerate(coords, start=1):
        marcadores[y, x] = i
    marcadores = ndi.label(marcadores)[0] if marcadores.max() == 0 else marcadores

    etiquetas = sk_watershed(-distancia, marcadores, mask=binaria_bool)

    if etiquetas.max() == 0:
        return np.zeros_like(img_gris, dtype=np.uint8)

    # Nos quedamos con la región de mayor área (candidata a pupila)
    tam = np.bincount(etiquetas.ravel())
    tam[0] = 0  # ignorar fondo
    region_mas_grande = np.argmax(tam)
    return (etiquetas == region_mas_grande).astype(np.uint8) * 255
