"""
Procesamiento temporal completo (Módulo 3.2): recorre el video frame a
frame, segmenta la pupila con el método elegido en config.yaml, mide
área/diámetro/centroide, arma la curva temporal y deriva los parámetros
fisiológicos (latencia, amplitud, velocidad de constricción/redilatación).

Robustez agregada (Módulo 3.2, punto 4 — artefactos):
  * Selección de la pupila entre todas las regiones oscuras por forma
    (circularidad, solidez, área) y por cercanía al frame anterior, en vez
    de quedarse con "la región más grande". Esto evita que el algoritmo
    salte a las pestañas, la ceja o sombras.
  * Seguimiento: cada frame se segmenta en una ventana centrada en la
    pupila del frame anterior, así un movimiento de cabeza no la saca de
    la ROI.
  * Relleno de huecos y envolvente convexa: el reflejo corneal (dentro de
    la pupila o "mordiendo" su borde) no achica el área medida.
  * Refinamiento del borde con watershed con marcadores (Módulo 2.1)
    sobre el gradiente: el diámetro se mide en el borde pupila/iris de
    máximo contraste, sin depender de un umbral de gris. El método del
    config.yaml ubica la pupila; la watershed mide su borde.
  * Elipse sobre el borde visible: si un reflejo "muerde" la pupila, la
    elipse se ajusta solo con los puntos del borde que no están ocluidos.
  * Seguimiento robusto: el tamaño de referencia es el percentil 75 de las
    mediciones de los últimos 0,2 s (las oclusiones solo achican la pupila
    medida), con un cambio máximo de 2,5 % del diámetro por frame.
  * Parpadeos: se detectan porque el párpado aclara la zona del ojo mucho
    más que el resto de la imagen (fuera de la ventana del estímulo); esos
    frames se descartan sin modificar el seguimiento.
  * Si se pierde la pupila > 0,3 s, se la busca de nuevo en todo el frame y
    solo se acepta una candidata de tamaño compatible (evita el iris).
  * Tolerancia adaptativa del region growing: en cada frame se calcula a
    partir del contraste pupila/iris, así un reflejo tenue sobre la pupila
    o el cambio de iluminación del estímulo no dejan parte de la pupila
    afuera.
  * Rechazo de artefactos: frames sin pupila válida (parpadeo) o con un
    salto de área imposible en 1 frame se marcan como inválidos y se
    interpolan.
  * Detección automática del estímulo luminoso por el escalón de brillo
    del video, para medir la latencia desde el estímulo (no desde el
    inicio del video).
  * Basal = último segundo antes del estímulo; extremo sobre una curva
    suavizada a 0,5 s (en B, percentil 90 post-estímulo); latencia =
    primer instante en que el diámetro se aleja del basal más que
    max(10 % de la amplitud, 2 x ruido basal) (en B, desde el mínimo de
    los primeros 0,7 s si la pupila baja un instante al apagar la luz).
"""
from collections import deque

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import savgol_filter
from skimage.segmentation import watershed as sk_watershed

from . import acquisition as acq
from . import segmentation as seg
from . import morphology as morf


# ---------------------------------------------------------------------------
# Parámetros de robustez (valores razonables para un ojo en primer plano)
# ---------------------------------------------------------------------------
CIRCULARIDAD_MIN = 0.70      # 1.0 = círculo; las pestañas dan valores bajos
SOLIDEZ_MIN = 0.60           # área / área de la envolvente convexa (baja si
                             # el reflejo corneal le "muerde" un borde)
AREA_MIN_FRAC = 0.0005       # área mínima como fracción del frame
AREA_MAX_FRAC = 0.15         # área máxima como fracción del frame
VENTANA_FACTOR = 3.0         # lado de la ventana de seguimiento / diámetro


# ---------------------------------------------------------------------------
# Selección de la pupila entre las regiones candidatas
# ---------------------------------------------------------------------------

def _forma_componente(comp):
    """Circularidad y solidez de una componente binaria (uint8 0/255)."""
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    area_hull = cv2.contourArea(hull)
    perim_hull = cv2.arcLength(hull, True)
    if area_hull <= 0 or perim_hull <= 0:
        return 0.0, 0.0
    circularidad = 4 * np.pi * area_hull / perim_hull ** 2
    solidez = area / area_hull
    return circularidad, solidez


def _envolvente(comp):
    """Rellena la envolvente convexa de una componente binaria."""
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    salida = np.zeros_like(comp)
    if cnts:
        cv2.fillPoly(salida, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 255)
    return salida


def elegir_pupila(mascara, centro_previo=None, area_previa=None, escala=None):
    """Devuelve la máscara de la componente más parecida a una pupila.

    Criterios: circular, sólida, de tamaño razonable y (si hay historia)
    cerca de la pupila del frame anterior y con área parecida.
    Devuelve None si ninguna componente es aceptable (p. ej. parpadeo).
    """
    h, w = mascara.shape
    area_frame = h * w if escala is None else escala
    a_min = AREA_MIN_FRAC * area_frame
    a_max = AREA_MAX_FRAC * area_frame

    # El reflejo corneal deja un hueco dentro de la pupila: lo rellenamos
    rellena = ndi.binary_fill_holes(mascara > 0).astype(np.uint8) * 255
    n, etiquetas, stats, centros = cv2.connectedComponentsWithStats(rellena, 8)

    mejor, mejor_score = None, -np.inf
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < a_min or area > a_max:
            continue
        comp = (etiquetas == k).astype(np.uint8) * 255
        circ, sol = _forma_componente(comp)
        if circ < CIRCULARIDAD_MIN or sol < SOLIDEZ_MIN:
            continue
        # La pupila se mide con su envolvente convexa: reconstruye la parte
        # tapada por el reflejo corneal cuando éste cae sobre el borde
        comp = _envolvente(comp)
        area = np.count_nonzero(comp)
        score = circ + sol
        if centro_previo is not None:
            ref = np.sqrt(area_previa / np.pi) if area_previa else 20.0
            score -= np.hypot(*(centros[k] - centro_previo)) / (4 * ref)
        if area_previa:
            score -= abs(area - area_previa) / area_previa
        if score > mejor_score:
            mejor, mejor_score = comp, score
    return mejor


# ---------------------------------------------------------------------------
# ROI y segmentación por frame
# ---------------------------------------------------------------------------

def _mascara_oscura(gris):
    """Máscara de las regiones más oscuras (candidatas a pupila)."""
    suave = cv2.medianBlur(gris, 5)
    t = min(seg.threshold_otsu(suave), np.percentile(suave, 5) + 25)
    return (suave <= t).astype(np.uint8) * 255


def localizar_pupila_inicial(frame_bgr):
    """Busca la pupila en el frame completo: la región oscura más circular.

    Prueba varios umbrales (de muy estricto a más permisivo) porque con un
    único umbral global la pupila suele quedar unida al iris o a las
    pestañas; se queda con la región más circular entre todos."""
    gris = cv2.medianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 5)
    umbrales = np.unique(np.linspace(np.percentile(gris, 0.2),
                                     np.percentile(gris, 10), 12).astype(int))
    candidatos = []
    for t in umbrales:
        mascara = morf.apertura((gris <= t).astype(np.uint8) * 255, 3)
        c = elegir_pupila(mascara)
        if c is not None:
            candidatos.append((float(gris[c > 0].mean()), np.count_nonzero(c), c))
    if not candidatos:
        return None
    # La pupila es la región circular MÁS OSCURA. Entre las casi igual de
    # oscuras (mismo objeto con distinto umbral) nos quedamos con la más
    # grande, que es la que llega al borde real; el iris queda afuera
    # porque es bastante más claro.
    media_min = min(m for m, _, _ in candidatos)
    comp = max((c for c in candidatos if c[0] <= media_min + 12), key=lambda c: c[1])[2]
    d = morf.descriptores_morfometricos(comp)
    return np.array([d["centroide_x"], d["centroide_y"]]), d["area_px"]


def roi_automatica(frame_bgr, margen=1.5):
    """ROI cuadrada alrededor de la pupila del primer frame (con margen
    para el iris). Se mantiene por compatibilidad con los notebooks; el
    pipeline además sigue a la pupila frame a frame."""
    ini = localizar_pupila_inicial(frame_bgr)
    H, W = frame_bgr.shape[:2]
    if ini is None:
        return 0, 0, W, H
    (cx, cy), area = ini
    lado = 2 * np.sqrt(area / np.pi) * 4 * margen
    x = int(max(0, cx - lado / 2))
    y = int(max(0, cy - lado / 2))
    return x, y, int(min(W - x, lado)), int(min(H - y, lado))


def segmentar_frame(gris, metodo, tolerancia=15, tam_apertura=3, tam_cierre=5,
                    semilla=None, solo_pupila=True):
    """Segmenta la pupila con el método elegido.

    solo_pupila=True (como antes, para los notebooks): devuelve una sola
    región, la más parecida a una pupila (o la más grande si ninguna pasa
    el filtro de forma).
    solo_pupila=False (lo usa el pipeline): devuelve la máscara con todas
    las regiones oscuras, y la pupila se elige después en `elegir_pupila`
    usando además la historia de los frames anteriores."""
    if metodo == "otsu":
        m, _ = seg.umbral_otsu(gris)
        mascara = cv2.bitwise_not(m)
    elif metodo == "li":
        m, _ = seg.umbral_li(gris)
        mascara = cv2.bitwise_not(m)
    elif metodo == "triangle":
        m, _ = seg.umbral_triangle(gris)
        mascara = cv2.bitwise_not(m)
    elif metodo == "adaptativo":
        m = seg.umbral_adaptativo(gris, tam_bloque=31, c=5)
        mascara = cv2.bitwise_not(m)
    elif metodo == "kmeans":
        etiquetas, centros = seg.segmentar_kmeans(gris, k=3)
        mascara = seg.mascara_desde_kmeans(etiquetas, centros, cluster_mas_oscuro=True)
    elif metodo == "region_growing":
        if semilla is None:
            # Semilla en el mínimo de la imagen MUY suavizada: la pupila es un
            # disco oscuro grande y sobrevive al blur; las pestañas (finas) no.
            sigma = max(3, min(gris.shape) // 20)
            muy_suave = cv2.GaussianBlur(gris, (0, 0), sigma)
            semilla = seg.elegir_semilla_automatica(muy_suave, modo="minimo")
        suave = cv2.medianBlur(gris, 5)
        mascara = seg.region_growing(suave, semilla, tolerancia=tolerancia)
    elif metodo == "chan_vese":
        mascara = seg.segmentar_chan_vese(gris, iteraciones=100)
    elif metodo == "snake":
        mascara = seg.segmentar_snake(gris)
    elif metodo == "watershed":
        mascara = seg.segmentar_watershed(gris)
    else:
        raise ValueError(f"Método de segmentación desconocido: {metodo}")

    limpia = morf.limpiar_mascara(mascara, tam_apertura, tam_cierre)
    if not solo_pupila:
        return limpia
    pupila = elegir_pupila(limpia)
    return pupila if pupila is not None else morf.mayor_componente(limpia)


def medir_pupila(mascara):
    """Área, diámetro equivalente y centroide del blob principal."""
    desc = morf.descriptores_morfometricos(mascara)
    area = desc["area_px"]
    diametro = 2 * np.sqrt(area / np.pi) if area and not np.isnan(area) else np.nan
    return area, diametro, desc["centroide_x"], desc["centroide_y"]


def _detectar_picos(diametro, fps, ventana_s=0.7, k=4.0):
    """Marca como artefacto los frames que se apartan bruscamente de la
    tendencia local (parpadeo parcial: el párpado tapa un pedazo de la
    pupila durante unos pocos frames y el diámetro cae de golpe).

    Criterio: |D - tendencia| > max(k · MAD, 4 % de D), con tendencia =
    percentil 75 móvil de 0,7 s. Se usa un percentil alto (y no la
    mediana) porque las oclusiones solo achican la pupila medida: si en un
    tramo la mitad de los frames sale recortada por un reflejo, la
    tendencia sigue al borde real y los recortados se marcan.
    El reflejo real es mucho más lento que un parpadeo, así que no lo
    confunde con un artefacto."""
    n = max(5, int(round(ventana_s * fps)) | 1)
    tendencia = diametro.rolling(n, center=True, min_periods=3).quantile(0.75)
    residuo = (diametro - tendencia).abs()
    mad = 1.4826 * residuo.rolling(n * 3, center=True, min_periods=3).median()
    limite = np.maximum(k * mad, 0.04 * tendencia)
    picos = residuo > limite
    # ensanchar 2 frames a cada lado: el inicio y el fin del parpadeo
    picos = picos.rolling(5, center=True, min_periods=1).max().astype(bool)
    return picos.fillna(False)


def _ventana(centro, area, shape, factor=VENTANA_FACTOR):
    H, W = shape[:2]
    d = 2 * np.sqrt(area / np.pi)
    lado = max(40, d * factor)
    x0 = int(np.clip(centro[0] - lado / 2, 0, W - 1))
    y0 = int(np.clip(centro[1] - lado / 2, 0, H - 1))
    x1 = int(np.clip(centro[0] + lado / 2, x0 + 1, W))
    y1 = int(np.clip(centro[1] + lado / 2, y0 + 1, H))
    return x0, y0, x1, y1


def _tolerancia_adaptativa(gris, centro, area):
    """(nivel del iris - nivel del fondo pupilar) / 2, medidos en la imagen
    suavizada: fondo = percentil 5 dentro de la pupila anterior, iris =
    mediana de un anillo entre 1,3 y 1,8 radios."""
    suave = cv2.medianBlur(gris, 5)
    r = np.sqrt(area / np.pi)
    yy, xx = np.ogrid[:gris.shape[0], :gris.shape[1]]
    dist = np.hypot(xx - centro[0], yy - centro[1])
    dentro = suave[dist <= 0.8 * r]
    anillo = suave[(dist >= 1.3 * r) & (dist <= 1.8 * r)]
    if dentro.size < 10 or anillo.size < 10:
        return 0
    fondo = np.percentile(dentro, 5)
    iris = np.median(anillo)
    return max(0.0, (iris - fondo) / 2)


def refinar_borde_watershed(gris, centro, r_prev):
    """Watershed con marcadores (Módulo 2.1) sobre la magnitud del
    gradiente, para ubicar el borde pupila/iris donde el contraste es
    máximo, sin depender de un umbral de gris.

    Marcador "pupila": disco de 0,25 r alrededor del punto más oscuro
    cercano al centro. Marcador "fondo": todo lo que está a más de 1,6 r.
    La línea divisoria cae en el borde más fuerte entre ambos, que es el
    borde pupilar: no se escapa por las pestañas ni se corta con un
    reflejo tenue (problemas típicos del region growing).
    Devuelve la máscara de la pupila (envolvente convexa) o None."""
    H, W = gris.shape
    suave = cv2.GaussianBlur(gris, (0, 0), 2).astype(np.float32)
    yy, xx = np.ogrid[:H, :W]
    dist = np.hypot(xx - centro[0], yy - centro[1])
    cerca = dist <= 0.5 * r_prev
    if not cerca.any():
        return None
    tmp = np.where(cerca, suave, np.inf)
    py, px = np.unravel_index(np.argmin(tmp), tmp.shape)
    marcadores = np.zeros((H, W), np.int32)
    marcadores[dist >= 1.6 * r_prev] = 2
    marcadores[np.hypot(xx - px, yy - py) <= max(2, 0.25 * r_prev)] = 1
    gx = cv2.Sobel(suave, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(suave, cv2.CV_32F, 0, 1)
    etiquetas = sk_watershed(np.hypot(gx, gy), marcadores)
    pupila = ndi.binary_fill_holes(etiquetas == 1).astype(np.uint8) * 255
    circ, sol = _forma_componente(pupila)
    if circ < CIRCULARIDAD_MIN or sol < SOLIDEZ_MIN:
        return None
    return _elipse_borde_visible(pupila)


def _elipse_borde_visible(mascara):
    """Ajusta una elipse SOLO con los puntos del borde que no están
    ocluidos. Un reflejo o una sombra que "muerde" la pupila deja una
    concavidad: esos puntos quedan lejos de la envolvente convexa y se
    descartan; los que coinciden con la envolvente son borde real.
    Devuelve la máscara de la elipse (o la envolvente si no alcanza)."""
    cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mascara
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    dist = np.array([abs(cv2.pointPolygonTest(hull, (float(p[0][0]), float(p[0][1])), True))
                     for p in c])
    visibles = c[dist <= 1.5]
    if len(visibles) < 0.4 * len(c) or len(visibles) < 10:
        return _envolvente(mascara)
    (ex, ey), (a, b), ang = cv2.fitEllipse(visibles.astype(np.float32))
    if min(a, b) <= 0 or min(a, b) / max(a, b) < 0.6:
        return _envolvente(mascara)
    salida = np.zeros_like(mascara)
    cv2.ellipse(salida, ((ex, ey), (a, b), ang), 255, -1)
    return salida


# ---------------------------------------------------------------------------
# Procesamiento del video
# ---------------------------------------------------------------------------

def procesar_video(cfg):
    video_cfg = cfg["video"]
    seg_cfg = cfg["segmentacion"]
    paso = video_cfg.get("cada_n_frames", 1)

    frames, fps = acq.extraer_frames(video_cfg["path"], paso)
    if not frames:
        raise RuntimeError(f"No se pudieron leer frames de: {video_cfg['path']}")

    # ROI: la del config si se definió; si no, el frame completo (el
    # seguimiento de la pupila se encarga de mirar solo el ojo).
    roi = video_cfg.get("roi")
    if roi is None:
        roi = (0, 0, frames[0].shape[1], frames[0].shape[0])
    rx, ry, rw, rh = [int(v) for v in roi]

    # Pupila inicial: primer frame (de los primeros) donde se encuentre
    centro, area_prev = None, None
    for f in frames[: max(1, int(fps))]:
        ini = localizar_pupila_inicial(f[ry:ry + rh, rx:rx + rw])
        if ini is not None:
            centro, area_prev = ini
            break
    if centro is None:
        raise RuntimeError("No se encontró la pupila en el primer segundo de video. "
                           "Probá definir 'roi' en config.yaml alrededor del ojo.")
    area_ref_frame = rw * rh

    filas = []
    perdidos = 0
    mediciones = deque(maxlen=max(5, int(0.2 * fps / paso)))
    relaciones = deque(maxlen=max(5, int(0.5 * fps / paso)))
    en_parpadeo = 0

    # Estímulo (escalón de brillo global), para no confundir el cambio de
    # iluminación del flash con un parpadeo
    brillos = pd.DataFrame({"brillo": [float(f.mean()) for f in frames]})
    idx_est, _ = detectar_estimulo(brillos, fps / paso)
    zona_estimulo = range(-1, -1) if idx_est is None else range(
        max(0, idx_est - int(0.3 * fps / paso)), idx_est + int(1.2 * fps / paso))
    for i, frame in enumerate(frames):
        recorte = frame[ry:ry + rh, rx:rx + rw]
        gris_total = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        brillo = float(gris_total.mean())

        x0, y0, x1, y1 = _ventana(centro, area_prev, gris_total.shape)
        gris = gris_total[y0:y1, x0:x1]

        # Parpadeo: el párpado aclara la zona del ojo mucho más que el resto
        # de la imagen (el flash, en cambio, aclara todo). Si la relación
        # brillo(ojo)/brillo(frame) se aparta > 8 % de su valor reciente,
        # el frame se descarta SIN tocar el seguimiento (centro y tamaño),
        # así al reabrirse el ojo se retoma desde la pupila correcta.
        # Un parpadeo dura < 0,4 s: si la "anomalía" sigue más tiempo, es el
        # nuevo estado normal (p. ej. otra iluminación) y se toma como
        # referencia.
        relacion = float(gris.mean()) / max(1.0, float(gris_total.mean()))
        if (i not in zona_estimulo and len(relaciones) >= 5
                and abs(relacion / np.median(relaciones) - 1) > 0.08
                and en_parpadeo < 0.4 * fps / paso):
            en_parpadeo += 1
            filas.append({
                "frame": i, "tiempo_s": (i * paso) / fps if fps > 0 else i,
                "area_px": np.nan, "diametro_px": np.nan,
                "centroide_x": np.nan, "centroide_y": np.nan,
                "valido": False, "brillo": round(brillo, 2),
            })
            continue
        if en_parpadeo >= 0.4 * fps / paso:
            relaciones.clear()
        en_parpadeo = 0
        relaciones.append(relacion)

        # Tolerancia adaptativa del region growing: mitad de la diferencia
        # entre el fondo de la pupila y el iris que la rodea. Con una
        # tolerancia fija, un reflejo tenue sobre la pupila (o el cambio de
        # iluminación al prender la luz) deja afuera parte de la pupila.
        tolerancia = seg_cfg.get("tolerancia_region_growing", 15)
        if seg_cfg["metodo"] == "region_growing":
            tolerancia = max(tolerancia, _tolerancia_adaptativa(
                gris, centro - [x0, y0], area_prev))

        mascara = segmentar_frame(
            gris, seg_cfg["metodo"],
            tolerancia,
            seg_cfg.get("tam_apertura", 3),
            seg_cfg.get("tam_cierre", 5),
            solo_pupila=False,
        )
        # Apertura proporcional al tamaño de la pupila: corta los "puentes"
        # finos que la unen a las pestañas o al borde del párpado (pasa
        # cuando el ojo mira hacia arriba), sin achicar la pupila.
        k = max(3, int(round(0.25 * np.sqrt(area_prev / np.pi))) | 1)
        mascara = morf.apertura(mascara, k)
        comp = elegir_pupila(mascara, centro - [x0, y0], area_prev, escala=area_ref_frame)

        # Refinamiento del borde con watershed, centrado en la pupila del
        # último frame bueno. El centro de la región del método solo se usa
        # si está cerca (si la región incluyó pestañas, su centroide se
        # corre y desubicaría a la watershed).
        c_ref = centro - [x0, y0]
        if comp is not None:
            m = cv2.moments(comp)
            c_met = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            if np.hypot(*(c_met - c_ref)) < 0.3 * np.sqrt(area_prev / np.pi):
                c_ref = c_met
        refinada = refinar_borde_watershed(gris, c_ref, np.sqrt(area_prev / np.pi))

        valido = False
        area = diametro = cx = cy = np.nan
        d_ref = 2 * np.sqrt(area_prev / np.pi)
        if refinada is not None:
            # El diámetro sale siempre de la watershed (borde por gradiente).
            # Se acepta toda medición plausible; los valores sueltos fuera
            # de la tendencia los elimina después _detectar_picos.
            a, d, cxl, cyl = medir_pupila(refinada)
            # Plausible: dentro de -25 % / +25 % del tamaño de referencia
            if 0.75 <= d / d_ref <= 1.25:
                valido = True
                area, diametro = a, d
                cx, cy = cxl + x0 + rx, cyl + y0 + ry
                centro = np.array([cxl + x0, cyl + y0])
                mediciones.append(d)

        # Tamaño de referencia del seguimiento: percentil 75 de las
        # mediciones de los últimos 0,2 s, con un cambio máximo de 2,5 % del
        # diámetro por frame (más rápido que cualquier respuesta pupilar
        # real). Se usa un percentil alto porque las oclusiones (reflejo,
        # párpado, pestañas) solo pueden ACHICAR la pupila medida: si la
        # watershed alterna entre el borde real y uno falso más chico, la
        # referencia se queda con el real y no queda "atrapada" en el falso.
        if mediciones:
            d_obj = float(np.percentile(mediciones, 75))
            paso_max = 0.025 * d_ref * paso
            d_nuevo = d_ref + float(np.clip(d_obj - d_ref, -paso_max, paso_max))
            area_prev = np.pi * d_nuevo ** 2 / 4
        perdidos = 0 if valido else perdidos + 1

        # Si se perdió la pupila más de 0,3 s, se la vuelve a buscar en
        # todo el frame. Solo se acepta una candidata de entre la mitad y
        # 1,4 veces el diámetro del último bueno (la pupila puede contraerse
        # mucho, pero no agrandarse tanto): evita "engancharse" al iris.
        if perdidos > 0.3 * fps / paso:
            ini = localizar_pupila_inicial(recorte)
            if ini is not None and 0.25 <= ini[1] / area_prev <= 2.0:
                centro, area_prev = ini
                mediciones.clear()
                perdidos = 0

        filas.append({
            "frame": i, "tiempo_s": (i * paso) / fps if fps > 0 else i,
            "area_px": area, "diametro_px": diametro,
            "centroide_x": cx, "centroide_y": cy,
            "valido": valido, "brillo": round(brillo, 2),
        })

    df = pd.DataFrame(filas)
    df["valido"] = df["valido"] & ~_detectar_picos(df["diametro_px"], fps / paso)

    # Artefactos (parpadeos, saltos): interpolación lineal de los frames
    # inválidos, para que la curva sea continua. La columna `valido`
    # indica cuáles se midieron y cuáles se interpolaron.
    for col in ["area_px", "diametro_px", "centroide_x", "centroide_y"]:
        df.loc[~df["valido"], col] = np.nan
    for col in ["area_px", "diametro_px", "centroide_x", "centroide_y"]:
        df[col] = df[col].interpolate(limit_direction="both")

    px_por_mm = cfg["calibracion"].get("px_por_mm")
    if px_por_mm:
        df["diametro_mm"] = df["diametro_px"] / px_por_mm
        df["area_mm2"] = df["area_px"] / (px_por_mm ** 2)

    return df, fps, roi


# ---------------------------------------------------------------------------
# Parámetros fisiológicos
# ---------------------------------------------------------------------------

def _suavizar(serie, fps, ventana_s=0.15):
    """Mediana (quita picos sueltos) + Savitzky-Golay (suaviza sin
    aplanar la pendiente, para derivar la velocidad)."""
    n = max(5, int(round(ventana_s * fps)) | 1)
    med = serie.rolling(window=n, center=True, min_periods=1).median()
    if len(med) > n + 2:
        return pd.Series(savgol_filter(med.values, n, 2), index=serie.index)
    return med


def detectar_estimulo(df, fps):
    """Tiempo del estímulo = escalón de brillo más grande del video.
    Devuelve (indice, signo): signo +1 si la luz sube (A), -1 si baja (B)."""
    b = df["brillo"].rolling(max(3, int(0.1 * fps)), center=True, min_periods=1).mean()
    salto = max(1, int(0.2 * fps))
    dif = (b.shift(-salto) - b.shift(salto)).fillna(0)
    idx = int(dif.abs().idxmax())
    escalon = dif.loc[idx]
    if abs(escalon) < 3:          # menos de 3 niveles de gris: sin escalón claro
        return None, 0
    # inicio del escalón: primer punto antes del máximo donde ya cambia
    inicio = idx
    while inicio > 0 and np.sign(dif.loc[inicio - 1]) == np.sign(escalon) \
            and abs(dif.loc[inicio - 1]) > 0.2 * abs(escalon):
        inicio -= 1
    return inicio + salto, int(np.sign(escalon))


def inferir_tipo(df, fps=30):
    """A si la luz sube (o, sin escalón de brillo, si la pupila se achica)."""
    if "brillo" in df.columns:
        _, signo = detectar_estimulo(df, fps)
        if signo != 0:
            return "A" if signo > 0 else "B"
    col = "diametro_mm" if "diametro_mm" in df.columns else "diametro_px"
    n = max(1, len(df) // 10)
    return "A" if df[col].iloc[-n:].median() < df[col].iloc[:n].median() else "B"


def calcular_parametros_fisiologicos(df, fps, tipo="auto"):
    col = "diametro_mm" if "diametro_mm" in df.columns else "diametro_px"
    unidad = "mm" if col == "diametro_mm" else "px"

    d = _suavizar(df[col], fps)
    t = df["tiempo_s"]
    if tipo == "auto":
        tipo = inferir_tipo(df, fps)

    # Estímulo: escalón de brillo; si no hay, se asume el inicio del video
    idx_est = None
    if "brillo" in df.columns:
        idx_est, _ = detectar_estimulo(df, fps)
    if idx_est is None:
        idx_est = 0
    t_est = t.loc[idx_est]

    # Basal: último segundo antes del estímulo (si no hay, primer 10 %).
    # Ruido basal robusto: MAD (no lo infla un pico suelto)
    pre = d.loc[max(0, idx_est - int(1 * fps)):max(0, idx_est - 1)]
    if len(pre) < 5:
        pre = d.iloc[: max(5, len(d) // 10)]
    d0 = pre.median()
    ruido = 1.4826 * np.median(np.abs(pre - d0)) if len(pre) > 5 else 0.0

    # Extremo (máxima constricción / redilatación): sobre una curva más
    # suavizada (0,5 s) para que un pico de artefacto no lo defina
    d_lenta = _suavizar(df[col], fps, ventana_s=0.5)
    post = d_lenta.loc[idx_est:]
    if tipo == "A":
        # constricción máxima dentro de los 5 s siguientes al estímulo
        idx_ext = post.loc[:idx_est + int(5 * fps)].idxmin()
    else:
        # Redilatación: es lenta y termina en una meseta, así que el extremo
        # se toma como el percentil 90 de la curva después del estímulo (el
        # máximo absoluto lo define cualquier pico de pocos frames)
        p90 = post.quantile(0.90)
        idx_ext = (post >= p90).idxmax()          # primer instante que lo alcanza
    d_ext = d_lenta.loc[idx_ext] if tipo == "A" else float(post.quantile(0.90))
    amplitud = abs(d0 - d_ext)

    # Velocidad: derivada de la curva suavizada a 0,3 s, para que el ruido
    # de segmentación frame a frame no infle el pico
    d_vel = _suavizar(df[col], fps, ventana_s=0.3)
    vel = pd.Series(np.gradient(d_vel.values, t.values), index=d.index)
    signo = -1 if tipo == "A" else 1          # A: el diámetro baja; B: sube
    v = signo * vel

    # Pico de velocidad de la respuesta (entre el estímulo y el extremo)
    tramo_v = v.loc[idx_est:idx_ext]
    v_pico = tramo_v.max() if len(tramo_v) > 1 and tramo_v.max() > 0 else np.nan

    # Latencia: primer instante en que el diámetro se aleja del basal más
    # que el 10 % de la amplitud o que 2 veces el ruido basal (lo mayor)
    umbral = max(0.10 * amplitud, 2 * ruido)
    tramo = d.loc[idx_est:idx_ext]
    d_lat = d0
    if tipo == "B":
        # Si justo después de apagar la luz la pupila baja un instante
        # (entrecerrar el ojo con el cambio de luz), la subida se mide
        # desde ese mínimo y no desde el basal
        inicio = d.loc[idx_est:idx_est + int(0.7 * fps)]
        if len(inicio) and inicio.min() < d0:
            d_lat = float(inicio.min())
            tramo = d.loc[inicio.idxmin():idx_ext]
    desvio = (d0 - tramo) if tipo == "A" else (tramo - d_lat)
    superan = desvio[desvio > umbral]
    idx_ini = superan.index.min() if len(superan) else idx_est
    latencia_s = t.loc[idx_ini] - t_est

    if tipo == "A":
        v_constr = v_pico
        resto = vel.loc[idx_ext:]
        v_redil = resto.max() if len(resto) > 1 and resto.max() > 0 else np.nan
    else:
        v_constr = np.nan
        v_redil = v_pico

    r = lambda v, n=3: round(float(v), n) if pd.notna(v) else np.nan
    return {
        "tipo_inferido": tipo,
        "t_estimulo_s": r(t_est),
        "diametro_basal": r(d0),
        "diametro_extremo": r(d_ext),
        "amplitud": r(amplitud),
        "amplitud_pct": r(100 * amplitud / d0 if d0 else np.nan, 2),
        "latencia_s": r(latencia_s),
        "velocidad_max": r(v_constr if tipo == "A" else v_redil),
        "velocidad_constriccion": r(v_constr),
        "velocidad_redilatacion": r(v_redil),
        "frames_validos_pct": r(100 * df["valido"].mean(), 1) if "valido" in df else np.nan,
        "unidad_longitud": unidad,
        "unidad_velocidad": f"{unidad}/s",
    }
