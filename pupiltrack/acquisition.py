"""
Módulo 0/1 - Adquisición de datos.
Extrae frames de un video (registro A: oscuridad->luz, registro B: luz->oscuridad)
y permite recortar la Región de Interés (ROI) del ojo.
"""
import cv2
import numpy as np


def extraer_frames(video_path, cada_n_frames=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % cada_n_frames == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, fps


def seleccionar_roi(frame, roi=None):
    if roi is None:
        roi = cv2.selectROI("Seleccionar ojo (ENTER para confirmar)", frame, False)
        cv2.destroyAllWindows()
    x, y, w, h = roi
    return frame[y:y + h, x:x + w], roi


def frame_representativo(frames, indice=None):
    if indice is None:
        indice = len(frames) // 2
    return frames[indice]
