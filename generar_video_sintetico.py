"""Genera un video sintético (pupila que se contrae) para probar el pipeline."""
import cv2
import numpy as np
import os

SALIDA = "data/registro_ejemplo.mp4"
FPS = 30
DURACION_S = 5
SIZE = 300


def dibujar_ojo(radio_pupila):
    img = np.full((SIZE, SIZE, 3), 230, dtype=np.uint8)
    centro = (SIZE // 2, SIZE // 2)
    cv2.circle(img, centro, 110, (180, 200, 210), -1)
    cv2.circle(img, centro, 70, (90, 60, 40), -1)
    cv2.circle(img, centro, int(radio_pupila), (10, 10, 10), -1)
    cv2.circle(img, (centro[0] + 12, centro[1] - 12), 6, (255, 255, 255), -1)
    ruido = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    return img


def main():
    os.makedirs("data", exist_ok=True)
    n_frames = FPS * DURACION_S
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(SALIDA, fourcc, FPS, (SIZE, SIZE))

    radio_basal, radio_contraido = 45, 18
    inicio_estimulo = int(n_frames * 0.15)
    fin_contraccion = int(n_frames * 0.55)

    for i in range(n_frames):
        if i < inicio_estimulo:
            radio = radio_basal
        elif i < fin_contraccion:
            frac = (i - inicio_estimulo) / (fin_contraccion - inicio_estimulo)
            frac = 1 - (1 - frac) ** 2
            radio = radio_basal + (radio_contraido - radio_basal) * frac
        else:
            radio = radio_contraido
        out.write(dibujar_ojo(radio))

    out.release()
    print(f"Video sintético generado: {SALIDA} ({n_frames} frames, {FPS} fps)")


if __name__ == "__main__":
    main()
