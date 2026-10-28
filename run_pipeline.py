"""
Pipeline PupilTrack — entrega final (Módulos 0 a 4).

Uso:
    python run_pipeline.py [ruta/a/config.yaml]

Para procesar un video nuevo, NO hay que tocar ningún .py: alcanza con
editar `video.path` en config.yaml.
"""
import sys
import os
import cv2
import pandas as pd

from pupiltrack.config import cargar_config
from pupiltrack.temporal import procesar_video, calcular_parametros_fisiologicos
from pupiltrack import acquisition as acq
from pupiltrack import texture as tex
from pupiltrack import compression as comp


def main():
    ruta_config = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = cargar_config(ruta_config)

    video_path = cfg["video"]["path"]
    if not video_path or not os.path.exists(video_path):
        print(f"[ERROR] No se encontró el video en: '{video_path}'")
        print("        Editá 'video: path:' en config.yaml.")
        sys.exit(1)

    print(f"Procesando: {video_path}")
    print(f"Método de segmentación: {cfg['segmentacion']['metodo']}")

    df, fps, roi = procesar_video(cfg)
    print(f"Frames procesados: {len(df)} | fps: {fps:.2f} | ROI: {roi}")

    frames_validos = df["diametro_px"].notna().sum()
    print(f"Frames con pupila detectada: {frames_validos}/{len(df)}")

    parametros = calcular_parametros_fisiologicos(df, fps, tipo=cfg["video"]["tipo"])

    carpeta_out = cfg["output"]["carpeta"]
    os.makedirs(carpeta_out, exist_ok=True)

    path_serie = os.path.join(carpeta_out, cfg["output"]["csv_serie_temporal"])
    df.to_csv(path_serie, index=False)
    print(f"Serie temporal guardada en: {path_serie}")

    path_params = os.path.join(carpeta_out, cfg["output"]["csv_parametros"])
    pd.DataFrame([parametros]).to_csv(path_params, index=False)
    print(f"Parámetros fisiológicos guardados en: {path_params}")

    # Módulo 3.1: textura (GLCM + 1er orden) sobre un frame representativo
    try:
        x, y, w, h = roi
        frames, _ = acq.extraer_frames(video_path, cfg["video"].get("cada_n_frames", 1))
        frame_rep = acq.frame_representativo(frames)[y:y + h, x:x + w]
        niveles = cfg["textura"]["niveles_glcm"]
        glcm_rgb = tex.textura_glcm_rgb(frame_rep, niveles=niveles)
        primer_orden = tex.textura_primer_orden_rgb(frame_rep)

        filas_textura = []
        for canal in ["R", "G", "B"]:
            fila = {"canal": canal}
            fila.update(primer_orden[canal])
            fila.update(glcm_rgb[canal])
            filas_textura.append(fila)

        path_textura = os.path.join(carpeta_out, cfg["output"]["csv_textura"])
        pd.DataFrame(filas_textura).to_csv(path_textura, index=False)
        print(f"Textura (GLCM, frame representativo) guardada en: {path_textura}")
    except Exception as e:
        print(f"[AVISO] No se pudo calcular textura: {e}")

    # Módulo 4: compresión (Huffman, RLE H/V, JPEG Q=90/50/10/5) sobre el
    # frame representativo y la ROI segmentada, y efecto de JPEG sobre el
    # diámetro pupilar medido
    try:
        if "frames" not in locals():
            frames, _ = acq.extraer_frames(video_path, cfg["video"].get("cada_n_frames", 1))
        validos = df.index[df["valido"]] if "valido" in df else df.index
        i_rep = validos[abs(validos - len(df) // 2).argmin()]
        fila = df.loc[i_rep]
        gris = cv2.cvtColor(frames[i_rep], cv2.COLOR_BGR2GRAY)
        mascara = comp.mascara_pupila_en_frame(frames[i_rep], fila["centroide_x"],
                                               fila["centroide_y"], fila["diametro_px"])
        tabla = comp.evaluar_compresion(gris, mascara)
        tabla.insert(0, "frame", int(i_rep))
        path_comp = os.path.join(carpeta_out, cfg["output"].get("csv_compresion", "compresion_metodos.csv"))
        tabla.to_csv(path_comp, index=False)
        print(f"Compresión (Módulo 4) guardada en: {path_comp}")

        efecto = comp.efecto_en_diametro(frames, df, comp.medir_diametro)
        path_ef = os.path.join(carpeta_out, cfg["output"].get("csv_compresion_diametro", "compresion_diametro.csv"))
        efecto.to_csv(path_ef, index=False)
        print(f"Efecto de JPEG en el diámetro guardado en: {path_ef}")
    except Exception as e:
        print(f"[AVISO] No se pudo calcular compresión: {e}")

    print("\n--- Resumen ---")
    for k, v in parametros.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
