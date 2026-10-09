"""Carga config.yaml a un diccionario de Python con valores por defecto."""
import yaml

DEFAULTS = {
    "video": {"path": "", "tipo": "auto", "roi": None, "cada_n_frames": 1},
    "segmentacion": {
        "metodo": "region_growing",
        "tolerancia_region_growing": 15,
        "tam_apertura": 3,
        "tam_cierre": 5,
    },
    "textura": {"niveles_glcm": 32},
    "calibracion": {"px_por_mm": None},
    "output": {
        "carpeta": "resultados",
        "csv_serie_temporal": "pupila_serie_temporal.csv",
        "csv_parametros": "parametros_fisiologicos.csv",
        "csv_textura": "textura_glcm.csv",
    },
}


def _merge(base, override):
    resultado = dict(base)
    for clave, valor in (override or {}).items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = _merge(resultado[clave], valor)
        else:
            resultado[clave] = valor
    return resultado


def cargar_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        crudo = yaml.safe_load(f) or {}
    cfg = dict(DEFAULTS)
    for seccion in DEFAULTS:
        cfg[seccion] = _merge(DEFAULTS[seccion], crudo.get(seccion))
    return cfg
