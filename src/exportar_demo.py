"""Guarda los pacientes de test en demo/*.npz para que la web funcione sin el dataset completo.

Cada archivo lleva los cortes, las máscaras del radiólogo, el número de cada corte, las
probabilidades de la red ya calculadas (en 0-255) y la máscara de la cabeza para la vista 3D.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu

from inferencia import cargar_modelo, predecir_lote


def mascara_cabeza(imagenes):
    """Contorno de la cabeza en cada corte, quitando los cortes de ruido de arriba."""
    cabeza = []
    for corte in imagenes:
        suave = ndimage.gaussian_filter(corte.max(-1).astype(float), 2)
        m = suave > threshold_otsu(suave) * 0.6
        m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=4))
        etiquetas, n = ndimage.label(m)
        if n:
            m = etiquetas == (np.argmax(ndimage.sum(m, etiquetas, range(1, n + 1))) + 1)
        cabeza.append(m)
    cabeza = np.array(cabeza)
    areas = cabeza.sum(axis=(1, 2))
    cabeza[areas > 0.85 * cabeza[0].size] = False      # cortes que son solo ruido
    areas = cabeza.sum(axis=(1, 2))
    k = int(areas[: int(0.75 * len(areas))].argmax())
    for i in range(k + 1, len(areas)):                   # hacia arriba el área solo puede bajar
        if areas[i - 1] == 0 or areas[i] > areas[i - 1] * 1.1:
            cabeza[i:] = False
            break
    return cabeza


if __name__ == "__main__":
    modelo = cargar_modelo("modelos/unet_v2.pt")
    division = pd.read_csv("data/division.csv")
    test = division[division["conjunto"] == "test"]
    Path("demo").mkdir(exist_ok=True)

    for paciente, cortes in test.groupby("paciente"):
        cortes = cortes.sort_values("corte")
        imagenes = np.stack([np.array(Image.open(r))[:, :, :3] for r in cortes["imagen"]]).astype(np.uint8)
        mascaras = np.stack([np.array(Image.open(r)) > 0 for r in cortes["mascara"]])
        prob = predecir_lote(modelo, list(imagenes))
        np.savez_compressed(
            f"demo/{paciente}.npz", imagenes=imagenes, mascaras=mascaras,
            cortes=cortes["corte"].to_numpy(), prob=np.round(prob * 255).astype(np.uint8),
            cabeza=mascara_cabeza(imagenes),
        )
        print(paciente, len(cortes), "cortes")
