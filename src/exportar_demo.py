"""Guarda los pacientes de test en demo/*.npz para que la demo funcione sin el dataset completo."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

division = pd.read_csv("data/division.csv")
test = division[division["conjunto"] == "test"]
Path("demo").mkdir(exist_ok=True)

for paciente, cortes in test.groupby("paciente"):
    cortes = cortes.sort_values("corte")
    imagenes = np.stack([np.array(Image.open(r))[:, :, :3] for r in cortes["imagen"]]).astype(np.uint8)
    mascaras = np.stack([np.array(Image.open(r)) > 0 for r in cortes["mascara"]])
    np.savez_compressed(f"demo/{paciente}.npz", imagenes=imagenes, mascaras=mascaras,
                        cortes=cortes["corte"].to_numpy())
    print(paciente, len(cortes), "cortes")
