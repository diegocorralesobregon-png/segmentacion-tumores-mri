from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from dataset import normalizar
from unet import UNet

MODELO = "modelos/unet_v2.pt"
SALIDA = Path("resultados")
PACIENTES = {
    "TCGA_DU_5871_19941206": "el mejor",
    "TCGA_DU_7018_19911220": "se queda corto",
    "TCGA_CS_6666_20011109": "se queda corto",
    "TCGA_CS_6668_20011025": "marca de más",
}
ACIERTO = [0.1, 0.5, 1.0, 0.55]
SIN_MARCAR = [1.0, 0.1, 0.1, 0.65]
DE_MAS = [1.0, 0.9, 0.0, 0.65]


def predecir(modelo, imagen):
    x = torch.from_numpy(imagen).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        logits = modelo(normalizar(x).unsqueeze(0))
    return (torch.sigmoid(logits)[0, 0] > 0.5).numpy()


if __name__ == "__main__":
    df = pd.read_csv(SALIDA / "test_por_corte.csv")
    con = df[df["tiene_tumor"]].copy()
    sin = df[~df["tiene_tumor"]]

    # 1. ¿Influye el tamaño del tumor?
    con["tamano"] = pd.cut(
        con["px_real"], [0, 500, 2000, 70000],
        labels=["Pequeño (<500 px)", "Mediano (500-2000)", "Grande (>2000)"],
    )
    tabla = con.groupby("tamano", observed=True).agg(
        cortes=("dice", "size"),
        dice_medio=("dice", "mean"),
        no_detectados=("px_pred", lambda s: (s == 0).mean()),
    )
    print("\n1. Dice según el tamaño del tumor en el corte")
    print(tabla.round(3).to_string())

    # 2. ¿Dónde están los falsos positivos?
    falsos = sin[sin["px_pred"] > 0]
    print(f"\n2. Cortes sanos con algo marcado: {len(falsos)} de {len(sin)}")
    if len(falsos):
        cortes_tumor = {p: set(g["corte"]) for p, g in con.groupby("paciente")}
        junto = sum(
            any(abs(f["corte"] - c) <= 2 for c in cortes_tumor.get(f["paciente"], []))
            for _, f in falsos.iterrows()
        )
        print(f"   A 1-2 cortes de donde empieza o acaba el tumor: {junto}")
        print(f"   Manchas de menos de 50 px: {(falsos['px_pred'] < 50).sum()}")
        print(f"   Manchas de más de 200 px:  {(falsos['px_pred'] > 200).sum()}")

    # Figura 1: Dice frente a tamaño
    fig1, eje = plt.subplots(figsize=(7, 5))
    eje.scatter(con["px_real"], con["dice"], s=14, alpha=0.6, color="#3b6ea5")
    eje.set_xscale("log")
    eje.set_xlabel("Tamaño real del tumor en el corte (píxeles, escala logarítmica)")
    eje.set_ylabel("Dice del corte")
    eje.set_title("Dice de cada corte según el tamaño del tumor")
    fig1.tight_layout()
    fig1.savefig(SALIDA / "dice_vs_tamano.png", dpi=150)

    # Figura 2: cuatro pacientes, cuatro cortes de cada uno
    modelo = UNet()
    modelo.load_state_dict(torch.load(MODELO, map_location="cpu"))
    modelo.eval()

    fig2, ejes = plt.subplots(len(PACIENTES), 4, figsize=(14, 3.6 * len(PACIENTES)))
    for f, (paciente, nota) in enumerate(PACIENTES.items()):
        cortes = df[
            (df["paciente"] == paciente) & ((df["px_real"] > 0) | (df["px_pred"] > 0))
        ].sort_values("corte")
        elegidos = cortes.iloc[np.linspace(0, len(cortes) - 1, 4).round().astype(int)]

        for c, (_, fila) in enumerate(elegidos.iterrows()):
            imagen = np.array(Image.open(fila["imagen"]))
            real = np.array(Image.open(fila["mascara"])) > 0
            pred = predecir(modelo, imagen)

            capa = np.zeros((256, 256, 4))
            capa[real & pred] = ACIERTO
            capa[real & ~pred] = SIN_MARCAR
            capa[~real & pred] = DE_MAS

            eje = ejes[f, c]
            eje.imshow(imagen[:, :, 1], cmap="gray")
            eje.imshow(capa)
            eje.axis("off")
            titulo = f"corte {int(fila['corte'])} · Dice {fila['dice']:.2f}"
            if c == 0:
                titulo = f"{paciente[5:12]} ({nota})\n" + titulo
            eje.set_title(titulo, fontsize=10)

    fig2.legend(
        handles=[
            Patch(color=ACIERTO[:3], label="Acierto"),
            Patch(color=SIN_MARCAR[:3], label="Tumor sin marcar"),
            Patch(color=DE_MAS[:3], label="Marcado de más"),
        ],
        loc="lower center", ncol=3, fontsize=11,
    )
    fig2.tight_layout(rect=[0, 0.04, 1, 1])
    fig2.savefig(SALIDA / "errores_ejemplos.png", dpi=130)

print(f"\nFiguras guardadas en {SALIDA}/")
plt.show()