from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import crear_loaders
from unet import UNet


def buscar(nombre):
    for carpeta in ("modelos", "."):
        ruta = Path(carpeta) / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(f"No encuentro {nombre} ni en modelos/ ni en la raíz del proyecto")


if __name__ == "__main__":
    modelo = UNet()
    modelo.load_state_dict(torch.load(buscar("unet_v2.pt"), map_location="cpu"))
    modelo.eval()

    _, val_loader, _ = crear_loaders(lote=8)
    df = val_loader.dataset.df.copy()

    # Píxeles reales, predichos y acertados de cada corte
    reales, predichos, aciertos = [], [], []
    with torch.no_grad():
        for imagenes, mascaras in tqdm(val_loader, desc="Prediciendo"):
            pred = (torch.sigmoid(modelo(imagenes)) > 0.5).float()
            reales += mascaras.sum(dim=(1, 2, 3)).tolist()
            predichos += pred.sum(dim=(1, 2, 3)).tolist()
            aciertos += (pred * mascaras).sum(dim=(1, 2, 3)).tolist()

    df["px_real"] = reales
    df["px_pred"] = predichos
    df["px_inter"] = aciertos
    eps = 1e-6
    df["dice"] = (2 * df["px_inter"] + eps) / (df["px_real"] + df["px_pred"] + eps)

    con = df[df["tiene_tumor"]]
    sin = df[~df["tiene_tumor"]]
    falsos = sin[sin["px_pred"] > 0]

    print(f"\n1. Dice medio por corte (todos): {df['dice'].mean():.3f}")

    print(f"\n2. Cortes CON tumor ({len(con)})")
    print(f"   Dice medio:   {con['dice'].mean():.3f}")
    print(f"   Dice mediano: {con['dice'].median():.3f}")
    print(f"   No detectados (no marca nada): {(con['px_pred'] == 0).mean():.1%}")

    print(f"\n3. Cortes SIN tumor ({len(sin)})")
    print(f"   Marca algo por error: {len(falsos) / len(sin):.1%}")
    if len(falsos):
        print(f"   Tamaño mediano de esas manchas: {falsos['px_pred'].median():.0f} píxeles")
        print(f"   Manchas de menos de 50 píxeles: {(falsos['px_pred'] < 50).mean():.1%}")

    # Dice por paciente: se suman los píxeles de todos sus cortes
    pac = df.groupby("paciente")[["px_real", "px_pred", "px_inter"]].sum()
    pac["dice"] = 2 * pac["px_inter"] / (pac["px_real"] + pac["px_pred"] + eps)

    print(f"\n4. Dice por paciente ({len(pac)})   <- el medio debe salir 0.816")
    print(f"   Medio: {pac['dice'].mean():.3f}   Mediano: {pac['dice'].median():.3f}")
    print(f"   Peor:  {pac['dice'].min():.3f}   Mejor:   {pac['dice'].max():.3f}")
    print(pac["dice"].sort_values().round(3).to_string())

    total = 2 * df["px_inter"].sum() / (df["px_real"].sum() + df["px_pred"].sum())
    print(f"\n5. Dice global (todos los píxeles juntos): {total:.3f}")

    # Curvas del entrenamiento
    h = pd.read_csv(buscar("historial_v2.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(h["epoca"], h["perdida_train"], label="Train")
    a.plot(h["epoca"], h["perdida_val"], label="Validación")
    a.set_xlabel("Época")
    a.set_ylabel("Pérdida")
    a.set_title("Pérdida")
    a.legend()
    b.plot(h["epoca"], h["dice_paciente_val"])
    b.set_xlabel("Época")
    b.set_ylabel("Dice")
    b.set_title("Dice por paciente en validación")
    plt.tight_layout()
    plt.show()