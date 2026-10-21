from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import crear_loaders
from unet import UNet

MODELO = "modelos/unet_v2.pt"
SALIDA = Path("resultados")

if __name__ == "__main__":
    modelo = UNet()
    modelo.load_state_dict(torch.load(MODELO, map_location="cpu"))
    modelo.eval()

    _, _, test_loader = crear_loaders(lote=8)
    df = test_loader.dataset.df.copy()

    reales, predichos, aciertos = [], [], []
    with torch.no_grad():
        for imagenes, mascaras in tqdm(test_loader, desc="Evaluando test"):
            pred = (torch.sigmoid(modelo(imagenes)) > 0.5).float()
            reales += mascaras.sum(dim=(1, 2, 3)).tolist()
            predichos += pred.sum(dim=(1, 2, 3)).tolist()
            aciertos += (pred * mascaras).sum(dim=(1, 2, 3)).tolist()

    eps = 1e-6
    df["px_real"] = reales
    df["px_pred"] = predichos
    df["px_inter"] = aciertos
    df["dice"] = (2 * df["px_inter"] + eps) / (df["px_real"] + df["px_pred"] + eps)

    # Métricas por paciente
    pac = df.groupby("paciente")[["px_real", "px_pred", "px_inter"]].sum()
    pac["dice"] = 2 * pac["px_inter"] / (pac["px_real"] + pac["px_pred"] + eps)
    pac["iou"] = pac["px_inter"] / (pac["px_real"] + pac["px_pred"] - pac["px_inter"] + eps)
    pac["sensibilidad"] = pac["px_inter"] / (pac["px_real"] + eps)
    pac["precision"] = pac["px_inter"] / (pac["px_pred"] + eps)
    pac = pac.sort_values("dice")

    con = df[df["tiene_tumor"]]
    sin = df[~df["tiene_tumor"]]
    dice_global = 2 * df["px_inter"].sum() / (df["px_real"].sum() + df["px_pred"].sum())

    lineas = [
        f"Conjunto de test: {len(pac)} pacientes, {len(df)} cortes ({len(con)} con tumor)",
        "",
        "Por paciente (media ± desviación, mediana):",
    ]
    for m in ["dice", "iou", "sensibilidad", "precision"]:
        lineas.append(
            f"  {m:<13} {pac[m].mean():.3f} ± {pac[m].std():.3f}   mediana {pac[m].median():.3f}"
        )
    lineas += [
        f"  Peor paciente: {pac['dice'].min():.3f}   Mejor: {pac['dice'].max():.3f}",
        "",
        "Por corte:",
        f"  Dice medio en cortes con tumor: {con['dice'].mean():.3f}",
        f"  Cortes con tumor no detectados: {(con['px_pred'] == 0).mean():.1%}",
        f"  Cortes sanos con algo marcado:  {(sin['px_pred'] > 0).mean():.1%}",
        "",
        f"Dice global (todos los píxeles): {dice_global:.3f}",
    ]
    texto = "\n".join(lineas)
    print("\n" + texto + "\n")
    print(pac[["dice", "iou", "sensibilidad", "precision"]].round(3).to_string())

    # Guardar resultados
    SALIDA.mkdir(exist_ok=True)
    (SALIDA / "test_resumen.txt").write_text(texto, encoding="utf-8")
    pac.round(4).to_csv(SALIDA / "test_por_paciente.csv")
    df.to_csv(SALIDA / "test_por_corte.csv", index=False)

    # Gráfica para el README
    fig, eje = plt.subplots(figsize=(8, 6))
    nombres = [p[5:12] for p in pac.index]
    eje.barh(nombres, pac["dice"], color="#3b6ea5")
    eje.axvline(pac["dice"].mean(), color="black", linestyle="--", linewidth=1)
    for i, v in enumerate(pac["dice"]):
        eje.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    eje.set_xlim(0, 1.08)
    eje.set_xlabel("Dice")
    eje.set_title(f"Dice por paciente en test (media {pac['dice'].mean():.3f})")
    plt.tight_layout()
    plt.savefig(SALIDA / "dice_por_paciente_test.png", dpi=150)
    print(f"\nResultados guardados en la carpeta {SALIDA}/")
    plt.show()