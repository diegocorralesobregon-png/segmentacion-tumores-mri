from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from dataset import normalizar
from unet import UNet

SALIDA = Path("resultados")
EJEMPLOS = ["TCGA_DU_5871_19941206", "TCGA_HT_7884_19980913", "TCGA_CS_6668_20011025"]


def otsu(valores):
    """Umbral (0-255) que mejor separa las intensidades en dos grupos."""
    hist = np.bincount(valores.ravel(), minlength=256).astype(float)
    niveles = np.arange(256)
    peso0 = np.cumsum(hist)
    peso1 = hist.sum() - peso0
    suma0 = np.cumsum(hist * niveles)
    media0 = suma0 / np.maximum(peso0, 1)
    media1 = (suma0[-1] - suma0) / np.maximum(peso1, 1)
    return int(np.argmax(peso0 * peso1 * (media0 - media1) ** 2))


def segmentar_clasico(flair, erosion=12, area_minima=100):
    """Segmentación sin aprendizaje: umbrales de Otsu encadenados y morfología."""
    vacio = np.zeros(flair.shape, dtype=bool)

    # 1. Separar la cabeza del fondo negro y quitar cuero cabelludo y cráneo
    cabeza = ndimage.binary_fill_holes(flair > otsu(flair))
    cerebro = ndimage.binary_erosion(cabeza, iterations=erosion)
    if cerebro.sum() < 500:
        return vacio

    # 2. Dentro del cerebro, quedarse con el tejido (fuera el líquido, que es oscuro)
    tejido = cerebro & (flair > otsu(flair[cerebro]))
    if tejido.sum() < area_minima:
        return vacio

    # 3. Dentro del tejido, quedarse con lo más brillante
    brillante = tejido & (flair > otsu(flair[tejido]))

    # 4. De todas las manchas brillantes, solo la más grande
    etiquetas, n = ndimage.label(brillante)
    if n == 0:
        return vacio
    areas = ndimage.sum(brillante, etiquetas, index=np.arange(1, n + 1))
    if areas.max() < area_minima:
        return vacio
    return etiquetas == (int(np.argmax(areas)) + 1)


def metricas_por_paciente(df):
    eps = 1e-6
    pac = df.groupby("paciente")[["px_real", "px_pred", "px_inter"]].sum()
    pac["dice"] = 2 * pac["px_inter"] / (pac["px_real"] + pac["px_pred"] + eps)
    pac["iou"] = pac["px_inter"] / (pac["px_real"] + pac["px_pred"] - pac["px_inter"] + eps)
    pac["sensibilidad"] = pac["px_inter"] / (pac["px_real"] + eps)
    pac["precision"] = pac["px_inter"] / (pac["px_pred"] + eps)
    return pac


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-6)


if __name__ == "__main__":
    df = pd.read_csv("data/division.csv")
    df = df[df["conjunto"] == "test"].reset_index(drop=True)

    # Otsu sobre los mismos 16 pacientes de test
    reales, predichos, aciertos = [], [], []
    for _, fila in tqdm(df.iterrows(), total=len(df), desc="Otsu en test"):
        flair = np.array(Image.open(fila["imagen"]))[:, :, 1]
        real = np.array(Image.open(fila["mascara"])) > 0
        pred = segmentar_clasico(flair)
        reales.append(int(real.sum()))
        predichos.append(int(pred.sum()))
        aciertos.append(int((pred & real).sum()))

    df["px_real"] = reales
    df["px_pred"] = predichos
    df["px_inter"] = aciertos
    clasico = metricas_por_paciente(df)

    # Resultados de la U-Net guardados el día 10
    unet = pd.read_csv(SALIDA / "test_por_paciente.csv", index_col="paciente").loc[clasico.index]
    unet_cortes = pd.read_csv(SALIDA / "test_por_corte.csv")

    print(f"\n{'':<18}{'Otsu clásico':>18}{'U-Net':>18}")
    for m in ["dice", "iou", "sensibilidad", "precision"]:
        a = f"{clasico[m].mean():.3f} ± {clasico[m].std():.3f}"
        b = f"{unet[m].mean():.3f} ± {unet[m].std():.3f}"
        print(f"{m:<18}{a:>18}{b:>18}")

    sanos_c = df[~df["tiene_tumor"]]
    sanos_u = unet_cortes[~unet_cortes["tiene_tumor"]]
    print(f"{'Sanos con marca':<18}{(sanos_c['px_pred'] > 0).mean():>18.1%}"
          f"{(sanos_u['px_pred'] > 0).mean():>18.1%}")

    gana = int((clasico["dice"] > unet["dice"]).sum())
    print(f"\nPacientes en los que Otsu supera a la U-Net: {gana} de {len(clasico)}")
    print("\nDice por paciente:")
    print(pd.DataFrame({"otsu": clasico["dice"], "unet": unet["dice"]}).round(3).to_string())

    clasico.round(4).to_csv(SALIDA / "clasico_por_paciente.csv")

    # Figura: los dos métodos sobre el mismo corte
    modelo = UNet()
    modelo.load_state_dict(torch.load("modelos/unet_v2.pt", map_location="cpu"))
    modelo.eval()

    fig, ejes = plt.subplots(len(EJEMPLOS), 4, figsize=(14, 3.6 * len(EJEMPLOS)))
    for f, paciente in enumerate(EJEMPLOS):
        cortes = df[df["paciente"] == paciente]
        fila = cortes.loc[cortes["pixeles_tumor"].idxmax()]
        imagen = np.array(Image.open(fila["imagen"]))
        flair = imagen[:, :, 1]
        real = np.array(Image.open(fila["mascara"])) > 0

        pred_otsu = segmentar_clasico(flair)
        x = torch.from_numpy(imagen).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            logits = modelo(normalizar(x).unsqueeze(0))
        pred_unet = (torch.sigmoid(logits)[0, 0] > 0.5).numpy()

        columnas = [
            (f"{paciente[5:12]} · FLAIR", None, None),
            ("Radiólogo", real, "red"),
            (f"Otsu · Dice {dice(pred_otsu, real):.2f}", pred_otsu, "yellow"),
            (f"U-Net · Dice {dice(pred_unet, real):.2f}", pred_unet, "cyan"),
        ]
        for c, (titulo, mascara, color) in enumerate(columnas):
            eje = ejes[f, c]
            eje.imshow(flair, cmap="gray")
            if mascara is not None and mascara.any():
                eje.contour(mascara, levels=[0.5], colors=color, linewidths=1.5)
            eje.set_title(titulo, fontsize=10)
            eje.axis("off")

    fig.tight_layout()
    fig.savefig(SALIDA / "comparacion_clasico.png", dpi=130)
    print(f"\nGuardado en {SALIDA}/")
    plt.show()