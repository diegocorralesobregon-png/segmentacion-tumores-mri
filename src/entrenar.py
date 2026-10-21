import torch
import pandas as pd
from tqdm import tqdm

from unet import UNet
from perdida import PerdidaDiceBCE, dice_por_imagen


def entrenar_epoca(modelo, loader, perdida, optimizador, dispositivo):
    """Una pasada completa por los datos de entrenamiento."""
    modelo.train()
    total = 0.0
    for imagenes, mascaras in tqdm(loader, desc="Entrenando", leave=False):
        imagenes, mascaras = imagenes.to(dispositivo), mascaras.to(dispositivo)

        optimizador.zero_grad()              # 1. borrar gradientes anteriores
        logits = modelo(imagenes)            # 2. predecir
        valor = perdida(logits, mascaras)    # 3. medir el error
        valor.backward()                     # 4. calcular gradientes
        optimizador.step()                   # 5. mover los pesos

        total += valor.item() * len(imagenes)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluar(modelo, loader, perdida, dispositivo):
    """Pérdida media y Dice medio por paciente. El loader no debe ir barajado."""
    modelo.eval()
    total = 0.0
    reales, predichos, aciertos = [], [], []
    for imagenes, mascaras in loader:
        imagenes, mascaras = imagenes.to(dispositivo), mascaras.to(dispositivo)
        logits = modelo(imagenes)
        total += perdida(logits, mascaras).item() * len(imagenes)
        pred = (torch.sigmoid(logits) > 0.5).float()
        reales += mascaras.sum(dim=(1, 2, 3)).tolist()
        predichos += pred.sum(dim=(1, 2, 3)).tolist()
        aciertos += (pred * mascaras).sum(dim=(1, 2, 3)).tolist()

    df = pd.DataFrame({
        "paciente": loader.dataset.df["paciente"].to_numpy(),
        "real": reales, "pred": predichos, "inter": aciertos,
    })
    pac = df.groupby("paciente").sum()
    dice = 2 * pac["inter"] / (pac["real"] + pac["pred"] + 1e-6)
    return total / len(loader.dataset), float(dice.mean())


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataset import DatasetTumores

    torch.manual_seed(0)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", dispositivo)

    # Cuatro cortes con tumor, siempre los mismos y sin aumentos
    ds = DatasetTumores("data/division.csv", "train")
    indices = ds.df[ds.df.tiene_tumor].sample(4, random_state=0).index.tolist()
    imagenes = torch.stack([ds[i][0] for i in indices]).to(dispositivo)
    mascaras = torch.stack([ds[i][1] for i in indices]).to(dispositivo)

    modelo = UNet().to(dispositivo)
    perdida = PerdidaDiceBCE()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    PASOS = 80
    modelo.train()
    for paso in range(1, PASOS + 1):
        optimizador.zero_grad()
        logits = modelo(imagenes)
        valor = perdida(logits, mascaras)
        valor.backward()
        optimizador.step()

        if paso == 1 or paso % 10 == 0:
            dice = dice_por_imagen(logits, mascaras).mean().item()
            print(f"Paso {paso:3d}   pérdida {valor.item():.3f}   Dice {dice:.3f}")

    # Real (rojo) frente a predicción (amarillo)
    pred = (torch.sigmoid(logits.detach()) > 0.5).float().cpu()
    fig, ejes = plt.subplots(1, 4, figsize=(14, 4))
    for i, eje in enumerate(ejes):
        eje.imshow(imagenes[i, 1].cpu(), cmap="gray")
        eje.contour(mascaras[i, 0].cpu(), levels=[0.5], colors="red", linewidths=1.5)
        eje.contour(pred[i, 0], levels=[0.5], colors="yellow", linewidths=1.5)
        eje.axis("off")
    plt.suptitle("Rojo: tumor real · Amarillo: predicción tras sobreajustar")
    plt.tight_layout()
    plt.show()