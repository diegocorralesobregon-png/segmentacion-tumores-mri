import csv
import random
import time
from pathlib import Path

import torch

from dataset import crear_loaders
from unet import UNet
from perdida import PerdidaDiceBCE
from entrenar import entrenar_epoca, evaluar

EPOCAS = 60
LOTE = 16
LR = 1e-3
SEMILLA = 42

if __name__ == "__main__":
    random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", dispositivo)

    train_loader, val_loader, _ = crear_loaders(lote=LOTE, trabajadores=2)
    modelo = UNet().to(dispositivo)
    perdida = PerdidaDiceBCE()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=EPOCAS)

    Path("modelos").mkdir(exist_ok=True)
    mejor_dice = 0.0
    historial = []

    for epoca in range(1, EPOCAS + 1):
        inicio = time.time()
        lr_actual = optimizador.param_groups[0]["lr"]
        perdida_train = entrenar_epoca(modelo, train_loader, perdida, optimizador, dispositivo)
        perdida_val, dice_val = evaluar(modelo, val_loader, perdida, dispositivo)
        planificador.step()
        historial.append([epoca, perdida_train, perdida_val, dice_val, lr_actual])

        marca = ""
        if dice_val > mejor_dice:
            mejor_dice = dice_val
            torch.save(modelo.state_dict(), "modelos/unet_v2.pt")
            marca = "  <- mejor, guardado"

        print(
            f"Época {epoca:2d}/{EPOCAS}   train {perdida_train:.3f}   val {perdida_val:.3f}   "
            f"Dice paciente {dice_val:.3f}   lr {lr_actual:.5f}   "
            f"({time.time() - inicio:.0f} s){marca}"
        )

        with open("modelos/historial_v2.csv", "w", newline="") as f:
            escritor = csv.writer(f)
            escritor.writerow(["epoca", "perdida_train", "perdida_val", "dice_paciente_val", "lr"])
            escritor.writerows(historial)

    print(f"\nMejor Dice por paciente en validación: {mejor_dice:.3f}")