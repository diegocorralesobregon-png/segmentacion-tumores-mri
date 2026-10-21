import torch
import torch.nn as nn


class PerdidaDiceBCE(nn.Module):
    """BCE + (1 - Dice suave). Recibe logits, no probabilidades."""

    def __init__(self, suavizado=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.suavizado = suavizado

    def forward(self, logits, mascaras):
        bce = self.bce(logits, mascaras)

        # Dice suave sobre todos los píxeles del lote
        prob = torch.sigmoid(logits)
        interseccion = (prob * mascaras).sum()
        total = prob.sum() + mascaras.sum()
        dice = (2 * interseccion + self.suavizado) / (total + self.suavizado)

        return bce + (1 - dice)


def dice_por_imagen(logits, mascaras, umbral=0.5, eps=1e-6):
    """Dice duro de cada imagen del lote. Devuelve un valor por imagen."""
    pred = (torch.sigmoid(logits) > umbral).float()
    interseccion = (pred * mascaras).sum(dim=(1, 2, 3))
    total = pred.sum(dim=(1, 2, 3)) + mascaras.sum(dim=(1, 2, 3))
    return (2 * interseccion + eps) / (total + eps)


if __name__ == "__main__":
    perdida = PerdidaDiceBCE()

    def logits_de(mascara_pred):
        # +10 donde se predice tumor, -10 donde no
        return mascara_pred * 20 - 10

    def exactitud(logits, mascara):
        pred = (torch.sigmoid(logits) > 0.5).float()
        return (pred == mascara).float().mean().item()

    # Máscara de juguete: un tumor cuadrado de 40x40 (2,4 % de la imagen)
    real = torch.zeros(1, 1, 256, 256)
    real[:, :, 100:140, 100:140] = 1

    nada = torch.zeros_like(real)
    desplazada = torch.zeros_like(real)
    desplazada[:, :, 100:140, 110:150] = 1

    casos = {
        "No predice nada": nada,
        "Desplazada 10 px": desplazada,
        "Perfecta": real,
    }

    print(f"{'Predicción':<20}{'Exactitud':>10}{'Dice':>8}{'Pérdida':>10}")
    for nombre, pred in casos.items():
        lg = logits_de(pred)
        print(
            f"{nombre:<20}"
            f"{exactitud(lg, real):>10.2%}"
            f"{dice_por_imagen(lg, real).item():>8.3f}"
            f"{perdida(lg, real).item():>10.3f}"
        )

    # Corte sin tumor en el que tampoco se predice nada
    vacia = torch.zeros(1, 1, 256, 256)
    print("\nDice con máscara vacía y predicción vacía:",
          round(dice_por_imagen(logits_de(vacia), vacia).item(), 3))

    # Todo junto: datos reales + U-Net sin entrenar + pérdida + gradientes
    from dataset import crear_loaders
    from unet import UNet

    train_loader, _, _ = crear_loaders(lote=4)
    imagenes, mascaras = next(iter(train_loader))
    modelo = UNet()
    logits = modelo(imagenes)
    valor = perdida(logits, mascaras)
    valor.backward()

    print(f"\nPérdida con la U-Net sin entrenar: {valor.item():.3f}")
    print(f"Dice medio del lote: {dice_por_imagen(logits, mascaras).mean().item():.3f}")
    print("Gradientes calculados:", modelo.enc1.capas[0].weight.grad is not None)