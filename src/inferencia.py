import torch

from dataset import normalizar
from unet import UNet


def cargar_modelo(ruta="modelos/unet_v2.pt"):
    modelo = UNet()
    modelo.load_state_dict(torch.load(ruta, map_location="cpu"))
    modelo.eval()
    return modelo


def predecir(modelo, imagen):
    """Recibe un corte (256, 256, 3) en 0-255 y devuelve el mapa de probabilidad (256, 256)."""
    x = torch.from_numpy(imagen).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        logits = modelo(normalizar(x).unsqueeze(0))
    return torch.sigmoid(logits)[0, 0].numpy()