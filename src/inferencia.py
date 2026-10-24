import numpy as np
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
    return predecir_lote(modelo, [imagen])[0]


def predecir_lote(modelo, imagenes, lote=8):
    """Recibe una lista de cortes (256, 256, 3) y devuelve sus mapas de probabilidad (n, 256, 256)."""
    salidas = []
    for i in range(0, len(imagenes), lote):
        x = torch.from_numpy(np.stack(imagenes[i:i + lote])).permute(0, 3, 1, 2).float() / 255
        x = torch.stack([normalizar(c) for c in x])
        with torch.no_grad():
            salidas.append(torch.sigmoid(modelo(x))[:, 0].numpy())
    return np.concatenate(salidas)
