import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2 import InterpolationMode


def normalizar(imagen):
    """Cada canal queda con media 0 y escala parecida, sea cual sea el brillo del escáner."""
    media = imagen.mean(dim=(1, 2), keepdim=True)
    desviacion = imagen.std(dim=(1, 2), keepdim=True)
    return (imagen - media) / (desviacion + 0.01)


class DatasetTumores(Dataset):
    def __init__(self, ruta_csv, conjunto, aumentar=False):
        df = pd.read_csv(ruta_csv)
        self.df = df[df.conjunto == conjunto].reset_index(drop=True)
        self.aumentar = aumentar

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        fila = self.df.iloc[i]
        imagen = np.array(Image.open(fila["imagen"]))
        mascara = np.array(Image.open(fila["mascara"])) > 0

        imagen = torch.from_numpy(imagen).permute(2, 0, 1).float() / 255
        mascara = torch.from_numpy(mascara).unsqueeze(0).float()

        if self.aumentar:
            imagen, mascara = self.aumento(imagen, mascara)
        return normalizar(imagen), mascara

    def aumento(self, imagen, mascara):
        if random.random() < 0.5:
            imagen, mascara = F.hflip(imagen), F.hflip(mascara)
        angulo = random.uniform(-15, 15)
        imagen = F.rotate(imagen, angulo, interpolation=InterpolationMode.BILINEAR)
        mascara = F.rotate(mascara, angulo, interpolation=InterpolationMode.NEAREST)
        return imagen, mascara


def crear_loaders(ruta_csv="data/division.csv", lote=16, trabajadores=0):
    train = DatasetTumores(ruta_csv, "train", aumentar=True)
    val = DatasetTumores(ruta_csv, "val")
    test = DatasetTumores(ruta_csv, "test")

    # Los cortes con tumor se eligen más a menudo, hasta ser la mitad de lo que ve la red
    con_tumor = train.df["tiene_tumor"].to_numpy()
    pesos = np.where(con_tumor, 0.5 / con_tumor.sum(), 0.5 / (~con_tumor).sum())
    muestreador = WeightedRandomSampler(
        torch.as_tensor(pesos, dtype=torch.double), num_samples=len(train), replacement=True
    )

    return (
        DataLoader(train, batch_size=lote, sampler=muestreador, num_workers=trabajadores),
        DataLoader(val, batch_size=lote, num_workers=trabajadores),
        DataLoader(test, batch_size=lote, num_workers=trabajadores),
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader = crear_loaders()
    imagenes, mascaras = next(iter(train_loader))
    print("Lote de imágenes:", tuple(imagenes.shape),
        f"- valores de {imagenes.min():.2f} a {imagenes.max():.2f}")
    print("Lote de máscaras:", tuple(mascaras.shape), "- valores:", mascaras.unique().tolist())

    con = total = 0
    for i, (_, m) in enumerate(train_loader):
        con += (m.sum(dim=(1, 2, 3)) > 0).sum().item()
        total += len(m)
        if i == 19:
            break
    print(f"Cortes con tumor en 20 lotes: {con / total:.0%}  (antes era el 34 %)")