import torch
import torch.nn as nn


class BloqueConv(nn.Module):
    """Dos convoluciones 3x3, cada una seguida de BatchNorm y ReLU."""

    def __init__(self, entrada, salida):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Conv2d(entrada, salida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            nn.Conv2d(salida, salida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.capas(x)


class UNet(nn.Module):
    def __init__(self, canales_entrada=3, canales_salida=1, base=32):
        super().__init__()
        # Encoder
        self.enc1 = BloqueConv(canales_entrada, base)
        self.enc2 = BloqueConv(base, base * 2)
        self.enc3 = BloqueConv(base * 2, base * 4)
        self.enc4 = BloqueConv(base * 4, base * 8)
        self.bajar = nn.MaxPool2d(2)

        # Cuello de botella
        self.cuello = BloqueConv(base * 8, base * 16)

        # Decoder: cada bloque recibe el doble de canales por la conexión de salto
        self.subir4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = BloqueConv(base * 16, base * 8)
        self.subir3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = BloqueConv(base * 8, base * 4)
        self.subir2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = BloqueConv(base * 4, base * 2)
        self.subir1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = BloqueConv(base * 2, base)

        # Un valor por píxel
        self.salida = nn.Conv2d(base, canales_salida, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                 # 32 canales, 256x256
        e2 = self.enc2(self.bajar(e1))    # 64, 128x128
        e3 = self.enc3(self.bajar(e2))    # 128, 64x64
        e4 = self.enc4(self.bajar(e3))    # 256, 32x32

        c = self.cuello(self.bajar(e4))   # 512, 16x16

        d4 = self.dec4(torch.cat([self.subir4(c), e4], dim=1))   # 256, 32x32
        d3 = self.dec3(torch.cat([self.subir3(d4), e3], dim=1))  # 128, 64x64
        d2 = self.dec2(torch.cat([self.subir2(d3), e2], dim=1))  # 64, 128x128
        d1 = self.dec1(torch.cat([self.subir1(d2), e1], dim=1))  # 32, 256x256

        return self.salida(d1)            # 1, 256x256 (logits)


if __name__ == "__main__":
    import time
    from dataset import crear_loaders

    modelo = UNet()
    parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Parámetros: {parametros:,}")

    # Prueba 1: un tensor aleatorio con la forma de un lote
    x = torch.randn(2, 3, 256, 256)
    modelo.eval()
    with torch.no_grad():
        inicio = time.time()
        y = modelo(x)
    print("Entrada:", tuple(x.shape), "-> Salida:", tuple(y.shape))
    print(f"Tiempo para 2 imágenes: {time.time() - inicio:.2f} s")

    # Prueba 2: un lote real del dataset
    _, val_loader, _ = crear_loaders(lote=4)
    imagenes, mascaras = next(iter(val_loader))
    with torch.no_grad():
        logits = modelo(imagenes)
    print("Lote real:", tuple(imagenes.shape), "->", tuple(logits.shape))
    print(f"Logits: de {logits.min():.2f} a {logits.max():.2f}")
    assert logits.shape == mascaras.shape, "La salida no coincide con la máscara"
    print("La salida tiene la misma forma que la máscara.")