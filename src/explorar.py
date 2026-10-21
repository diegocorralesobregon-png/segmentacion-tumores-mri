from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATOS = Path("data/kaggle_3m")


def numero_corte(ruta):
    # TCGA_CS_4941_19960909_12.tif -> 12
    return int(ruta.stem.split("_")[-1])


def ruta_mascara(ruta_corte):
    return ruta_corte.with_name(ruta_corte.stem + "_mask.tif")


def cargar(ruta):
    return np.array(Image.open(ruta))


# 1. Elegir un paciente
pacientes = sorted(p for p in DATOS.iterdir() if p.is_dir())
paciente = pacientes[0]
print("Paciente:", paciente.name)

# 2. Sus cortes, ordenados de abajo a arriba de la cabeza
cortes = sorted(
    (f for f in paciente.glob("*.tif") if not f.name.endswith("_mask.tif")),
    key=numero_corte,
)
print("Cortes de este paciente:", len(cortes))

# 3. Buscar el corte donde el tumor ocupa más píxeles
areas = [(cargar(ruta_mascara(c)) > 0).sum() for c in cortes]
corte = cortes[int(np.argmax(areas))]

imagen = cargar(corte)
mascara = cargar(ruta_mascara(corte)) > 0

print("Corte elegido:", corte.name)
print("Imagen:", imagen.shape, imagen.dtype, "- valores de", imagen.min(), "a", imagen.max())
print("Píxeles de tumor:", mascara.sum(), "de", mascara.size)
print("Cortes con tumor en este paciente:", sum(a > 0 for a in areas), "de", len(cortes))

# 4. Dibujar los tres canales, la máscara y la superposición
nombres = ["Precontraste", "FLAIR", "Poscontraste"]
fig, ejes = plt.subplots(1, 5, figsize=(18, 4))

for i in range(3):
    ejes[i].imshow(imagen[:, :, i], cmap="gray")
    ejes[i].set_title(nombres[i])

ejes[3].imshow(mascara, cmap="gray")
ejes[3].set_title("Máscara")

ejes[4].imshow(imagen[:, :, 1], cmap="gray")
ejes[4].contour(mascara.astype(float), levels=[0.5], colors="red", linewidths=1.5)
ejes[4].set_title("FLAIR + tumor")

for eje in ejes:
    eje.axis("off")

plt.suptitle(f"{paciente.name} - corte {numero_corte(corte)}")
plt.tight_layout()
plt.show()