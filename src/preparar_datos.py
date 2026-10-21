from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

DATOS = Path("data/kaggle_3m")
SEMILLA = 42

# 1. Una fila por corte, con cuántos píxeles de tumor tiene
filas = []
pacientes = sorted(p for p in DATOS.iterdir() if p.is_dir())

for paciente in tqdm(pacientes, desc="Leyendo máscaras"):
    for ruta_mascara in paciente.glob("*_mask.tif"):
        ruta_imagen = ruta_mascara.with_name(ruta_mascara.name.replace("_mask", ""))
        pixeles = int((np.array(Image.open(ruta_mascara)) > 0).sum())
        filas.append({
            "paciente": paciente.name,
            "corte": int(ruta_imagen.stem.split("_")[-1]),
            "imagen": ruta_imagen.as_posix(),
            "mascara": ruta_mascara.as_posix(),
            "pixeles_tumor": pixeles,
            "tiene_tumor": pixeles > 0,
        })

df = pd.DataFrame(filas).sort_values(["paciente", "corte"]).reset_index(drop=True)

# 2. Estadísticas generales
con_tumor = df[df.tiene_tumor]
por_paciente = df.groupby("paciente").size()

print("\nCortes totales:", len(df))
print("Con tumor:", len(con_tumor), f"({df.tiene_tumor.mean():.1%})")
print("Tamaño mediano del tumor:", f"{con_tumor.pixeles_tumor.median() / 65536:.2%} de la imagen")
print("Cortes por paciente: de", por_paciente.min(), "a", por_paciente.max())

# 3. División por paciente: 78 train, 16 val, 16 test
rng = np.random.default_rng(SEMILLA)
nombres = np.array([p.name for p in pacientes])
rng.shuffle(nombres)

conjunto = {}
for nombre in nombres[:78]:
    conjunto[nombre] = "train"
for nombre in nombres[78:94]:
    conjunto[nombre] = "val"
for nombre in nombres[94:]:
    conjunto[nombre] = "test"

df["conjunto"] = df.paciente.map(conjunto)

# 4. Comprobar que ningún paciente está en dos conjuntos
assert df.groupby("paciente").conjunto.nunique().max() == 1, "Hay fuga de datos"

# 5. Resumen y guardado
resumen = df.groupby("conjunto").agg(
    pacientes=("paciente", "nunique"),
    cortes=("imagen", "size"),
    pct_con_tumor=("tiene_tumor", "mean"),
).loc[["train", "val", "test"]]
resumen["pct_con_tumor"] = (resumen.pct_con_tumor * 100).round(1)

print("\n", resumen)

df.to_csv("data/division.csv", index=False)
print("\nGuardado en data/division.csv")

# 6. Histograma del tamaño del tumor
plt.hist(con_tumor.pixeles_tumor / 65536 * 100, bins=40)
plt.xlabel("Tumor (% de la imagen)")
plt.ylabel("Número de cortes")
plt.title("Tamaño del tumor en los cortes que lo tienen")
plt.tight_layout()
plt.show()