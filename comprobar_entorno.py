from pathlib import Path
import torch

print("PyTorch:", torch.__version__)
print("GPU disponible:", torch.cuda.is_available())

datos = Path("data/kaggle_3m")
pacientes = [p for p in datos.iterdir() if p.is_dir()]
mascaras = list(datos.glob("*/*_mask.tif"))
imagenes = [f for f in datos.glob("*/*.tif") if not f.name.endswith("_mask.tif")]

print("Pacientes:", len(pacientes))
print("Cortes:", len(imagenes))
print("Máscaras:", len(mascaras))