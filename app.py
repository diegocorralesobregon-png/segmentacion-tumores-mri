import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

from inferencia import cargar_modelo, predecir

RUTA_MODELO = "modelos/unet_v2.pt"
RUTA_DIVISION = Path("data/division.csv")

st.set_page_config(page_title="Segmentación de tumores cerebrales", layout="wide")


@st.cache_resource
def obtener_modelo():
    return cargar_modelo(RUTA_MODELO)


@st.cache_data
def pacientes_de_test():
    df = pd.read_csv(RUTA_DIVISION)
    return df[df["conjunto"] == "test"].sort_values(["paciente", "corte"])


def leer_imagen(archivo):
    """Devuelve un corte (256, 256, 3) en 0-255 y avisa si hubo que adaptarlo."""
    original = np.array(Image.open(archivo))
    avisos = []
    matriz = original.astype(np.float32)
    if original.dtype != np.uint8 and matriz.max() > 0:
        matriz = matriz / matriz.max() * 255
    if matriz.ndim == 2:
        matriz = np.stack([matriz] * 3, axis=-1)
        avisos.append("La imagen tiene un solo canal y el modelo se entrenó con tres secuencias "
                      "(precontraste, FLAIR y poscontraste). Se ha repetido el canal, así que el "
                      "resultado es menos fiable.")
    matriz = matriz[:, :, :3]
    if matriz.shape[:2] != (256, 256):
        avisos.append(f"La imagen medía {matriz.shape[1]}×{matriz.shape[0]} y se ha reescalado a 256×256.")
        matriz = np.array(Image.fromarray(matriz.astype(np.uint8)).resize((256, 256), Image.BILINEAR))
    return matriz.astype(np.uint8), avisos


def a_gris(canal):
    """Canal en escala de grises con el contraste ajustado, como imagen RGB entre 0 y 1."""
    canal = canal.astype(np.float32)
    canal = canal / max(canal.max(), 1)
    return np.stack([canal] * 3, axis=-1)


def borde(mascara):
    return mascara & ~ndimage.binary_erosion(mascara)


def superponer(flair, prediccion, real=None):
    """FLAIR con la predicción en cian translúcido y, si existe, el contorno real en rojo."""
    imagen = a_gris(flair)
    cian = np.array([0.0, 0.85, 1.0])
    imagen[prediccion] = 0.55 * imagen[prediccion] + 0.45 * cian
    if real is not None and real.any():
        imagen[borde(real)] = [1.0, 0.15, 0.15]
    return imagen


st.title("Segmentación de tumores cerebrales en resonancia magnética")
st.caption(
    "U-Net entrenada con gliomas de bajo grado (dataset LGG de TCGA). "
    "Proyecto de aprendizaje: no es una herramienta de diagnóstico."
)

if not Path(RUTA_MODELO).exists():
    st.error(f"No encuentro el modelo en {RUTA_MODELO}.")
    st.stop()
modelo = obtener_modelo()

# --- Panel lateral: de dónde sale la imagen ---
opciones = ["Subir una imagen"]
if RUTA_DIVISION.exists():
    opciones.insert(0, "Paciente de test")
origen = st.sidebar.radio("Imagen", opciones)

imagen, real, avisos = None, None, []
if origen == "Paciente de test":
    df = pacientes_de_test()
    paciente = st.sidebar.selectbox(
        "Paciente", sorted(df["paciente"].unique()), format_func=lambda p: p[5:12]
    )
    cortes = df[df["paciente"] == paciente]
    con_mas_tumor = int(cortes.loc[cortes["pixeles_tumor"].idxmax(), "corte"])
    numero = st.sidebar.slider(
        "Corte", int(cortes["corte"].min()), int(cortes["corte"].max()), con_mas_tumor
    )
    fila = cortes[cortes["corte"] == numero].iloc[0]
    imagen, avisos = leer_imagen(fila["imagen"])
    real = np.array(Image.open(fila["mascara"])) > 0
    st.sidebar.caption("Pacientes que el modelo no vio durante el entrenamiento.")
else:
    archivo = st.sidebar.file_uploader("Corte de resonancia", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if archivo is not None:
        imagen, avisos = leer_imagen(archivo)

umbral = st.sidebar.slider("Umbral de decisión", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption(
    "Por encima de este valor de probabilidad, un píxel se marca como tumor. "
    "El modelo se evaluó con 0,50."
)

if imagen is None:
    st.info("Sube un corte de resonancia desde el panel de la izquierda.")
    st.stop()

for aviso in avisos:
    st.warning(aviso)

# --- Predicción ---
probabilidad = predecir(modelo, imagen)
prediccion = probabilidad > umbral
flair = imagen[:, :, 1]

# --- Cifras ---
area = int(prediccion.sum())
c1, c2, c3 = st.columns(3)
c1.metric("Área marcada", f"{area} px", f"{area / prediccion.size:.1%} de la imagen",
          delta_color="off", delta_arrow="off")
c2.metric("Probabilidad máxima", f"{probabilidad.max():.2f}")
if real is not None:
    if real.any() or prediccion.any():
        dice = 2 * (prediccion & real).sum() / (prediccion.sum() + real.sum())
        c3.metric("Dice frente al radiólogo", f"{dice:.2f}", f"tumor real: {int(real.sum())} px",
                  delta_color="off", delta_arrow="off")
    else:
        c3.metric("Dice frente al radiólogo", "—", "sin tumor y sin marca: correcto",
                  delta_color="off", delta_arrow="off")

# --- Imágenes ---
i1, i2, i3 = st.columns(3)
i1.image(a_gris(flair), caption="FLAIR", width="stretch")
leyenda = "Predicción (cian)" + (" y contorno del radiólogo (rojo)" if real is not None else "")
i2.image(superponer(flair, prediccion, real), caption=leyenda, width="stretch")
i3.image(plt.get_cmap("magma")(probabilidad)[:, :, :3], caption="Mapa de probabilidad", width="stretch")

with st.expander("Ver las tres secuencias"):
    s1, s2, s3 = st.columns(3)
    for columna, canal, nombre in zip((s1, s2, s3), range(3), ("Precontraste", "FLAIR", "Poscontraste")):
        columna.image(a_gris(imagen[:, :, canal]), caption=nombre, width="stretch")

with st.expander("Cómo de fiable es"):
    st.markdown(
        "- En 16 pacientes de test, el Dice medio por paciente es **0,83 ± 0,12** (mediana 0,87).\n"
        "- Falla más en tumores pequeños (menos de 500 píxeles) y se queda corto en los muy grandes.\n"
        "- En torno al 11 % de los cortes sanos marca algo, sobre todo en las órbitas, "
        "la base del cráneo y los cortes más altos.\n"
        "- Solo se ha entrenado con gliomas de bajo grado y cortes 2D de 256×256."
    )