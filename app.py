import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from scipy import ndimage
from skimage.draw import line as linea_pixeles
from skimage.measure import marching_cubes

from inferencia import cargar_modelo, predecir, predecir_lote
from medidas import diametros, volumen_cm3, masa_g, DENSIDAD_TEJIDO

RUTA_MODELO = "modelos/unet_v2.pt"
CARPETA_DEMO = Path("demo")  # pacientes de test en .npz (los genera src/exportar_demo.py)
MM_PIXEL_DEFECTO = 0.94   # campo de visión típico de 240 mm en una matriz de 256
MM_CORTE_DEFECTO = 5.0    # grosor habitual de un FLAIR axial clínico

st.set_page_config(page_title="Brain tumour segmentation", layout="wide")


# ---------- Carga y utilidades ----------

@st.cache_resource
def obtener_modelo():
    return cargar_modelo(RUTA_MODELO)


def pacientes_de_test():
    return sorted(ruta.stem for ruta in CARPETA_DEMO.glob("*.npz"))


@st.cache_data(show_spinner=False)
def datos_paciente(paciente):
    """Cortes (n, 256, 256, 3), máscaras del radiólogo (n, 256, 256) y número de cada corte."""
    datos = np.load(CARPETA_DEMO / f"{paciente}.npz")
    return datos["imagenes"], datos["mascaras"], datos["cortes"]


@st.cache_data(show_spinner=False)
def volumen_paciente(paciente, _modelo):
    """Probabilidades y máscara real de todos los cortes del paciente, de abajo arriba."""
    imagenes, reales, numeros = datos_paciente(paciente)
    return predecir_lote(_modelo, list(imagenes)), reales, numeros


def leer_imagen(archivo):
    """Devuelve un corte (256, 256, 3) en 0-255 y avisa si hubo que adaptarlo."""
    original = np.array(Image.open(archivo))
    avisos = []
    matriz = original.astype(np.float32)
    if original.dtype != np.uint8 and matriz.max() > 0:
        matriz = matriz / matriz.max() * 255
    if matriz.ndim == 2:
        matriz = np.stack([matriz] * 3, axis=-1)
        avisos.append("The image has a single channel, but the model was trained with three sequences "
                      "(pre-contrast, FLAIR and post-contrast). The channel has been repeated, so the "
                      "result is less reliable.")
    matriz = matriz[:, :, :3]
    if matriz.shape[:2] != (256, 256):
        avisos.append(f"The image was {matriz.shape[1]}×{matriz.shape[0]} and has been resized to 256×256.")
        matriz = np.array(Image.fromarray(matriz.astype(np.uint8)).resize((256, 256), Image.BILINEAR))
    return matriz.astype(np.uint8), avisos


def a_gris(canal):
    canal = canal.astype(np.float32)
    canal = canal / max(canal.max(), 1)
    return np.stack([canal] * 3, axis=-1)


def borde(mascara):
    return mascara & ~ndimage.binary_erosion(mascara)


def pintar_linea(imagen, extremos, color):
    if extremos is None:
        return
    (f1, c1), (f2, c2) = extremos
    filas, columnas = linea_pixeles(int(round(f1)), int(round(c1)), int(round(f2)), int(round(c2)))
    dentro = (filas >= 0) & (filas < 256) & (columnas >= 0) & (columnas < 256)
    imagen[filas[dentro], columnas[dentro]] = color


def superponer(flair, prediccion, real=None, ejes=None):
    """FLAIR con la predicción en cian, el contorno real en rojo y los diámetros en amarillo."""
    imagen = a_gris(flair)
    cian = np.array([0.0, 0.85, 1.0])
    imagen[prediccion] = 0.55 * imagen[prediccion] + 0.45 * cian
    if real is not None and real.any():
        imagen[borde(real)] = [1.0, 0.15, 0.15]
    if ejes is not None:
        for extremos in ejes:
            pintar_linea(imagen, extremos, [1.0, 0.9, 0.0])
    return imagen


def superficie(volumen, mm_pixel, mm_corte, nombre, color, opacidad):
    """Malla 3D de una máscara (cortes, filas, columnas) en mm, o None si está vacía."""
    if not volumen.any():
        return None
    suave = ndimage.gaussian_filter(np.pad(volumen.astype(np.float32), 1), sigma=(0.5, 1, 1))
    nivel = 0.5 if suave.max() > 0.5 else suave.max() / 2
    espaciado = (mm_corte, mm_pixel, mm_pixel)
    vertices, caras, _, _ = marching_cubes(suave, level=nivel, spacing=espaciado)
    vertices -= np.array(espaciado)  # deshace el relleno de un vóxel
    return go.Mesh3d(
        x=vertices[:, 2], y=-vertices[:, 1], z=vertices[:, 0],
        i=caras[:, 0], j=caras[:, 1], k=caras[:, 2],
        color=color, opacity=opacidad, name=nombre, showlegend=True,
        flatshading=False, lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
        hoverinfo="name",
    )


def a_stl(malla):
    """Convierte una malla de Plotly en un archivo STL binario (en mm) para imprimirla en 3D."""
    vertices = np.column_stack([np.asarray(malla.x), np.asarray(malla.y), np.asarray(malla.z)])
    triangulos = vertices[np.column_stack([np.asarray(malla.i), np.asarray(malla.j), np.asarray(malla.k)])]
    triangulos = triangulos[:, ::-1]  # el eje y está invertido: se recupera la orientación de las caras
    normales = np.cross(triangulos[:, 1] - triangulos[:, 0], triangulos[:, 2] - triangulos[:, 0])
    normales /= np.linalg.norm(normales, axis=1, keepdims=True) + 1e-12
    registros = np.zeros(len(triangulos), dtype=[("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("atributo", "<u2")])
    registros["normal"], registros["vertices"] = normales, triangulos
    return b"STL tumor".ljust(80, b" ") + np.uint32(len(triangulos)).tobytes() + registros.tobytes()


def ejes_3d(extremos, z, mm_pixel, nombre, longitud):
    (f1, c1), (f2, c2) = extremos
    return go.Scatter3d(
        x=[c1 * mm_pixel, c2 * mm_pixel], y=[-f1 * mm_pixel, -f2 * mm_pixel], z=[z, z],
        mode="lines", line=dict(color="#f5c400", width=7),
        name="Measurement axes", legendgroup="ejes", showlegend=nombre == "mayor",
        hovertemplate=f"{nombre.capitalize()} diameter: {longitud:.1f} mm<extra></extra>",
    )


# ---------- Cabecera ----------

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(120deg, #B23A1E 0%, #D9622B 45%, #F2A541 100%);
           padding: 1.6rem 2rem; border-radius: 18px; color: #FFF8F0; margin-bottom: 1.2rem;}
    .hero h1 {color: #FFF8F0; margin: 0 0 .3rem 0; font-size: 2.3rem;}
    .hero .slogan {font-size: 1.25rem; font-style: italic; color: #FFE7C7; margin: 0;}
    .summary {background: #FCE9D6; border-left: 6px solid #D9622B; border-radius: 10px;
              padding: 1rem 1.3rem; margin-bottom: .6rem;}
    div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #F3C9A3;
              border-top: 4px solid #E07B39; border-radius: 12px; padding: .6rem .9rem;}
    section[data-testid="stSidebar"] img {border-radius: 50%; border: 3px solid #E07B39;}
    </style>
    <div class="hero">
      <h1>Brain Tumour Segmentation on MRI</h1>
      <p class="slogan">Turning grey MRI slices into a tumour you can measure, rotate and hold.</p>
    </div>
    <div class="summary">
      <b>About the project.</b> Following up gliomas means measuring the tumour by hand on every
      MRI scan, which is slow and varies between experts. This web tool uses a U-Net neural network
      to outline the tumour automatically, measures it (diameters, volume and estimated mass) and
      shows it as an interactive 3D model that can be exported for 3D printing.<br>
      <b>Objective:</b> to give clinicians fast and reproducible tumour measurements from any
      browser, without installing software. On 16 patients the model never saw, it reaches a mean
      Dice of 0.83.
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Learning project: this is not a diagnostic tool.")

EQUIPO = [("Diego Corrales", "diego"), ("Blanca Acón", "blanca"),
          ("Telmo Careaga", "telmo"), ("Leire Badiola", "leire")]
with st.sidebar.expander("👥 Group 08 · The team", expanded=False):
    for fila in (EQUIPO[:2], EQUIPO[2:]):
        cols = st.columns(2)
        for col, (nombre, foto) in zip(cols, fila):
            ruta = Path("equipo") / f"{foto}.jpg"
            if ruta.exists():
                col.image(str(ruta), width="stretch")
            col.markdown(f"<div style='text-align:center;font-weight:600;font-size:.85rem;line-height:1.2;margin-bottom:.6rem'>{nombre}</div>",
                         unsafe_allow_html=True)
    st.markdown(
        "Biomedical Engineering students  \n"
        "**Course:** Introduction to Biomedical Engineering, Final Project 2026–2027  \n"
        "**University of Deusto**  \n"
        "**Supervisors:** Begoña García-Zapirain Soto · Cyrille Kesiku Yetuyetu  \n"
        "**Data:** LGG Segmentation Dataset (110 patients, The Cancer Genome Atlas)  \n"
        "[Code on GitHub](https://github.com/diegocorralesobregon-png/segmentacion-tumores-mri)"
    )
st.sidebar.divider()

if not Path(RUTA_MODELO).exists():
    st.error(f"Model not found at {RUTA_MODELO}.")
    st.stop()
modelo = obtener_modelo()

# ---------- Panel lateral ----------

opciones = ["Upload an image"]
if pacientes_de_test():
    opciones.insert(0, "Test patient")
origen = st.sidebar.radio("Image", opciones)

imagen, real, avisos, paciente, numero = None, None, [], None, None
if origen == "Test patient":
    paciente = st.sidebar.selectbox("Patient", pacientes_de_test(), format_func=lambda p: p[5:12])
    imagenes, mascaras, numeros_cortes = datos_paciente(paciente)
    con_mas_tumor = int(numeros_cortes[mascaras.sum(axis=(1, 2)).argmax()])
    numero = st.sidebar.slider(
        "Slice", int(numeros_cortes.min()), int(numeros_cortes.max()), con_mas_tumor
    )
    indice = int(np.where(numeros_cortes == numero)[0][0])
    imagen, real = imagenes[indice], mascaras[indice]
    st.sidebar.caption("Patients the model never saw during training.")
else:
    archivo = st.sidebar.file_uploader("MRI slice", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if archivo is not None:
        imagen, avisos = leer_imagen(archivo)

umbral = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption(
    "Pixels with a probability above this value are marked as tumour. "
    "The model was evaluated at 0.50."
)

st.sidebar.markdown("**Scale**")
mm_pixel = st.sidebar.number_input("Pixel size (mm)", 0.1, 3.0, MM_PIXEL_DEFECTO, 0.01)
mm_corte = st.sidebar.number_input("Slice spacing (mm)", 0.5, 10.0, MM_CORTE_DEFECTO, 0.5)
st.sidebar.caption(
    "The dataset files do not store the real scale. The default values are typical of a clinical "
    "FLAIR scan, not measured on these images, so values in mm are approximate."
)

if imagen is None:
    st.info("Upload an MRI slice from the panel on the left.")
    st.stop()

for aviso in avisos:
    st.warning(aviso)

# ---------- Predicción del corte ----------

probabilidad = predecir(modelo, imagen)
prediccion = probabilidad > umbral
flair = imagen[:, :, 1]
mayor, perpendicular, ext_mayor, ext_perp = diametros(prediccion, mm_pixel)

pestana_corte, pestana_3d = st.tabs(["Slice", "3D model"])

# ---------- Pestaña: corte ----------

with pestana_corte:
    area = int(prediccion.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Marked area", f"{area * mm_pixel ** 2 / 100:.2f} cm²", f"{area} px",
              delta_color="off", delta_arrow="off")
    c2.metric("Maximum probability", f"{probabilidad.max():.2f}")
    if real is not None:
        if real.any() or prediccion.any():
            dice = 2 * (prediccion & real).sum() / (prediccion.sum() + real.sum())
            c3.metric("Dice vs. radiologist", f"{dice:.2f}", f"real tumour: {int(real.sum())} px",
                      delta_color="off", delta_arrow="off")
        else:
            c3.metric("Dice vs. radiologist", "—", "no tumour and nothing marked: correct",
                      delta_color="off", delta_arrow="off")

    d1, d2, d3 = st.columns(3)
    d1.metric("Longest diameter", f"{mayor:.1f} mm" if mayor else "—")
    d2.metric("Perpendicular diameter", f"{perpendicular:.1f} mm" if perpendicular else "—")
    d3.metric("Product of diameters", f"{mayor * perpendicular / 100:.2f} cm²" if mayor else "—")

    mostrar_ejes = st.checkbox("Show the diameters on the image", value=True)
    ejes = (ext_mayor, ext_perp) if mostrar_ejes else None

    i1, i2, i3 = st.columns(3)
    i1.image(a_gris(flair), caption="FLAIR", width="stretch")
    leyenda = "Prediction (cyan)" + (", radiologist (red)" if real is not None else "")
    if mostrar_ejes and mayor:
        leyenda += " and diameters (yellow)"
    i2.image(superponer(flair, prediccion, real, ejes), caption=leyenda, width="stretch")
    i3.image(plt.get_cmap("magma")(probabilidad)[:, :, :3], caption="Probability map", width="stretch")

    with st.expander("Show the three sequences"):
        s1, s2, s3 = st.columns(3)
        for columna, canal, nombre in zip((s1, s2, s3), range(3), ("Pre-contrast", "FLAIR", "Post-contrast")):
            columna.image(a_gris(imagen[:, :, canal]), caption=nombre, width="stretch")

    with st.expander("How reliable is it?"):
        st.markdown(
            "- On 16 test patients, the mean Dice per patient is **0.83 ± 0.12** (median 0.87).\n"
            "- It fails more often on small tumours (under 500 pixels) and misses part of very large ones.\n"
            "- It marks something on about 11% of healthy slices, mostly in the orbits, "
            "the skull base and the highest slices.\n"
            "- It was trained only on lower-grade gliomas and 2D slices of 256×256 pixels.\n"
            "- The mask covers the abnormality visible on FLAIR, which includes tumour and oedema."
        )

# ---------- Pestaña: modelo 3D ----------

with pestana_3d:
    if paciente is None:
        st.info("The 3D model needs every slice of a patient. "
                "Choose \"Test patient\" in the panel on the left.")
        st.stop()

    with st.spinner("Predicting every slice of the patient..."):
        probs, reales, numeros = volumen_paciente(paciente, modelo)
    pred_vol = probs > umbral

    vol_pred = volumen_cm3(pred_vol, mm_pixel, mm_corte)
    vol_real = volumen_cm3(reales, mm_pixel, mm_corte)
    suma = pred_vol.sum() + reales.sum()
    dice_3d = 2 * (pred_vol & reales).sum() / suma if suma else 1.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted volume", f"{vol_pred:.1f} cm³")
    m2.metric("Estimated mass", f"{masa_g(vol_pred):.1f} g")
    m3.metric("Radiologist volume", f"{vol_real:.1f} cm³")
    m4.metric("Patient Dice", f"{dice_3d:.2f}")

    trazas = []
    malla_pred = superficie(pred_vol, mm_pixel, mm_corte, "Predicted tumour", "#00c8ff", 0.85)
    malla_real = superficie(reales, mm_pixel, mm_corte, "Radiologist tumour", "#ff3b3b", 0.35)
    trazas += [t for t in (malla_pred, malla_real) if t is not None]

    # Ejes de medida en el corte donde la predicción es mayor
    areas = pred_vol.sum(axis=(1, 2))
    if areas.max() > 0:
        k = int(np.argmax(areas))
        may, per, e_may, e_per = diametros(pred_vol[k], mm_pixel)
        z = k * mm_corte
        trazas.append(ejes_3d(e_may, z, mm_pixel, "longest", may))
        if e_per is not None:
            trazas.append(ejes_3d(e_per, z, mm_pixel, "perpendicular", per))
        texto_ejes = (f"The yellow axes are on slice {int(numeros[k])}, where the prediction "
                      f"is largest: {may:.1f} × {per:.1f} mm.")
    else:
        texto_ejes = "With this threshold the network marks no tumour on any slice."

    if not trazas:
        st.info("There is no tumour to draw with this threshold.")
    else:
        figura = go.Figure(trazas)
        figura.update_layout(
            height=620, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers",
                        orientation="h", y=1.02, x=0),
            scene=dict(
                aspectmode="data",
                xaxis_title="mm", yaxis_title="mm", zaxis_title="mm (bottom to top)",
            ),
        )
        st.plotly_chart(figura, width="stretch")

    if malla_pred is not None:
        st.download_button(
            "Download the predicted tumour for 3D printing (STL)", a_stl(malla_pred),
            file_name=f"tumor_{paciente[5:12]}.stl", mime="model/stl",
        )

    st.caption(
        f"Click a legend item to show or hide it; double-click to show it alone. "
        f"Drag to rotate and use the mouse wheel to zoom. {texto_ejes} "
        f"The mass is the volume multiplied by {DENSIDAD_TEJIDO} g/cm³, an approximate density "
        f"of brain tissue. This patient has {len(numeros)} slices: with the default slice spacing "
        f"({MM_CORTE_DEFECTO} mm) the volume may be far off if the original scan had thinner slices."
    )
