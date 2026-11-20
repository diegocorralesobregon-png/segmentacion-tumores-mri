import io
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
from PIL import Image
from scipy import ndimage
from skimage.draw import line as linea_pixeles
from skimage.measure import marching_cubes

from medidas import diametros, volumen_cm3, masa_g, DENSIDAD_TEJIDO

RUTA_MODELO = "modelos/unet_v2.pt"
CARPETA_DEMO = Path("demo")        # pacientes de test en .npz (los genera src/exportar_demo.py)
CARPETA_RESULTADOS = Path("resultados/web")
MM_PIXEL_DEFECTO = 0.94   # campo de visión típico de 240 mm en una matriz de 256
MM_CORTE_DEFECTO = 5.0    # grosor habitual de un FLAIR axial clínico

st.set_page_config(page_title="Brain tumour segmentation", layout="wide")

# El diseño para móvil solo se activa si el navegador dice que es un móvil
try:
    _agente = st.context.headers.get("User-Agent", "") or ""
except Exception:
    _agente = ""
MOVIL = any(clave in _agente for clave in ("Mobi", "Android", "iPhone", "iPad"))
if st.query_params.get("mobile") in ("1", "0"):   # para probar: ?mobile=1 o ?mobile=0
    MOVIL = st.query_params.get("mobile") == "1"


# ---------- Carga y utilidades ----------

@st.cache_resource
def obtener_modelo():
    from inferencia import cargar_modelo
    return cargar_modelo(RUTA_MODELO)


def predecir_cortes(imagenes):
    from inferencia import predecir_lote
    return predecir_lote(obtener_modelo(), list(imagenes))


def pacientes_de_test():
    return sorted(ruta.stem for ruta in CARPETA_DEMO.glob("*.npz"))


@st.cache_data(show_spinner=False)
def datos_paciente(paciente):
    """Cortes, máscaras del radiólogo, número de corte y, si están guardadas,
    las probabilidades ya calculadas por la red y la máscara de la cabeza."""
    datos = np.load(CARPETA_DEMO / f"{paciente}.npz")
    prob = datos["prob"].astype(np.float32) / 255 if "prob" in datos else None
    cabeza = datos["cabeza"] if "cabeza" in datos else None
    return datos["imagenes"], datos["mascaras"], datos["cortes"], prob, cabeza


@st.cache_data(show_spinner=False)
def probabilidades_paciente(paciente):
    imagenes, _, _, prob, _ = datos_paciente(paciente)
    return prob if prob is not None else predecir_cortes(imagenes)


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


def malla(volumen, mm_pixel, mm_corte, paso=1):
    """Vértices (x, y, z en mm) y caras de la superficie de una máscara (cortes, filas, columnas)."""
    if not volumen.any():
        return None
    if paso > 1:
        volumen = volumen[:, ::paso, ::paso]
        mm_pixel = mm_pixel * paso
    suave = ndimage.gaussian_filter(np.pad(volumen.astype(np.float32), 1), sigma=(0.5, 1, 1))
    nivel = 0.5 if suave.max() > 0.5 else suave.max() / 2
    espaciado = (mm_corte, mm_pixel, mm_pixel)
    vertices, caras, _, _ = marching_cubes(suave, level=nivel, spacing=espaciado)
    vertices -= np.array(espaciado)  # deshace el relleno de un vóxel
    xyz = np.column_stack([vertices[:, 2], -vertices[:, 1], vertices[:, 0]])
    return xyz, caras


def superficie(volumen, mm_pixel, mm_corte, nombre, color, opacidad, paso=1, hover=True):
    resultado = malla(volumen, mm_pixel, mm_corte, paso)
    if resultado is None:
        return None
    xyz, caras = resultado
    return go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        i=caras[:, 0], j=caras[:, 1], k=caras[:, 2],
        color=color, opacity=opacidad, name=nombre, showlegend=True,
        flatshading=False, lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
        hoverinfo="name" if hover else "skip",
    )


def a_stl(malla_plotly):
    """Convierte una malla de Plotly en un archivo STL binario (en mm) para imprimirla en 3D."""
    vertices = np.column_stack([np.asarray(malla_plotly.x), np.asarray(malla_plotly.y), np.asarray(malla_plotly.z)])
    triangulos = vertices[np.column_stack([np.asarray(malla_plotly.i), np.asarray(malla_plotly.j),
                                           np.asarray(malla_plotly.k)])]
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
        name="Measurement axes", legendgroup="ejes", showlegend=nombre == "longest",
        hovertemplate=f"{nombre.capitalize()} diameter: {longitud:.1f} mm<extra></extra>",
    )


@st.cache_data(show_spinner=False)
def informe_pdf(paciente, umbral, mm_pixel, mm_corte):
    """Informe en PDF de un paciente de test: medidas, corte con más tumor y vista 3D."""
    imagenes, reales, numeros, _, _ = datos_paciente(paciente)
    probs = probabilidades_paciente(paciente)
    pred = probs > umbral
    vol_pred = volumen_cm3(pred, mm_pixel, mm_corte)
    vol_real = volumen_cm3(reales, mm_pixel, mm_corte)
    suma = pred.sum() + reales.sum()
    dice = 2 * (pred & reales).sum() / suma if suma else 1.0
    areas = pred.sum(axis=(1, 2))
    k = int(np.argmax(areas)) if areas.max() > 0 else int(np.argmax(reales.sum(axis=(1, 2))))
    may, per, e_may, e_per = diametros(pred[k], mm_pixel)

    azul, gris = "#14284B", "#5F6B7A"
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.95, "Brain Tumour Segmentation on MRI", fontsize=18, weight="bold", color=azul)
        fig.text(0.07, 0.925, "Automatic measurement report · Group 08 · University of Deusto",
                 fontsize=10, color=gris)
        fig.add_artist(plt.Line2D([0.07, 0.93], [0.915, 0.915], color="#0B62AE", lw=2))
        filas = [
            ("Patient (test set)", paciente[5:12]),
            ("Date of report", date.today().strftime("%d/%m/%Y")),
            ("Decision threshold", f"{umbral:.2f}"),
            ("Scale used", f"{mm_pixel:.2f} mm/pixel, {mm_corte:.1f} mm between slices"),
            ("Predicted tumour volume", f"{vol_pred:.1f} cm³"),
            ("Estimated mass", f"{masa_g(vol_pred):.1f} g (density {DENSIDAD_TEJIDO} g/cm³)"),
            ("Radiologist volume", f"{vol_real:.1f} cm³"),
            ("Agreement with radiologist (Dice)", f"{dice:.2f}"),
            (f"Largest slice (no. {int(numeros[k])})",
             f"{may:.1f} × {per:.1f} mm (product {may * per / 100:.2f} cm²)" if may else "no tumour marked"),
        ]
        y = 0.88
        for etiqueta, valor in filas:
            fig.text(0.07, y, etiqueta, fontsize=10.5, color=gris)
            fig.text(0.47, y, valor, fontsize=10.5, color=azul, weight="bold")
            y -= 0.027

        ax1 = fig.add_axes([0.06, 0.30, 0.42, 0.30])
        ax1.imshow(superponer(imagenes[k][:, :, 1], pred[k], reales[k],
                              (e_may, e_per) if may else None))
        ax1.set_title(f"Slice {int(numeros[k])}: prediction (cyan), radiologist (red), diameters (yellow)",
                      fontsize=7.5, color=azul)
        ax1.axis("off")

        ax2 = fig.add_axes([0.52, 0.28, 0.44, 0.34], projection="3d")
        for vol, color, alfa in ((pred, "#00a6d6", 0.9), (reales, "#e8443a", 0.25)):
            resultado = malla(vol, mm_pixel, mm_corte)
            if resultado is not None:
                xyz, caras = resultado
                ax2.plot_trisurf(xyz[:, 0], xyz[:, 1], caras, xyz[:, 2], color=color, alpha=alfa,
                                 linewidth=0, shade=True, rasterized=True)
        ax2.set_title("3D model: predicted (blue), radiologist (red)", fontsize=7.5, color=azul)
        ax2.set_xlabel("mm", fontsize=6); ax2.set_ylabel("mm", fontsize=6); ax2.set_zlabel("mm", fontsize=6)
        ax2.tick_params(labelsize=5)
        try:
            ax2.set_box_aspect(None, zoom=0.9)
            ax2.set_aspect("equal")
        except Exception:
            pass

        fig.text(0.07, 0.20, "Notes", fontsize=11, weight="bold", color=azul)
        notas = [
            "Automatic segmentation by a U-Net trained on lower-grade gliomas (LGG Segmentation Dataset).",
            "On 16 test patients the mean Dice per patient was 0.83 ± 0.12; it is less accurate on small tumours.",
            "The public images do not store their physical scale: values in mm, cm³ and g are approximate.",
            "The mask covers the abnormality visible on FLAIR, which includes tumour and oedema.",
            "This is a learning project and not a diagnostic device. Results must be reviewed by a clinician.",
        ]
        for i, nota in enumerate(notas):
            fig.text(0.08, 0.175 - i * 0.022, "•  " + nota, fontsize=8.5, color="#333333")
        fig.text(0.07, 0.04, "segmentacion-tumores-mri.streamlit.app", fontsize=8, color=gris)
        pdf.savefig(fig, dpi=200)
        plt.close(fig)
    return buffer.getvalue()


# ---------- Cabecera ----------

estilo_movil = """
    .hero {padding: 1rem 1.1rem !important; border-radius: 14px !important;}
    .hero h1 {font-size: 1.45rem !important;}
    .hero .slogan {font-size: 1rem !important;}
    .summary {font-size: .9rem; padding: .8rem 1rem !important;}
    div[data-testid="stMetric"] {padding: .4rem .6rem !important;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem !important;}
    div[data-testid="stVerticalBlockBorderWrapper"] img {border-radius: 50%; border: 3px solid #0B62AE;}
""" if MOVIL else ""

st.markdown(
    f"""
    <style>
    .hero {{background: linear-gradient(120deg, #14284B 0%, #0B62AE 60%, #3A8FD6 100%);
           padding: 1.6rem 2rem; border-radius: 18px; color: #FFFFFF; margin-bottom: 1.2rem;}}
    .hero h1 {{color: #FFFFFF; margin: 0 0 .3rem 0; font-size: 2.3rem;}}
    .hero .slogan {{font-size: 1.25rem; font-style: italic; color: #CFE3F7; margin: 0;}}
    .summary {{background: #E8F0F8; border-left: 6px solid #0B62AE; border-radius: 10px;
              padding: 1rem 1.3rem; margin-bottom: .6rem;}}
    div[data-testid="stMetric"] {{background: #FFFFFF; border: 1px solid #C9D8EA;
              border-top: 4px solid #0B62AE; border-radius: 12px; padding: .6rem .9rem;}}
    section[data-testid="stSidebar"] img {{border-radius: 50%; border: 3px solid #0B62AE;}}
    {estilo_movil}
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

# En ordenador los controles van en el panel lateral; en móvil, en la propia página
panel = st.container(border=True) if MOVIL else st.sidebar

EQUIPO = [("Diego Corrales", "diego"), ("Blanca Acón", "blanca"),
          ("Telmo Careaga", "telmo"), ("Leire Badiola", "leire")]
with panel.expander("👥 Group 08 · The team", expanded=False):
    for fila in (EQUIPO[:2], EQUIPO[2:]):
        cols = st.columns(2)
        for col, (nombre, foto) in zip(cols, fila):
            ruta = Path("equipo") / f"{foto}.jpg"
            if ruta.exists():
                col.image(str(ruta), width="stretch")
            col.markdown(f"<div style='text-align:center;font-weight:600;font-size:.85rem;"
                         f"line-height:1.2;margin-bottom:.6rem'>{nombre}</div>", unsafe_allow_html=True)
    st.markdown(
        "Biomedical Engineering students  \n"
        "**Course:** Introduction to Biomedical Engineering, Final Project 2026–2027  \n"
        "**University of Deusto**  \n"
        "**Supervisors:** Begoña García-Zapirain Soto · Cyrille Kesiku Yetuyetu  \n"
        "**Data:** LGG Segmentation Dataset (110 patients, The Cancer Genome Atlas)  \n"
        "[Code on GitHub](https://github.com/diegocorralesobregon-png/segmentacion-tumores-mri)"
    )
if not MOVIL:
    st.sidebar.divider()

# ---------- Controles ----------

opciones = ["Upload an image"]
if pacientes_de_test():
    opciones.insert(0, "Test patient")
origen = panel.radio("Image", opciones, horizontal=MOVIL)

imagen, real, avisos, paciente, numero, indice = None, None, [], None, None, None
if origen == "Test patient":
    paciente = panel.selectbox("Patient", pacientes_de_test(), format_func=lambda p: p[5:12])
    imagenes, mascaras, numeros_cortes, _, cabeza = datos_paciente(paciente)
    con_mas_tumor = int(numeros_cortes[mascaras.sum(axis=(1, 2)).argmax()])
    numero = panel.slider(
        "Slice", int(numeros_cortes.min()), int(numeros_cortes.max()), con_mas_tumor
    )
    indice = int(np.where(numeros_cortes == numero)[0][0])
    imagen, real = imagenes[indice], mascaras[indice]
    panel.caption("Patients the model never saw during training.")
else:
    archivo = panel.file_uploader("MRI slice", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if archivo is not None:
        imagen, avisos = leer_imagen(archivo)

ajustes = panel.expander("Threshold and scale") if MOVIL else panel
umbral = ajustes.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
ajustes.caption(
    "Pixels with a probability above this value are marked as tumour. "
    "The model was evaluated at 0.50."
)
ajustes.markdown("**Scale**")
mm_pixel = ajustes.number_input("Pixel size (mm)", 0.1, 3.0, MM_PIXEL_DEFECTO, 0.01)
mm_corte = ajustes.number_input("Slice spacing (mm)", 0.5, 10.0, MM_CORTE_DEFECTO, 0.5)
ajustes.caption(
    "The dataset files do not store the real scale. The default values are typical of a clinical "
    "FLAIR scan, not measured on these images, so values in mm are approximate."
)

for aviso in avisos:
    st.warning(aviso)

pestana_3d, pestana_corte, pestana_resultados = st.tabs(["3D model", "Slice", "Results"])

# ---------- Pestaña: modelo 3D ----------

with pestana_3d:
    if paciente is None:
        st.info("The 3D model needs every slice of a patient. "
                "Choose \"Test patient\" in the controls.")
    else:
        with st.spinner("Building the 3D model..."):
            probs = probabilidades_paciente(paciente)
        reales = mascaras
        pred_vol = probs > umbral

        vol_pred = volumen_cm3(pred_vol, mm_pixel, mm_corte)
        vol_real = volumen_cm3(reales, mm_pixel, mm_corte)
        suma = pred_vol.sum() + reales.sum()
        dice_3d = 2 * (pred_vol & reales).sum() / suma if suma else 1.0

        m1, m2, m3, m4 = (st.columns(2) + st.columns(2)) if MOVIL else st.columns(4)
        m1.metric("Predicted volume", f"{vol_pred:.1f} cm³")
        m2.metric("Estimated mass", f"{masa_g(vol_pred):.1f} g")
        m3.metric("Radiologist volume", f"{vol_real:.1f} cm³")
        m4.metric("Patient Dice", f"{dice_3d:.2f}")

        trazas = []
        malla_pred = superficie(pred_vol, mm_pixel, mm_corte, "Predicted tumour", "#00c8ff", 0.85)
        malla_real = superficie(reales, mm_pixel, mm_corte, "Radiologist tumour", "#ff3b3b", 0.35)
        malla_cabeza = (superficie(cabeza, mm_pixel, mm_corte, "Head (outline)", "#B8C4D2", 0.12,
                                   paso=2, hover=False) if cabeza is not None else None)
        trazas += [t for t in (malla_cabeza, malla_pred, malla_real) if t is not None]

        # Ejes de medida en el corte donde la predicción es mayor
        areas = pred_vol.sum(axis=(1, 2))
        if areas.max() > 0:
            k = int(np.argmax(areas))
            may, per, e_may, e_per = diametros(pred_vol[k], mm_pixel)
            z = k * mm_corte
            trazas.append(ejes_3d(e_may, z, mm_pixel, "longest", may))
            if e_per is not None:
                trazas.append(ejes_3d(e_per, z, mm_pixel, "perpendicular", per))
            texto_ejes = (f"The yellow axes are on slice {int(numeros_cortes[k])}, where the prediction "
                          f"is largest: {may:.1f} × {per:.1f} mm.")
        else:
            texto_ejes = "With this threshold the network marks no tumour on any slice."

        if not trazas:
            st.info("There is no tumour to draw with this threshold.")
        else:
            figura = go.Figure(trazas)
            figura.update_layout(
                height=430 if MOVIL else 620, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(itemclick="toggle", itemdoubleclick="toggleothers",
                            orientation="h", y=1.02, x=0),
                scene=dict(
                    aspectmode="data",
                    xaxis_title="mm", yaxis_title="mm", zaxis_title="mm (bottom to top)",
                ),
            )
            st.plotly_chart(figura, width="stretch")

        b1, b2 = st.columns(2)
        if malla_pred is not None:
            b1.download_button(
                "Download the predicted tumour for 3D printing (STL)", a_stl(malla_pred),
                file_name=f"tumour_{paciente[5:12]}.stl", mime="model/stl", width="stretch",
            )
        with st.spinner("Preparing the PDF report..."):
            pdf = informe_pdf(paciente, umbral, mm_pixel, mm_corte)
        b2.download_button(
            "Download the patient report (PDF)", pdf,
            file_name=f"report_{paciente[5:12]}.pdf", mime="application/pdf", width="stretch",
        )

        st.caption(
            f"Click a legend item to show or hide it; double-click to show it alone. "
            f"Drag to rotate and use the mouse wheel to zoom. {texto_ejes} "
            f"The grey head outline is the skin surface detected on the MRI, shown only for orientation. "
            f"The mass is the volume multiplied by {DENSIDAD_TEJIDO} g/cm³, an approximate density "
            f"of brain tissue. This patient has {len(numeros_cortes)} slices: with the default slice spacing "
            f"({MM_CORTE_DEFECTO} mm) the volume may be far off if the original scan had thinner slices."
        )

# ---------- Pestaña: corte ----------

with pestana_corte:
    if imagen is None:
        st.info("Upload an MRI slice in the controls.")
    else:
        if paciente is not None:
            probabilidad = probabilidades_paciente(paciente)[indice]
        else:
            probabilidad = predecir_cortes([imagen])[0]
        prediccion = probabilidad > umbral
        flair = imagen[:, :, 1]
        mayor, perpendicular, ext_mayor, ext_perp = diametros(prediccion, mm_pixel)

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

# ---------- Pestaña: resultados ----------

def figura_resultados(nombre, pie):
    ruta = CARPETA_RESULTADOS / nombre
    if ruta.exists():
        st.image(str(ruta), caption=pie, width="stretch")


with pestana_resultados:
    st.subheader("How well does it work?")
    st.markdown(
        "The 110 patients of the public LGG Segmentation Dataset were split **by patient** into "
        "training (78), validation (16) and test (16). All design decisions were taken on the "
        "validation set, and the 16 test patients were evaluated **only once**, at the end. "
        "Metrics are computed per patient, adding up the pixels of all their slices."
    )
    r1, r2, r3, r4 = (st.columns(2) + st.columns(2)) if MOVIL else st.columns(4)
    r1.metric("Dice per patient", "0.83", "± 0.12 · median 0.87", delta_color="off", delta_arrow="off")
    r2.metric("Sensitivity", "0.84", "tumour pixels found", delta_color="off", delta_arrow="off")
    r3.metric("Precision", "0.85", "marked pixels that are tumour", delta_color="off", delta_arrow="off")
    r4.metric("Tumour slices missed", "2.5%", "of 204 slices with tumour", delta_color="off", delta_arrow="off")

    st.markdown("#### Comparison with a classical method")
    st.markdown(
        "To check whether deep learning was needed, the U-Net was compared with a method that does not "
        "learn: chained **Otsu thresholds** on the FLAIR image. The U-Net is better in **15 of the 16** "
        "patients (Wilcoxon signed-rank test, p < 0.001)."
    )
    st.table({
        "Method": ["Otsu thresholding", "U-Net (ours)", "Buda et al., 2019 (same dataset)"],
        "Dice": ["0.435 ± 0.186", "0.828 ± 0.124", "0.82"],
        "IoU": ["0.294", "0.721", "–"],
        "Sensitivity": ["0.528", "0.843", "–"],
        "Precision": ["0.443", "0.848", "–"],
        "Healthy slices marked": ["63.0%", "11.1%", "–"],
    })
    st.caption("Buda et al. evaluated all 110 patients with a different protocol, so their value is not "
               "directly comparable.")
    c1, c2 = st.columns(1) * 2 if MOVIL else st.columns(2)
    with c1:
        figura_resultados("dice_patients.png", "Dice of each test patient for the U-Net and the Otsu baseline.")
    with c2:
        figura_resultados("volume.png", "Predicted vs. radiologist tumour volume (Pearson r = 0.83; "
                                         "median absolute difference 12.8%).")

    st.markdown("#### Where does it fail?")
    st.table({
        "Tumour area in the slice": ["Small (< 500 px)", "Medium (500–2,000 px)", "Large (> 2,000 px)"],
        "Slices": ["34", "87", "83"],
        "Mean Dice": ["0.559", "0.839", "0.813"],
        "Missed": ["11.8%", "1.1%", "0%"],
    })
    figura_resultados("examples.png", "(a) U-Net success; (b) Otsu on the same slice; (c) a large, "
                                      "heterogeneous tumour only partly found; (d) a false positive in the orbit.")
    st.markdown(
        "- **Small tumours** are the hardest: Dice falls to 0.56 and 12% of those slices are missed.\n"
        "- In the **two worst patients** the network finds only the bright part of a very large, "
        "heterogeneous lesion; lowering the threshold does not fix it.\n"
        "- **False positives** appear in 41 of 368 healthy slices, mostly in the orbits, the skull base "
        "and the top of the head.\n"
        "- **Limitations:** one dataset and tumour type, 16 test patients, 2D slices without 3D context "
        "and an assumed physical scale. The tool has not yet been tested with clinicians."
    )
    with st.expander("How the system works"):
        figura_resultados("architecture.png", "Architecture of the tool.")
