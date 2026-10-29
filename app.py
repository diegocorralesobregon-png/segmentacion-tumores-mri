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

st.set_page_config(page_title="Segmentación de tumores cerebrales", layout="wide")


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
        avisos.append("La imagen tiene un solo canal y el modelo se entrenó con tres secuencias "
                      "(precontraste, FLAIR y poscontraste). Se ha repetido el canal, así que el "
                      "resultado es menos fiable.")
    matriz = matriz[:, :, :3]
    if matriz.shape[:2] != (256, 256):
        avisos.append(f"La imagen medía {matriz.shape[1]}×{matriz.shape[0]} y se ha reescalado a 256×256.")
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
        name="Ejes de medida", legendgroup="ejes", showlegend=nombre == "mayor",
        hovertemplate=f"Diámetro {nombre}: {longitud:.1f} mm<extra></extra>",
    )


# ---------- Cabecera ----------

st.title("Segmentación de tumores cerebrales en resonancia magnética")
st.caption(
    "U-Net entrenada con gliomas de bajo grado (dataset LGG de TCGA). "
    "Proyecto de aprendizaje: no es una herramienta de diagnóstico."
)

if not Path(RUTA_MODELO).exists():
    st.error(f"No encuentro el modelo en {RUTA_MODELO}.")
    st.stop()
modelo = obtener_modelo()

# ---------- Panel lateral ----------

opciones = ["Subir una imagen"]
if pacientes_de_test():
    opciones.insert(0, "Paciente de test")
origen = st.sidebar.radio("Imagen", opciones)

imagen, real, avisos, paciente, numero = None, None, [], None, None
if origen == "Paciente de test":
    paciente = st.sidebar.selectbox("Paciente", pacientes_de_test(), format_func=lambda p: p[5:12])
    imagenes, mascaras, numeros_cortes = datos_paciente(paciente)
    con_mas_tumor = int(numeros_cortes[mascaras.sum(axis=(1, 2)).argmax()])
    numero = st.sidebar.slider(
        "Corte", int(numeros_cortes.min()), int(numeros_cortes.max()), con_mas_tumor
    )
    indice = int(np.where(numeros_cortes == numero)[0][0])
    imagen, real = imagenes[indice], mascaras[indice]
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

st.sidebar.markdown("**Escala**")
mm_pixel = st.sidebar.number_input("Tamaño de píxel (mm)", 0.1, 3.0, MM_PIXEL_DEFECTO, 0.01)
mm_corte = st.sidebar.number_input("Distancia entre cortes (mm)", 0.5, 10.0, MM_CORTE_DEFECTO, 0.5)
st.sidebar.caption(
    "Los archivos del dataset no guardan la escala real. Los valores por defecto son típicos "
    "de un FLAIR clínico, no medidos en estas imágenes: las cifras en mm son aproximadas."
)

if imagen is None:
    st.info("Sube un corte de resonancia desde el panel de la izquierda.")
    st.stop()

for aviso in avisos:
    st.warning(aviso)

# ---------- Predicción del corte ----------

probabilidad = predecir(modelo, imagen)
prediccion = probabilidad > umbral
flair = imagen[:, :, 1]
mayor, perpendicular, ext_mayor, ext_perp = diametros(prediccion, mm_pixel)

pestana_corte, pestana_3d = st.tabs(["Corte", "Modelo 3D"])

# ---------- Pestaña: corte ----------

with pestana_corte:
    area = int(prediccion.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Área marcada", f"{area * mm_pixel ** 2 / 100:.2f} cm²", f"{area} px",
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

    d1, d2, d3 = st.columns(3)
    d1.metric("Diámetro mayor", f"{mayor:.1f} mm" if mayor else "—")
    d2.metric("Diámetro perpendicular", f"{perpendicular:.1f} mm" if perpendicular else "—")
    d3.metric("Producto de diámetros", f"{mayor * perpendicular / 100:.2f} cm²" if mayor else "—")

    mostrar_ejes = st.checkbox("Mostrar los diámetros sobre la imagen", value=True)
    ejes = (ext_mayor, ext_perp) if mostrar_ejes else None

    i1, i2, i3 = st.columns(3)
    i1.image(a_gris(flair), caption="FLAIR", width="stretch")
    leyenda = "Predicción (cian)" + (", radiólogo (rojo)" if real is not None else "")
    if mostrar_ejes and mayor:
        leyenda += " y diámetros (amarillo)"
    i2.image(superponer(flair, prediccion, real, ejes), caption=leyenda, width="stretch")
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
            "- Solo se ha entrenado con gliomas de bajo grado y cortes 2D de 256×256.\n"
            "- La máscara marca la anomalía visible en FLAIR, que incluye tumor y edema."
        )

# ---------- Pestaña: modelo 3D ----------

with pestana_3d:
    if paciente is None:
        st.info("El modelo 3D necesita todos los cortes de un paciente. "
                "Elige «Paciente de test» en el panel de la izquierda.")
        st.stop()

    with st.spinner("Prediciendo todos los cortes del paciente..."):
        probs, reales, numeros = volumen_paciente(paciente, modelo)
    pred_vol = probs > umbral

    vol_pred = volumen_cm3(pred_vol, mm_pixel, mm_corte)
    vol_real = volumen_cm3(reales, mm_pixel, mm_corte)
    suma = pred_vol.sum() + reales.sum()
    dice_3d = 2 * (pred_vol & reales).sum() / suma if suma else 1.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volumen predicho", f"{vol_pred:.1f} cm³")
    m2.metric("Masa estimada", f"{masa_g(vol_pred):.1f} g")
    m3.metric("Volumen del radiólogo", f"{vol_real:.1f} cm³")
    m4.metric("Dice del paciente", f"{dice_3d:.2f}")

    trazas = []
    malla_pred = superficie(pred_vol, mm_pixel, mm_corte, "Tumor predicho", "#00c8ff", 0.85)
    malla_real = superficie(reales, mm_pixel, mm_corte, "Tumor del radiólogo", "#ff3b3b", 0.35)
    trazas += [t for t in (malla_pred, malla_real) if t is not None]

    # Ejes de medida en el corte donde la predicción es mayor
    areas = pred_vol.sum(axis=(1, 2))
    if areas.max() > 0:
        k = int(np.argmax(areas))
        may, per, e_may, e_per = diametros(pred_vol[k], mm_pixel)
        z = k * mm_corte
        trazas.append(ejes_3d(e_may, z, mm_pixel, "mayor", may))
        if e_per is not None:
            trazas.append(ejes_3d(e_per, z, mm_pixel, "perpendicular", per))
        texto_ejes = (f"Los ejes amarillos están en el corte {int(numeros[k])}, donde la predicción "
                      f"es más grande: {may:.1f} × {per:.1f} mm.")
    else:
        texto_ejes = "Con este umbral la red no marca tumor en ningún corte."

    if not trazas:
        st.info("No hay tumor que dibujar con este umbral.")
    else:
        figura = go.Figure(trazas)
        figura.update_layout(
            height=620, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers",
                        orientation="h", y=1.02, x=0),
            scene=dict(
                aspectmode="data",
                xaxis_title="mm", yaxis_title="mm", zaxis_title="mm (de abajo arriba)",
            ),
        )
        st.plotly_chart(figura, width="stretch")

    if malla_pred is not None:
        st.download_button(
            "Descargar el tumor predicho para imprimir en 3D (STL)", a_stl(malla_pred),
            file_name=f"tumor_{paciente[5:12]}.stl", mime="model/stl",
        )

    st.caption(
        f"Pulsa un elemento de la leyenda para mostrarlo u ocultarlo; doble clic para dejarlo solo. "
        f"Arrastra para girar y usa la rueda para acercar. {texto_ejes} "
        f"La masa es el volumen multiplicado por {DENSIDAD_TEJIDO} g/cm³, una densidad aproximada "
        f"del tejido cerebral. Este paciente tiene {len(numeros)} cortes: con la distancia entre "
        f"cortes por defecto ({MM_CORTE_DEFECTO} mm) el volumen puede estar muy desviado si el "
        f"estudio original tenía cortes más finos."
    )
