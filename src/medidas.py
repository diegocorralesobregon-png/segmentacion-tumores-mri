import numpy as np
from scipy import ndimage

DENSIDAD_TEJIDO = 1.04  # g/cm³, valor aproximado del tejido cerebral


def bordes(mascara):
    """Coordenadas (fila, columna) de los píxeles del borde de la región."""
    return np.argwhere(mascara & ~ndimage.binary_erosion(mascara))


def diametros(mascara, mm_pixel):
    """Diámetro mayor y diámetro perpendicular máximo de una máscara 2D, en mm.

    Devuelve (mayor_mm, perpendicular_mm, extremos_mayor, extremos_perpendicular),
    donde cada par de extremos son dos puntos (fila, columna) para dibujar la línea.
    """
    vacio = (0.0, 0.0, None, None)
    if mascara.sum() < 2:
        return vacio

    # Diámetro mayor: la mayor distancia entre dos puntos del borde
    puntos = bordes(mascara).astype(float)
    if len(puntos) > 600:
        puntos = puntos[np.linspace(0, len(puntos) - 1, 600).astype(int)]
    distancias = np.linalg.norm(puntos[:, None, :] - puntos[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(distancias), distancias.shape)
    p1, p2 = puntos[i], puntos[j]
    mayor = distancias[i, j]
    if mayor == 0:
        return vacio

    # Perpendicular: se giran todos los píxeles para que el diámetro mayor quede horizontal
    # y se busca la columna con más extensión en vertical
    eje = (p2 - p1) / mayor
    normal = np.array([-eje[1], eje[0]])
    pixeles = np.argwhere(mascara).astype(float)
    a_lo_largo = (pixeles - p1) @ eje
    a_lo_ancho = (pixeles - p1) @ normal
    columnas = np.round(a_lo_largo).astype(int)
    mejor, extremos_perp = 0.0, None
    for c in np.unique(columnas):
        valores = a_lo_ancho[columnas == c]
        ancho = valores.max() - valores.min() + 1
        if ancho > mejor:
            mejor = ancho
            q1 = p1 + c * eje + valores.min() * normal
            q2 = p1 + c * eje + valores.max() * normal
            extremos_perp = (q1, q2)

    return (mayor + 1) * mm_pixel, mejor * mm_pixel, (p1, p2), extremos_perp


def volumen_cm3(volumen_mascara, mm_pixel, mm_corte):
    """Volumen de una máscara 3D (cortes, filas, columnas) en cm³."""
    return volumen_mascara.sum() * mm_pixel * mm_pixel * mm_corte / 1000


def masa_g(volumen_cm3_valor):
    return volumen_cm3_valor * DENSIDAD_TEJIDO
