"""Símbolos de los apoyos según los grados de libertad que restringen.

* **Empotrado** (los 6 grados): bloque macizo.
* **Articulado** (las 3 traslaciones, giros libres): pirámide.
* **Rodillo** (una o dos traslaciones): pirámide con ruedas sobre una placa,
  orientada según la traslación restringida (una por cada traslación).
* **Giros restringidos**: un anillo alrededor de cada eje de giro (Rx, Ry, Rz),
  que se suma al símbolo de las traslaciones.

Es solo geometría de PyVista (sin OpenGL), de modo que se puede probar sin ventana.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyvista as pv

COLOR_EMPOTRADO = "navy"
COLOR_ARTICULADO = "royalblue"
COLOR_RUEDA = "dimgray"
COLOR_GIRO = "darkorange"
COLORES_SIMBOLO = (COLOR_EMPOTRADO, COLOR_ARTICULADO, COLOR_RUEDA, COLOR_GIRO)

NOMBRES_GDL = ("Ux", "Uy", "Uz", "Rx", "Ry", "Rz")
_EJES = np.eye(3)


def nombre_apoyo(grados: Sequence[bool]) -> str:
    """Nombre corto: «empotrado», «articulado», «rodillo Uz» o la lista de grados."""

    g = tuple(bool(x) for x in grados)
    activos = " ".join(n for n, a in zip(NOMBRES_GDL, g) if a)
    if all(g):
        return "empotrado"
    if g[:3] == (True, True, True) and not any(g[3:]):
        return "articulado"
    if sum(g[:3]) in (1, 2) and not any(g[3:]):
        return f"rodillo {activos}"
    return activos or "libre"


def _perpendiculares(eje: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    auxiliar = _EJES[2] if abs(eje[2]) < 0.9 else _EJES[0]
    a = np.cross(eje, auxiliar)
    a /= np.linalg.norm(a)
    return a, np.cross(eje, a)


def _rodillo(centro: np.ndarray, eje: np.ndarray, largo: float) -> Dict[str, List[pv.PolyData]]:
    """Pirámide con la punta en ``centro``, hacia el lado negativo de ``eje``, con ruedas y placa."""

    a, b = _perpendiculares(eje)
    base = centro - eje * 0.8 * largo
    piramide = pv.Cone(center=centro - eje * 0.4 * largo, direction=eje, height=0.8 * largo, radius=0.42 * largo, resolution=4)
    ruedas = [
        pv.Cylinder(center=base - eje * 0.1 * largo + a * lado * 0.22 * largo, direction=b, radius=0.1 * largo,
                    height=0.5 * largo, resolution=16)
        for lado in (-1, 1)
    ]
    placa = pv.Box(bounds=(-0.5, 0.5, -0.5, 0.5, -0.04, 0.04))
    placa.scale([0.9 * largo, 0.9 * largo, largo], inplace=True)
    # la placa es plana en la dirección del eje: se lleva a la base girando Z hacia ``eje``
    giro = np.eye(4)
    giro[:3, :3] = np.column_stack([a, b, eje])
    placa.transform(giro, inplace=True)
    placa.translate(base - eje * 0.22 * largo, inplace=True)
    return {COLOR_ARTICULADO: [piramide], COLOR_RUEDA: ruedas + [placa]}


def _anillo(centro: np.ndarray, eje_giro: int, largo: float) -> pv.PolyData:
    toro = pv.ParametricTorus(ringradius=0.55 * largo, crosssectionradius=0.07 * largo)
    if eje_giro == 0:
        toro.rotate_y(90, inplace=True)
    elif eje_giro == 1:
        toro.rotate_x(90, inplace=True)
    toro.translate(centro, inplace=True)
    return toro


def simbolo_apoyo(grados: Sequence[bool], centro: Sequence[float], largo: float) -> Dict[str, List[pv.PolyData]]:
    """Mallas del símbolo agrupadas por color. ``grados`` = (Ux, Uy, Uz, Rx, Ry, Rz)."""

    g = [bool(x) for x in grados]
    centro = np.asarray(centro, dtype=float)
    partes: Dict[str, List[pv.PolyData]] = {}

    def agregar(color: str, malla) -> None:
        partes.setdefault(color, []).extend(malla if isinstance(malla, list) else [malla])

    if all(g):
        agregar(COLOR_EMPOTRADO, pv.Cube(center=centro - np.array([0, 0, 0.45 * largo]),
                                         x_length=1.1 * largo, y_length=1.1 * largo, z_length=0.9 * largo))
        return partes

    traslaciones = [i for i in range(3) if g[i]]
    if len(traslaciones) == 3:
        agregar(COLOR_ARTICULADO, pv.Cone(center=centro - np.array([0, 0, 0.4 * largo]), direction=(0, 0, 1),
                                          height=0.8 * largo, radius=0.45 * largo, resolution=4))
    else:
        for i in traslaciones:
            for color, mallas in _rodillo(centro, _EJES[i], largo).items():
                agregar(color, mallas)
    for i in range(3):
        if g[3 + i]:
            agregar(COLOR_GIRO, _anillo(centro, i, largo))
    if not partes:  # sin ningún grado restringido: una marca mínima
        agregar(COLOR_GIRO, pv.Sphere(radius=0.15 * largo, center=centro))
    return partes
