"""Malla 2D (cuadriláteros) de la sección transversal de una barra.

Las coordenadas son ``(u, v)`` según los ejes locales ``e1`` y ``e2`` de la
barra (ver ``core.geometria.ejes_seccion``) y los cuadriláteros se recorren en
sentido antihorario visto desde el eje ``t`` de la barra, para que al barrerlos
a lo largo del eje los hexaedros tengan volumen positivo.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from ..core.geometria import EjesSeccion, contorno_perfil_i, es_perfil_compuesto
from ..core.modelo import PerfilSeccion

_REDONDEO = 9


def _lineas(a: float, b: float, tamano: float) -> np.ndarray:
    n = max(1, math.ceil((b - a) / tamano - 1e-9))
    return np.linspace(a, b, n + 1)


class _Malla2D:
    """Acumula cuadriláteros uniendo los nodos que coinciden."""

    def __init__(self) -> None:
        self.indice: Dict[Tuple[float, float], int] = {}
        self.puntos: List[Tuple[float, float]] = []
        self.quads: List[Tuple[int, int, int, int]] = []

    def nodo(self, u: float, v: float) -> int:
        clave = (round(u, _REDONDEO) + 0.0, round(v, _REDONDEO) + 0.0)
        if clave not in self.indice:
            self.indice[clave] = len(self.puntos)
            self.puntos.append((u, v))
        return self.indice[clave]

    def cuadricula(self, lineas_u: np.ndarray, lineas_v: np.ndarray) -> None:
        for i in range(len(lineas_u) - 1):
            for j in range(len(lineas_v) - 1):
                u0, u1, v0, v1 = lineas_u[i], lineas_u[i + 1], lineas_v[j], lineas_v[j + 1]
                self.quads.append((self.nodo(u0, v0), self.nodo(u1, v0), self.nodo(u1, v1), self.nodo(u0, v1)))

    def resultado(self, intercambiar: bool) -> Tuple[np.ndarray, np.ndarray]:
        puntos = np.array(self.puntos, dtype=float)
        quads = np.array(self.quads, dtype=np.int64)
        if intercambiar:
            puntos = puntos[:, ::-1].copy()
        # orientación antihoraria en (u, v)
        a = puntos[quads]
        area = 0.5 * ((a[:, 0, 0] * a[:, 1, 1] - a[:, 1, 0] * a[:, 0, 1]) + (a[:, 1, 0] * a[:, 2, 1] - a[:, 2, 0] * a[:, 1, 1])
                      + (a[:, 2, 0] * a[:, 3, 1] - a[:, 3, 0] * a[:, 2, 1]) + (a[:, 3, 0] * a[:, 0, 1] - a[:, 0, 0] * a[:, 3, 1]))
        quads[area < 0] = quads[area < 0][:, ::-1]
        return puntos, quads


def malla_seccion(perfil: PerfilSeccion, ejes: EjesSeccion, tamano: float) -> Tuple[np.ndarray, np.ndarray]:
    """(puntos (n, 2), cuadriláteros (m, 4)) de la sección de una barra.

    Perfil en I con su forma real (alas y alma conformes); en cualquier otro
    caso, el rectángulo ``dim1`` × ``dim2`` con el que se analiza la barra.
    """

    malla = _Malla2D()
    if es_perfil_compuesto(perfil):
        h, b, tf, tw = perfil.altura, perfil.ancho, perfil.espesor_ala, perfil.espesor_alma
        H, B, w, y1 = h / 2.0, b / 2.0, tw / 2.0, h / 2.0 - tf
        # coordenadas (profundidad d, ancho a); la zona central de las alas coincide con el alma
        a_alas = np.unique(np.round(np.concatenate([_lineas(-B, -w, tamano), [-w, w], _lineas(w, B, tamano)]), _REDONDEO))
        d_alma = _lineas(-y1, y1, tamano)
        malla.cuadricula(_lineas(y1, H, tamano / 2.0 if tf > tamano else tamano), a_alas)
        malla.cuadricula(_lineas(-H, -y1, tamano / 2.0 if tf > tamano else tamano), a_alas)
        malla.cuadricula(d_alma, np.array([-w, w]))
        return malla.resultado(intercambiar=not ejes.profundidad_en_e1)

    malla.cuadricula(_lineas(-ejes.dim1 / 2, ejes.dim1 / 2, tamano), _lineas(-ejes.dim2 / 2, ejes.dim2 / 2, tamano))
    return malla.resultado(intercambiar=False)


def contorno_local(perfil: PerfilSeccion, ejes: EjesSeccion) -> List[Tuple[float, float]]:
    """Contorno de la sección en coordenadas (u, v) (para recortes en uniones)."""

    if es_perfil_compuesto(perfil):
        contorno = contorno_perfil_i(perfil)
        return [(p, q) for p, q in contorno] if ejes.profundidad_en_e1 else [(q, p) for p, q in contorno]
    a, b = ejes.dim1 / 2.0, ejes.dim2 / 2.0
    return [(a, b), (-a, b), (-a, -b), (a, -b)]


def pesos_nodales(puntos: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Área tributaria de cada nodo (un cuarto de cada cuadrilátero que lo usa)."""

    p = puntos[quads]
    area = 0.5 * np.abs(
        (p[:, 0, 0] * p[:, 1, 1] - p[:, 1, 0] * p[:, 0, 1]) + (p[:, 1, 0] * p[:, 2, 1] - p[:, 2, 0] * p[:, 1, 1])
        + (p[:, 2, 0] * p[:, 3, 1] - p[:, 3, 0] * p[:, 2, 1]) + (p[:, 3, 0] * p[:, 0, 1] - p[:, 0, 0] * p[:, 3, 1])
    )
    pesos = np.zeros(len(puntos))
    for k in range(4):
        np.add.at(pesos, quads[:, k], area / 4.0)
    return pesos
