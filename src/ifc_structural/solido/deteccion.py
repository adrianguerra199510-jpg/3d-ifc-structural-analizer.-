"""Detección automática de los elementos que se tocan (o casi) para conectarlos con constraints.

Se trabaja con la **geometría real** de cada elemento (su superficie triangulada: la del IFC en los sólidos libres, las
caras del barrido en vigas, columnas, losas y muros), no con una malla: se reparten puntos sobre la superficie de cada
pieza y se mide su distancia con signo a la superficie de la otra (negativa dentro). Las partes de la superficie a menos
de la separación máxima, o metidas dentro de la otra pieza (un perno en su agujero, una tuerca solapada con la placa),
son el contacto; su área y la separación mínima describen cada pareja.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.modelo import Proyecto

#: Puntos de muestra por elemento (como mucho; el espaciado no baja de 2 mm ni pasa de 5 cm).
MUESTRAS_POR_ELEMENTO = 4000


@dataclass
class Contacto:
    a_tipo: str
    a_id: str
    b_tipo: str
    b_id: str
    area: float  # m² de superficie en contacto
    separacion: float  # m: 0 si se tocan o se solapan

    def descripcion(self) -> str:
        sep = "se tocan" if self.separacion < 5e-4 else f"separados {self.separacion * 1000:.1f} mm"
        return f"{self.a_tipo.title()} {self.a_id} ↔ {self.b_tipo.title()} {self.b_id}  ({self.area * 1e4:.0f} cm², {sep})"


@dataclass
class Superficie:
    """Superficie triangulada de un elemento: vértices, triángulos y cara (superficie del elemento) de cada triángulo."""

    vertices: np.ndarray
    triangulos: np.ndarray
    cara: np.ndarray

    @property
    def area(self) -> float:
        p = self.vertices[self.triangulos]
        return float(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())

    @property
    def caja(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def superficie_de(proyecto: Proyecto, tipo: str, id_: str) -> Superficie:
    """Superficie real del elemento (vigas, columnas, losas y muros: las caras planas de su sólido, trianguladas)."""

    if tipo == "SOLIDO":
        s = proyecto.solidos[id_]
        return Superficie(np.asarray(s.vertices, float), np.asarray(s.triangulos, np.int64), np.asarray(s.cara_de_triangulo))
    import pyvista as pv

    from ..core.geometria import solido_elemento

    sol = solido_elemento(proyecto, tipo, id_)
    vertices, triangulos, caras = [], [], []
    base = 0
    for i, c in enumerate(sol.caras):
        puntos = np.asarray(c.puntos, dtype=float)
        poligono = pv.PolyData(puntos, np.r_[len(puntos), np.arange(len(puntos))]).triangulate()  # también no convexas
        tri = np.asarray(poligono.faces).reshape(-1, 4)[:, 1:]
        vertices.append(np.asarray(poligono.points, dtype=float))
        triangulos.append(tri + base)
        caras.append(np.full(len(tri), i))
        base += len(poligono.points)
    return Superficie(np.vstack(vertices), np.vstack(triangulos).astype(np.int64), np.concatenate(caras))


def _polidata(sup: Superficie):
    import pyvista as pv

    return pv.PolyData(sup.vertices, np.hstack([np.full((len(sup.triangulos), 1), 3), sup.triangulos]).ravel())


def muestras(sup: Superficie, maximo: int = MUESTRAS_POR_ELEMENTO):
    """(puntos, área que representa cada uno, triángulo de cada uno) repartidos sobre la superficie."""

    from ..core.solidos import muestras_de_superficie

    espaciado = min(0.05, max(0.002, math.sqrt(max(sup.area, 1e-12) / maximo)))
    puntos, triangulo = muestras_de_superficie(sup.vertices, sup.triangulos, espaciado)
    p = sup.vertices[sup.triangulos]
    area_tri = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    cuenta = np.bincount(triangulo, minlength=len(sup.triangulos))
    return np.asarray(puntos, float), area_tri[triangulo] / np.maximum(cuenta[triangulo], 1), np.asarray(triangulo)


def distancia_con_signo(puntos: np.ndarray, sup: Superficie, malla=None) -> np.ndarray:
    """Distancia de cada punto a la superficie (negativa si queda dentro del sólido, con su superficie cerrada)."""

    import pyvista as pv

    if len(puntos) == 0:
        return np.zeros(0)
    malla = _polidata(sup) if malla is None else malla
    nube = pv.PolyData(np.asarray(puntos, float))
    return np.asarray(nube.compute_implicit_distance(malla)["implicit_distance"], dtype=float)


def elementos_del_proyecto(proyecto: Proyecto) -> List[Tuple[str, str]]:
    elementos = [("VIGA", i) for i in proyecto.vigas] + [("COLUMNA", i) for i in proyecto.columnas]
    elementos += [("LOSA", i) for i in proyecto.losas] + [("MURO", i) for i in proyecto.muros]
    return elementos + [("SOLIDO", i) for i in proyecto.solidos]


def contacto_entre(a: Superficie, b: Superficie, tolerancia: float, datos_a=None, datos_b=None, malla_a=None,
                   malla_b=None) -> Optional[Tuple[float, float]]:
    """(área de contacto, separación) entre dos superficies, o ``None`` si no hay contacto: los puntos de cada una a
    menos de ``tolerancia`` de la otra o dentro de ella. El área es la menor de las dos (la de la pieza que apoya)."""

    resultado = []
    for (puntos, peso, _), otra, malla in ((datos_a or muestras(a), b, malla_b), (datos_b or muestras(b), a, malla_a)):
        minimo, maximo = otra.caja
        cerca = np.all((puntos >= minimo - tolerancia) & (puntos <= maximo + tolerancia), axis=1)
        if not cerca.any():
            return None
        d = distancia_con_signo(puntos[cerca], otra, malla)
        en_contacto = d <= tolerancia
        if not en_contacto.any():
            return None
        resultado.append((float(peso[cerca][en_contacto].sum()), float(max(0.0, d[en_contacto].min()))))
    return min(resultado[0][0], resultado[1][0]), min(resultado[0][1], resultado[1][1])


def detectar_contactos(proyecto: Proyecto, tolerancia: float, area_minima: float = 5e-5) -> List[Contacto]:
    """Parejas de elementos que se tocan, se solapan o están a menos de ``tolerancia`` (m), con al menos ``area_minima``
    m² de superficie en contacto (menos es un contacto de arista o de punta, no una unión)."""

    from ..core.validacion import exigir_geometria_valida

    exigir_geometria_valida(proyecto)
    tolerancia = max(float(tolerancia), 5e-4)
    elementos = elementos_del_proyecto(proyecto)
    superficies: Dict[Tuple[str, str], Superficie] = {}
    for e in elementos:
        try:
            superficies[e] = superficie_de(proyecto, *e)
        except (KeyError, ValueError, IndexError):  # elemento sin geometría
            continue
    claves = list(superficies)
    cajas = {e: superficies[e].caja for e in claves}
    datos: Dict[Tuple[str, str], tuple] = {}
    mallas: Dict[Tuple[str, str], object] = {}
    resultado: List[Contacto] = []
    for i, ea in enumerate(claves):
        for eb in claves[i + 1:]:
            (amin, amax), (bmin, bmax) = cajas[ea], cajas[eb]
            if np.any(amin - tolerancia > bmax) or np.any(bmin - tolerancia > amax):
                continue
            for e in (ea, eb):
                if e not in datos:
                    datos[e] = muestras(superficies[e])
                    mallas[e] = _polidata(superficies[e])
            contacto = contacto_entre(superficies[ea], superficies[eb], tolerancia, datos[ea], datos[eb], mallas[ea], mallas[eb])
            if contacto is None or contacto[0] < area_minima:
                continue
            resultado.append(Contacto(ea[0], ea[1], eb[0], eb[1], contacto[0], contacto[1]))
    return sorted(resultado, key=lambda c: -c.area)
