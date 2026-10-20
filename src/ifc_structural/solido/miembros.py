"""Miembros sólidos del modelo: barras unidas y recortadas en las uniones.

En el análisis de barras todo se une por los ejes, de modo que una viga que
llega a una columna empieza en el eje de la columna y los dos sólidos se
solaparían. Para el modelo sólido:

* las barras consecutivas, alineadas y con la misma sección y material (la
  columna partida por una unión) se funden en un solo miembro;
* el extremo de una barra que llega a otra se recorta hasta la superficie de
  la que atraviesa (o de la de mayor sección, si ninguna atraviesa el nudo), de
  modo que su cara extrema quede apoyada sobre ella.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

from ..core.geometria import ejes_seccion
from ..core.modelo import ElementoBarra, Proyecto
from .seccion2d import contorno_local

_TOL = 1e-6


@dataclass
class MiembroBarra:
    id: str
    barras: List[str]  # barras del modelo que lo forman, en orden a lo largo del eje
    perfil: str
    material: str
    p0: np.ndarray  # extremos del eje sin recortar
    p1: np.ndarray
    t: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    ejes_ref: object  # EjesSeccion de la barra de referencia (dimensiones y orientación)
    tramos: List[Tuple[str, float, float]] = field(default_factory=list)  # (barra, s_ini, s_fin) medidos desde p0
    recorte0: float = 0.0
    recorte1: float = 0.0

    @property
    def longitud(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def s_util(self) -> Tuple[float, float]:
        return self.recorte0, self.longitud - self.recorte1

    def punto(self, s: float) -> np.ndarray:
        return self.p0 + s * self.t

    def coordenada(self, punto: np.ndarray) -> float:
        return float(np.dot(np.asarray(punto, dtype=float) - self.p0, self.t))

    def distancia_al_eje(self, punto: np.ndarray) -> float:
        d = np.asarray(punto, dtype=float) - self.p0
        return float(np.linalg.norm(d - np.dot(d, self.t) * self.t))

    def barra_en(self, s: float) -> str:
        for barra, s0, s1 in self.tramos:
            if s0 - 1e-9 <= s <= s1 + 1e-9:
                return barra
        return min(self.tramos, key=lambda tr: min(abs(s - tr[1]), abs(s - tr[2])))[0]


def _unitario(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _direccion(proyecto: Proyecto, barra: ElementoBarra) -> np.ndarray:
    a = np.array(proyecto.nodos[barra.nodo_inicio].coordenadas, dtype=float)
    b = np.array(proyecto.nodos[barra.nodo_fin].coordenadas, dtype=float)
    return _unitario(b - a)


def _continuas(proyecto: Proyecto, a: ElementoBarra, b: ElementoBarra, nodo: str) -> bool:
    """¿Es ``b`` la continuación de ``a`` por el nodo (alineadas, mismo perfil, material y orientación)?"""

    if a.perfil != b.perfil or a.material != b.material:
        return False
    da = _direccion(proyecto, a) * (1.0 if a.nodo_fin == nodo else -1.0)  # sale de a hacia el nodo
    db = _direccion(proyecto, b) * (1.0 if b.nodo_inicio == nodo else -1.0)  # sale del nodo por b
    if np.linalg.norm(da - db) > 1e-6:
        return False
    ea, eb = ejes_seccion(proyecto, a), ejes_seccion(proyecto, b)
    return abs(abs(float(np.dot(ea.e1, eb.e1))) - 1.0) < 1e-6 and ea.profundidad_en_e1 == eb.profundidad_en_e1


def barras_de_longitud_cero(proyecto: Proyecto) -> List[str]:
    """Barras con los dos extremos en el mismo punto (no se pueden mallar ni analizar)."""

    resultado = []
    for id_, b in proyecto.todas_las_barras().items():
        a, c = np.array(proyecto.nodos[b.nodo_inicio].coordenadas), np.array(proyecto.nodos[b.nodo_fin].coordenadas)
        if not np.isfinite(a).all() or not np.isfinite(c).all() or float(np.linalg.norm(c - a)) < 1e-6:
            resultado.append(id_)
    return resultado


def construir_miembros(proyecto: Proyecto) -> List[MiembroBarra]:
    """Miembros del modelo con las barras continuas fundidas (sin recortar). Las barras de longitud cero se omiten."""

    degeneradas = set(barras_de_longitud_cero(proyecto))
    barras = {i: b for i, b in proyecto.todas_las_barras().items() if i not in degeneradas}
    por_nodo: Dict[str, List[str]] = {}
    for id_, b in barras.items():
        for n in (b.nodo_inicio, b.nodo_fin):
            por_nodo.setdefault(n, []).append(id_)

    padre = {i: i for i in barras}

    def buscar(x: str) -> str:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for nodo, ids in por_nodo.items():
        # De las parejas de barras que se continúan por el nudo solo se funde la de mayor
        # sección (la columna que atraviesa la unión); las demás se recortan contra ella.
        parejas = [
            (a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
            if _continuas(proyecto, barras[a], barras[b], nodo) or _continuas(proyecto, barras[b], barras[a], nodo)
        ]
        if not parejas:
            continue
        a, b = max(parejas, key=lambda ab: (proyecto.perfiles[barras[ab[0]].perfil].area,
                                            barras[ab[0]].tipo.value == "COLUMNA"))
        padre[buscar(a)] = buscar(b)

    grupos: Dict[str, List[str]] = {}
    for id_ in barras:
        grupos.setdefault(buscar(id_), []).append(id_)

    miembros: List[MiembroBarra] = []
    for ids in grupos.values():
        ref = barras[ids[0]]
        ejes = ejes_seccion(proyecto, ref)
        t = ejes.t
        puntos = {}
        for i in ids:
            for n in (barras[i].nodo_inicio, barras[i].nodo_fin):
                puntos[n] = np.array(proyecto.nodos[n].coordenadas, dtype=float)
        origen = puntos[barras[ids[0]].nodo_inicio]
        coord = {n: float(np.dot(p - origen, t)) for n, p in puntos.items()}
        n_ini, n_fin = min(coord, key=coord.get), max(coord, key=coord.get)
        p0, p1 = puntos[n_ini], puntos[n_fin]
        tramos = []
        for i in sorted(ids, key=lambda i: min(coord[barras[i].nodo_inicio], coord[barras[i].nodo_fin])):
            c0, c1 = coord[barras[i].nodo_inicio], coord[barras[i].nodo_fin]
            tramos.append((i, min(c0, c1) - coord[n_ini], max(c0, c1) - coord[n_ini]))
        miembros.append(MiembroBarra(
            id=ids[0] if len(ids) == 1 else "+".join(sorted(ids)), barras=[tr[0] for tr in tramos],
            perfil=ref.perfil, material=ref.material, p0=p0, p1=p1, t=t, e1=ejes.e1, e2=ejes.e2,
            ejes_ref=ejes, tramos=tramos,
        ))
    return miembros


def _salida_del_contorno(miembro: MiembroBarra, proyecto: Proyecto, direccion: np.ndarray) -> Optional[float]:
    """Distancia (medida a lo largo de ``direccion``) desde el eje del miembro hasta
    su superficie, o None si la dirección es paralela al eje."""

    perp = direccion - np.dot(direccion, miembro.t) * miembro.t
    norma = float(np.linalg.norm(perp))
    if norma < 1e-6:
        return None
    u, v = float(np.dot(perp, miembro.e1) / norma), float(np.dot(perp, miembro.e2) / norma)
    perfil = proyecto.perfiles[miembro.perfil]
    poligono = Polygon(contorno_local(perfil, miembro.ejes_ref))
    rayo = LineString([(0.0, 0.0), (u * 1e3, v * 1e3)])
    interseccion = poligono.intersection(rayo)
    if interseccion.is_empty:
        return None
    partes = list(interseccion.geoms) if hasattr(interseccion, "geoms") else [interseccion]
    primera = min(partes, key=lambda g: g.distance(__import__("shapely.geometry", fromlist=["Point"]).Point(0.0, 0.0)))
    distancia_plano = max(np.hypot(x, y) for x, y in primera.coords)
    return float(distancia_plano / norma)


def _area(proyecto: Proyecto, miembro: MiembroBarra) -> float:
    return proyecto.perfiles[miembro.perfil].area


def recortar_uniones(proyecto: Proyecto, miembros: List[MiembroBarra]) -> List[str]:
    """Recorta los extremos de los miembros que llegan a otro. Devuelve avisos."""

    avisos: List[str] = []
    nodos = {n: np.array(c.coordenadas, dtype=float) for n, c in proyecto.nodos.items()}
    for nombre, punto in nodos.items():
        interiores, extremos = [], []
        for m in miembros:
            if m.distancia_al_eje(punto) > 1e-5:
                continue
            s = m.coordenada(punto)
            if _TOL < s < m.longitud - _TOL:
                interiores.append(m)
            elif abs(s) <= _TOL:
                extremos.append((m, 0))
            elif abs(s - m.longitud) <= _TOL:
                extremos.append((m, 1))
        if not extremos:
            continue
        if interiores:
            dominante = max(interiores, key=lambda m: _area(proyecto, m))
        elif len(extremos) >= 2:
            dominante = max((m for m, _ in extremos), key=lambda m: _area(proyecto, m))
        else:
            continue
        for m, extremo in extremos:
            if m is dominante:
                continue
            direccion = m.t if extremo == 0 else -m.t  # hacia el interior del miembro
            salida = _salida_del_contorno(dominante, proyecto, direccion)
            if salida is None or salida < 1e-6:
                continue
            if salida > 0.5 * m.longitud:
                avisos.append(f"{m.id}: no se recortó en el nudo {nombre} (la unión es mayor que la mitad de la barra).")
                continue
            if extremo == 0:
                m.recorte0 = max(m.recorte0, salida)
            else:
                m.recorte1 = max(m.recorte1, salida)
    return avisos
