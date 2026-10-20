"""Malla de hexaedros de todo el modelo y sus utilidades (caras, anillos).

Orden de nodos de cada hexaedro (CalculiX ``C3D8``): 1-4 la cara inferior
recorrida en sentido antihorario visto desde la superior, 5-8 la cara superior.
Cara ``S1`` = 1-2-3-4, ``S2`` = 5-8-7-6, ``S3`` = 1-5-6-2, ``S4`` = 2-6-7-3,
``S5`` = 3-7-8-4, ``S6`` = 4-8-5-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.geometria import normal_losa
from ..core.modelo import Proyecto
from .miembros import MiembroBarra
from .libres import estimar_hexaedros_solido, mallar_solido
from .seccion2d import malla_seccion, pesos_nodales

#: Caras de un hexaedro (índices 0-based de sus 8 nodos), en el orden S1..S6 de CalculiX.
CARAS_HEXAEDRO = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))


class ErrorMalla(RuntimeError):
    pass


@dataclass
class CarasFrontera:
    hexaedro: np.ndarray  # (n,) índice global del hexaedro
    cara: np.ndarray  # (n,) número de cara 1..6
    nodos: np.ndarray  # (n, 4) índices globales de nodo
    centro: np.ndarray  # (n, 3)
    normal: np.ndarray  # (n, 3) unitaria, hacia fuera
    area: np.ndarray  # (n,)


@dataclass
class ParteMalla:
    """Un miembro (barra, losa o muro) ya mallado."""

    id: str
    tipo: str  # "BARRA", "LOSA", "MURO" o "SOLIDO" (sólido de forma libre)
    material: str
    primer_hexaedro: int
    ultimo_hexaedro: int  # exclusivo
    miembro: Optional[MiembroBarra] = None
    estaciones: np.ndarray = field(default_factory=lambda: np.zeros(0))
    anillos: List[np.ndarray] = field(default_factory=list)
    pesos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    superficies: Dict[int, np.ndarray] = field(default_factory=dict)  # solo sólidos libres: superficie -> (k, 2) (hexaedro local, cara 1..6)
    _fronteras: Optional[CarasFrontera] = None

    @property
    def hexaedros(self) -> range:
        return range(self.primer_hexaedro, self.ultimo_hexaedro)


@dataclass
class MallaSolida:
    coordenadas: np.ndarray  # (N, 3)
    hexaedros: np.ndarray  # (M, 8) 0-based
    parte_de_hexaedro: np.ndarray  # (M,) índice en ``partes``
    elemento_de_hexaedro: List[str]  # id de la barra, losa o muro del modelo al que pertenece cada hexaedro
    partes: List[ParteMalla]
    avisos: List[str] = field(default_factory=list)

    def fronteras(self, parte: ParteMalla) -> CarasFrontera:
        if parte._fronteras is None:
            parte._fronteras = _caras_frontera(self, parte)
        return parte._fronteras


# ---------------------------------------------------------------------------
# Barras
# ---------------------------------------------------------------------------


def _estaciones(inicio: float, fin: float, especiales: Sequence[float], tamano: float) -> np.ndarray:
    puntos = {round(inicio, 6), round(fin, 6)}
    puntos.update(round(s, 6) for s in especiales if inicio + 1e-6 < s < fin - 1e-6)
    ordenados = sorted(puntos)
    estaciones = [ordenados[0]]
    for a, b in zip(ordenados, ordenados[1:]):
        n = max(1, math.ceil((b - a) / tamano - 1e-9))
        estaciones.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(estaciones)


def _malla_barra(proyecto: Proyecto, miembro: MiembroBarra, tamano: float, especiales: Sequence[float],
                 nodos: List[np.ndarray], n_nodos: int, n_hex: int):
    perfil = proyecto.perfiles[miembro.perfil]
    puntos2d, quads = malla_seccion(perfil, miembro.ejes_ref, tamano)
    inicio, fin = miembro.s_util
    estaciones = _estaciones(inicio, fin, [miembro.coordenada(p) for p in []] + list(especiales)
                             + [tr[1] for tr in miembro.tramos[1:]], tamano)
    plantilla = puntos2d[:, 0:1] * miembro.e1 + puntos2d[:, 1:2] * miembro.e2  # (n2, 3)
    n2 = len(puntos2d)
    anillos, bloques = [], []
    for k, s in enumerate(estaciones):
        bloques.append(miembro.punto(s) + plantilla)
        anillos.append(np.arange(n_nodos + k * n2, n_nodos + (k + 1) * n2))
    nodos.append(np.vstack(bloques))
    abajo = np.arange(len(estaciones) - 1)[:, None, None] * n2 + n_nodos
    hexaedros = np.concatenate([quads[None] + abajo, quads[None] + abajo + n2], axis=2).reshape(-1, 8)
    elementos = []
    for k in range(len(estaciones) - 1):
        elementos.extend([miembro.barra_en(0.5 * (estaciones[k] + estaciones[k + 1]))] * len(quads))
    return estaciones, anillos, pesos_nodales(puntos2d, quads), hexaedros, elementos, len(estaciones) * n2


# ---------------------------------------------------------------------------
# Losas y muros
# ---------------------------------------------------------------------------


def _quads_poligono(contorno2d: np.ndarray, tamano: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cuadriláteros de un polígono convexo en el plano (coordenadas 2D antihorarias)."""

    n = len(contorno2d)
    a = contorno2d[1] - contorno2d[0]
    b = contorno2d[-1] - contorno2d[0]
    es_rectangulo = n == 4 and abs(np.dot(a, b)) < 1e-6 * np.linalg.norm(a) * np.linalg.norm(b) and np.allclose(
        contorno2d[2], contorno2d[0] + a + b, atol=1e-6)
    puntos: Dict[Tuple[float, float], int] = {}
    lista: List[Tuple[float, float]] = []
    quads: List[Tuple[int, int, int, int]] = []

    def nodo(p) -> int:
        clave = (round(float(p[0]), 8) + 0.0, round(float(p[1]), 8) + 0.0)
        if clave not in puntos:
            puntos[clave] = len(lista)
            lista.append(clave)
        return puntos[clave]

    def cuadricula(origen, du, dv, nu, nv) -> None:
        for i in range(nu):
            for j in range(nv):
                p = [origen + du * i / nu + dv * j / nv, origen + du * (i + 1) / nu + dv * j / nv,
                     origen + du * (i + 1) / nu + dv * (j + 1) / nv, origen + du * i / nu + dv * (j + 1) / nv]
                quads.append(tuple(nodo(q) for q in p))

    if es_rectangulo:
        cuadricula(contorno2d[0], a, b, max(1, math.ceil(np.linalg.norm(a) / tamano - 1e-9)),
                   max(1, math.ceil(np.linalg.norm(b) / tamano - 1e-9)))
    else:
        centro = contorno2d.mean(axis=0)
        medios = [(contorno2d[i] + contorno2d[(i + 1) % n]) / 2.0 for i in range(n)]
        mitad_max = max(np.linalg.norm(contorno2d[(i + 1) % n] - contorno2d[i]) for i in range(n)) / 2.0
        k = max(1, math.ceil(mitad_max / tamano - 1e-9))
        for i in range(n):
            vertice, m_ant, m_sig = contorno2d[i], medios[i - 1], medios[i]
            # cuadrilátero [m_ant, vértice, m_sig, centro] como cuadrícula k×k (interpolación bilineal)
            for p in range(k):
                for q in range(k):
                    def f(s, t):
                        return ((1 - s) * (1 - t) * m_ant + s * (1 - t) * vertice + s * t * m_sig + (1 - s) * t * centro)

                    quads.append(tuple(nodo(f(*st)) for st in (
                        (p / k, q / k), ((p + 1) / k, q / k), ((p + 1) / k, (q + 1) / k), (p / k, (q + 1) / k))))
    puntos2d = np.array(lista)
    quads_a = np.array(quads, dtype=np.int64)
    c = puntos2d[quads_a]
    area = 0.5 * ((c[:, 0, 0] * c[:, 1, 1] - c[:, 1, 0] * c[:, 0, 1]) + (c[:, 1, 0] * c[:, 2, 1] - c[:, 2, 0] * c[:, 1, 1])
                  + (c[:, 2, 0] * c[:, 3, 1] - c[:, 3, 0] * c[:, 2, 1]) + (c[:, 3, 0] * c[:, 0, 1] - c[:, 0, 0] * c[:, 3, 1]))
    quads_a[area < 0] = quads_a[area < 0][:, ::-1]
    return puntos2d, quads_a


def _malla_plano(proyecto: Proyecto, elemento, tamano: float, nodos: List[np.ndarray], n_nodos: int):
    contorno = np.array([proyecto.nodos[n].coordenadas for n in elemento.nodos_contorno], dtype=float)
    normal = normal_losa(contorno)
    a = contorno[1] - contorno[0]
    a = a - np.dot(a, normal) * normal
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    centro = contorno.mean(axis=0)
    plano = np.column_stack([(contorno - centro) @ a, (contorno - centro) @ b])
    # sentido antihorario
    area = 0.5 * sum(plano[i, 0] * plano[(i + 1) % len(plano), 1] - plano[(i + 1) % len(plano), 0] * plano[i, 1]
                     for i in range(len(plano)))
    if area < 0:
        plano = plano[::-1].copy()
    puntos2d, quads = _quads_poligono(plano, tamano)
    capas = max(2, math.ceil(elemento.espesor / tamano - 1e-9)) if elemento.espesor > 0.08 else 1
    base = centro + puntos2d[:, 0:1] * a + puntos2d[:, 1:2] * b
    n2 = len(puntos2d)
    bloques = [base + normal * (-elemento.espesor / 2.0 + elemento.espesor * k / capas) for k in range(capas + 1)]
    nodos.append(np.vstack(bloques))
    abajo = np.arange(capas)[:, None, None] * n2 + n_nodos
    hexaedros = np.concatenate([quads[None] + abajo, quads[None] + abajo + n2], axis=2).reshape(-1, 8)
    return hexaedros, len(bloques) * n2


# ---------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------


def construir_malla_solida(proyecto: Proyecto, miembros: Sequence[MiembroBarra], tamano: float,
                           especiales: Optional[Dict[str, Sequence[float]]] = None) -> MallaSolida:
    """Malla de hexaedros de todos los miembros y sólidos libres; ``especiales`` da, por miembro,
    coordenadas axiales donde debe haber una estación (cargas y apoyos)."""

    especiales = especiales or {}
    bloques_nodos: List[np.ndarray] = []
    lista_hex: List[np.ndarray] = []
    elementos: List[str] = []
    partes: List[ParteMalla] = []
    n_nodos = n_hex = 0

    for miembro in miembros:
        estaciones, anillos, pesos, hexa, elems, n_nuevos = _malla_barra(
            proyecto, miembro, tamano, especiales.get(miembro.id, ()), bloques_nodos, n_nodos, n_hex)
        partes.append(ParteMalla(
            id=miembro.id, tipo="BARRA", material=miembro.material, primer_hexaedro=n_hex,
            ultimo_hexaedro=n_hex + len(hexa), miembro=miembro, estaciones=estaciones, anillos=anillos, pesos=pesos))
        lista_hex.append(hexa)
        elementos.extend(elems)
        n_nodos += n_nuevos
        n_hex += len(hexa)

    for tipo, coleccion in (("LOSA", proyecto.losas), ("MURO", proyecto.muros)):
        for id_, elemento in coleccion.items():
            hexa, n_nuevos = _malla_plano(proyecto, elemento, tamano, bloques_nodos, n_nodos)
            partes.append(ParteMalla(id=id_, tipo=tipo, material=elemento.material, primer_hexaedro=n_hex,
                                     ultimo_hexaedro=n_hex + len(hexa)))
            lista_hex.append(hexa)
            elementos.extend([id_] * len(hexa))
            n_nodos += n_nuevos
            n_hex += len(hexa)

    avisos: List[str] = []
    # Sólidos de forma libre: el de mayor volumen conserva el volumen común con los menores y las barras
    # (pernos, placas y soldaduras suelen solapar con los perfiles que unen).
    ordenados = sorted(proyecto.solidos.values(), key=lambda x: -x.volumen)
    recortados = 0
    for k, solido in enumerate(ordenados):
        if not solido.cerrado:
            raise ErrorMalla(f"El sólido {solido.id} ({solido.nombre}) no es una superficie cerrada y no se puede mallar en 3D.")
        try:
            libre = mallar_solido(solido, tamano, ordenados[:k], lambda p: _dentro_de_barras(proyecto, miembros, p))
        except ValueError as exc:  # coordenadas no válidas
            raise ErrorMalla(str(exc)) from exc
        if libre is None:
            avisos.append(f"El sólido {solido.id} ({solido.nombre}) queda dentro de otras piezas y no se analiza.")
            continue
        recortados += bool(libre.descartados)
        bloques_nodos.append(libre.coordenadas)
        lista_hex.append(libre.hexaedros + n_nodos)
        partes.append(ParteMalla(id=solido.id, tipo="SOLIDO", material=solido.material, primer_hexaedro=n_hex,
                                 ultimo_hexaedro=n_hex + len(libre.hexaedros), superficies=libre.superficies))
        elementos.extend([solido.id] * len(libre.hexaedros))
        n_nodos += len(libre.coordenadas)
        n_hex += len(libre.hexaedros)

    if recortados:
        avisos.append(f"{recortados} sólido(s) se recortaron donde solapan con otras piezas (se conserva el volumen de la pieza "
                      "mayor o de la viga o columna, para no contarlo dos veces).")
    if not partes:
        raise ErrorMalla("No hay elementos que mallar.")
    parte_de_hex = np.concatenate([np.full(p.ultimo_hexaedro - p.primer_hexaedro, i) for i, p in enumerate(partes)])
    n_nodos_barras = sum(len(b) for b in bloques_nodos[: len(miembros)])
    malla = MallaSolida(
        coordenadas=np.vstack(bloques_nodos), hexaedros=np.vstack(lista_hex).astype(np.int64),
        parte_de_hexaedro=parte_de_hex, elemento_de_hexaedro=elementos, partes=partes, avisos=avisos,
    )
    _recortar_planos(proyecto, malla, miembros, n_nodos_barras)
    return malla


def _dentro_de_barras(proyecto: Proyecto, miembros: Sequence[MiembroBarra], puntos: np.ndarray) -> np.ndarray:
    """Máscara de los ``puntos`` que caen dentro del volumen de alguna viga o columna."""

    from shapely import contains_xy
    from shapely.geometry import Polygon

    from .seccion2d import contorno_local

    dentro = np.zeros(len(puntos), dtype=bool)
    for m in miembros:
        rel = puntos - m.p0
        s, u, v = rel @ m.t, rel @ m.e1, rel @ m.e2
        poligono = Polygon(contorno_local(proyecto.perfiles[m.perfil], m.ejes_ref))
        dentro |= (s > m.recorte0 + 1e-9) & (s < m.longitud - m.recorte1 - 1e-9) & contains_xy(poligono, u, v)
    return dentro


def _recortar_planos(proyecto: Proyecto, malla: MallaSolida, miembros: Sequence[MiembroBarra], n_nodos_barras: int) -> None:
    """Quita de las losas y los muros los hexaedros que quedan dentro de una viga o columna
    (el IFC suele colocar la losa solapada con las vigas): así no se cuenta dos veces el volumen
    y la losa queda apoyada contra los lados de las vigas."""

    indices_planos = [i for i, p in enumerate(malla.partes) if p.tipo in ("LOSA", "MURO")]
    if not indices_planos:
        return
    centros = malla.coordenadas[malla.hexaedros].mean(axis=1)
    mantener = np.ones(len(malla.hexaedros), dtype=bool)
    primero = malla.partes[indices_planos[0]].primer_hexaedro
    ultimo = malla.partes[indices_planos[-1]].ultimo_hexaedro
    candidatos = np.arange(primero, ultimo)
    mantener[candidatos[_dentro_de_barras(proyecto, miembros, centros[candidatos])]] = False
    if mantener.all():
        return
    usados = np.zeros(len(malla.coordenadas), dtype=bool)
    usados[:n_nodos_barras] = True
    usados[malla.hexaedros[mantener].ravel()] = True
    nuevo = np.cumsum(usados) - 1
    malla.coordenadas = malla.coordenadas[usados]
    malla.hexaedros = nuevo[malla.hexaedros[mantener]]
    malla.parte_de_hexaedro = malla.parte_de_hexaedro[mantener]
    malla.elemento_de_hexaedro = [e for e, k in zip(malla.elemento_de_hexaedro, mantener) if k]
    inicio = 0
    for i, parte in enumerate(malla.partes):
        cuenta = int((malla.parte_de_hexaedro == i).sum())
        parte.primer_hexaedro, parte.ultimo_hexaedro, parte._fronteras = inicio, inicio + cuenta, None
        inicio += cuenta
    if any(p.ultimo_hexaedro == p.primer_hexaedro for p in malla.partes):
        vacias = [p.id for p in malla.partes if p.ultimo_hexaedro == p.primer_hexaedro]
        raise ErrorMalla(f"Las piezas {', '.join(vacias)} quedan totalmente dentro de otras: revise la geometría.")


def estimar_hexaedros(proyecto: Proyecto, miembros: Sequence[MiembroBarra], tamano: float) -> int:
    """Número aproximado de hexaedros con ese tamaño de elemento (sin mallar)."""

    total = 0.0
    for m in miembros:
        perfil = proyecto.perfiles[m.perfil]
        largo = max(m.longitud - m.recorte0 - m.recorte1, tamano)
        _, quads = malla_seccion(perfil, m.ejes_ref, tamano)
        total += len(quads) * (math.ceil(largo / tamano) + len(m.tramos))
    for coleccion in (proyecto.losas, proyecto.muros):
        for e in coleccion.values():
            c = np.array([proyecto.nodos[n].coordenadas for n in e.nodos_contorno], dtype=float)
            area = 0.5 * np.linalg.norm(sum(np.cross(c[i], c[(i + 1) % len(c)]) for i in range(len(c))))
            capas = max(2, math.ceil(e.espesor / tamano)) if e.espesor > 0.08 else 1
            total += area / tamano**2 * capas
    total += sum(estimar_hexaedros_solido(x, tamano) for x in proyecto.solidos.values())
    return int(total)


# ---------------------------------------------------------------------------
# Caras frontera y anillos
# ---------------------------------------------------------------------------


def _caras_frontera(malla: MallaSolida, parte: ParteMalla) -> CarasFrontera:
    hexa = malla.hexaedros[parte.primer_hexaedro: parte.ultimo_hexaedro]
    n = len(hexa)
    todas = np.stack([hexa[:, list(c)] for c in CARAS_HEXAEDRO], axis=1)  # (n, 6, 4)
    ordenadas = np.sort(todas.reshape(-1, 4), axis=1)
    _, primero, cuenta = np.unique(ordenadas, axis=0, return_index=True, return_counts=True)
    indices = primero[cuenta == 1]
    h, c = np.divmod(indices, 6)
    nodos = todas[h, c]
    p = malla.coordenadas[nodos]
    centro = p.mean(axis=1)
    normal = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    centro_hex = malla.coordenadas[hexa[h]].mean(axis=1)
    hacia_dentro = np.einsum("ij,ij->i", normal, centro - centro_hex) < 0
    normal[hacia_dentro] *= -1.0
    return CarasFrontera(hexaedro=h + parte.primer_hexaedro, cara=c + 1, nodos=nodos, centro=centro, normal=normal, area=area)
