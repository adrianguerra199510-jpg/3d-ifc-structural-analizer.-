"""Apoyos y cargas sobre una superficie cualquiera, aplicados a la malla de hexaedros.

Una superficie del modelo (cara de un sólido libre, de una barra, de una losa o de un muro)
se traduce en las caras de hexaedro que la forman: en un sólido libre ya están asociadas al
mallar; en barras, losas y muros se buscan las caras de borde que están sobre el plano de
la cara y dentro de su contorno. Las presiones y las fuerzas se reparten por área en los
nodos; un momento se reparte con fuerzas proporcionales a la distancia al centro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..core.asignacion_superficie import direccion_efectiva
from ..core.cargas import ApoyoSuperficie, CargaSuperficie
from ..core.geometria import solido_elemento
from ..core.modelo import Proyecto
from .malla import CARAS_HEXAEDRO, MallaSolida, ParteMalla
from .tetraedros import CARAS_TETRAEDRO, MEDIOS_CARA


@dataclass
class SuperficieMallada:
    parte: int  # índice en ``malla.partes``
    nodos: np.ndarray  # (n, 4) índices de nodo (hexaedros) o (n, 6) con los nodos medios (tetraedros)
    centro: np.ndarray  # (n, 3)
    normal: np.ndarray  # (n, 3) unitaria, hacia fuera
    area: np.ndarray  # (n,)
    hexaedro: np.ndarray = None  # (n,) índice global (0-based, numeración de CalculiX) del elemento de cada cara
    cara: np.ndarray = None  # (n,) número de cara de CalculiX (1..6 hexaedro, 1..4 tetraedro)
    pesos_nodo: np.ndarray = None  # (k,) fracción del área de cada cara que lleva cada nodo (None = 1/k)
    nodos_solo: np.ndarray = None  # láminas: nodos que están sobre la superficie (en un canto, solo los del borde)

    @property
    def area_total(self) -> float:
        return float(self.area.sum())

    @property
    def nodos_unicos(self) -> np.ndarray:
        if self.nodos_solo is not None and len(self.nodos_solo):
            return np.unique(self.nodos_solo)
        return np.unique(self.nodos)


def _geometria_tetra(malla: MallaSolida, parte: ParteMalla, local: np.ndarray, cara: np.ndarray) -> SuperficieMallada:
    tetra = malla.tetraedros[parte.primer_tetraedro + local]
    nodos = np.array([np.concatenate([tetra[k, list(CARAS_TETRAEDRO[int(c) - 1])], tetra[k, list(MEDIOS_CARA[int(c) - 1])]])
                      for k, c in enumerate(cara)]).reshape(-1, 6)
    p = malla.coordenadas[nodos[:, :3]]
    centro = p.mean(axis=1)
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    normal[np.einsum("ij,ij->i", normal, centro - malla.coordenadas[tetra[:, :4]].mean(axis=1)) < 0] *= -1.0
    return SuperficieMallada(parte=malla.partes.index(parte), nodos=nodos, centro=centro, normal=normal, area=area,
                             hexaedro=len(malla.hexaedros) + parte.primer_tetraedro + np.asarray(local), cara=np.asarray(cara),
                             pesos_nodo=np.array([0, 0, 0, 1 / 3, 1 / 3, 1 / 3]))


def _geometria_lamina(malla: MallaSolida, parte: ParteMalla, local: np.ndarray, lado: np.ndarray) -> SuperficieMallada:
    """Elementos de lámina cargados por un lado: ``lado`` 1 = el de la normal de la lámina, 2 = el opuesto."""

    k = parte.primer_lamina + np.asarray(local)
    nodos = malla.laminas[k]
    p = malla.coordenadas[nodos[:, :3]]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    normal = malla.normal_de_lamina[k] * np.where(np.asarray(lado) == 1, 1.0, -1.0)[:, None]
    espesor = malla.espesor_de_lamina[k]
    return SuperficieMallada(parte=malla.partes.index(parte), nodos=nodos, centro=p.mean(axis=1) + 0.5 * espesor[:, None] * normal,
                             normal=normal, area=area,
                             hexaedro=len(malla.hexaedros) + len(malla.tetraedros) + k, cara=np.asarray(lado),
                             pesos_nodo=np.array([0, 0, 0, 1 / 3, 1 / 3, 1 / 3]))


def _geometria(malla: MallaSolida, parte: ParteMalla, hexaedro_local: np.ndarray, cara: np.ndarray) -> SuperficieMallada:
    if parte.lamina:
        return _geometria_lamina(malla, parte, hexaedro_local, cara)
    if parte.tetra:
        return _geometria_tetra(malla, parte, hexaedro_local, cara)
    hexa = malla.hexaedros[parte.primer_hexaedro + hexaedro_local]
    nodos = np.stack([hexa[k, list(CARAS_HEXAEDRO[int(c) - 1])] for k, c in enumerate(cara)]) if len(cara) else np.zeros((0, 4), int)
    p = malla.coordenadas[nodos]
    centro = p.mean(axis=1)
    normal = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    centro_hex = malla.coordenadas[hexa].mean(axis=1)
    normal[np.einsum("ij,ij->i", normal, centro - centro_hex) < 0] *= -1.0
    return SuperficieMallada(parte=malla.partes.index(parte), nodos=nodos, centro=centro, normal=normal, area=area,
                             hexaedro=parte.primer_hexaedro + np.asarray(hexaedro_local), cara=np.asarray(cara))


def _parte_de(malla: MallaSolida, tipo: str, id_: str) -> Optional[ParteMalla]:
    for parte in malla.partes:
        if tipo == "SOLIDO" and parte.tipo == "SOLIDO" and parte.id == id_:
            return parte
        if tipo in ("VIGA", "COLUMNA") and parte.miembro is not None and id_ in parte.miembro.barras:
            return parte
        if tipo in ("LOSA", "MURO") and parte.tipo == tipo and parte.id == id_:
            return parte
    return None


def superficie_en_malla(proyecto: Proyecto, malla: MallaSolida, tipo: str, id_: str, cara: int, tamano: float) -> Optional[SuperficieMallada]:
    """Caras de hexaedro que forman la superficie ``cara`` del elemento, o ``None`` si no se encuentra."""

    parte = _parte_de(malla, tipo, id_)
    if parte is None:
        return None
    if tipo == "SOLIDO":
        pares = parte.superficies.get(cara)
        if pares is None or len(pares) == 0:
            return None
        sup = _geometria(malla, parte, pares[:, 0], pares[:, 1])
        if parte.lamina:
            sup.nodos_solo = parte.nodos_superficie.get(cara)
        return sup

    try:
        cara_geom = solido_elemento(proyecto, tipo, id_).caras[cara]
    except (KeyError, IndexError):
        return None
    f = malla.fronteras(parte)
    if len(f.area) == 0:
        return None
    tol = 2e-3 + 0.05 * tamano
    if cara_geom.tipo.startswith("EXTREMO") and parte.miembro is not None and parte.anillos:
        c0 = malla.coordenadas[parte.anillos[0]].mean(axis=0)
        c1 = malla.coordenadas[parte.anillos[-1]].mean(axis=0)
        inicio = np.linalg.norm(c0 - cara_geom.centro) <= np.linalg.norm(c1 - cara_geom.centro)
        referencia, normal = (c0, c0 - c1) if inicio else (c1, c1 - c0)
        normal = normal / np.linalg.norm(normal)
        marca = (np.abs((f.centro - referencia) @ normal) <= tol) & (f.normal @ normal > 0.9)
    else:
        normal = np.asarray(cara_geom.normal, dtype=float)
        puntos = np.asarray(cara_geom.puntos, dtype=float)
        centro = puntos.mean(axis=0)
        marca = (np.abs((f.centro - centro) @ normal) <= tol) & (f.normal @ normal > 0.9)
        if marca.any():
            from shapely import contains_xy
            from shapely.geometry import Polygon

            u = puntos[1] - puntos[0]
            u = u - np.dot(u, normal) * normal
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            poligono = Polygon(np.column_stack([(puntos - centro) @ u, (puntos - centro) @ v])).buffer(tol)
            rel = f.centro[marca] - centro
            dentro = contains_xy(poligono, rel @ u, rel @ v)
            indices = np.where(marca)[0]
            marca[:] = False
            marca[indices[dentro]] = True
    if not marca.any():
        return None
    return SuperficieMallada(parte=malla.partes.index(parte), nodos=f.nodos[marca], centro=f.centro[marca],
                             normal=f.normal[marca], area=f.area[marca], hexaedro=f.hexaedro[marca], cara=f.cara[marca],
                             pesos_nodo=f.pesos_nodo)


# ---------------------------------------------------------------------------
# Cargas
# ---------------------------------------------------------------------------


def fuerzas_nodales(malla: MallaSolida, sup: SuperficieMallada, carga: CargaSuperficie):
    """(nodos, fuerzas (n, 3) en kN) equivalentes a la carga sobre la superficie."""

    area_total = sup.area_total
    k = sup.nodos.shape[1]
    reparto = np.full(k, 1.0 / k) if sup.pesos_nodo is None else np.asarray(sup.pesos_nodo, dtype=float)  # fracción del área por nodo
    if carga.tipo == "MOMENTO":
        nodos = sup.nodos.ravel()
        pesos = (sup.area[:, None] * reparto[None, :]).ravel()
        unicos, inverso = np.unique(nodos, return_inverse=True)
        w = np.bincount(inverso, weights=pesos)
        w /= w.sum()
        x = malla.coordenadas[unicos]
        r = x - (w[:, None] * x).sum(axis=0)
        K = np.zeros((3, 3))
        for wi, ri in zip(w, r):
            K += wi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
        momento = carga.valor * np.asarray(carga.direccion, dtype=float)
        m_eq = np.linalg.lstsq(K, momento, rcond=None)[0]
        return unicos, w[:, None] * np.cross(m_eq[None, :], r)
    if carga.tipo == "PRESION":
        if carga.direccion is not None:
            direccion = np.tile(np.asarray(carga.direccion, dtype=float), (len(sup.area), 1))
        else:
            direccion = -sup.normal
        por_cara = (carga.valor * sup.area)[:, None] * direccion
    else:  # FUERZA: fuerza total repartida por área
        total = carga.valor * np.asarray(carga.direccion, dtype=float)
        por_cara = (sup.area / area_total)[:, None] * total[None, :]
    nodos = sup.nodos.ravel()
    fuerzas = (por_cara[:, None, :] * reparto[None, :, None]).reshape(-1, 3)
    unicos, inverso = np.unique(nodos, return_inverse=True)
    acumuladas = np.zeros((len(unicos), 3))
    np.add.at(acumuladas, inverso, fuerzas)
    return unicos, acumuladas
