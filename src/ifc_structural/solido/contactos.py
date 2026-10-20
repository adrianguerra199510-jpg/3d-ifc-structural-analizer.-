"""Uniones entre los sólidos: caras coplanares enfrentadas que se tocan.

Dos piezas que se apoyan una sobre otra (la cara extrema de una viga contra
el ala de la columna, una losa sobre una viga, dos tramos de columna) se unen
con ``*TIE`` de CalculiX, que enlaza mallas no coincidentes: equivale a una
soldadura continua sobre toda la superficie de contacto. No se modelan
pernos ni contacto con separación o fricción.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .malla import CarasFrontera, MallaSolida

_COS_ENFRENTADAS = -0.99


@dataclass
class Union:
    nombre: str
    parte_esclava: int
    parte_maestra: int
    esclava: List[Tuple[int, int]]  # (hexaedro global, cara 1..6)
    maestra: List[Tuple[int, int]]
    tolerancia: float


def detectar_uniones(malla: MallaSolida, tamano: float, espesores: Optional[Dict[str, float]] = None):
    """Uniones entre las piezas de la malla. Devuelve ``(uniones, avisos)``.

    Una losa o un muro que no toca a ninguna otra pieza (típico cuando el IFC la coloca
    a la altura del eje de las vigas, o con una pequeña holgura) se une a las caras más
    cercanas que la miran, a una distancia de hasta medio espesor; se avisa de ello.
    """

    tol = max(2e-3, 0.25 * tamano)
    espesores = espesores or {}
    avisos: List[str] = []
    uniones: List[Union] = []
    usados: Set[int] = set()  # nodos ya esclavos de otra unión
    cajas = []
    tolerancias: Dict[Tuple[int, int], float] = {}
    for parte in malla.partes:
        pts = malla.coordenadas[np.unique(malla.hexaedros[parte.primer_hexaedro: parte.ultimo_hexaedro])]
        cajas.append((pts.min(axis=0) - tol, pts.max(axis=0) + tol))

    candidatas = []
    cercanas_avisadas: Set[str] = set()
    for i in range(len(malla.partes)):
        for j in range(i + 1, len(malla.partes)):
            pi, pj = malla.partes[i], malla.partes[j]
            planos = [p for p in (pi, pj) if p.tipo in ("LOSA", "MURO")]
            # Con losas o muros, el IFC las coloca a veces solapadas con las vigas o con holgura:
            # si no se tocan exactamente se buscan caras que las miran a corta distancia.
            holgura = max([tol] + [max(0.6 * tamano, 0.5 * espesores.get(p.id, 0.0)) for p in planos])
            for h in ([tol, holgura] if holgura > tol else [tol]):
                if np.all(cajas[i][0] - (h - tol) <= cajas[j][1]) and np.all(cajas[j][0] - (h - tol) <= cajas[i][1]):
                    par = _caras_en_contacto(malla, i, j, h)
                    if par is not None:
                        candidatas.append(par)
                        tolerancias[(i, j)] = h
                        if h > tol:
                            cercanas_avisadas.update(p.id for p in planos)
                        break
    if cercanas_avisadas:
        avisos.append(f"{', '.join(sorted(cercanas_avisadas))}: se unió a las piezas vecinas por las caras que se miran a corta "
                      "distancia (el IFC la coloca solapada o con holgura respecto de las vigas).")
    # primero los contactos con más caras (los principales)
    for i, j, fi, fj, area_i, area_j in sorted(candidatas, key=lambda c: -(len(c[2]) + len(c[3]))):
        # esclava: la superficie de contacto más pequeña (la que queda entera dentro de la otra)
        esclava_es_i = area_i <= area_j
        pi, pj = (i, j) if esclava_es_i else (j, i)
        f_esc, f_mae = (fi, fj) if esclava_es_i else (fj, fi)
        frontera_esc = malla.fronteras(malla.partes[pi])
        nodos_caras = frontera_esc.nodos
        mapa = {(int(h), int(c)): k for k, (h, c) in enumerate(zip(frontera_esc.hexaedro, frontera_esc.cara))}
        elegidas = []
        for h, c in f_esc:
            nodos = set(int(n) for n in nodos_caras[mapa[(h, c)]])
            if nodos & usados:
                continue
            elegidas.append((h, c))
            usados |= nodos
        if elegidas:
            uniones.append(Union(f"U{len(uniones) + 1}", pi, pj, elegidas, f_mae, 1.5 * tolerancias.get((i, j), tol)))
    return uniones, avisos


def _caras_en_contacto(malla: MallaSolida, i: int, j: int, tol: float):
    a: CarasFrontera = malla.fronteras(malla.partes[i])
    b: CarasFrontera = malla.fronteras(malla.partes[j])
    if len(a.area) == 0 or len(b.area) == 0:
        return None
    sa, sb = np.sqrt(a.area), np.sqrt(b.area)
    arbol = cKDTree(b.centro)
    radio = 0.5 * (sa.max() + sb.max()) * 1.05 + tol
    vecinas = arbol.query_ball_point(a.centro, radio)
    en_a, en_b = set(), set()
    for k, lista in enumerate(vecinas):
        for m in lista:
            if np.dot(a.normal[k], b.normal[m]) > _COS_ENFRENTADAS:
                continue
            d = b.centro[m] - a.centro[k]
            if abs(np.dot(a.normal[k], d)) > tol:
                continue
            plano = d - np.dot(a.normal[k], d) * a.normal[k]
            if np.linalg.norm(plano) > 0.5 * (sa[k] + sb[m]) * 0.99 * 1.1:  # solapan en el plano
                continue
            en_a.add(k)
            en_b.add(m)
    if not en_a:
        return None
    f_a = [(int(a.hexaedro[k]), int(a.cara[k])) for k in sorted(en_a)]
    f_b = [(int(b.hexaedro[m]), int(b.cara[m])) for m in sorted(en_b)]
    return i, j, f_a, f_b, float(a.area[sorted(en_a)].sum()), float(b.area[sorted(en_b)].sum())


def partes_sin_conexion(malla: MallaSolida, uniones: List[Union], partes_apoyadas: Set[int],
                        enlaces: Sequence[Tuple[int, int]] = ()) -> List[int]:
    """Partes que no están unidas (directa o indirectamente) a ninguna parte con apoyos."""

    padre = list(range(len(malla.partes)))

    def buscar(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for u in uniones:
        padre[buscar(u.parte_esclava)] = buscar(u.parte_maestra)
    for a, b in enlaces:  # constraints manuales entre superficies de dos partes
        padre[buscar(a)] = buscar(b)
    raices_apoyadas = {buscar(p) for p in partes_apoyadas}
    return [i for i in range(len(malla.partes)) if buscar(i) not in raices_apoyadas]
