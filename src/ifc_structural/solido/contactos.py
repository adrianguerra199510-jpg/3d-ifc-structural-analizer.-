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
        pts = malla.coordenadas[malla.nodos_de_parte(parte)]
        cajas.append((pts.min(axis=0) - tol, pts.max(axis=0) + tol))

    candidatas = []
    cercanas_avisadas: Set[str] = set()
    for i in range(len(malla.partes)):
        for j in range(i + 1, len(malla.partes)):
            pi, pj = malla.partes[i], malla.partes[j]
            if pi.lamina or pj.lamina:  # las láminas se acoplan por nodos (``detectar_acoples``), no con *TIE
                continue
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
            else:
                # piezas que se solapaban (una tuerca metida en la placa): la menor cedió el volumen común y su borde allí
                # es un corte sin caras planas enfrentadas; se unen las caras que quedan a menos de un elemento
                if frozenset((pi.id, pj.id)) in malla.solapes:
                    h = max(tol, 0.6 * tamano)
                    par = _caras_cercanas(malla, i, j, h)
                    if par is not None:
                        candidatas.append(par)
                        tolerancias[(i, j)] = h
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


def _distancia_a_triangulos(puntos: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distancia de cada punto (n, 3) a su triángulo (a, b, c) (n, 3): proyección sobre el plano, recortada al triángulo."""

    ab, ac, ap = b - a, c - a, puntos - a
    d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
    d20, d21 = (ap * ab).sum(1), (ap * ac).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
    v, w = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    dentro = (v >= 0) & (w >= 0) & (v + w <= 1)
    cercano = a + v[:, None] * ab + w[:, None] * ac
    resultado = np.linalg.norm(puntos - cercano, axis=1)
    if (~dentro).any():  # fuera del triángulo: el punto más cercano está en una de sus aristas
        k = ~dentro
        mejor = np.full(int(k.sum()), np.inf)
        for p0, p1 in ((a, b), (b, c), (c, a)):
            e = p1[k] - p0[k]
            t = np.clip(((puntos[k] - p0[k]) * e).sum(1) / np.maximum((e * e).sum(1), 1e-300), 0.0, 1.0)
            mejor = np.minimum(mejor, np.linalg.norm(puntos[k] - (p0[k] + t[:, None] * e), axis=1))
        resultado[k] = mejor
    return resultado


def _distancia_a_borde(malla: MallaSolida, puntos: np.ndarray, borde: CarasFrontera, vecinas: int = 8) -> np.ndarray:
    """Distancia de cada punto a las caras de borde de una parte (las ``vecinas`` de centro más cercano)."""

    k = min(vecinas, len(borde.centro))
    _, idx = cKDTree(borde.centro).query(puntos, k=k)
    idx = np.asarray(idx).reshape(len(puntos), k)
    # caras triangulares (tetraedros: vértices 0-2) o cuadriláteros (hexaedros de 20 nodos: vértices 0-3, en 2 triángulos)
    triangulos = ((0, 1, 2), (0, 2, 3)) if borde.nodos.shape[1] == 8 else ((0, 1, 2),)
    distancia = np.full(len(puntos), np.inf)
    for columna in range(k):
        caras = borde.nodos[idx[:, columna]]
        for a, b, c in triangulos:
            distancia = np.minimum(distancia, _distancia_a_triangulos(puntos, malla.coordenadas[caras[:, a]],
                                                                      malla.coordenadas[caras[:, b]], malla.coordenadas[caras[:, c]]))
    return distancia


def _caras_cercanas(malla: MallaSolida, i: int, j: int, tol: float):
    """Como ``_caras_en_contacto`` pero sin exigir caras planas enfrentadas: las caras de borde de cada parte cuyo centro
    está a menos de ``tol`` de las caras de borde de la otra. Para superficies curvas que se tocan (un perno en su
    agujero) o cortadas por un solape; ``*TIE`` proyecta cada nodo esclavo sobre la cara más cercana. Solo para caras
    triangulares (tetraedros) o cuadriláteros de 8 nodos (hexaedros barridos); con hexaedros lineales no se usa."""

    a: CarasFrontera = malla.fronteras(malla.partes[i])
    b: CarasFrontera = malla.fronteras(malla.partes[j])
    if len(a.area) == 0 or len(b.area) == 0 or a.nodos.shape[1] not in (6, 8) or b.nodos.shape[1] not in (6, 8):
        return None
    en_a = _distancia_a_borde(malla, a.centro, b) <= tol
    en_b = _distancia_a_borde(malla, b.centro, a) <= tol
    if en_a.sum() < 3 or en_b.sum() < 3:
        return None
    f_a = [(int(h), int(c)) for h, c in zip(a.hexaedro[en_a], a.cara[en_a])]
    f_b = [(int(h), int(c)) for h, c in zip(b.hexaedro[en_b], b.cara[en_b])]
    return i, j, f_a, f_b, float(a.area[en_a].sum()), float(b.area[en_b].sum())


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


def _nodos_de_borde(malla: MallaSolida, indice: int) -> np.ndarray:
    parte = malla.partes[indice]
    if parte.lamina:
        return malla.nodos_de_parte(parte)
    return np.unique(malla.fronteras(parte).nodos)


def detectar_acoples(malla: MallaSolida, tamano: float, ya_acopladas: Sequence[Tuple[int, int]] = (),
                     prohibidos: Optional[Set[int]] = None,
                     tolerancias: Optional[Dict[frozenset, object]] = None) -> Tuple[List["Acople"], List[str]]:
    """Acoples de las láminas con las piezas vecinas: cada nodo de la lámina a menos de ``tol`` de un nodo del borde de
    la otra pieza (``tol`` = 0,6 veces el tamaño de elemento más medio espesor) se une a él por ``*EQUATION``
    (traslaciones; también giros si la otra pieza es una lámina). Un nodo solo depende de otro una vez.
    ``prohibidos`` son nodos que no pueden depender de otro (los que tienen apoyo); ``tolerancias`` da, por pareja de
    partes, una tolerancia mayor pedida por un constraint manual: un número, o (tolerancia, nodos) para que la tolerancia
    mayor valga solo en esos nodos (las superficies elegidas en el constraint; el resto, con la normal)."""

    from .generador import Acople

    acoples: List[Acople] = []
    avisos: List[str] = []
    dependientes: Set[int] = set(prohibidos or ())
    maestros: Set[int] = set()
    saltadas = {frozenset(par) for par in ya_acopladas}
    laminas = [i for i, p in enumerate(malla.partes) if p.lamina]
    for i in laminas:
        pi = malla.partes[i]
        espesor_i = float(malla.espesor_de_lamina[pi.primer_lamina: pi.ultimo_lamina].max())
        nodos_i = malla.nodos_de_parte(pi)
        caja_i = (malla.coordenadas[nodos_i].min(axis=0), malla.coordenadas[nodos_i].max(axis=0))
        for j, pj in enumerate(malla.partes):
            if j == i or (pj.lamina and j < i) or frozenset((i, j)) in saltadas:
                continue
            espesor_j = float(malla.espesor_de_lamina[pj.primer_lamina: pj.ultimo_lamina].max()) if pj.lamina else 0.0
            tol_normal = 0.6 * tamano + 0.5 * (espesor_i + espesor_j)
            pedido = (tolerancias or {}).get(frozenset((i, j)), 0.0)
            tol_pedida, permitidos = pedido if isinstance(pedido, tuple) else (pedido, None)
            tol = max(tol_normal, tol_pedida)
            nodos_j = _nodos_de_borde(malla, j)
            cj = malla.coordenadas[nodos_j]
            if np.any(caja_i[0] - tol > cj.max(axis=0)) or np.any(caja_i[1] + tol < cj.min(axis=0)):
                continue
            esclavos_posibles = np.array([n for n in nodos_i if int(n) not in dependientes and int(n) not in maestros], dtype=np.int64)
            maestros_posibles = np.array([n for n in nodos_j if int(n) not in dependientes], dtype=np.int64)
            if len(esclavos_posibles) == 0 or len(maestros_posibles) == 0:
                continue
            distancia, vecino = cKDTree(malla.coordenadas[maestros_posibles]).query(
                malla.coordenadas[esclavos_posibles], distance_upper_bound=tol)
            bien = np.isfinite(distancia)
            if permitidos is not None and tol > tol_normal:  # la tolerancia mayor, solo en las superficies del constraint
                en_superficie = np.isin(esclavos_posibles, list(permitidos))
                bien &= (distancia <= tol_normal) | en_superficie
            if bien.sum() < 3:
                continue
            a, b = esclavos_posibles[bien], maestros_posibles[vecino[bien]]
            dependientes.update(int(n) for n in a)
            maestros.update(int(n) for n in b)
            giro = [] if pj.lamina else _ecuaciones_giro(malla, a, maestros_posibles, tol, tamano)
            acoples.append(Acople(f"AC{len(acoples) + 1}", i, j, a, b, giros=pj.lamina, ecuaciones_giro=giro))
    return acoples, avisos


def _ecuaciones_giro(malla: MallaSolida, nodos_lamina: np.ndarray, nodos_solido: np.ndarray, tol: float, tamano: float,
                     vecinos: int = 8) -> List[List[Tuple[int, int, float]]]:
    """Ecuaciones que igualan el giro de cada nodo de la lámina al del parche de nodos del sólido que lo rodea.

    Para el parche (sus ``vecinos`` nodos más cercanos, centrados en su centroide ``c``) el giro de cuerpo rígido que
    mejor ajusta sus desplazamientos es ``θ = Σ M⁻¹ [r_j]× u_j`` con ``r_j = x_j - c`` y ``M = Σ (|r_j|² I - r_j r_jᵀ)``:
    una combinación lineal de los desplazamientos, que CalculiX acepta en ``*EQUATION``. Sin esto una lámina soldada
    por su canto a un sólido quedaría articulada alrededor de la línea de unión."""

    if len(nodos_solido) < 4:
        return []
    coordenadas = malla.coordenadas
    arbol = cKDTree(coordenadas[nodos_solido])
    k = min(vecinos, len(nodos_solido))
    distancia, indice = arbol.query(coordenadas[nodos_lamina], k=k)
    alcance = max(2.0 * tol, 1.5 * tamano)
    ecuaciones: List[List[Tuple[int, int, float]]] = []
    for fila, n in enumerate(nodos_lamina):
        cerca = indice[fila][np.isfinite(distancia[fila]) & (distancia[fila] <= alcance)]
        if len(cerca) < 4:
            continue
        parche = nodos_solido[cerca]
        r = coordenadas[parche] - coordenadas[parche].mean(axis=0)
        M = np.eye(3) * float((r * r).sum()) - r.T @ r
        valores = np.linalg.eigvalsh(M)
        if valores.min() <= 1e-6 * max(valores.max(), 1e-30):
            continue  # parche alineado: el giro alrededor de la línea no está definido
        inversa = np.linalg.inv(M)
        for g in range(3):
            terminos: List[Tuple[int, int, float]] = [(int(n), 4 + g, 1.0)]
            for nodo, rj in zip(parche, r):
                cruz = np.array([[0.0, -rj[2], rj[1]], [rj[2], 0.0, -rj[0]], [-rj[1], rj[0], 0.0]])  # [r]× u = r × u
                fila_c = inversa[g] @ cruz
                for d in range(3):
                    if abs(fila_c[d]) > 1e-12 * max(1.0, float(np.abs(inversa).max())):
                        terminos.append((int(nodo), d + 1, -float(fila_c[d])))
            ecuaciones.append(terminos)
    return ecuaciones
