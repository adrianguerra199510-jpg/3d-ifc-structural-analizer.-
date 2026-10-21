"""Láminas (cáscaras ``S6`` de CalculiX) para sólidos de pared delgada.

Un sólido cuyo espesor es mucho menor que su tamaño (el cuerpo de un depósito, una tolva, una chapa curva) no se puede
representar bien con tetraedros ni con hexaedros: haría falta un elemento por cada milímetro de pared. Se modela como
lámina: se obtiene su **superficie media** a partir de la superficie triangulada del sólido (cada triángulo de una piel se
lanza un rayo hacia dentro hasta la piel opuesta; la superficie media está a medio camino y el espesor es la distancia),
se refina con bisección de aristas hasta el tamaño de elemento y se convierte a triángulos cuadráticos de 6 nodos
(``S6``) con el espesor de cada elemento. No necesita que la superficie sea una variedad cerrada perfecta: sirve también
para las piezas con aristas compartidas por cuatro triángulos que Gmsh no puede mallar.

Orden de nodos de ``S6``: vértices 1-3 y medios 4-6 de las aristas 1-2, 2-3 y 3-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..core.solidos import SolidoGeneral, normales_y_areas

#: Máximo de triángulos de una lámina tras el refinamiento.
MAXIMO_TRIANGULOS = 120_000
#: Fracción del tamaño de elemento por debajo de la cual el espesor medio hace que un sólido sea una lámina.
RELACION_PARED_DELGADA = 0.4
#: La bisección de aristas deja elementos de entre 0,5 y 1 vez el límite: con este factor el lado medio es ~ el tamaño de elemento.
FACTOR_REFINAMIENTO = 2.2


class ErrorLamina(RuntimeError):
    """No se pudo obtener la superficie media del sólido."""


@dataclass
class MallaLamina:
    coordenadas: np.ndarray  # (N, 3) vértices y nodos medios
    elementos: np.ndarray  # (M, 6) orden de CalculiX
    espesores: np.ndarray  # (M,) m
    normales: np.ndarray  # (M, 3) unitarias, hacia fuera de la piel conservada
    superficies: Dict[int, np.ndarray] = field(default_factory=dict)  # superficie -> (k, 2) (elemento local, lado 1 | 2)
    nodos_superficie: Dict[int, np.ndarray] = field(default_factory=dict)  # superficie -> nodos que están sobre ella
    tamano: float = 0.0
    volumen: float = 0.0  # suma de área × espesor


def espesor_medio(solido: SolidoGeneral) -> float:
    """Espesor medio si el sólido es una pared delgada: 2 V / A."""

    area = sum(c.area for c in solido.caras)
    return 2.0 * float(solido.volumen) / area if area > 0 else float("inf")


def es_pared_delgada(solido: SolidoGeneral, tamano: float) -> bool:
    return espesor_medio(solido) <= max(0.015, RELACION_PARED_DELGADA * tamano)


def _vecinos_por_arista(triangulos: np.ndarray, validos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pares de triángulos válidos que comparten una arista usada por exactamente dos triángulos (las aristas donde
    se tocan dos cascarones, con cuatro triángulos, no unen pieles distintas)."""

    m = len(triangulos)
    aristas = np.sort(np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]]), axis=1)
    propietario = np.tile(np.arange(m), 3)
    unicas, inverso, cuenta = np.unique(aristas, axis=0, return_inverse=True, return_counts=True)
    inverso = inverso.ravel()
    orden = np.argsort(inverso, kind="stable")
    ordenado = propietario[orden]
    inicio = np.concatenate([[0], np.cumsum(cuenta)[:-1]])
    dobles = np.where(cuenta == 2)[0]
    a, b = ordenado[inicio[dobles]], ordenado[inicio[dobles] + 1]
    bien = validos[a] & validos[b]
    return a[bien], b[bien]


def _piel_conservada(solido: SolidoGeneral) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triángulos de una de las dos pieles y su espesor.

    Devuelve (índices de los triángulos conservados, espesor de cada uno, triángulo opuesto de cada triángulo del sólido
    (-1 si no tiene), espesor de cada triángulo del sólido)."""

    import pyvista as pv

    v, t = solido.vertices, solido.triangulos
    n, area = normales_y_areas(v, t)
    centro = v[t].mean(axis=1)
    medio = espesor_medio(solido)
    if not np.isfinite(medio) or medio <= 0:
        raise ErrorLamina("el sólido no tiene espesor")
    limite = 3.0 * medio
    malla = pv.PolyData(v, np.hstack([np.full((len(t), 1), 3), t]).ravel().astype(np.int64))
    opuesto = np.full(len(t), -1, dtype=np.int64)
    espesor = np.zeros(len(t))
    eps = 1e-4 * medio
    for i in range(len(t)):
        puntos, celdas = malla.ray_trace(centro[i] - n[i] * eps, centro[i] - n[i] * limite)
        if len(celdas) == 0:
            continue
        d = np.linalg.norm(np.asarray(puntos) - centro[i], axis=1)
        k = int(np.argmin(d))
        j = int(celdas[k])
        if j != i and float(np.dot(n[i], n[j])) < -0.3:
            opuesto[i], espesor[i] = j, d[k]
    validos = opuesto >= 0
    if not validos.any():
        raise ErrorLamina("no se encontró la pared opuesta de ningún triángulo")

    # Pieles: componentes conexas de triángulos válidos; de cada pareja de pieles enfrentadas se conserva una.
    a, b = _vecinos_por_arista(t, validos)
    grafo = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(t), len(t)))
    _, etiqueta = connected_components(grafo, directed=False)
    etiqueta = np.where(validos, etiqueta, -1)
    conservada: List[int] = []
    descartadas = set()
    for c in sorted(set(etiqueta[validos].tolist())):
        if c in descartadas:
            continue
        conservada.append(c)
        enfrentadas = etiqueta[opuesto[(etiqueta == c)]]
        enfrentadas = enfrentadas[enfrentadas >= 0]
        if len(enfrentadas):
            pareja = int(np.bincount(enfrentadas).argmax())
            if pareja != c:
                descartadas.add(pareja)
    mantener = np.isin(etiqueta, conservada) & validos
    return np.where(mantener)[0], espesor[mantener], opuesto, espesor


def _soldar(v: np.ndarray, t: np.ndarray, atributo: np.ndarray, tolerancia: float):
    """Une los vértices a menos de ``tolerancia`` (las teselas diminutas de esquinas redondeadas dan elementos más
    estrechos que su espesor) y descarta los triángulos que quedan degenerados."""

    if tolerancia <= 0:
        return v, t, atributo
    arbol = cKDTree(v)
    pares = arbol.query_pairs(tolerancia, output_type="ndarray")
    if len(pares) == 0:
        return v, t, atributo
    grafo = coo_matrix((np.ones(len(pares)), (pares[:, 0], pares[:, 1])), shape=(len(v), len(v)))
    n, grupo = connected_components(grafo, directed=False)
    suma = np.zeros((n, 3))
    np.add.at(suma, grupo, v)
    centros = suma / np.bincount(grupo, minlength=n)[:, None]
    nt = grupo[t]
    bien = (nt[:, 0] != nt[:, 1]) & (nt[:, 1] != nt[:, 2]) & (nt[:, 0] != nt[:, 2])
    usados, inverso = np.unique(nt[bien], return_inverse=True)
    return centros[usados], inverso.reshape(-1, 3), atributo[bien]


def _refinar(v: np.ndarray, t: np.ndarray, atributos: np.ndarray, h: float):
    """Bisección conforme de aristas más largas que ``h``. ``atributos`` (M,) se hereda en los triángulos hijos."""

    v = v.copy()
    for _ in range(30):
        arista = np.stack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=1)  # (M, 3, 2)
        largo = np.linalg.norm(v[arista[..., 0]] - v[arista[..., 1]], axis=2)
        cortar = largo > h
        if not cortar.any():
            break
        claves = np.sort(arista, axis=2)
        todas = claves[cortar]
        unicas, inverso = np.unique(todas, axis=0, return_inverse=True)
        medios = len(v) + np.arange(len(unicas))
        v = np.vstack([v, 0.5 * (v[unicas[:, 0]] + v[unicas[:, 1]])])
        indice = np.full(cortar.shape, -1, dtype=np.int64)
        indice[cortar] = medios[inverso.ravel()]
        mascara = cortar[:, 0] * 1 + cortar[:, 1] * 2 + cortar[:, 2] * 4
        nuevos, hereda = [], []
        for m in range(8):
            sel = np.where(mascara == m)[0]
            if len(sel) == 0:
                continue
            a, b, c = t[sel, 0], t[sel, 1], t[sel, 2]
            m01, m12, m20 = indice[sel, 0], indice[sel, 1], indice[sel, 2]
            if m == 0:
                partes = [(a, b, c)]
            elif m == 1:
                partes = [(a, m01, c), (m01, b, c)]
            elif m == 2:
                partes = [(b, m12, a), (m12, c, a)]
            elif m == 4:
                partes = [(c, m20, b), (m20, a, b)]
            elif m == 3:
                partes = [(m01, b, m12), (a, m01, m12), (a, m12, c)]
            elif m == 6:
                partes = [(m12, c, m20), (b, m12, m20), (b, m20, a)]
            elif m == 5:
                partes = [(m20, a, m01), (m01, b, c), (m01, c, m20)]
            else:
                partes = [(a, m01, m20), (m01, b, m12), (m20, m12, c), (m01, m12, m20)]
            for p in partes:
                nuevos.append(np.stack(p, axis=1))
                hereda.append(atributos[sel])
        t = np.vstack(nuevos)
        atributos = np.concatenate(hereda)
        if len(t) > MAXIMO_TRIANGULOS:
            raise ErrorLamina(f"la lámina tendría más de {MAXIMO_TRIANGULOS:,} elementos".replace(",", " "))
    return v, t, atributos


def mallar_lamina(solido: SolidoGeneral, tamano: float) -> Optional[MallaLamina]:
    """Lámina ``S6`` de la superficie media del sólido. Lanza ``ErrorLamina`` si no se puede obtener."""

    v, t = solido.vertices, solido.triangulos
    if not (np.isfinite(v).all() and len(t)):
        raise ErrorLamina("coordenadas no válidas")
    n, _ = normales_y_areas(v, t)
    conservados, espesores, opuesto, espesor_tri = _piel_conservada(solido)

    # superficie media: cada vértice de la piel conservada se desplaza la mitad del espesor hacia dentro
    tc = t[conservados]
    usados, local = np.unique(tc, return_inverse=True)
    local = local.reshape(-1, 3)
    desplazamiento = np.zeros((len(usados), 3))
    cuenta = np.zeros(len(usados))
    for k in range(3):
        np.add.at(desplazamiento, local[:, k], -0.5 * espesores[:, None] * n[conservados])
        np.add.at(cuenta, local[:, k], 1.0)
    vm = v[usados] + desplazamiento / cuenta[:, None]
    vm, local, origen = _soldar(vm, local, np.arange(len(conservados)),
                              max(0.4 * float(np.median(espesores)), 0.12 * FACTOR_REFINAMIENTO * tamano))
    if len(local) == 0:
        raise ErrorLamina("la superficie media quedó vacía")
    vm, tm, origen = _refinar(vm, local, np.arange(len(conservados)), FACTOR_REFINAMIENTO * tamano)

    e = espesores[origen]
    p = vm[tm]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    keep = area > 1e-12
    tm, origen, e, normal, area = tm[keep], origen[keep], e[keep], normal[keep], area[keep]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    # la normal de cada elemento debe seguir la de la piel conservada (hacia fuera)
    previa = n[conservados][origen]
    voltea = np.einsum("ij,ij->i", normal, previa) < 0
    tm[voltea] = tm[voltea][:, [0, 2, 1]]
    normal[voltea] *= -1.0

    # nodos medios de las aristas
    arista = np.concatenate([tm[:, [0, 1]], tm[:, [1, 2]], tm[:, [2, 0]]])
    claves = np.sort(arista, axis=1)
    unicas, inverso = np.unique(claves, axis=0, return_inverse=True)
    inverso = inverso.reshape(3, -1).T
    coordenadas = np.vstack([vm, 0.5 * (vm[unicas[:, 0]] + vm[unicas[:, 1]])])
    elementos = np.hstack([tm, len(vm) + inverso]).astype(np.int64)

    # superficies del sólido: cada triángulo original se asocia al elemento más cercano (y a su lado)
    centro_e = vm[tm].mean(axis=1)
    arbol = cKDTree(centro_e)
    centro_o = v[t].mean(axis=1)
    distancia, cercano = arbol.query(centro_o)
    alcance = max(2.0 * tamano, 4.0 * float(espesores.max()))
    lado = np.where(np.einsum("ij,ij->i", n, normal[cercano]) >= 0, 1, 2)
    superficies: Dict[int, np.ndarray] = {}
    cara_de = solido.cara_de_triangulo
    valido = distancia <= alcance
    for s in np.unique(cara_de[valido]):
        marca = valido & (cara_de == s)
        pares = np.unique(np.stack([cercano[marca], lado[marca]], axis=1), axis=0)
        superficies[int(s)] = pares
    # nodos sobre cada superficie del sólido (en un canto solo los del borde; en una piel, todos los de la lámina bajo ella)
    from ..core.solidos import muestras_de_superficie

    espaciado = max(0.5 * float(np.median(espesores)), 0.25 * tamano)
    arbol_nodos = cKDTree(coordenadas)
    nodos_superficie: Dict[int, np.ndarray] = {}
    for s_id in superficies:
        triangulos_cara = t[cara_de == s_id]
        puntos, _ = muestras_de_superficie(v, triangulos_cara, espaciado)
        cerca = arbol_nodos.query_ball_point(puntos, 0.5 * float(espesores.max()) + 0.55 * espaciado)
        nodos_superficie[s_id] = np.unique(np.fromiter((n for lista in cerca for n in lista), dtype=np.int64))
    return MallaLamina(coordenadas=coordenadas, elementos=elementos, espesores=e, normales=normal, superficies=superficies,
                       nodos_superficie=nodos_superficie, tamano=tamano, volumen=float((area * e).sum()))
