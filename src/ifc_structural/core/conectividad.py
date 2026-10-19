"""Conexión de los nodos de barras que casi coinciden.

En un IFC de Revit, la viga que llega a una columna termina en la cara de
la columna (o a media altura de su apoyo), no exactamente en el punto de
eje donde termina la columna. Si cada extremo se deja como un nodo
independiente, la estructura queda desconectada y el análisis resulta
inestable. Idealizar el pórtico por sus ejes implica unir esos nodos.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree

from .modelo import Proyecto

#: Distancia máxima (m) entre dos extremos de barras para considerarlos el
#: mismo nodo estructural.
TOLERANCIA_UNION = 0.4


def conectar_nodos(proyecto: Proyecto, tolerancia: float = TOLERANCIA_UNION) -> int:
    """Une los extremos de barras que distan menos que ``tolerancia``.

    El nodo que se conserva es el de una columna, si el grupo tiene alguna
    (el eje de una columna es la referencia más fiable); si no, el más
    cercano al centro del grupo. Nunca se une una barra consigo misma
    (barras más cortas que la tolerancia no se colapsan). Devuelve cuántos
    nodos desaparecieron al unirse a otro.
    """

    barras = proyecto.todas_las_barras()
    usados: List[str] = sorted({n for b in barras.values() for n in (b.nodo_inicio, b.nodo_fin)})
    if len(usados) < 2:
        return 0

    indice = {nodo: i for i, nodo in enumerate(usados)}
    coordenadas = np.array([proyecto.nodos[n].coordenadas for n in usados], dtype=float)
    pares = cKDTree(coordenadas).query_pairs(tolerancia)

    excluidos: set = set()
    padre: List[int] = list(range(len(usados)))

    def buscar(x: int) -> int:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for _ in range(len(barras) + 1):
        padre = list(range(len(usados)))
        for i, j in pares:
            if i not in excluidos and j not in excluidos:
                padre[buscar(i)] = buscar(j)
        colapsadas = [
            b for b in barras.values() if buscar(indice[b.nodo_inicio]) == buscar(indice[b.nodo_fin])
        ]
        if not colapsadas:
            break
        for b in colapsadas:
            excluidos.update((indice[b.nodo_inicio], indice[b.nodo_fin]))

    grupos: Dict[int, List[int]] = {}
    for i in range(len(usados)):
        grupos.setdefault(buscar(i), []).append(i)

    nodos_de_columna = {n for c in proyecto.columnas.values() for n in (c.nodo_inicio, c.nodo_fin)}
    reemplazo: Dict[str, str] = {}
    for miembros in grupos.values():
        if len(miembros) < 2:
            continue
        nombres = [usados[i] for i in miembros]
        de_columna = [n for n in nombres if n in nodos_de_columna]
        if de_columna:
            representante = de_columna[0]
        else:
            centro = coordenadas[miembros].mean(axis=0)
            representante = min(nombres, key=lambda n: np.linalg.norm(np.array(proyecto.nodos[n].coordenadas) - centro))
        for n in nombres:
            if n != representante:
                reemplazo[n] = representante

    if not reemplazo:
        return 0

    for barra in barras.values():
        barra.nodo_inicio = reemplazo.get(barra.nodo_inicio, barra.nodo_inicio)
        barra.nodo_fin = reemplazo.get(barra.nodo_fin, barra.nodo_fin)
    for elemento in list(proyecto.losas.values()) + list(proyecto.muros.values()):
        elemento.nodos_contorno = [reemplazo.get(n, n) for n in elemento.nodos_contorno]

    en_uso = {n for b in barras.values() for n in (b.nodo_inicio, b.nodo_fin)}
    for elemento in list(proyecto.losas.values()) + list(proyecto.muros.values()):
        en_uso.update(elemento.nodos_contorno)
    for nodo in reemplazo:
        if nodo not in en_uso:
            del proyecto.nodos[nodo]

    return len(reemplazo)
