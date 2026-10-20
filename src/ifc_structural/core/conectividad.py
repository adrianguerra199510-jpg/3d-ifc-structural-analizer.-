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


#: Holgura (m) que se suma a la mitad de la sección de una barra para decidir
#: si el extremo de otra llega "a su eje" (vigas que terminan en la cara de la
#: columna, con o sin placa de extremo).
HOLGURA_UNION_T = 0.05


def conectar_uniones_en_t(proyecto: Proyecto, holgura: float = HOLGURA_UNION_T) -> int:
    """Conecta las barras que llegan al eje de otra por un punto intermedio
    (unión en T: viga contra una columna continua, viga secundaria contra
    una principal).

    El extremo que llega se traslada al punto del eje de la otra barra más
    cercano (idealización por ejes) y la barra atravesada se parte en ese
    punto, de modo que ambas queden unidas por un nodo. Un extremo solo se
    conecta si su distancia al eje no supera la mitad de la sección mayor de
    la barra atravesada más ``holgura``, y si el punto cae lejos de los
    extremos de esa barra (los extremos próximos entre sí los resuelve
    ``conectar_nodos``). Devuelve cuántas barras se partieron.
    """

    from .geometria import dimensiones_seccion  # evita importación circular

    nodos_de_barras = {n for b in proyecto.todas_las_barras().values() for n in (b.nodo_inicio, b.nodo_fin)}
    cortes: Dict[str, List[tuple]] = {}  # barra atravesada -> [(s, nodo_nuevo)]
    trasladados: Dict[str, str] = {}  # nodo viejo -> nodo nuevo
    for nodo_id in sorted(nodos_de_barras):
        p = np.array(proyecto.nodos[nodo_id].coordenadas, dtype=float)
        mejor = None
        for barra_id, barra in proyecto.todas_las_barras().items():
            if nodo_id in (barra.nodo_inicio, barra.nodo_fin):
                continue
            a = np.array(proyecto.nodos[barra.nodo_inicio].coordenadas, dtype=float)
            b = np.array(proyecto.nodos[barra.nodo_fin].coordenadas, dtype=float)
            longitud = float(np.linalg.norm(b - a))
            if longitud < 1e-9:
                continue
            s = float(np.dot(p - a, b - a) / longitud**2)
            margen = holgura / longitud
            if not margen < s < 1.0 - margen:
                continue
            punto = a + s * (b - a)
            distancia = float(np.linalg.norm(p - punto))
            altura, ancho = dimensiones_seccion(proyecto.perfiles[barra.perfil])
            if distancia > max(altura, ancho) / 2.0 + holgura:
                continue
            if mejor is None or distancia < mejor[0]:
                mejor = (distancia, barra_id, s, punto)
        if mejor is not None:
            _, barra_id, s, punto = mejor
            nuevo = proyecto.nodo_o_existente(*(round(float(c), 6) + 0.0 for c in punto))
            trasladados[nodo_id] = nuevo.id
            if not any(abs(s - s0) < 1e-6 for s0, n0 in cortes.get(barra_id, []) if n0 == nuevo.id):
                cortes.setdefault(barra_id, []).append((s, nuevo.id))

    if not trasladados:
        return 0

    for barra in proyecto.todas_las_barras().values():
        barra.nodo_inicio = trasladados.get(barra.nodo_inicio, barra.nodo_inicio)
        barra.nodo_fin = trasladados.get(barra.nodo_fin, barra.nodo_fin)

    partidas = 0
    for barra_id, puntos in cortes.items():
        barra = proyecto.elemento_barra(barra_id)
        original_fin = barra.nodo_fin
        puntos = sorted(puntos)
        es_viga = barra.tipo.value == "VIGA"
        agregar = proyecto.agregar_viga if es_viga else proyecto.agregar_columna
        anterior = barra
        for k, (_, nodo_corte) in enumerate(puntos):
            anterior.nodo_fin = nodo_corte
            nuevo_id = proyecto._contador.siguiente("B" if es_viga else "C")
            while nuevo_id in proyecto.vigas or nuevo_id in proyecto.columnas:
                nuevo_id = proyecto._contador.siguiente("B" if es_viga else "C")
            siguiente = agregar(
                nodo_corte, original_fin, barra.material, barra.perfil, id_=nuevo_id,
                nombre=f"{barra.nombre} (tramo {k + 2})", ifc_guid=barra.ifc_guid, vector_n1=barra.vector_n1,
            )
            anterior = siguiente
        barra.nombre = f"{barra.nombre} (tramo 1)"
        partidas += 1

    en_uso = {n for b in proyecto.todas_las_barras().values() for n in (b.nodo_inicio, b.nodo_fin)}
    for losa in list(proyecto.losas.values()) + list(proyecto.muros.values()):
        en_uso.update(losa.nodos_contorno)
    for nodo in list(trasladados):
        if nodo not in en_uso:
            proyecto.nodos.pop(nodo, None)
    return partidas
