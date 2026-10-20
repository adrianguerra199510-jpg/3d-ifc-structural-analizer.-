"""Revisión de la geometría del modelo: valores no finitos (NaN, infinito) o imposibles.

Un NaN en una coordenada, una sección o un espesor se propaga por el mallador y acaba como un error críptico
(«cannot convert float NaN to integer»). Aquí se detecta de antemano y se dice qué elemento lo causa.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .modelo import Proyecto


def _finito(*valores) -> bool:
    return all(v is not None and math.isfinite(float(v)) for v in valores)


def problemas_de_geometria(proyecto: Proyecto) -> List[str]:
    """Descripción (una por elemento) de cada elemento con geometría no finita o inutilizable."""

    problemas: List[str] = []
    malos = {i for i, n in proyecto.nodos.items() if not _finito(n.x, n.y, n.z)}
    for i in sorted(malos):
        problemas.append(f"El nodo {i} tiene coordenadas no válidas.")
    for coleccion, etiqueta in ((proyecto.vigas, "La viga"), (proyecto.columnas, "La columna")):
        for b in coleccion.values():
            nombre = f"{etiqueta} {b.id}" + (f" ({b.nombre})" if b.nombre else "")
            if b.nodo_inicio in malos or b.nodo_fin in malos:
                problemas.append(f"{nombre} usa un nodo con coordenadas no válidas.")
            p = proyecto.perfiles.get(b.perfil)
            if p is None:
                problemas.append(f"{nombre} usa el perfil «{b.perfil}», que no existe.")
            elif not _finito(p.area, p.inercia_y, p.inercia_z, p.inercia_torsional) or p.area <= 0 \
                    or (p.altura is not None and not _finito(p.altura)) or (p.ancho is not None and not _finito(p.ancho)):
                problemas.append(f"{nombre}: el perfil «{p.nombre}» tiene propiedades no válidas.")
    for coleccion, etiqueta in ((proyecto.losas, "La losa"), (proyecto.muros, "El muro")):
        for e in coleccion.values():
            nombre = f"{etiqueta} {e.id}" + (f" ({e.nombre})" if e.nombre else "")
            if not _finito(e.espesor) or e.espesor <= 0:
                problemas.append(f"{nombre} tiene un espesor no válido ({e.espesor}).")
            elif any(n in malos or n not in proyecto.nodos for n in e.nodos_contorno):
                problemas.append(f"{nombre} usa un nodo con coordenadas no válidas.")
    for s in proyecto.solidos.values():
        nombre = f"El sólido {s.id}" + (f" ({s.nombre})" if s.nombre else "")
        if not np.isfinite(s.vertices).all():
            problemas.append(f"{nombre} tiene vértices con coordenadas no válidas.")
        elif not math.isfinite(float(s.volumen)):
            problemas.append(f"{nombre} tiene un volumen no válido.")
    return problemas


def exigir_geometria_valida(proyecto: Proyecto) -> None:
    """Lanza ``ValueError`` con la lista de elementos con geometría no válida, si los hay."""

    problemas = problemas_de_geometria(proyecto)
    if problemas:
        resto = f" (y {len(problemas) - 8} más)" if len(problemas) > 8 else ""
        raise ValueError("La geometría del modelo no es válida; revise o elimine estos elementos:\n- "
                         + "\n- ".join(problemas[:8]) + resto)
