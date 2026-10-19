"""Superposición lineal de esfuerzos por caso para obtener combinaciones.

Cada caso de carga se resuelve una única vez en CalculiX (ver
``solver.ejecutor_ccx``). Las combinaciones (``core.casos_carga.Combinacion``)
se obtienen combinando esos resultados por superposición lineal del
**tensor de esfuerzos** nodal, sin volver a invocar el solver — válido
porque el análisis es elástico lineal. Las magnitudes derivadas (von
Mises, principales, cortante máximo) se calculan DESPUÉS de combinar el
tensor, nunca combinando las magnitudes, que no son lineales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..core.casos_carga import Combinacion
from . import esfuerzos


def superponer_combinacion(tensiones_por_caso: Dict[str, np.ndarray], combinacion: Combinacion) -> np.ndarray:
    """Tensor (N, 6) combinado: suma de factor × tensor de cada caso."""

    total = None
    for caso_id, factor in combinacion.factores.items():
        if abs(factor) < 1e-12:
            continue
        tensor = tensiones_por_caso.get(caso_id)
        if tensor is None:
            continue
        total = factor * tensor if total is None else total + factor * tensor

    if total is None:
        n = next((t.shape[0] for t in tensiones_por_caso.values()), 0)
        total = np.zeros((n, 6))
    return total


def superponer_todas_las_combinaciones(
    tensiones_por_caso: Dict[str, np.ndarray], combinaciones: List[Combinacion]
) -> Dict[str, np.ndarray]:
    return {c.id: superponer_combinacion(tensiones_por_caso, c) for c in combinaciones}


@dataclass
class Envolvente:
    """Envolvente de una magnitud sobre un conjunto de combinaciones.

    ``valores`` es el valor más desfavorable en cada nodo y
    ``combinacion_por_nodo`` el id de la combinación que lo produjo.
    """

    clave: str
    valores: np.ndarray
    combinacion_por_nodo: np.ndarray


def calcular_envolvente(tensiones_por_combinacion: Dict[str, np.ndarray], clave: str) -> Envolvente:
    """Para cada nodo, el valor más desfavorable de la magnitud ``clave``
    entre todas las combinaciones, y qué combinación lo produjo."""

    if not tensiones_por_combinacion:
        raise ValueError("No hay combinaciones sobre las que calcular la envolvente.")

    ids = list(tensiones_por_combinacion)
    valores = np.vstack([esfuerzos.calcular(clave, tensiones_por_combinacion[i]) for i in ids])  # (C, N)
    if esfuerzos.MAGNITUDES[clave].envolvente == "min":
        indice = np.argmin(valores, axis=0)
    else:
        indice = np.argmax(valores, axis=0)
    columnas = np.arange(valores.shape[1])
    return Envolvente(
        clave=clave,
        valores=valores[indice, columnas],
        combinacion_por_nodo=np.array(ids, dtype=object)[indice],
    )
