"""Superposición lineal de resultados por caso para obtener combinaciones.

Cada caso de carga se resuelve una única vez en CalculiX (ver
``solver.ejecutor_ccx``). Las combinaciones (``core.casos_carga.Combinacion``)
se obtienen combinando esos resultados por superposición lineal, sin volver
a invocar el solver — válido porque el análisis es elástico lineal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..core.casos_carga import Combinacion
from ..solver.lector_resultados import ResultadosCaso, Vector3, Vector6

#: Nombres de las 6 componentes de fuerza de sección, en el mismo orden
#: que ``ResultadosCaso.fuerzas_seccion`` (ver solver/lector_resultados.py).
COMPONENTES_FUERZA = ("N", "V1", "V2", "T", "M1", "M2")


def _combinar_vector(valores_por_caso: Dict[str, Tuple[float, ...]], combinacion: Combinacion, n: int) -> Tuple[float, ...]:
    total = [0.0] * n
    for caso_id, factor in combinacion.factores.items():
        if abs(factor) < 1e-12:
            continue
        valores = valores_por_caso.get(caso_id)
        if valores is None:
            continue
        for i in range(n):
            total[i] += factor * valores[i]
    return tuple(total)


def superponer_combinacion(
    resultados_por_caso: Dict[str, ResultadosCaso], combinacion: Combinacion
) -> ResultadosCaso:
    """Combina los resultados de varios casos según los factores de ``combinacion``."""

    nodos = set()
    for caso_id, factor in combinacion.factores.items():
        if abs(factor) < 1e-12:
            continue
        resultados = resultados_por_caso.get(caso_id)
        if resultados is None:
            continue
        nodos |= set(resultados.desplazamientos) | set(resultados.fuerzas_seccion)

    desplazamientos: Dict[int, Vector3] = {}
    fuerzas_seccion: Dict[int, Vector6] = {}
    for nodo in nodos:
        desplazamientos_por_caso = {
            caso_id: r.desplazamientos.get(nodo, (0.0, 0.0, 0.0))
            for caso_id, r in resultados_por_caso.items()
        }
        fuerzas_por_caso = {
            caso_id: r.fuerzas_seccion.get(nodo, (0.0,) * 6)
            for caso_id, r in resultados_por_caso.items()
        }
        desplazamientos[nodo] = _combinar_vector(desplazamientos_por_caso, combinacion, 3)
        fuerzas_seccion[nodo] = _combinar_vector(fuerzas_por_caso, combinacion, 6)

    return ResultadosCaso(desplazamientos=desplazamientos, fuerzas_seccion=fuerzas_seccion)


def superponer_todas_las_combinaciones(
    resultados_por_caso: Dict[str, ResultadosCaso], combinaciones: List[Combinacion]
) -> Dict[str, ResultadosCaso]:
    return {c.id: superponer_combinacion(resultados_por_caso, c) for c in combinaciones}


@dataclass
class Envolvente:
    """Envolvente (máximos y mínimos) de las fuerzas de sección de un nodo,
    evaluada sobre un conjunto de combinaciones (y/o casos) de carga.
    """

    maximo: Dict[int, Vector6] = field(default_factory=dict)
    minimo: Dict[int, Vector6] = field(default_factory=dict)
    combinacion_del_maximo: Dict[Tuple[int, int], str] = field(default_factory=dict)
    combinacion_del_minimo: Dict[Tuple[int, int], str] = field(default_factory=dict)

    def maximo_de(self, nodo: int, componente: str) -> float:
        return self.maximo[nodo][COMPONENTES_FUERZA.index(componente)]

    def minimo_de(self, nodo: int, componente: str) -> float:
        return self.minimo[nodo][COMPONENTES_FUERZA.index(componente)]


def calcular_envolvente(resultados_por_combinacion: Dict[str, ResultadosCaso]) -> Envolvente:
    """Calcula, para cada nodo de la malla y cada componente de fuerza de
    sección (N, V1, V2, T, M1, M2), el máximo y el mínimo entre todas las
    combinaciones evaluadas, junto con qué combinación lo produjo.
    """

    envolvente = Envolvente()
    nodos = set()
    for resultados in resultados_por_combinacion.values():
        nodos |= set(resultados.fuerzas_seccion)

    for nodo in nodos:
        maximos = [float("-inf")] * 6
        minimos = [float("inf")] * 6
        comb_max = [""] * 6
        comb_min = [""] * 6
        for combinacion_id, resultados in resultados_por_combinacion.items():
            valores = resultados.fuerzas_seccion.get(nodo)
            if valores is None:
                continue
            for i in range(6):
                if valores[i] > maximos[i]:
                    maximos[i] = valores[i]
                    comb_max[i] = combinacion_id
                if valores[i] < minimos[i]:
                    minimos[i] = valores[i]
                    comb_min[i] = combinacion_id
        envolvente.maximo[nodo] = tuple(maximos)
        envolvente.minimo[nodo] = tuple(minimos)
        for i, nombre_componente in enumerate(COMPONENTES_FUERZA):
            envolvente.combinacion_del_maximo[(nodo, i)] = comb_max[i]
            envolvente.combinacion_del_minimo[(nodo, i)] = comb_min[i]

    return envolvente
