"""Apoyos y cargas sobre una superficie cualquiera (una cara de un sólido de forma libre o de una
barra, losa o muro). Solo se aplican en el análisis 3D.

Sin dependencias de la interfaz gráfica, para poder probarlo aisladamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .asignacion import ErrorAsignacion, Objetivo, cara_de, descripcion
from .cargas import ApoyoSuperficie, CargaSuperficie
from .modelo import Proyecto

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Superficie:
    """Datos de una superficie elegida, vengan de donde vengan."""

    tipo: str
    elemento_id: str
    cara: int
    nombre: str
    area: float
    normal: Vector3  # unitaria, hacia fuera del sólido
    centro: Vector3


def superficie_de(proyecto: Proyecto, objetivo: Objetivo) -> Superficie:
    cara = cara_de(proyecto, objetivo)
    if cara is None:
        raise ErrorAsignacion(
            f"{descripcion(proyecto, objetivo)}: elija una superficie (modo «Cara») para asignarle apoyos o cargas."
        )
    return Superficie(
        tipo=objetivo.tipo, elemento_id=objetivo.id, cara=objetivo.cara, nombre=descripcion(proyecto, objetivo),
        area=float(cara.area), normal=tuple(float(c) for c in cara.normal), centro=tuple(float(c) for c in cara.centro),
    )


def apoyo_superficie_para(proyecto: Proyecto, objetivo: Objetivo, id_: str, grados: Sequence[bool]) -> ApoyoSuperficie:
    """Apoyo sobre la superficie elegida con los grados de libertad (Ux, Uy, Uz, Rx, Ry, Rz) restringidos."""

    sup = superficie_de(proyecto, objetivo)
    if not any(grados):
        raise ErrorAsignacion(f"{sup.nombre}: marque al menos un grado de libertad restringido.")
    nombres = "".join(n for n, g in zip(("Ux", "Uy", "Uz", "Rx", "Ry", "Rz"), grados) if g)
    return ApoyoSuperficie(
        id=id_, elemento_tipo=sup.tipo, elemento_id=sup.elemento_id, cara=sup.cara,
        ux=bool(grados[0]), uy=bool(grados[1]), uz=bool(grados[2]),
        rx=bool(grados[3]), ry=bool(grados[4]), rz=bool(grados[5]),
        nombre=f"{sup.nombre}: apoyo {nombres}",
    )


def carga_superficie_para(proyecto: Proyecto, objetivo: Objetivo, caso_id: str, id_: str, tipo: str, valor: float,
                          direccion: Optional[Vector3]) -> CargaSuperficie:
    """Carga sobre la superficie elegida: presión (kPa), fuerza total (kN) o momento (kN·m)."""

    sup = superficie_de(proyecto, objetivo)
    if abs(valor) < 1e-12:
        raise ErrorAsignacion(f"{sup.nombre}: el valor es cero.")
    if direccion is not None:
        v = np.asarray(direccion, dtype=float)
        if np.linalg.norm(v) < 1e-9:
            raise ErrorAsignacion(f"{sup.nombre}: la dirección no puede ser el vector nulo.")
        direccion = tuple(float(c) for c in v / np.linalg.norm(v))
    if tipo in ("FUERZA", "MOMENTO") and direccion is None:
        raise ErrorAsignacion(f"{sup.nombre}: indique la dirección de la fuerza o del eje del momento.")
    unidad = {"PRESION": "kPa", "FUERZA": "kN", "MOMENTO": "kN·m"}[tipo]
    donde = {"PRESION": "presión", "FUERZA": "fuerza", "MOMENTO": "momento"}[tipo]
    return CargaSuperficie(
        id=id_, caso_id=caso_id, elemento_tipo=sup.tipo, elemento_id=sup.elemento_id, cara=sup.cara,
        tipo=tipo, valor=float(valor), direccion=direccion, nombre=f"{sup.nombre}: {donde} {valor:g} {unidad}",
    )


def direccion_efectiva(carga: CargaSuperficie, normal_hacia_fuera: Sequence[float]) -> np.ndarray:
    """Sentido en que actúa una presión: el de la dirección elegida o, si no hay, hacia dentro de la superficie."""

    if carga.direccion is not None:
        return np.asarray(carga.direccion, dtype=float)
    return -np.asarray(normal_hacia_fuera, dtype=float)
