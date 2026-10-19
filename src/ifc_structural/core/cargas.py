"""Cargas y apoyos.

Unidades: fuerzas puntuales en kN, cargas lineales en kN/m, cargas de área
en kPa (kN/m2), momentos en kN·m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

Vector3 = Tuple[float, float, float]


class SistemaReferencia(str, Enum):
    GLOBAL = "GLOBAL"
    LOCAL = "LOCAL"


@dataclass
class Apoyo:
    """Restricción de desplazamiento/rotación en un nodo.

    Cada bandera indica si ese grado de libertad está restringido (True) o
    libre (False). ``ux, uy, uz`` son traslaciones; ``rx, ry, rz`` rotaciones
    (estas últimas solo aplican a nodos que pertenecen a elementos tipo
    barra, ya que los elementos de losa en modo transferencia no tienen
    grados de libertad rotacionales propios en Fase 1).
    """

    id: str
    nodo_id: str
    ux: bool = True
    uy: bool = True
    uz: bool = True
    rx: bool = True
    ry: bool = True
    rz: bool = True
    nombre: str = ""

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id

    def grados_restringidos(self) -> List[int]:
        """Números de grado de libertad (1-6, convención CalculiX) restringidos."""

        banderas = [self.ux, self.uy, self.uz, self.rx, self.ry, self.rz]
        return [i + 1 for i, restringido in enumerate(banderas) if restringido]


@dataclass
class CargaPuntual:
    """Fuerza y/o momento puntual, en el sistema global.

    Puede aplicarse directamente sobre un nodo existente (``nodo_id``) o
    sobre un punto intermedio de una viga/columna (``elemento_id`` +
    ``posicion``, fracción 0..1 desde su nodo de inicio). En este segundo
    caso, el generador de .inp inserta un nodo real en esa posición al
    construir la malla de análisis.
    """

    id: str
    caso_id: str
    nodo_id: Optional[str] = None
    elemento_id: Optional[str] = None
    posicion: Optional[float] = None
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    nombre: str = ""

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id
        sobre_nodo = self.nodo_id is not None
        sobre_elemento = self.elemento_id is not None and self.posicion is not None
        if sobre_nodo == sobre_elemento:
            raise ValueError(
                "CargaPuntual debe indicar exactamente una ubicación: "
                "'nodo_id', o bien 'elemento_id' + 'posicion'."
            )
        if sobre_elemento and not (0.0 <= self.posicion <= 1.0):
            raise ValueError("'posicion' debe estar entre 0.0 y 1.0")


@dataclass
class CargaLineal:
    """Carga distribuida sobre una viga o columna, en el sistema global.

    En el caso general (carga transferida desde una losa) la intensidad
    puede variar linealmente a lo largo del elemento entre ``w_inicio`` y
    ``w_fin`` (en kN/m); para una carga uniforme simple ambos valores son
    iguales. La dirección de la carga es el vector unitario
    ``direccion`` (por defecto, gravedad: -Z global).
    """

    id: str
    caso_id: str
    elemento_id: str
    w_inicio: float
    w_fin: float
    s_inicio: float = 0.0  # posición relativa (0..1) sobre el elemento, desde nodo_inicio
    s_fin: float = 1.0
    direccion: Vector3 = (0.0, 0.0, -1.0)
    sistema: SistemaReferencia = SistemaReferencia.GLOBAL
    nombre: str = ""
    origen: str = "MANUAL"  # "MANUAL" o "TRANSFERENCIA_LOSA"
    losa_origen_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id

    @property
    def es_uniforme(self) -> bool:
        return abs(self.w_inicio - self.w_fin) < 1e-9

    @property
    def es_parcial(self) -> bool:
        return self.s_inicio > 1e-9 or self.s_fin < 1.0 - 1e-9


@dataclass
class CargaArea:
    """Presión uniforme sobre una losa o muro, en kPa (kN/m2).

    ``direccion`` es el vector unitario de aplicación de la presión
    (por defecto, gravedad: -Z global, típico de cargas de piso/techo).
    En Fase 1 solo se contempla presión uniforme sobre toda la superficie
    (no variable punto a punto).
    """

    id: str
    caso_id: str
    presion: float
    losa_id: Optional[str] = None
    muro_id: Optional[str] = None
    direccion: Vector3 = (0.0, 0.0, -1.0)
    nombre: str = ""
    origen: str = "MANUAL"  # "MANUAL" o "PESO_PROPIO"

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id
        if not self.losa_id and not self.muro_id:
            raise ValueError("CargaArea debe referenciar una losa o un muro")
