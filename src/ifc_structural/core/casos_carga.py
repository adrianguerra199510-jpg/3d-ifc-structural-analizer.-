"""Casos de carga y combinaciones."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class TipoCaso(str, Enum):
    """Tipos de caso de carga habituales en AASHTO LRFD / ACI 318."""

    D = "D"  # carga muerta (peso propio + sobreimpuesta estructural)
    SCP = "SCP"  # sobrecarga permanente (acabados, instalaciones)
    L = "L"  # carga viva
    W = "W"  # viento
    E = "E"  # sismo
    OTRO = "OTRO"


@dataclass
class CasoCarga:
    """Caso de carga independiente. Se resuelve una única vez en CalculiX."""

    id: str
    nombre: str
    tipo: TipoCaso = TipoCaso.OTRO

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id


@dataclass
class Combinacion:
    """Combinación de casos de carga, resuelta por superposición lineal.

    ``factores`` mapea ``caso_id -> factor``. Un caso ausente del diccionario
    se interpreta con factor 0 (no participa en la combinación).
    """

    id: str
    nombre: str
    factores: Dict[str, float] = field(default_factory=dict)
    norma: str = ""

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id

    def factor(self, caso_id: str) -> float:
        return self.factores.get(caso_id, 0.0)


# Combinaciones de referencia frecuentes (ACI 318-19 / AASHTO LRFD Resistencia I),
# ofrecidas como plantillas editables en la GUI; no se aplican automáticamente.
PLANTILLAS_COMBINACIONES = {
    "ACI318_1.4D": {"D": 1.4},
    "ACI318_1.2D_1.6L": {"D": 1.2, "L": 1.6},
    "ACI318_1.2D_1.0L_1.0W": {"D": 1.2, "L": 1.0, "W": 1.0},
    "ACI318_0.9D_1.0W": {"D": 0.9, "W": 1.0},
    "ACI318_1.2D_1.0L_1.0E": {"D": 1.2, "L": 1.0, "E": 1.0},
    "AASHTO_RESISTENCIA_I": {"D": 1.25, "SCP": 1.5, "L": 1.75},
}
