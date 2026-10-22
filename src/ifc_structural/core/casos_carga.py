"""Casos de carga y combinaciones."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence


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
    #: Solo en los casos de carga muerta (D): el caso incluye el peso propio de los elementos (peso específico del
    #: material × volumen).
    peso_propio: bool = True

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id

    @property
    def incluye_peso_propio(self) -> bool:
        return self.tipo == TipoCaso.D and self.peso_propio


def casos_con_peso_propio(casos: Dict[str, CasoCarga]) -> List[str]:
    """Casos que llevan el peso propio: los de carga muerta con la casilla marcada."""

    return [c.id for c in casos.values() if c.incluye_peso_propio]


def combinaciones_calculables(combinaciones: Dict[str, "Combinacion"], casos: Sequence[str]) -> Dict[str, "Combinacion"]:
    """Combinaciones cuyos casos (los de factor no nulo) están todos en ``casos``."""

    elegidos = set(casos)
    return {i: c for i, c in combinaciones.items() if {k for k, f in c.factores.items() if f} <= elegidos}


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


def casos_y_combinaciones_por_defecto():
    """Casos D (carga muerta) y L (carga viva) y dos combinaciones ACI 318
    habituales, para poder analizar sin configurar nada primero."""

    casos = {
        "D": CasoCarga(id="D", nombre="Carga muerta", tipo=TipoCaso.D),
        "L": CasoCarga(id="L", nombre="Carga viva", tipo=TipoCaso.L),
    }
    combinaciones = {
        "COMB1": Combinacion(id="COMB1", nombre="1.4D", factores={"D": 1.4}, norma="ACI 318"),
        "COMB2": Combinacion(id="COMB2", nombre="1.2D + 1.6L", factores={"D": 1.2, "L": 1.6}, norma="ACI 318"),
    }
    return casos, combinaciones
