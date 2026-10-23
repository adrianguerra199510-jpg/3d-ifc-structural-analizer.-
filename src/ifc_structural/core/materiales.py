"""Plantillas de materiales habituales y utilidades de unidades.

El modelo guarda el módulo de elasticidad en kPa y el peso específico en
kN/m³; en la interfaz se muestra el módulo en MPa, que es lo usual.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .modelo import Material, Proyecto

KPA_POR_MPA = 1000.0


def _hormigon(fc: float) -> Tuple[float, float, float]:
    return (4700.0 * math.sqrt(fc), 0.20, 24.0)  # E = 4700·√f'c (ACI 318), MPa


#: nombre -> (E [MPa], coeficiente de Poisson, peso específico [kN/m³])
PLANTILLAS_MATERIALES: Dict[str, Tuple[float, float, float]] = {
    "Hormigón f'c = 21 MPa": _hormigon(21.0),
    "Hormigón f'c = 28 MPa": _hormigon(28.0),
    "Hormigón f'c = 35 MPa": _hormigon(35.0),
    "Hormigón f'c = 42 MPa": _hormigon(42.0),
    "Acero estructural A36": (200000.0, 0.30, 78.5),
    "Acero estructural A572 Gr. 50": (200000.0, 0.30, 78.5),
    "Aluminio 6061-T6": (69000.0, 0.33, 26.6),
    "Madera (pino)": (10000.0, 0.30, 6.0),
    "Mampostería de ladrillo": (5000.0, 0.20, 18.0),
}


#: nombre de plantilla -> esfuerzo de fluencia fy [MPa] (las que no están son elásticas)
FLUENCIA_PLANTILLAS: Dict[str, float] = {
    "Acero estructural A36": 250.0,
    "Acero estructural A572 Gr. 50": 345.0,
    "Aluminio 6061-T6": 240.0,
}


def material_desde_valores(nombre: str, e_mpa: float, poisson: float, peso_especifico: float,
                           fy_mpa: float = 0.0, et_mpa: float = 0.0) -> Material:
    """Material con E en MPa; con ``fy_mpa`` > 0, elastoplástico bilineal (``et_mpa`` = 0: Et = E/100)."""

    return Material(
        nombre=nombre, modulo_elasticidad=e_mpa * KPA_POR_MPA, modulo_poisson=poisson, densidad=peso_especifico,
        fluencia=fy_mpa * KPA_POR_MPA if fy_mpa > 0 else None,
        modulo_tangente=et_mpa * KPA_POR_MPA if fy_mpa > 0 and et_mpa > 0 else None,
    )


def modulo_en_mpa(material: Material) -> float:
    return material.modulo_elasticidad / KPA_POR_MPA


def descripcion_material(material: Material) -> str:
    texto = (
        f"{material.nombre} — E = {modulo_en_mpa(material):,.0f} MPa, ν = {material.modulo_poisson:g}, "
        f"γ = {material.densidad:g} kN/m³"
    )
    if material.es_plastico:
        texto += (f", fy = {material.fluencia / KPA_POR_MPA:,.0f} MPa (bilineal, Et = "
                  f"{material.tangente / KPA_POR_MPA:,.0f} MPa)")
    return texto


def usos_de_material(proyecto: Proyecto, nombre: str) -> list:
    """Elementos (vigas, columnas, losas, muros) que usan el material."""

    usos = []
    for coleccion in (proyecto.vigas, proyecto.columnas, proyecto.losas, proyecto.muros):
        usos.extend(e.id for e in coleccion.values() if e.material == nombre)
    return usos
