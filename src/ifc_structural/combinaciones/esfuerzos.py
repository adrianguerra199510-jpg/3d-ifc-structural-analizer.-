"""Magnitudes de esfuerzo derivadas del tensor nodal.

Todas son invariantes (no dependen del sistema de ejes en que CalculiX
exprese el tensor), por lo que se pueden calcular directamente sobre el
tensor ya combinado (superposición lineal de tensores por combinación).

Unidades: las mismas del tensor (kPa con el modelo en kN y m). La
conversión a MPa para mostrar es ``FACTOR_KPA_A_MPA``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

FACTOR_KPA_A_MPA = 1e-3


def tensores_3x3(t: np.ndarray) -> np.ndarray:
    """(N, 6) en el orden SXX, SYY, SZZ, SXY, SYZ, SZX -> (N, 3, 3)."""

    t = np.atleast_2d(t)
    m = np.zeros((t.shape[0], 3, 3))
    m[:, 0, 0], m[:, 1, 1], m[:, 2, 2] = t[:, 0], t[:, 1], t[:, 2]
    m[:, 0, 1] = m[:, 1, 0] = t[:, 3]
    m[:, 1, 2] = m[:, 2, 1] = t[:, 4]
    m[:, 2, 0] = m[:, 0, 2] = t[:, 5]
    return m


def von_mises(t: np.ndarray) -> np.ndarray:
    t = np.atleast_2d(t)
    sxx, syy, szz, sxy, syz, szx = (t[:, i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )


def principales(t: np.ndarray) -> np.ndarray:
    """Esfuerzos principales (N, 3) en orden ascendente: σ3 ≤ σ2 ≤ σ1."""

    return np.linalg.eigvalsh(tensores_3x3(t))


def principal_maximo(t: np.ndarray) -> np.ndarray:
    return principales(t)[:, 2]


def principal_minimo(t: np.ndarray) -> np.ndarray:
    return principales(t)[:, 0]


def cortante_maximo(t: np.ndarray) -> np.ndarray:
    p = principales(t)
    return (p[:, 2] - p[:, 0]) / 2.0


@dataclass(frozen=True)
class Magnitud:
    clave: str
    nombre: str
    funcion: Callable[[np.ndarray], np.ndarray]
    #: En la envolvente de varias combinaciones: "max" toma el mayor valor
    #: y "min" el menor (σ3 es negativo en compresión: el más desfavorable
    #: es el mínimo).
    envolvente: str


MAGNITUDES: Dict[str, Magnitud] = {
    "von_mises": Magnitud("von_mises", "Von Mises (σvm)", von_mises, "max"),
    "s1": Magnitud("s1", "Principal máximo σ1 (tracción +)", principal_maximo, "max"),
    "s3": Magnitud("s3", "Principal mínimo σ3 (compresión −)", principal_minimo, "min"),
    "tau_max": Magnitud("tau_max", "Cortante máximo τmax", cortante_maximo, "max"),
}


def calcular(clave: str, tensor: np.ndarray) -> np.ndarray:
    """Valor nodal (N,) de la magnitud ``clave`` para un tensor (N, 6)."""

    return MAGNITUDES[clave].funcion(tensor)
