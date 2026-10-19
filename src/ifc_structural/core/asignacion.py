"""Conversión de lo que el usuario selecciona en el visor (elementos o
caras) en apoyos y cargas del modelo.

Todo se traduce a los tipos de carga que ya entiende el solver
(``CargaArea``, ``CargaLineal``, ``CargaPuntual``) y a ``Apoyo``:

=====================================  =====================================
Selección                              Resultado
=====================================  =====================================
Losa (elemento o cara sup./inf.)       ``CargaArea`` (kPa) hacia -Z o, si se
                                       eligió una cara, hacia dentro de ella
Viga/columna completa                  ``CargaLineal`` uniforme (kN/m)
Cara lateral de viga/columna           ``CargaLineal`` = presión × ancho de
                                       la cara, perpendicular a la cara
Cara extrema de viga/columna           ``CargaPuntual`` en ese nodo =
                                       presión × área de la cara, axial
Cara extrema de viga/columna           ``Apoyo`` en el nodo de ese extremo
=====================================  =====================================

Sin dependencias de la interfaz gráfica, para poder probarlo aisladamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .cargas import CargaArea, CargaLineal, CargaPuntual
from .geometria import Cara, solido_elemento
from .modelo import Proyecto

Vector3 = Tuple[float, float, float]
Carga = Union[CargaArea, CargaLineal, CargaPuntual]

GRAVEDAD: Vector3 = (0.0, 0.0, -1.0)

#: Direcciones globales ofrecidas para cargas que no salen de una cara.
DIRECCIONES = {
    "Gravedad (−Z)": (0.0, 0.0, -1.0),
    "Hacia arriba (+Z)": (0.0, 0.0, 1.0),
    "+X": (1.0, 0.0, 0.0),
    "−X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "−Y": (0.0, -1.0, 0.0),
}


class ErrorAsignacion(ValueError):
    """La selección no admite lo que se pidió; el mensaje explica por qué."""


@dataclass(frozen=True)
class Objetivo:
    """Elemento (o una de sus caras) seleccionado en el visor."""

    tipo: str  # "VIGA", "COLUMNA", "LOSA" o "MURO"
    id: str
    cara: Optional[int] = None  # índice de cara; None = el elemento completo


_ETIQUETA_TIPO = {"VIGA": "Viga", "COLUMNA": "Columna", "LOSA": "Losa", "MURO": "Muro"}


def cara_de(proyecto: Proyecto, objetivo: Objetivo) -> Optional[Cara]:
    if objetivo.cara is None:
        return None
    return solido_elemento(proyecto, objetivo.tipo, objetivo.id).caras[objetivo.cara]


def descripcion(proyecto: Proyecto, objetivo: Objetivo) -> str:
    """Texto legible: «Viga B1», «Losa L1 — cara superior»."""

    base = f"{_ETIQUETA_TIPO[objetivo.tipo]} {objetivo.id}"
    cara = cara_de(proyecto, objetivo)
    return base if cara is None else f"{base} — cara {cara.nombre}"


def _formato(valor: float) -> str:
    return f"{valor:g}"


def _vector(normal) -> Vector3:
    return (float(normal[0]), float(normal[1]), float(normal[2]))


# ---------------------------------------------------------------------------
# Apoyos
# ---------------------------------------------------------------------------


def nodo_para_apoyo(proyecto: Proyecto, objetivo: Objetivo) -> str:
    """Nodo donde va un apoyo asignado sobre la selección."""

    if objetivo.tipo not in ("VIGA", "COLUMNA"):
        raise ErrorAsignacion(
            f"{descripcion(proyecto, objetivo)}: los apoyos se asignan en las caras extremas de vigas y "
            "columnas. Las losas y muros no tienen apoyos propios en la Fase 1 (se apoyan en las vigas)."
        )
    cara = cara_de(proyecto, objetivo)
    if cara is None or cara.nodo_id is None:
        raise ErrorAsignacion(
            f"{descripcion(proyecto, objetivo)}: seleccione la cara extrema (inicial o final) de la "
            "viga o columna donde va el apoyo (modo «Cara»)."
        )
    return cara.nodo_id


# ---------------------------------------------------------------------------
# Cargas
# ---------------------------------------------------------------------------


def cargas_para_objetivo(
    proyecto: Proyecto,
    objetivo: Objetivo,
    caso_id: str,
    id_: str,
    presion: float,
    w: float,
    direccion: Vector3 = GRAVEDAD,
) -> List[Carga]:
    """Cargas que resultan de asignar ``presion`` (kPa) y/o ``w`` (kN/m,
    magnitud positiva en el sentido de ``direccion``) sobre la selección.

    La presión se usa en losas y en las caras de vigas/columnas; la carga
    lineal, al seleccionar la viga o columna completa. Una presión
    positiva empuja hacia dentro de la cara.
    """

    nombre_objetivo = descripcion(proyecto, objetivo)
    cara = cara_de(proyecto, objetivo)

    if objetivo.tipo == "MURO":
        raise ErrorAsignacion(f"{nombre_objetivo}: las cargas sobre muros no se transfieren en la Fase 1 (son informativos).")

    if objetivo.tipo == "LOSA":
        if cara is not None and cara.tipo not in ("SUPERIOR", "INFERIOR"):
            raise ErrorAsignacion(
                f"{nombre_objetivo}: las cargas de área van sobre la cara superior o inferior de la losa, "
                "no sobre sus bordes."
            )
        if abs(presion) < 1e-12:
            raise ErrorAsignacion(f"{nombre_objetivo}: la presión es cero.")
        dir_carga = direccion if cara is None else tuple(-c for c in _vector(cara.normal))
        return [
            CargaArea(
                id=id_, caso_id=caso_id, presion=presion, losa_id=objetivo.id, direccion=dir_carga,
                nombre=f"{nombre_objetivo}: {_formato(presion)} kPa",
            )
        ]

    # Vigas y columnas
    if cara is None:
        if abs(w) < 1e-12:
            raise ErrorAsignacion(f"{nombre_objetivo}: la carga lineal es cero.")
        return [
            CargaLineal(
                id=id_, caso_id=caso_id, elemento_id=objetivo.id, w_inicio=w, w_fin=w, direccion=direccion,
                nombre=f"{nombre_objetivo}: {_formato(w)} kN/m",
            )
        ]

    if abs(presion) < 1e-12:
        raise ErrorAsignacion(f"{nombre_objetivo}: la presión es cero.")
    dir_carga = tuple(-c for c in _vector(cara.normal))

    if cara.tipo == "LATERAL":
        linea = presion * cara.ancho
        return [
            CargaLineal(
                id=id_, caso_id=caso_id, elemento_id=objetivo.id, w_inicio=linea, w_fin=linea, direccion=dir_carga,
                nombre=f"{nombre_objetivo}: {_formato(presion)} kPa × {cara.ancho:.3g} m = {linea:.4g} kN/m",
            )
        ]

    # Cara extrema: fuerza axial resultante en el nodo del extremo.
    fuerza = presion * cara.area
    fx, fy, fz = (fuerza * c for c in dir_carga)
    return [
        CargaPuntual(
            id=id_, caso_id=caso_id, nodo_id=cara.nodo_id, fx=fx, fy=fy, fz=fz,
            nombre=f"{nombre_objetivo}: {_formato(presion)} kPa × {cara.area:.3g} m² = {fuerza:.4g} kN",
        )
    ]


def vista_previa_cargas(
    proyecto: Proyecto, objetivos: List[Objetivo], presion: float, w: float, direccion: Vector3 = GRAVEDAD
) -> List[Tuple[bool, str]]:
    """Qué cargas se crearían para cada objetivo, o por qué no se puede.

    Devuelve [(es_valido, texto)] en el orden de ``objetivos``.
    """

    resultado: List[Tuple[bool, str]] = []
    for objetivo in objetivos:
        try:
            cargas = cargas_para_objetivo(proyecto, objetivo, "-", "-", presion, w, direccion)
            resultado.append((True, "; ".join(c.nombre for c in cargas)))
        except ErrorAsignacion as exc:
            resultado.append((False, str(exc)))
    return resultado
