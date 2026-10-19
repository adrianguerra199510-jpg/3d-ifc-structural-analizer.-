"""Lectura de resultados de CalculiX: desplazamientos y fuerzas de sección.

Ver el docstring de ``generador_inp.py`` (puntos 3 y 4) para la
justificación de por qué las fuerzas internas y las reacciones se leen del
.frd (bloque de fuerzas de sección) en vez del .dat: CalculiX no ofrece
una opción de fuerzas de sección para ``*EL PRINT``.

Formato del .frd relevante aquí (``*EL FILE, OUTPUT=2D, SECTION FORCES``,
confirmado empíricamente con CalculiX 2.21): es un archivo de texto con
bloques delimitados por líneas que empiezan con ``-4`` (encabezado de
bloque, p. ej. ``" -4  STRESS      6    1"``) y terminan en una línea
``-3``. Dentro del bloque, cada línea de datos empieza con ``-1`` y tiene
el id de nodo en las columnas 4-13 seguido de N valores de 12 caracteres
cada uno (formato Fortran ``E12.5``), sin separador entre ellos (los
signos "pegan" los números), por lo que se debe parsear por ancho fijo y
no por separación en espacios.

Para "fuerzas de sección" (SECTION FORCES), las 6 columnas del .frd
(etiquetadas SXX,SYY,SZZ,SXY,SYZ,SZX por herencia del formato de
tensores) representan, según el manual de CalculiX:

    SXX -> V1 (cortante, dirección local 1)
    SYY -> V2 (cortante, dirección local 2)
    SZZ -> N  (fuerza normal/axial)
    SXY -> T  (momento torsor)
    SYZ -> M1 (momento flector respecto del eje local 1)
    SZX -> M2 (momento flector respecto del eje local 2)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

Vector3 = Tuple[float, float, float]
Vector6 = Tuple[float, float, float, float, float, float]

_ANCHO_CAMPO = 12
_INICIO_VALORES = 13
_LARGO_CAMPO_NODO = 10


@dataclass
class ResultadosCaso:
    """Resultados crudos (por nodo de la malla ccx) de un caso de carga."""

    desplazamientos: Dict[int, Vector3] = field(default_factory=dict)
    fuerzas_seccion: Dict[int, Vector6] = field(default_factory=dict)


class ErrorLecturaResultados(ValueError):
    pass


def leer_desplazamientos_dat(ruta_dat: Path) -> Dict[int, Vector3]:
    """Lee el bloque "displacements (vx,vy,vz)" del .dat de CalculiX."""

    texto = ruta_dat.read_text(encoding="utf-8", errors="ignore")
    resultado: Dict[int, Vector3] = {}
    match = re.search(r"displacements[^\n]*\n\n(.*?)(?:\n\s*\n|\Z)", texto, re.S)
    if not match:
        return resultado
    for linea in match.group(1).splitlines():
        partes = linea.split()
        if len(partes) != 4:
            continue
        try:
            nodo = int(partes[0])
            resultado[nodo] = (float(partes[1]), float(partes[2]), float(partes[3]))
        except ValueError:
            continue
    return resultado


def _parsear_linea_ancho_fijo(linea: str, n_valores: int) -> Tuple[int, List[float]]:
    nodo = int(linea[3 : 3 + _LARGO_CAMPO_NODO])
    valores = []
    inicio = _INICIO_VALORES
    for _ in range(n_valores):
        campo = linea[inicio : inicio + _ANCHO_CAMPO]
        valores.append(float(campo))
        inicio += _ANCHO_CAMPO
    return nodo, valores


def leer_fuerzas_seccion_frd(ruta_frd: Path) -> Dict[int, Vector6]:
    """Lee el bloque de fuerzas de sección (SECTION FORCES) del .frd.

    Devuelve, por nodo de la malla ccx, la tupla ``(N, V1, V2, T, M1, M2)``
    ya reordenada según la convención documentada en el docstring del
    módulo (y verificada empíricamente).
    """

    texto = ruta_frd.read_text(encoding="utf-8", errors="ignore")
    resultado: Dict[int, Vector6] = {}
    en_bloque = False
    for linea in texto.splitlines():
        if linea.startswith(" -4"):
            en_bloque = "STRESS" in linea
            continue
        if linea.startswith(" -3"):
            en_bloque = False
            continue
        if en_bloque and linea.startswith(" -1"):
            try:
                nodo, valores = _parsear_linea_ancho_fijo(linea, 6)
            except (ValueError, IndexError):
                continue
            sxx, syy, szz, sxy, syz, szx = valores
            resultado[nodo] = (szz, sxx, syy, sxy, syz, szx)  # N,V1,V2,T,M1,M2
    return resultado


def leer_resultados_caso(ruta_dat: Path, ruta_frd: Path) -> ResultadosCaso:
    return ResultadosCaso(
        desplazamientos=leer_desplazamientos_dat(ruta_dat),
        fuerzas_seccion=leer_fuerzas_seccion_frd(ruta_frd),
    )
