"""Lectura de resultados de CalculiX: malla expandida de las vigas y tensor
de esfuerzos en cada uno de sus nodos.

Con ``*EL FILE`` + ``S`` (ver ``generador_inp.py``, punto 3) CalculiX
escribe en el .frd (ASCII) la malla expandida de las vigas B31 —
hexaedros de 8 nodos— y el tensor de esfuerzos nodal. Los hexaedros
conservan el número del elemento B31 del que provienen, lo que permite
asociar cada celda con su barra.

Formato del .frd (confirmado empíricamente con CalculiX 2.21): bloques
delimitados por líneas ``-4`` (encabezado de variable) y ``-3`` (fin).
Cada línea de datos empieza con `` -1`` y tiene el identificador en las
columnas 4-13 seguido de valores de 12 caracteres (formato ``E12.5``) sin
separador entre ellos (los signos "pegan" los números), por lo que se
parsea por ancho fijo y no por separación en espacios.

- Bloque ``2C``: nodos (id + x, y, z).
- Bloque ``3C``: elementos. Línea `` -1`` (id, tipo, grupo, material)
  seguida de una línea `` -2`` con los ids de sus nodos (campos de 10
  caracteres). Tipo 1 = hexaedro de 8 nodos.
- Bloque ``-4  STRESS``: SXX, SYY, SZZ, SXY, SYZ, SZX por nodo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

_ANCHO_CAMPO = 12
_INICIO_VALORES = 13
_LARGO_ID = 10
_TIPO_HEXAEDRO_8 = 1
_TIPO_TETRAEDRO_10 = 6
_TIPOS_CUNA = (2, 5)  # cuña de 6 o 15 nodos: lámina expandida (``S3``/``S6``)


class ErrorLecturaResultados(ValueError):
    pass


@dataclass
class MallaTensiones:
    """Malla expandida (hexaedros) sobre la que se evalúan los esfuerzos."""

    ids_nodos: np.ndarray  # (N,) etiquetas de nodo de ccx, ascendentes
    coordenadas: np.ndarray  # (N, 3)
    ids_elementos: np.ndarray  # (M,) coinciden con los ids de ElementoMalla
    conectividad: np.ndarray  # (M, 8) índices 0-based en ``ids_nodos``

    @property
    def n_nodos(self) -> int:
        return len(self.ids_nodos)


def recortar_malla(malla: MallaTensiones, mantener: np.ndarray):
    """Malla con solo los hexaedros marcados en ``mantener`` (máscara por
    celda) y los nodos que usan. Devuelve (malla recortada, índices de los
    nodos conservados en la malla original) para recortar igual cualquier
    campo nodal: ``campo[indices]``."""

    conectividad = malla.conectividad[mantener]
    usados = np.unique(conectividad)
    nueva = np.searchsorted(usados, conectividad)
    recortada = MallaTensiones(
        ids_nodos=malla.ids_nodos[usados],
        coordenadas=malla.coordenadas[usados],
        ids_elementos=malla.ids_elementos[mantener],
        conectividad=nueva,
    )
    return recortada, usados


def _parsear_valores(linea: str, n_valores: int) -> List[float]:
    valores = []
    inicio = _INICIO_VALORES
    for _ in range(n_valores):
        valores.append(float(linea[inicio : inicio + _ANCHO_CAMPO]))
        inicio += _ANCHO_CAMPO
    return valores


def leer_malla_frd(ruta_frd: Path) -> MallaTensiones:
    """Lee la malla expandida (nodos y hexaedros) del .frd."""

    lineas = ruta_frd.read_text(encoding="utf-8", errors="ignore").splitlines()

    ids_nodos: List[int] = []
    coordenadas: List[List[float]] = []
    elementos: Dict[int, List[int]] = {}

    bloque = None
    elemento_actual = None
    for linea in lineas:
        if linea.startswith("    2C"):
            bloque = "NODOS"
            continue
        if linea.startswith("    3C"):
            bloque = "ELEMENTOS"
            continue
        if linea.startswith(" -3"):
            bloque = None
            continue
        if bloque == "NODOS" and linea.startswith(" -1"):
            ids_nodos.append(int(linea[3 : 3 + _LARGO_ID]))
            coordenadas.append(_parsear_valores(linea, 3))
        elif bloque == "ELEMENTOS":
            if linea.startswith(" -1"):
                id_el = int(linea[3 : 3 + _LARGO_ID])
                tipo = int(linea[13:18])
                if tipo not in (_TIPO_HEXAEDRO_8, _TIPO_TETRAEDRO_10) + _TIPOS_CUNA:
                    raise ErrorLecturaResultados(
                        f"Tipo de elemento no soportado en el .frd: {tipo} (se esperaba 1, hexaedro; 6, tetraedro; 2 o 5, cuña de lámina)"
                    )
                elemento_actual = id_el
                elementos[id_el] = []
            elif linea.startswith(" -2") and elemento_actual is not None:
                campos = linea[3:]
                elementos[elemento_actual].extend(
                    int(campos[i : i + _LARGO_ID]) for i in range(0, len(campos.rstrip()), _LARGO_ID)
                )

    if not ids_nodos or not elementos:
        raise ErrorLecturaResultados(f"El archivo {ruta_frd.name} no contiene malla expandida (nodos/elementos).")

    ids = np.array(ids_nodos, dtype=np.int64)
    orden = np.argsort(ids)
    ids = ids[orden]
    coords = np.array(coordenadas, dtype=float)[orden]

    ids_el = np.array(sorted(elementos), dtype=np.int64)
    # Un tetraedro se dibuja como hexaedro degenerado con sus 4 vértices (a, b, c, c, d, d, d, d): así toda la
    # malla tiene el mismo número de nodos por celda. Los nodos medios quedan como puntos con resultados.
    # Una cuña (lámina expandida: tres nodos abajo y tres arriba) se dibuja como (a, b, c, c, d, e, f, f).
    def celda(e):
        if len(e) == 8:
            return e
        if len(e) == 10:
            return [e[0], e[1], e[2], e[2], e[3], e[3], e[3], e[3]]
        return [e[0], e[1], e[2], e[2], e[3], e[4], e[5], e[5]]

    conectividad = np.array([celda(elementos[i]) for i in ids_el], dtype=np.int64)
    indices = np.searchsorted(ids, conectividad)
    if not np.array_equal(ids[indices], conectividad):
        raise ErrorLecturaResultados("La conectividad del .frd referencia nodos inexistentes.")

    return MallaTensiones(ids_nodos=ids, coordenadas=coords, ids_elementos=ids_el, conectividad=indices)


def leer_tensiones_frd(ruta_frd: Path, malla: MallaTensiones) -> np.ndarray:
    """Tensor de esfuerzos nodal, (N, 6) en el orden SXX, SYY, SZZ, SXY,
    SYZ, SZX y en el orden de ``malla.ids_nodos`` (unidades: kPa si el
    modelo está en kN y m)."""

    lineas = ruta_frd.read_text(encoding="utf-8", errors="ignore").splitlines()
    por_nodo: Dict[int, List[float]] = {}
    en_bloque = False
    for linea in lineas:
        if linea.startswith(" -4"):
            en_bloque = "STRESS" in linea
            continue
        if linea.startswith(" -3"):
            en_bloque = False
            continue
        if en_bloque and linea.startswith(" -1"):
            por_nodo[int(linea[3 : 3 + _LARGO_ID])] = _parsear_valores(linea, 6)

    if not por_nodo:
        raise ErrorLecturaResultados(f"El archivo {ruta_frd.name} no contiene el bloque de esfuerzos (STRESS).")

    faltantes = [n for n in malla.ids_nodos if n not in por_nodo]
    if faltantes:
        raise ErrorLecturaResultados(
            f"Faltan esfuerzos en {len(faltantes)} nodos de la malla (p. ej. el nodo {faltantes[0]})."
        )
    return np.array([por_nodo[n] for n in malla.ids_nodos], dtype=float)


def leer_desplazamientos_frd(ruta_frd: Path, malla: MallaTensiones) -> np.ndarray:
    """Desplazamientos nodales (N, 3) en el orden de ``malla.ids_nodos`` (m con el modelo
    en kN y m). Requiere ``*NODE FILE, U`` en el .inp."""

    lineas = ruta_frd.read_text(encoding="utf-8", errors="ignore").splitlines()
    por_nodo: Dict[int, List[float]] = {}
    en_bloque = False
    for linea in lineas:
        if linea.startswith(" -4"):
            en_bloque = "DISP" in linea
            continue
        if linea.startswith(" -3"):
            en_bloque = False
            continue
        if en_bloque and linea.startswith(" -1"):
            por_nodo[int(linea[3 : 3 + _LARGO_ID])] = _parsear_valores(linea, 3)

    if not por_nodo:
        raise ErrorLecturaResultados(f"El archivo {ruta_frd.name} no contiene el bloque de desplazamientos (DISP).")
    faltantes = [n for n in malla.ids_nodos if n not in por_nodo]
    if faltantes:
        raise ErrorLecturaResultados(f"Faltan desplazamientos en {len(faltantes)} nodos de la malla (p. ej. el nodo {faltantes[0]}).")
    return np.array([por_nodo[n] for n in malla.ids_nodos], dtype=float)
