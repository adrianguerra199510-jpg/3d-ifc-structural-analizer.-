"""Pruebas del parseo del .frd: malla expandida y tensor de esfuerzos.

El texto de ejemplo es un fragmento real (dos primeros elementos) de la
salida de CalculiX 2.21 para una viga en voladizo de 0.5 x 0.3 m con una
carga puntual en el extremo (ver solver/generador_inp.py).
"""

from pathlib import Path

import numpy as np
import pytest

from ifc_structural.solver.lector_resultados import (
    ErrorLecturaResultados,
    leer_malla_frd,
    leer_tensiones_frd,
)

_FRD_EJEMPLO = """    1C
    1UUSER
    1UVERSION           Version 2.21
    2C                            12                                     1
 -1         6 0.00000E+00-1.50000E-01-2.50000E-01
 -1         7 0.00000E+00 1.50000E-01-2.50000E-01
 -1         8 0.00000E+00 1.50000E-01 2.50000E-01
 -1         9 0.00000E+00-1.50000E-01 2.50000E-01
 -1        14 1.00000E+00-1.50000E-01-2.50000E-01
 -1        15 1.00000E+00 1.50000E-01-2.50000E-01
 -1        16 1.00000E+00 1.50000E-01 2.50000E-01
 -1        17 1.00000E+00-1.50000E-01 2.50000E-01
 -1        22 2.00000E+00-1.50000E-01-2.50000E-01
 -1        23 2.00000E+00 1.50000E-01-2.50000E-01
 -1        24 2.00000E+00 1.50000E-01 2.50000E-01
 -1        25 2.00000E+00-1.50000E-01 2.50000E-01
 -3
    3C                             2                                     1
 -1         1    1    0    1
 -2         6        14        15         7         9        17        16         8
 -1         2    1    0    1
 -2        14        22        23        15        17        25        24        16
 -3
    1PSTEP                         1           1           1
  100CL  101 1.000000000          12                     0    1           1
 -4  STRESS      6    1
 -5  SXX         1    4    1    1
 -5  SYY         1    4    2    2
 -5  SZZ         1    4    3    3
 -5  SXY         1    4    1    2
 -5  SYZ         1    4    2    3
 -5  SZX         1    4    3    1
 -1         6-2.78649E+03 3.29543E+01 1.34967E+01 6.07351E+00 1.21470E+01-6.66640E+01
 -1         7-2.78649E+03 3.29543E+01 1.34967E+01-6.07351E+00-1.21470E+01-6.66640E+01
 -1         8 2.78649E+03-3.29543E+01-1.34967E+01 6.07351E+00-1.21470E+01-6.66640E+01
 -1         9 2.78649E+03-3.29543E+01-1.34967E+01-6.07351E+00 1.21470E+01-6.66640E+01
 -1        14-2.39490E+03 1.24118E+01 5.08335E+00 8.36102E+00 4.57501E+00-6.66640E+01
 -1        15-2.39490E+03 1.24118E+01 5.08335E+00-8.36102E+00-4.57501E+00-6.66640E+01
 -1        16 2.39490E+03-1.24118E+01-5.08335E+00 8.36102E+00-4.57501E+00-6.66640E+01
 -1        17 2.39490E+03-1.24118E+01-5.08335E+00-8.36102E+00 4.57501E+00-6.66640E+01
 -1        22-1.59999E+03-8.65441E-12-1.93978E-11 1.06485E+01-1.27010E-12-6.66640E+01
 -1        23-1.59999E+03-6.38778E-12-8.81606E-12-1.06485E+01 7.47136E-12-6.66640E+01
 -1        24 1.59999E+03-4.64695E-12 2.50822E-12 1.06485E+01-1.87228E-12-6.66640E+01
 -1        25 1.59999E+03-4.32721E-12-8.84093E-12-1.06485E+01 8.81784E-12-6.66640E+01
 -3
    1PSTEP                         2           1           1
  100CL  101 1.000000000          12                     0    1           1
 -4  ERROR       1    1
 -5  STR(%)      1    1    0    0
 -1         6 1.21612E+02
 -3
 9999
"""


@pytest.fixture()
def ruta_frd(tmp_path: Path) -> Path:
    ruta = tmp_path / "ejemplo.frd"
    ruta.write_text(_FRD_EJEMPLO, encoding="utf-8")
    return ruta


def test_leer_malla_frd(ruta_frd: Path):
    malla = leer_malla_frd(ruta_frd)

    assert malla.n_nodos == 12
    assert list(malla.ids_elementos) == [1, 2]
    assert malla.conectividad.shape == (2, 8)

    # El primer hexaedro usa los nodos 6,14,15,7,9,17,16,8 (índices 0-based
    # dentro de la lista ordenada de nodos).
    nodos_del_primero = [int(malla.ids_nodos[i]) for i in malla.conectividad[0]]
    assert nodos_del_primero == [6, 14, 15, 7, 9, 17, 16, 8]

    # Coordenadas del nodo 8: (0, 0.15, 0.25)
    i8 = int(np.searchsorted(malla.ids_nodos, 8))
    assert malla.coordenadas[i8] == pytest.approx([0.0, 0.15, 0.25])


def test_leer_tensiones_frd_usa_el_bloque_stress(ruta_frd: Path):
    malla = leer_malla_frd(ruta_frd)
    tensor = leer_tensiones_frd(ruta_frd, malla)

    assert tensor.shape == (12, 6)
    # Nodo 6 (fila 0): SXX = -2786.49, SZX = -66.664 (la última columna).
    assert tensor[0, 0] == pytest.approx(-2786.49)
    assert tensor[0, 5] == pytest.approx(-66.664)
    # Los signos pegados ("-2.39490E+03-...") se separan bien por ancho fijo.
    i14 = int(np.searchsorted(malla.ids_nodos, 14))
    assert tensor[i14, 0] == pytest.approx(-2394.90)
    assert tensor[i14, 1] == pytest.approx(12.4118)


def test_sin_bloque_stress_lanza_error(tmp_path: Path):
    ruta = tmp_path / "sin_stress.frd"
    texto = _FRD_EJEMPLO.split("    1PSTEP                         1")[0]
    ruta.write_text(texto, encoding="utf-8")
    malla = leer_malla_frd(ruta)
    with pytest.raises(ErrorLecturaResultados):
        leer_tensiones_frd(ruta, malla)


def test_sin_malla_lanza_error(tmp_path: Path):
    ruta = tmp_path / "vacio.frd"
    ruta.write_text("    1C\n 9999\n", encoding="utf-8")
    with pytest.raises(ErrorLecturaResultados):
        leer_malla_frd(ruta)
