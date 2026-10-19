"""Pruebas del parseo de .dat (desplazamientos) y .frd (fuerzas de sección).

Los textos de ejemplo usados aquí son fragmentos reales, verificados a
mano, de la salida de CalculiX 2.21 para una viga en voladizo simple (ver
el docstring de solver/generador_inp.py).
"""

from pathlib import Path

from ifc_structural.solver.lector_resultados import (
    leer_desplazamientos_dat,
    leer_fuerzas_seccion_frd,
)

_DAT_EJEMPLO = """
                        S T E P       1


                                INCREMENT     1


 displacements (vx,vy,vz) for set NALL and time  0.1000000E+01

         1  0.000000E+00  0.000000E+00  0.000000E+00
         2  2.263272E-18 -4.653223E-15 -2.073600E-03

 forces (fx,fy,fz) for set NALL and time  0.1000000E+01

         1 -4.000000E+01  7.857340E-12  1.000000E+01
         2  2.849276E-12 -7.857340E-12 -1.000000E+01
"""

_FRD_EJEMPLO = """    1PSTEP                         1           1           1
  100CL  101 1.000000000           3                     0    1           1
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -5  ALL         1    2    0    0    1ALL
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2-2.07360E-03 0.00000E+00 0.00000E+00
 -3
    1PSTEP                         2           1           1
  100CL  101 1.000000000           3                     0    1           1
 -4  STRESS      6    1
 -5  SXX         1    4    1    1
 -5  SYY         1    4    2    2
 -5  SZZ         1    4    3    3
 -5  SXY         1    4    1    2
 -5  SYZ         1    4    2    3
 -5  SZX         1    4    3    1
 -1         1-9.99960E+00 7.80282E-12 1.80478E-12-2.39012E-12-1.64189E-11-1.99999E+01
 -1         2-9.99960E+00 7.91123E-12 3.82983E-12-2.36777E-12-1.60365E-11-1.99999E+01
 -3
 9999
"""


def test_leer_desplazamientos_dat(tmp_path: Path):
    ruta = tmp_path / "ejemplo.dat"
    ruta.write_text(_DAT_EJEMPLO, encoding="utf-8")

    desplazamientos = leer_desplazamientos_dat(ruta)

    assert desplazamientos[1] == (0.0, 0.0, 0.0)
    assert desplazamientos[2] == (2.263272e-18, -4.653223e-15, -2.073600e-03)


def test_leer_fuerzas_seccion_frd(tmp_path: Path):
    ruta = tmp_path / "ejemplo.frd"
    ruta.write_text(_FRD_EJEMPLO, encoding="utf-8")

    fuerzas = leer_fuerzas_seccion_frd(ruta)

    assert set(fuerzas.keys()) == {1, 2}
    n, v1, v2, t, m1, m2 = fuerzas[1]
    assert abs(n - 1.80478e-12) < 1e-6
    assert abs(v1 - (-9.99960)) < 1e-4
    assert abs(m2 - (-19.9999)) < 1e-3


def test_leer_fuerzas_seccion_frd_ignora_bloque_disp(tmp_path: Path):
    ruta = tmp_path / "ejemplo.frd"
    ruta.write_text(_FRD_EJEMPLO, encoding="utf-8")

    fuerzas = leer_fuerzas_seccion_frd(ruta)

    # El bloque DISP tiene solo 3 valores por línea; si el parser lo
    # confundiera con el bloque STRESS (6 valores), fallaría al parsear o
    # daría resultados sin sentido. Verificamos que use el bloque correcto.
    assert all(len(v) == 6 for v in fuerzas.values())
