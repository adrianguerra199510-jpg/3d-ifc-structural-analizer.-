"""Pruebas de integración: ejecutan ccx de verdad y comprueban los esfuerzos
contra la teoría de vigas y la coherencia de orientación con lo que se
dibuja. Se omiten si no hay un ejecutable `ccx` en el sistema.
"""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.combinaciones import esfuerzos
from ifc_structural.combinaciones.motor_superposicion import calcular_envolvente
from ifc_structural.core.cargas import Apoyo, CargaPuntual
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ifc_structural.core.geometria import ejes_seccion
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.solver.analisis import ErrorAnalisis, ejecutar_analisis

pytestmark = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")

S = 1.0 / math.sqrt(2.0)


def _proyecto_con_barra(a, b, **kwargs) -> Proyecto:
    p = Proyecto("Integración ccx")
    p.agregar_material(Material(nombre="HORMIGON", modulo_elasticidad=2.5e7, densidad=24.0, modulo_poisson=0.2))
    p.agregar_perfil(
        PerfilSeccion(nombre="V30X50", area=0.15, inercia_y=3.125e-4, inercia_z=1.125e-3,
                      inercia_torsional=1e-4, altura=0.5, ancho=0.3)
    )
    p.agregar_nodo(*a, id_="N1")
    p.agregar_nodo(*b, id_="N2")
    p.agregar_viga("N1", "N2", "HORMIGON", "V30X50", id_="B1", **kwargs)
    return p


def _casos():
    return (
        {"D": CasoCarga("D", "Carga muerta", TipoCaso.D), "L": CasoCarga("L", "Carga viva", TipoCaso.L)},
        {"C1": Combinacion("C1", "1.2D+1.6L", {"D": 1.2, "L": 1.6})},
    )


def _analizar(p: Proyecto, tmp_path: Path, carga_puntual: CargaPuntual):
    casos, combinaciones = _casos()
    return ejecutar_analisis(
        p, casos, combinaciones, [Apoyo(id="AP1", nodo_id="N1")], {}, {},
        {"L": [carga_puntual]}, tmp_path,
    )


def test_voladizo_con_carga_puntual_coincide_con_la_teoria_de_vigas(tmp_path: Path):
    p = _proyecto_con_barra((0, 0, 0), (4, 0, 0))
    resultado = _analizar(p, tmp_path, CargaPuntual(id="CP1", caso_id="L", nodo_id="N2", fz=-10.0))

    x = resultado.malla_tensiones.coordenadas[:, 0]
    sxx = resultado.tensiones_por_caso["L"][:, 0]
    inercia = 0.3 * 0.5**3 / 12.0
    for x_i in (0.5, 1.0, 2.0, 3.0):
        teorico = 10.0 * (4.0 - x_i) * 0.25 / inercia  # M·c/I en kPa
        mayor_abs = np.abs(sxx[np.isclose(x, x_i, atol=0.02)]).max()
        assert mayor_abs == pytest.approx(teorico, rel=0.03), f"x = {x_i} m"


def test_la_combinacion_supera_a_cada_caso_por_separado_y_la_envolvente_la_encuentra(tmp_path: Path):
    p = _proyecto_con_barra((0, 0, 0), (4, 0, 0))
    resultado = _analizar(p, tmp_path, CargaPuntual(id="CP1", caso_id="L", nodo_id="N2", fz=-10.0))

    vm_d = esfuerzos.von_mises(resultado.tensiones_por_caso["D"]).max()  # peso propio
    vm_l = esfuerzos.von_mises(resultado.tensiones_por_caso["L"]).max()
    vm_c = esfuerzos.von_mises(resultado.tensiones_por_combinacion["C1"]).max()
    assert vm_c > max(vm_d, vm_l)

    envolvente = calcular_envolvente(resultado.tensiones_por_combinacion, "von_mises")
    assert envolvente.valores.max() == pytest.approx(vm_c)
    assert set(envolvente.combinacion_por_nodo) == {"C1"}


@pytest.mark.parametrize(
    "a, b, kwargs",
    [
        ((0, 0, 0), (4, 0, 0), {}),  # horizontal
        ((0, 0, 0), (0, 0, 3), {}),  # columna vertical
        ((0, 0, 0), (3, 3, 0), {}),  # diagonal horizontal
        ((0, 0, 0), (3, 0, 1.7320508), {}),  # inclinada 30°
        ((0, 0, 0), (3, 0, 3), {"vector_n1": (-S, 0, S)}),  # 45° exactos: caso que rompía ccx
    ],
)
def test_la_malla_de_esfuerzos_coincide_con_el_solido_que_se_dibuja(tmp_path: Path, a, b, kwargs):
    """Los nodos expandidos por CalculiX deben estar sobre el prisma que
    calcula core.geometria (y que dibuja el visor)."""

    p = _proyecto_con_barra(a, b, **kwargs)
    resultado = _analizar(p, tmp_path, CargaPuntual(id="CP1", caso_id="L", nodo_id="N2", fx=0.5, fy=0.5, fz=-1.0))

    ejes = ejes_seccion(p, p.vigas["B1"])
    origen = np.array(a, dtype=float)
    relativas = resultado.malla_tensiones.coordenadas - origen
    a_lo_largo = relativas @ ejes.t
    en_e1 = relativas @ ejes.e1
    en_e2 = relativas @ ejes.e2

    longitud = np.linalg.norm(np.array(b, dtype=float) - origen)
    assert a_lo_largo.min() == pytest.approx(0.0, abs=1e-4)
    assert a_lo_largo.max() == pytest.approx(longitud, abs=1e-4)
    assert np.abs(en_e1).max() == pytest.approx(ejes.dim1 / 2.0, abs=1e-4)
    assert np.abs(en_e2).max() == pytest.approx(ejes.dim2 / 2.0, abs=1e-4)


def test_sin_apoyos_el_analisis_se_niega_con_un_mensaje_claro(tmp_path: Path):
    p = _proyecto_con_barra((0, 0, 0), (4, 0, 0))
    casos, combinaciones = _casos()
    with pytest.raises(ErrorAnalisis, match="apoyos"):
        ejecutar_analisis(p, casos, combinaciones, [], {}, {}, {}, tmp_path)
