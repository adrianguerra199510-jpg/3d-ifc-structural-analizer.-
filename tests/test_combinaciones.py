"""Pruebas del motor de superposición de combinaciones."""

from ifc_structural.combinaciones.motor_superposicion import (
    calcular_envolvente,
    superponer_combinacion,
    superponer_todas_las_combinaciones,
)
from ifc_structural.core.casos_carga import Combinacion
from ifc_structural.solver.lector_resultados import ResultadosCaso


def _resultados(desplazamiento_z: float, n: float, m2: float) -> ResultadosCaso:
    return ResultadosCaso(
        desplazamientos={1: (0.0, 0.0, desplazamiento_z)},
        fuerzas_seccion={1: (n, 0.0, 0.0, 0.0, 0.0, m2)},
    )


def test_superponer_combinacion_aplica_factores():
    resultados_por_caso = {
        "D": _resultados(desplazamiento_z=-0.01, n=5.0, m2=-20.0),
        "L": _resultados(desplazamiento_z=-0.02, n=2.0, m2=-10.0),
    }
    combinacion = Combinacion(id="C1", nombre="1.2D+1.6L", factores={"D": 1.2, "L": 1.6})

    resultado = superponer_combinacion(resultados_por_caso, combinacion)

    assert resultado.desplazamientos[1][2] == 1.2 * -0.01 + 1.6 * -0.02
    n_comb, *_resto, m2_comb = resultado.fuerzas_seccion[1]
    assert n_comb == 1.2 * 5.0 + 1.6 * 2.0
    assert m2_comb == 1.2 * -20.0 + 1.6 * -10.0


def test_superponer_combinacion_ignora_casos_con_factor_cero():
    resultados_por_caso = {
        "D": _resultados(desplazamiento_z=-0.01, n=5.0, m2=-20.0),
        "W": _resultados(desplazamiento_z=1000.0, n=1000.0, m2=1000.0),
    }
    combinacion = Combinacion(id="C1", nombre="Solo D", factores={"D": 1.0})

    resultado = superponer_combinacion(resultados_por_caso, combinacion)

    assert resultado.desplazamientos[1][2] == -0.01
    assert resultado.fuerzas_seccion[1][0] == 5.0


def test_calcular_envolvente_identifica_maximo_y_minimo():
    combinaciones = [
        Combinacion(id="C1", nombre="C1", factores={"D": 1.4}),
        Combinacion(id="C2", nombre="C2", factores={"D": 1.2, "L": 1.6}),
    ]
    resultados_por_caso = {
        "D": _resultados(desplazamiento_z=-0.01, n=5.0, m2=-20.0),
        "L": _resultados(desplazamiento_z=-0.02, n=2.0, m2=-10.0),
    }

    resultados_por_combinacion = superponer_todas_las_combinaciones(resultados_por_caso, combinaciones)
    envolvente = calcular_envolvente(resultados_por_combinacion)

    # C1: N=1.4*5=7.0 ; C2: N=1.2*5+1.6*2=9.2 -> maximo es C2
    assert envolvente.maximo_de(1, "N") == 9.2
    assert envolvente.combinacion_del_maximo[(1, 0)] == "C2"
    assert envolvente.minimo_de(1, "N") == 7.0
    assert envolvente.combinacion_del_minimo[(1, 0)] == "C1"
