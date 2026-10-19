"""Pruebas de la superposición de tensores y de las magnitudes de esfuerzo."""

import numpy as np
import pytest

from ifc_structural.combinaciones import esfuerzos
from ifc_structural.combinaciones.motor_superposicion import (
    calcular_envolvente,
    superponer_combinacion,
    superponer_todas_las_combinaciones,
)
from ifc_structural.core.casos_carga import Combinacion


def _tensor(sxx=0.0, syy=0.0, szz=0.0, sxy=0.0, syz=0.0, szx=0.0):
    return np.array([[sxx, syy, szz, sxy, syz, szx]], dtype=float)


def test_von_mises_traccion_uniaxial_es_el_mismo_esfuerzo():
    assert esfuerzos.von_mises(_tensor(sxx=100.0))[0] == pytest.approx(100.0)


def test_von_mises_cortante_puro_es_raiz_de_3_por_tau():
    assert esfuerzos.von_mises(_tensor(sxy=10.0))[0] == pytest.approx(10.0 * np.sqrt(3.0))


def test_principales_y_cortante_maximo():
    # Estado plano: sxx=100, syy=-60, sxy=0 -> principales 100 y -60 (y 0 en z).
    t = _tensor(sxx=100.0, syy=-60.0)
    s3, s2, s1 = esfuerzos.principales(t)[0]
    assert (s1, s2, s3) == pytest.approx((100.0, 0.0, -60.0))
    assert esfuerzos.cortante_maximo(t)[0] == pytest.approx(80.0)


def test_principales_con_cortante_rotan_el_tensor():
    # sxx=syy=0, sxy=50 -> principales +50 y -50.
    s3, _, s1 = esfuerzos.principales(_tensor(sxy=50.0))[0]
    assert (s1, s3) == pytest.approx((50.0, -50.0))


def test_superponer_combinacion_aplica_factores_al_tensor():
    tensiones = {"D": _tensor(sxx=10.0, sxy=2.0), "L": _tensor(sxx=5.0, sxy=-1.0)}
    combinacion = Combinacion(id="C1", nombre="1.2D+1.6L", factores={"D": 1.2, "L": 1.6})

    resultado = superponer_combinacion(tensiones, combinacion)

    assert resultado[0, 0] == pytest.approx(1.2 * 10.0 + 1.6 * 5.0)
    assert resultado[0, 3] == pytest.approx(1.2 * 2.0 + 1.6 * -1.0)


def test_las_magnitudes_se_calculan_despues_de_combinar_no_antes():
    # Von Mises NO es lineal: vm(D) + vm(L) != vm(D + L) cuando los
    # tensores tienen signos distintos. La combinación debe usar el tensor.
    tensiones = {"D": _tensor(sxx=100.0), "L": _tensor(sxx=-100.0)}
    combinacion = Combinacion(id="C", nombre="D+L", factores={"D": 1.0, "L": 1.0})

    combinado = superponer_combinacion(tensiones, combinacion)

    assert esfuerzos.von_mises(combinado)[0] == pytest.approx(0.0)


def test_superponer_ignora_casos_con_factor_cero_o_inexistentes():
    tensiones = {"D": _tensor(sxx=10.0), "W": _tensor(sxx=1000.0)}
    combinacion = Combinacion(id="C", nombre="Solo D", factores={"D": 1.0, "W": 0.0, "FANTASMA": 5.0})

    assert superponer_combinacion(tensiones, combinacion)[0, 0] == pytest.approx(10.0)


def test_calcular_envolvente_von_mises_toma_el_maximo_y_dice_de_cual_combinacion():
    tensiones = {"D": _tensor(sxx=10.0), "L": _tensor(sxx=5.0)}
    combinaciones = [
        Combinacion(id="C1", nombre="1.4D", factores={"D": 1.4}),
        Combinacion(id="C2", nombre="1.2D+1.6L", factores={"D": 1.2, "L": 1.6}),
    ]
    por_combinacion = superponer_todas_las_combinaciones(tensiones, combinaciones)

    envolvente = calcular_envolvente(por_combinacion, "von_mises")

    assert envolvente.valores[0] == pytest.approx(20.0)  # C2: 1.2*10 + 1.6*5
    assert envolvente.combinacion_por_nodo[0] == "C2"


def test_envolvente_de_compresion_usa_el_minimo():
    tensiones = {"D": _tensor(sxx=-10.0), "L": _tensor(sxx=-5.0)}
    combinaciones = [
        Combinacion(id="C1", nombre="1.4D", factores={"D": 1.4}),
        Combinacion(id="C2", nombre="1.2D+1.6L", factores={"D": 1.2, "L": 1.6}),
    ]
    por_combinacion = superponer_todas_las_combinaciones(tensiones, combinaciones)

    envolvente = calcular_envolvente(por_combinacion, "s3")

    assert envolvente.valores[0] == pytest.approx(-20.0)  # la más comprimida
    assert envolvente.combinacion_por_nodo[0] == "C2"
