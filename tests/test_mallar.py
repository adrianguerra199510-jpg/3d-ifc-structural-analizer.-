"""Botón Mallar: la malla se genera sin resolver y ANALIZAR la reutiliza mientras el modelo no cambie."""

import shutil
from pathlib import Path

import numpy as np
import pytest

import ifc_structural.solido.analisis_solido as modulo
from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.casos_carga import CasoCarga, TipoCaso
from ifc_structural.solido.analisis_solido import celdas_vtk_de_malla, ejecutar_analisis_solido, huella_geometria, mallar_modelo

from .test_solidos_libres import caja, cara_hacia, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def _modelo():
    s = caja("S1", 0, 0, 0, 1.0, 0.1, 0.1)
    p = proyecto_con(s)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="S1", cara=cara_hacia(s, (-1, 0, 0)),
                            ux=True, uy=True, uz=True)
    return p, apoyo


def test_mallar_da_la_malla_y_sus_celdas_para_dibujarla():
    p, apoyo = _modelo()
    previa = mallar_modelo(p, 0.05, [apoyo], barridos=False)
    m = previa.malla
    assert len(m.tetraedros) > 0 and previa.tamano == 0.05 and previa.huella == huella_geometria(p)
    assert "C3D10" in previa.resumen() and "50 mm" in previa.resumen()
    celdas, tipos, es_lamina = celdas_vtk_de_malla(m)
    assert len(tipos) == m.n_elementos and not es_lamina.any()
    assert len(celdas) == 5 * len(m.tetraedros) and set(tipos) == {10}


def test_la_caja_prismatica_se_malla_con_hexaedros_barridos():
    p, apoyo = _modelo()
    previa = mallar_modelo(p, 0.05, [apoyo])
    m = previa.malla
    assert len(m.tetraedros) == 0 and len(m.hexaedros20) > 0 and len(m.hexaedros20) % 20 == 0  # 20 capas de 5 cm
    assert "C3D20R" in previa.resumen() and previa.barridos
    celdas, tipos, _ = celdas_vtk_de_malla(m)
    assert set(tipos) == {12} and len(celdas) == 9 * len(m.hexaedros20)


def test_la_huella_cambia_con_la_geometria():
    p, _ = _modelo()
    h = huella_geometria(p)
    p.solidos["S1"].vertices = np.asarray(p.solidos["S1"].vertices) * 1.01
    assert huella_geometria(p) != h


@requiere_ccx
def test_analizar_reutiliza_la_malla_si_nada_cambio(tmp_path: Path, monkeypatch):
    p, apoyo = _modelo()
    previa = mallar_modelo(p, 0.05, [apoyo])
    llamadas = []
    original = modulo.mallar_modelo
    monkeypatch.setattr(modulo, "mallar_modelo", lambda *a, **k: llamadas.append(1) or original(*a, **k))
    casos = {"D": CasoCarga("D", "Muerta", TipoCaso.D)}

    def analizar(tamano, carpeta):
        return ejecutar_analisis_solido(p, casos, {}, [apoyo], {}, {}, {}, tmp_path / carpeta, tamano=tamano,
                                        malla_previa=previa)

    r = analizar(0.05, "a")
    assert llamadas == [] and len(r.malla_tensiones.coordenadas) > 0
    ejecutar_analisis_solido(p, casos, {}, [apoyo], {}, {}, {}, tmp_path / "t", tamano=0.05, malla_previa=previa,
                             barridos=False)  # con tetraedros: es otra malla
    assert llamadas == [1]
    llamadas.clear()
    analizar(0.04, "b")  # otro tamaño: se vuelve a mallar
    assert llamadas == [1]
    p.solidos["S1"].vertices = np.asarray(p.solidos["S1"].vertices) * 1.01  # el modelo cambió
    analizar(0.05, "c")
    assert llamadas == [1, 1]
