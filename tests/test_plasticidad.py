"""Análisis elastoplástico bilineal (no lineal): material con fluencia, carga en incrementos y combinaciones resueltas
directamente (con plasticidad no se pueden sumar los casos)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ifc_structural.core.materiales import material_desde_valores
from ifc_structural.core.persistencia import cargar_proyecto, guardar_proyecto
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido
from ifc_structural.solido.generador import CargasCaso, ModeloSolido, escribir_caso, escribir_comun
from ifc_structural.solido.malla import construir_malla_solida

from .test_solidos_libres import _carga, caja, cara_hacia, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")

E, FY, ET = 200000.0, 250.0, 2000.0  # MPa (Et = E/100)
A = 0.1 * 0.1  # m²


def _barra():
    s = caja("B", 0, 0, 0, 1.0, 0.1, 0.1)
    p = proyecto_con(s)
    p.materiales["ACERO"] = material_desde_valores("ACERO", E, 0.3, 0.0, FY)
    # extremo −x: Ux en toda la cara (sin impedir la contracción lateral) y el centro fijo en Uy, Uz (pasador)
    cara = cara_hacia(s, (-1, 0, 0))
    apoyos = [ApoyoSuperficie(id="A", elemento_tipo="SOLIDO", elemento_id="B", cara=cara, ux=True, uy=False, uz=False),
              ApoyoSuperficie(id="P", elemento_tipo="SOLIDO", elemento_id="B", cara=cara, ux=False, uy=True, uz=True, pasador=True)]
    return s, p, apoyos


def _alargamiento_teorico(sigma_mpa: float) -> float:
    """Alargamiento (mm) de 1 m de barra con el material bilineal."""

    eps = sigma_mpa / E if sigma_mpa <= FY else FY / E + (sigma_mpa - FY) / ET
    return eps * 1000.0


def test_la_curva_bilineal_y_el_material_se_guardan(tmp_path: Path):
    m = material_desde_valores("A36", E, 0.3, 78.5, FY)
    assert m.es_plastico and m.tangente == pytest.approx(ET * 1000)
    (s0, e0), (s1, e1) = m.curva_plastica()
    assert s0 == pytest.approx(FY * 1000) and e0 == 0
    h = (s1 - s0) / (e1 - e0)  # módulo plástico: da la pendiente Et en la curva esfuerzo–deformación total
    assert 1 / (1 / (E * 1000) + 1 / h) == pytest.approx(ET * 1000)
    assert not material_desde_valores("H", 25000, 0.2, 24).es_plastico
    p = proyecto_con(caja("B", 0, 0, 0, 1, 0.1, 0.1))
    p.materiales["A36"] = m
    ruta = tmp_path / "p.json"
    guardar_proyecto(ruta, p, [], {}, {}, {}, {}, {})
    leido = cargar_proyecto(ruta)[0].materiales["A36"]
    assert leido.fluencia == pytest.approx(FY * 1000) and leido.es_plastico


def test_las_cargas_de_una_combinacion_suman_los_casos_con_sus_factores():
    d = CargasCaso(nodales={1: np.array([0.0, 0.0, -1.0])}, peso_propio=True)
    l_ = CargasCaso(nodales={1: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 2.0, 0.0])})
    c = CargasCaso.combinar({"D": d, "L": l_}, {"D": 1.2, "L": 1.6})
    assert np.allclose(c.nodales[1], [1.6, 0, -1.2]) and np.allclose(c.nodales[2], [0, 3.2, 0])
    assert c.peso_propio and c.factor_peso_propio == pytest.approx(1.2)
    assert not CargasCaso.combinar({"D": d, "L": l_}, {"L": 1.0}).peso_propio


def test_el_inp_lleva_la_plasticidad_solo_en_el_analisis_no_lineal():
    s, p, _ = _barra()
    malla = construir_malla_solida(p, [], 0.1)
    for no_lineal in (False, True):
        modelo = ModeloSolido(malla=malla, uniones=[], apoyos=[], materiales={"ACERO": "ACERO"}, no_lineal=no_lineal)
        texto = "\n".join(escribir_caso(escribir_comun(modelo, p), CargasCaso(peso_propio=True), "L", no_lineal))
        assert ("*PLASTIC" in texto) == no_lineal and ("PEEQ" in texto) == no_lineal
        assert ("0.25,1.0" in texto) == no_lineal  # carga en incrementos


@requiere_ccx
def test_barra_traccionada_mas_alla_de_la_fluencia(tmp_path: Path):
    s, p, apoyos = _barra()
    fuerza = 200.0 * 1000 * A  # 200 MPa: elástico
    casos = {"L": CasoCarga("L", "Viva", TipoCaso.L)}
    combinaciones = {"C": Combinacion("C", "1.5L", {"L": 1.5})}  # 300 MPa: plastifica
    r = ejecutar_analisis_solido(p, casos, combinaciones, apoyos, {}, {}, {}, tmp_path, tamano=0.05, no_lineal=True,
                                 cargas_superficie_por_caso={"L": [_carga(s, (1, 0, 0), "FUERZA", fuerza, (1, 0, 0))]})
    assert r.no_lineal and any("elastoplástico" in a for a in r.advertencias)
    x = r.malla_tensiones.coordenadas
    extremo, centro = np.isclose(x[:, 0], 1.0), (x[:, 0] > 0.3) & (x[:, 0] < 0.7)
    caso = float(r.desplazamientos_por_caso["L"][extremo, 0].mean()) * 1000
    comb = float(r.desplazamientos_por_combinacion["C"][extremo, 0].mean()) * 1000
    assert caso == pytest.approx(_alargamiento_teorico(200.0), rel=0.02)
    assert float(r.plastica_por_caso["L"].max()) == 0.0
    # la combinación se resolvió con su carga (1.5 × 200 = 300 MPa > fy): 26 veces más alargamiento, no 1.5 veces
    assert comb == pytest.approx(_alargamiento_teorico(300.0), rel=0.02)
    peeq = float(np.median(r.plastica_por_combinacion["C"][centro]))
    assert peeq == pytest.approx(_alargamiento_teorico(300.0) / 1000 - 300.0 / E, rel=0.02)
    sx = np.median(r.tensiones_por_combinacion["C"][centro, 0]) / 1000
    assert sx == pytest.approx(300.0, rel=0.02)


@requiere_ccx
def test_sin_materiales_con_fluencia_se_hace_el_analisis_elastico(tmp_path: Path):
    s = caja("B", 0, 0, 0, 1.0, 0.1, 0.1)
    p = proyecto_con(s)
    apoyo = ApoyoSuperficie(id="A", elemento_tipo="SOLIDO", elemento_id="B", cara=cara_hacia(s, (-1, 0, 0)))
    r = ejecutar_analisis_solido(p, {"L": CasoCarga("L", "Viva", TipoCaso.L)}, {}, [apoyo], {}, {}, {}, tmp_path,
                                 tamano=0.1, no_lineal=True,
                                 cargas_superficie_por_caso={"L": [_carga(s, (1, 0, 0), "FUERZA", 10.0, (0, 0, -1))]})
    assert not r.no_lineal and any("ningún material tiene esfuerzo de fluencia" in a for a in r.advertencias)


def test_la_casilla_del_analisis_elastoplastico_y_la_opcion_peeq():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ifc_structural.core.casos_carga import casos_y_combinaciones_por_defecto
    from ifc_structural.gui.dialogo_analizar import DialogoAnalizar
    from ifc_structural.gui.panel_resultados import DEFORMACION_PLASTICA, PanelEsfuerzos

    QApplication.instance() or QApplication([])
    casos, combinaciones = casos_y_combinaciones_por_defecto()
    sin = DialogoAnalizar(casos, combinaciones, plasticos=[], no_lineal=True)
    assert not sin.check_no_lineal.isEnabled() and not sin.no_lineal()
    con = DialogoAnalizar(casos, combinaciones, plasticos=["A36"], no_lineal=True)
    assert con.no_lineal()
    assert DialogoAnalizar(casos, combinaciones).check_no_lineal is None  # modelo de barras
    panel = PanelEsfuerzos()
    panel.establecer_resultados([("D", "Muerta")], [], plastica=True)
    panel.combo_magnitud.setCurrentIndex(panel.combo_magnitud.findData(DEFORMACION_PLASTICA))
    assert panel.unidad == "%"
    panel.establecer_resultados([("D", "Muerta")], [], plastica=False)
    assert panel.combo_magnitud.findData(DEFORMACION_PLASTICA) < 0 and panel.unidad == "MPa"
