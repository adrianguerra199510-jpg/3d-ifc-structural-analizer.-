"""Elementos «solo carga»: no se mallan; su peso y sus cargas pasan, por equilibrio, a las piezas malladas que tocan."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.casos_carga import CasoCarga, TipoCaso
from ifc_structural.core.modelo import Material
from ifc_structural.core.persistencia import cargar_proyecto, guardar_proyecto
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido, huella_geometria
from ifc_structural.solido.solo_carga import calcular_transferencias, proyecto_mallado, repartir
from ifc_structural.solido.validacion_modelo import ERROR, validar_modelo_solido

from .test_solidos_libres import E, _carga, caja, cara_hacia, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def test_reparto_entre_dos_apoyos_por_la_ley_de_la_palanca():
    # dos contactos iguales en x = 0 y x = 1; la carga W en x = 0.25: 3/4 a la izquierda, 1/4 a la derecha
    rejilla = np.array([[dx, dy, 0.0] for dx in (-0.01, 0.01) for dy in (-0.01, 0.01)])
    puntos = np.vstack([rejilla, rejilla + [1.0, 0, 0]])
    areas = np.ones(len(puntos))
    w = 10.0
    g, falta = repartir(puntos, areas, np.array([0, 0, -w]), np.zeros(3), np.array([0.25, 0, 0]))
    assert falta < 1e-9
    # (cada contacto tiene un área: lleva una parte muy pequeña del momento con su propio par)
    assert -g[:4, 2].sum() == pytest.approx(-0.75 * w, rel=1e-3) and -g[4:, 2].sum() == pytest.approx(-0.25 * w, rel=1e-3)
    assert np.abs(g[:, :2]).max() < 1e-9
    # en una línea no puede equilibrar un vuelco alrededor de ella
    linea = np.array([[x, 0.0, 0.0] for x in np.linspace(0, 1, 5)])
    _, falta = repartir(linea, np.ones(5), np.array([0, 0, -w]), np.zeros(3), np.array([0.5, 0.3, 0]))
    assert falta > 0.1


def _proyecto_con_bloque():
    viga = caja("V", 0, 0, 0, 1.0, 0.1, 0.1)
    bloque = caja("B", 0.9, 0, 0.1, 0.1, 0.1, 0.1)  # apoyado sobre el extremo de la viga
    p = proyecto_con(viga, bloque)
    p.materiales["LIGERO"] = Material(nombre="LIGERO", modulo_elasticidad=E, densidad=0.0, modulo_poisson=0.3)
    p.materiales["PESADO"] = Material(nombre="PESADO", modulo_elasticidad=E, densidad=1e5, modulo_poisson=0.3)
    viga.material, bloque.material = "LIGERO", "PESADO"
    p.solo_carga = {("SOLIDO", "B")}
    return p, viga, bloque


def test_el_proyecto_mallado_no_lleva_los_elementos_solo_carga_y_se_guarda(tmp_path: Path):
    p, viga, bloque = _proyecto_con_bloque()
    m = proyecto_mallado(p)
    assert set(m.solidos) == {"V"} and set(p.solidos) == {"V", "B"}
    antes = huella_geometria(p)
    p.solo_carga = set()
    assert huella_geometria(p) != antes  # marcar o desmarcar cambia la malla
    p.solo_carga = {("SOLIDO", "B")}
    ruta = tmp_path / "p.json"
    guardar_proyecto(ruta, p, [], {}, {}, {}, {}, {})
    assert cargar_proyecto(ruta)[0].solo_carga == {("SOLIDO", "B")}


def test_el_peso_y_las_cargas_del_bloque_pasan_a_la_viga():
    p, viga, bloque = _proyecto_con_bloque()
    casos = {"D": CasoCarga("D", "Muerta", TipoCaso.D), "L": CasoCarga("L", "Viva", TipoCaso.L)}
    cargas = {"L": [_carga(bloque, (0, 0, 1), "FUERZA", 5.0, (1, 0, 0))]}  # 5 kN horizontales sobre el bloque
    r = calcular_transferencias(p, casos, cargas, 0.002)
    assert len(r.grupos) == 1 and r.grupos[0].destinos == [("SOLIDO", "V")]
    peso = 1e5 * 0.001
    total_d = sum(t.fuerzas["D"].sum(axis=0) for t in r.transferencias)
    total_l = sum(t.fuerzas["L"].sum(axis=0) for t in r.transferencias)
    assert total_d == pytest.approx([0, 0, -peso], abs=1e-4 * peso)
    assert total_l == pytest.approx([5.0, 0, 0], abs=1e-4 * 5.0)
    # el momento también: la fuerza horizontal en la cara superior (brazo 0,1 m sobre el contacto) produce un par
    t = r.transferencias[0]
    momento = np.cross(t.puntos - t.puntos.mean(axis=0), t.fuerzas["L"]).sum(axis=0)
    assert momento[1] == pytest.approx(5.0 * 0.1, rel=0.05)


def test_la_validacion_avisa_si_un_elemento_solo_carga_no_toca_nada():
    p, viga, bloque = _proyecto_con_bloque()
    suelto = caja("X", 3, 0, 0, 0.1, 0.1, 0.1)
    p.solidos["X"] = suelto
    p.solo_carga.add(("SOLIDO", "X"))
    apoyo = ApoyoSuperficie(id="AP", elemento_tipo="SOLIDO", elemento_id="V", cara=cara_hacia(viga, (-1, 0, 0)))
    informe = validar_modelo_solido(p, {"D": CasoCarga("D", "Muerta", TipoCaso.D)}, {}, [apoyo], {}, tamano=0.05)
    avisos = [h for h in informe.avisos if h.seccion == "Solo carga"]
    assert len(avisos) == 1 and "X" in avisos[0].texto
    assert not [h for h in informe.errores if h.seccion == "Solo carga"]
    # con un apoyo propio (un anclaje) su peso va a ese apoyo: no hay aviso
    anclaje = ApoyoSuperficie(id="AX", elemento_tipo="SOLIDO", elemento_id="X", cara=0)
    informe = validar_modelo_solido(p, {"D": CasoCarga("D", "Muerta", TipoCaso.D)}, {}, [apoyo, anclaje], {}, tamano=0.05)
    assert not [h for h in informe.avisos if h.seccion == "Solo carga"]
    assert any("apoyos propios" in h.texto for h in informe.hallazgos if h.seccion == "Solo carga")
    assert any("apoyan en V" in h.texto for h in informe.hallazgos if h.seccion == "Solo carga")
    assert not [h for h in informe.errores if h.seccion in ("Conectividad", "Estabilidad")]  # solo cuenta lo mallado


@requiere_ccx
def test_voladizo_con_un_bloque_solo_carga_en_la_punta(tmp_path: Path):
    p, viga, bloque = _proyecto_con_bloque()
    apoyo = ApoyoSuperficie(id="AP", elemento_tipo="SOLIDO", elemento_id="V", cara=cara_hacia(viga, (-1, 0, 0)))
    r = ejecutar_analisis_solido(p, {"D": CasoCarga("D", "Muerta", TipoCaso.D)}, {}, [apoyo], {}, {}, {}, tmp_path,
                                 tamano=0.025)
    assert any("solo carga" in a for a in r.advertencias)
    assert set(r.barra_de_cada_celda()) == {"V"}  # el bloque no se malló
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["D"]
    punta = np.isclose(x[:, 0], 1.0)
    flecha = -float(u[punta, 2].mean())
    # carga puntual P = 100 kN a 0,95 m del empotramiento: δ = P·a²·(3L − a) / (6EI)
    pc, a, largo, inercia = 1e5 * 0.001, 0.95, 1.0, 0.1 * 0.1 ** 3 / 12
    teoria = pc * a ** 2 * (3 * largo - a) / (6 * E * inercia)
    assert flecha == pytest.approx(teoria, rel=0.06)


def test_la_ventana_marca_y_desmarca_elementos_solo_carga():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ifc_structural.core.asignacion import Objetivo
    from ifc_structural.gui.ventana_principal import VentanaPrincipal
    from ifc_structural.gui.visor_3d import OPACIDAD_SOLO_CARGA

    QApplication.instance() or QApplication([])
    p, viga, bloque = _proyecto_con_bloque()
    p.solo_carga = set()
    v = VentanaPrincipal()
    v._establecer_proyecto(p, [], {"D": CasoCarga("D", "Muerta", TipoCaso.D)}, {}, {}, {}, {})
    v.visor.seleccionar([Objetivo("SOLIDO", "B")])
    v.accion_solo_carga.trigger()
    assert v.proyecto.solo_carga == {("SOLIDO", "B")}
    actor = v.visor._visuales[("SOLIDO", "B")].actor
    assert actor.GetProperty().GetOpacity() == pytest.approx(OPACIDAD_SOLO_CARGA)
    v.deshacer()
    assert v.proyecto.solo_carga == set()
    v.rehacer()
    assert v.proyecto.solo_carga == {("SOLIDO", "B")}
    v.visor.seleccionar([Objetivo("SOLIDO", "B")])
    v.alternar_solo_carga()  # ya lo era: se vuelve a mallar
    assert v.proyecto.solo_carga == set()
    v.close()


def test_el_menu_contextual_permite_volver_a_incluir_los_solo_carga_en_el_analisis():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ifc_structural.core.asignacion import Objetivo
    from ifc_structural.gui.ventana_principal import VentanaPrincipal

    QApplication.instance() or QApplication([])
    p, viga, bloque = _proyecto_con_bloque()  # el bloque B es «solo carga»
    v = VentanaPrincipal()
    v._establecer_proyecto(p, [], {"D": CasoCarga("D", "Muerta", TipoCaso.D)}, {}, {}, {}, {})

    def textos():
        return [a.text() for a in v.construir_menu_contextual().actions()]

    assert v.accion_incluir_todos.isEnabled() and "(1)" in v.accion_incluir_todos.text()
    v.visor.seleccionar([Objetivo("SOLIDO", "B")])
    assert "Incluir en el análisis (volver a mallar)" in textos()
    assert not any(t.startswith("Solo carga") for t in textos())
    # selección mezclada: se ofrecen las dos cosas
    v.visor.seleccionar([Objetivo("SOLIDO", "B"), Objetivo("SOLIDO", "V")])
    assert any(t.startswith("Incluir en el análisis (volver a mallar)") for t in textos())
    assert any(t.startswith("Solo carga") for t in textos())
    v.incluir_en_analisis()
    assert v.proyecto.solo_carga == set() and not v.accion_incluir_todos.isEnabled()
    v.deshacer()
    assert v.proyecto.solo_carga == {("SOLIDO", "B")}
    # con la viga seleccionada (o desde el fondo) se puede incluir a todos los «solo carga»
    v.visor.seleccionar([Objetivo("SOLIDO", "V")])
    assert any(t.startswith("Incluir en el análisis todos") for t in textos())
    assert any(a.text().startswith("Incluir en el análisis todos") for a in v.construir_menu_fondo().actions())
    v.incluir_todos_en_analisis()
    assert v.proyecto.solo_carga == set()
    v.close()
