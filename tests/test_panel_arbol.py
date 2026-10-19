"""Panel único del modelo (árbol): estructura, selección, filtro y eliminación."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ifc_structural.core.cargas import Apoyo, ApoyoArea, CargaArea, CargaLineal, CargaPuntual, Conexion
from ifc_structural.core.casos_carga import casos_y_combinaciones_por_defecto
from ifc_structural.gui.panel_arbol import PanelArbol
from ifc_structural.gui.panel_resultados import ENVOLVENTE

from .test_asignacion import p  # noqa: F401  (proyecto de ejemplo)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Resultado:  # solo lo que usa el árbol
    tensiones_por_caso = {"D": None, "L": None}
    tensiones_por_combinacion = {"COMB1": None, "COMB2": None}


@pytest.fixture
def panel(app, p):
    casos, combinaciones = casos_y_combinaciones_por_defecto()
    arbol = PanelArbol()
    apoyos = [Apoyo(id="AP1", nodo_id="N0"), ApoyoArea(id="AP2", elemento_tipo="VIGA", elemento_id="B1", uz=True)]
    conexiones = [Conexion(id="CX1", nodo_id="N1", barra_id="B1")]
    lineales = {"D": [CargaLineal(id="C1", caso_id="D", elemento_id="B1", w_inicio=5, w_fin=5, nombre="Viga B1: uniforme 5 kN/m")]}
    areas = {"L": [CargaArea(id="C2", caso_id="L", presion=3.0, losa_id="L1", nombre="Losa L1 — cara superior: 3 kPa")]}
    puntuales = {"L": [CargaPuntual(id="C3", caso_id="L", nodo_id="N2", fz=-4, nombre="Viga B1: 4 kN (fin)")]}
    arbol.actualizar(p, apoyos, conexiones, casos, combinaciones, areas, lineales, puntuales, _Resultado())
    arbol.resize(400, 600)
    return arbol


def _items(panel):
    resultado = {}

    def recorrer(padre):
        for i in range(padre.childCount()):
            item = padre.child(i)
            resultado.setdefault(item.data(0, Qt.UserRole), []).append(item)
            recorrer(item)

    recorrer(panel.arbol.invisibleRootItem())
    return resultado


def test_estructura_del_arbol(panel):
    items = _items(panel)
    grupos = [i.text(0) for i in panel.arbol.topLevelItem(0).takeChildren()] if False else [
        panel.arbol.topLevelItem(0).child(k).text(0) for k in range(panel.arbol.topLevelItem(0).childCount())
    ]
    assert grupos == ["Elementos", "Apoyos (2)", "Conexiones (1)", "Cargas (3)", "Materiales (1)",
                      "Casos de carga (2)", "Combinaciones (2)", "Resultados"]
    assert ("ELEMENTO", "VIGA", "B1") in items and ("ELEMENTO", "COLUMNA", "C1") in items
    assert ("ELEMENTO", "LOSA", "L1") in items and ("ELEMENTO", "MURO", "M1") in items
    assert ("MATERIAL", "H") in items and ("CASO", "D") in items and ("COMBINACION", "COMB2") in items
    assert ("RESULTADO", ENVOLVENTE) in items and ("RESULTADO", "COMB1") in items and ("RESULTADO", "L") in items


def test_las_asignaciones_cuelgan_de_su_elemento_y_del_grupo(panel):
    items = _items(panel)
    # La carga lineal C1 aparece bajo la viga B1 y bajo el caso D
    ocurrencias = items[("ASIGNACION", "Carga lineal", "C1")]
    assert len(ocurrencias) == 2
    padres = {o.parent().text(0) for o in ocurrencias}
    assert any(t.startswith("B1") for t in padres) and any(t.startswith("Caso D") for t in padres)
    # La carga de área está bajo la losa
    assert any(o.parent().text(0).startswith("L1") for o in items[("ASIGNACION", "Carga de área", "C2")])
    # La viga B1 muestra cuántas asignaciones tiene
    assert "(" in items[("ELEMENTO", "VIGA", "B1")][0].text(1)


def test_seleccionar_en_el_arbol_emite_los_elementos(panel):
    recibidos = []
    panel.elementos_seleccionados.connect(recibidos.append)
    _items(panel)[("ELEMENTO", "COLUMNA", "C1")][0].setSelected(True)
    assert recibidos[-1] == [("COLUMNA", "C1")]
    panel.arbol.clearSelection()
    _items(panel)[("ASIGNACION", "Conexión", "CX1")][0].setSelected(True)  # una asignación selecciona su elemento
    assert recibidos[-1] == [("VIGA", "B1")]


def test_resaltar_desde_el_visor_no_emite(panel):
    recibidos = []
    panel.elementos_seleccionados.connect(recibidos.append)
    panel.resaltar([("VIGA", "B1"), ("LOSA", "L1")])
    assert not recibidos
    assert {c[1:] for c in panel.claves_seleccionadas()} == {("VIGA", "B1"), ("LOSA", "L1")}


def test_filtro_deja_solo_coincidencias_y_sus_padres(panel):
    panel.arbol.collapseAll()  # también con las ramas contraídas
    panel.buscador.setText("columna")
    claves = set(_items(panel))
    assert ("ELEMENTO", "COLUMNA", "C1") in claves
    assert ("ELEMENTO", "VIGA", "B1") not in claves
    assert ("GRUPO", "Elementos") in claves  # los padres se conservan
    panel.buscador.setText("carga")
    claves = set(_items(panel))
    assert ("GRUPO", "Cargas") in claves and ("ASIGNACION", "Carga lineal", "C1") in claves
    assert ("GRUPO", "Materiales") not in claves
    panel.buscador.setText("")
    assert ("ELEMENTO", "VIGA", "B1") in set(_items(panel))  # sin filtro vuelve todo


def test_supr_elimina_asignaciones_seleccionadas(panel, app):
    pedidos = []
    panel.eliminar_solicitado.connect(pedidos.append)
    panel.arbol.clearSelection()
    _items(panel)[("ASIGNACION", "Apoyo de área", "AP2")][0].setSelected(True)
    QTest.keyClick(panel, Qt.Key_Delete)
    assert pedidos == [[("Apoyo de área", "AP2")]]


def test_doble_clic_en_un_resultado_lo_pide(panel):
    pedidos = []
    panel.resultado_solicitado.connect(pedidos.append)
    item = _items(panel)[("RESULTADO", "COMB1")][0]
    panel._al_doble_clic(item, 0)
    assert pedidos == ["COMB1"]


def test_el_estado_expandido_se_conserva_al_actualizar(panel, p):
    items = _items(panel)
    item = items[("GRUPO", "Materiales")][0]
    item.setExpanded(True)
    casos, combinaciones = casos_y_combinaciones_por_defecto()
    panel.actualizar(p, [], [], casos, combinaciones, {}, {}, {}, None)
    assert _items(panel)[("GRUPO", "Materiales")][0].isExpanded()
    assert ("GRUPO", "Resultados") not in _items(panel)
