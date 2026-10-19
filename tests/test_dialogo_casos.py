"""Formulario de casos de carga y combinaciones (sin necesidad de OpenGL)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from ifc_structural.core.casos_carga import TipoCaso, casos_y_combinaciones_por_defecto
from ifc_structural.gui.dialogo_casos import DialogoCasos, formula


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def avisos(monkeypatch):
    mensajes = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: mensajes.append(a[2])))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: mensajes.append(a[2])))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    return mensajes


@pytest.fixture
def dialogo(app, avisos):
    casos, combinaciones = casos_y_combinaciones_por_defecto()
    return DialogoCasos(casos, combinaciones, casos_con_cargas={"L"})


def crear_caso(d, id_, nombre, tipo):
    d._pedir_nuevo_caso()
    d.edit_caso_id.setText(id_)
    d.edit_caso_nombre.setText(nombre)
    d.combo_caso_tipo.setCurrentIndex(d.combo_caso_tipo.findText(tipo.value))
    return d._guardar_caso()


def test_crear_caso_nuevo(dialogo):
    assert crear_caso(dialogo, "W", "Viento en X", TipoCaso.W)
    casos, _ = dialogo.resultado()
    assert list(casos) == ["D", "L", "W"]
    assert casos["W"].nombre == "Viento en X" and casos["W"].tipo == TipoCaso.W
    assert dialogo.lista_casos.count() == 3
    assert not dialogo.edit_caso_id.isEnabled()  # el id no se cambia después de guardar


def test_identificador_invalido_o_repetido(dialogo, avisos):
    assert not crear_caso(dialogo, "carga viento", "x", TipoCaso.W)
    assert "identificador" in avisos[-1].lower()
    dialogo._nuevo_caso()  # descarta el formulario inválido
    assert not crear_caso(dialogo, "D", "otra", TipoCaso.D)
    assert "Ya existe" in avisos[-1]
    assert list(dialogo.resultado()[0]) == ["D", "L"]


def test_crear_combinacion_con_factores(dialogo):
    crear_caso(dialogo, "W", "Viento", TipoCaso.W)
    dialogo.pestanas.setCurrentIndex(1)
    dialogo._pedir_nueva_combinacion()
    assert dialogo.edit_comb_id.text() == "COMB3"
    dialogo.edit_comb_nombre.setText("Con viento")
    dialogo._spins["D"].setValue(1.2)
    dialogo._spins["L"].setValue(1.0)
    dialogo._spins["W"].setValue(-1.0)
    assert dialogo.etiqueta_formula.text() == "Resulta: 1.2·D + 1·L − 1·W"
    assert dialogo._guardar_combinacion()
    _, combinaciones = dialogo.resultado()
    assert combinaciones["COMB3"].factores == {"D": 1.2, "L": 1.0, "W": -1.0}
    assert combinaciones["COMB3"].nombre == "Con viento"
    assert dialogo.lista_combinaciones.count() == 3


def test_combinacion_vacia_se_rechaza(dialogo, avisos):
    dialogo.pestanas.setCurrentIndex(1)
    dialogo._pedir_nueva_combinacion()
    assert not dialogo._guardar_combinacion()
    assert avisos and "factor" in avisos[0]


def test_editar_combinacion_existente(dialogo):
    dialogo.pestanas.setCurrentIndex(1)
    dialogo.lista_combinaciones.setCurrentRow(1)  # COMB2 = 1.2D + 1.6L
    assert dialogo._spins["L"].value() == pytest.approx(1.6)
    dialogo._spins["L"].setValue(1.5)
    assert dialogo._guardar_combinacion()
    assert dialogo.resultado()[1]["COMB2"].factores == {"D": 1.2, "L": 1.5}


def test_plantilla(dialogo):
    dialogo.pestanas.setCurrentIndex(1)
    dialogo._pedir_nueva_combinacion()
    dialogo.combo_plantilla.setCurrentText("ACI318_1.2D_1.6L")
    dialogo._aplicar_plantilla()
    assert dialogo._spins["D"].value() == pytest.approx(1.2)
    assert dialogo._spins["L"].value() == pytest.approx(1.6)
    assert dialogo._guardar_combinacion()


def test_plantilla_con_caso_inexistente_avisa(dialogo, avisos):
    dialogo.pestanas.setCurrentIndex(1)
    dialogo._pedir_nueva_combinacion()
    dialogo.combo_plantilla.setCurrentText("ACI318_1.2D_1.0L_1.0W")
    dialogo._aplicar_plantilla()
    assert avisos and "W" in avisos[0]
    assert dialogo._spins["D"].value() == pytest.approx(1.2)


def test_eliminar_caso_lo_quita_de_las_combinaciones(dialogo):
    dialogo.lista_casos.setCurrentRow(1)  # L
    dialogo._eliminar_caso()
    casos, combinaciones = dialogo.resultado()
    assert list(casos) == ["D"]
    assert "L" not in combinaciones["COMB2"].factores
    assert dialogo.casos_eliminados() == {"L"}


def test_cancelar_no_modifica_los_originales(app, avisos):
    casos, combinaciones = casos_y_combinaciones_por_defecto()
    d = DialogoCasos(casos, combinaciones)
    crear_caso(d, "W", "Viento", TipoCaso.W)
    d.pestanas.setCurrentIndex(1)
    d.lista_combinaciones.setCurrentRow(0)
    d._spins["D"].setValue(9.0)
    d._guardar_combinacion()
    d.reject()
    assert list(casos) == ["D", "L"]
    assert combinaciones["COMB1"].factores == {"D": 1.4}


def test_cambios_sin_guardar_se_guardan_al_aceptar(dialogo):
    dialogo.edit_caso_nombre.setText("Peso propio y acabados")  # caso D seleccionado
    dialogo.accept()
    assert dialogo.resultado()[0]["D"].nombre == "Peso propio y acabados"


def test_formula():
    casos, combinaciones = casos_y_combinaciones_por_defecto()
    assert formula(combinaciones["COMB2"], casos) == "1.2·D + 1.6·L"
    assert formula(combinaciones["COMB1"], casos) == "1.4·D"
