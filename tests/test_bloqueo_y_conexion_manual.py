"""Botón de bloqueo (tras analizar) y «Conectar manualmente» eligiendo las piezas con dos clics."""

import shutil
from pathlib import Path

import pytest

from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.casos_carga import CasoCarga, TipoCaso
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido

from .test_solo_carga import _proyecto_con_bloque
from .test_solidos_libres import cara_hacia

pytest.importorskip("PySide6")


def _ventana():
    from PySide6.QtWidgets import QApplication

    from ifc_structural.gui.ventana_principal import VentanaPrincipal

    QApplication.instance() or QApplication([])
    p, viga, _bloque = _proyecto_con_bloque()
    p.solo_carga = set()
    casos = {"D": CasoCarga("D", "Muerta", TipoCaso.D)}
    v = VentanaPrincipal()
    v._establecer_proyecto(p, [], casos, {}, {}, {}, {})
    return v, p, viga, casos


@pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")
def test_al_analizar_se_bloquea_y_al_desbloquear_se_borran_los_resultados(tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from ifc_structural.core.asignacion import Objetivo

    v, p, viga, casos = _ventana()
    assert not v._bloqueado and not v.accion_bloqueo.isEnabled() and not v.accion_bloqueo.isChecked()
    apoyo = ApoyoSuperficie(id="AP", elemento_tipo="SOLIDO", elemento_id="V", cara=cara_hacia(viga, (-1, 0, 0)))
    p.solo_carga = {("SOLIDO", "B")}
    r = ejecutar_analisis_solido(p, casos, {}, [apoyo], {}, {}, {}, tmp_path, tamano=0.03)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)  # advertencias del análisis
    v._al_terminar_analisis(r)
    assert v._bloqueado and v.accion_bloqueo.isChecked() and v.accion_bloqueo.isEnabled()
    assert v.accion_bloqueo.text() == "Bloqueado"
    assert v.boton_bloqueo.objectName() == "botonBloqueo"  # azul cuando está marcado (estilo :checked)
    for accion in (v.accion_analizar, v.accion_apoyo, v.accion_carga, v.accion_eliminar, v.accion_solo_carga):
        assert not accion.isEnabled()
    avisos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: avisos.append(a))
    v.visor.seleccionar([Objetivo("SOLIDO", "B")])
    v.alternar_solo_carga()  # bloqueado: no cambia nada
    assert avisos and p.solo_carga == {("SOLIDO", "B")}

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    v.accion_bloqueo.trigger()  # no confirma: sigue bloqueado y con resultados
    assert v._bloqueado and v.accion_bloqueo.isChecked() and v.resultado is not None

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    v.accion_bloqueo.trigger()
    assert not v._bloqueado and not v.accion_bloqueo.isChecked() and not v.accion_bloqueo.isEnabled()
    assert v.resultado is None and v.accion_bloqueo.text() == "Desbloqueado"
    assert v.accion_analizar.isEnabled() and v.accion_apoyo.isEnabled()
    v.close()


def test_conectar_manual_sin_seleccion_espera_dos_clics_y_abre_el_dialogo(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from ifc_structural.core.asignacion import Objetivo
    from ifc_structural.gui import ventana_principal as vp

    abiertos = []

    class DialogoFalso:
        def __init__(self, proyecto, a, b, parent=None):
            abiertos.append((a, b))

        def exec(self):
            return False

    monkeypatch.setattr(vp, "DialogoUnionSuperficie", DialogoFalso)
    v, p, _viga, _casos = _ventana()
    v.visor.seleccionar([])
    v.asignar_conexion()  # sin selección: entra en el modo de elegir con clics
    assert v._conexion_pendiente == [] and "PRIMERA" in v.statusBar().currentMessage()
    v.visor.seleccionar([Objetivo("SOLIDO", "V")])  # primer clic
    assert "SEGUNDA" in v.statusBar().currentMessage()
    v.visor.seleccionar([Objetivo("SOLIDO", "V")])  # otro clic en la misma pieza no cuenta
    assert len(v._conexion_pendiente) == 1
    v.visor.seleccionar([Objetivo("SOLIDO", "B")])  # segundo clic (sin Ctrl): se abre el diálogo
    for _ in range(5):
        QApplication.processEvents()
    assert v._conexion_pendiente is None
    assert [(a.id, b.id) for a, b in abiertos] == [("V", "B")]
    assert {o.id for o in v.visor.seleccion} == {"V", "B"}

    v.visor.seleccionar([])
    v.asignar_conexion()
    v._atajo_cancelar_conexion.activated.emit()  # Esc
    assert v._conexion_pendiente is None and "cancelado" in v.statusBar().currentMessage()
    v.visor.seleccionar([Objetivo("SOLIDO", "V")])
    v.visor.seleccionar([Objetivo("SOLIDO", "B")])
    QApplication.processEvents()
    assert len(abiertos) == 1
    v.close()
