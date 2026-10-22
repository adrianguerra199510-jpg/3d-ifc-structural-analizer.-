"""Peso propio en los casos de carga muerta y elección de los casos que se analizan."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.casos_carga import (CasoCarga, Combinacion, TipoCaso, casos_con_peso_propio, combinaciones_calculables,
                                              casos_y_combinaciones_por_defecto)
from ifc_structural.core.persistencia import cargar_proyecto, guardar_proyecto
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido

from .test_solidos_libres import caja, cara_hacia, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def test_solo_los_casos_de_carga_muerta_marcados_llevan_el_peso_propio():
    casos = {
        "D": CasoCarga("D", "Muerta", TipoCaso.D),
        "SD": CasoCarga("SD", "Sobreimpuesta", TipoCaso.D, peso_propio=False),
        "L": CasoCarga("L", "Viva", TipoCaso.L, peso_propio=True),  # la casilla no cuenta fuera de los casos D
    }
    assert casos_con_peso_propio(casos) == ["D"]
    assert casos_con_peso_propio(casos_y_combinaciones_por_defecto()[0]) == ["D"]


def test_combinaciones_que_se_pueden_calcular_con_los_casos_elegidos():
    combinaciones = {"C1": Combinacion("C1", "1.4D", {"D": 1.4}), "C2": Combinacion("C2", "1.2D+1.6L", {"D": 1.2, "L": 1.6}),
                     "C3": Combinacion("C3", "D (L con factor 0)", {"D": 1.0, "L": 0.0})}
    assert set(combinaciones_calculables(combinaciones, ["D"])) == {"C1", "C3"}
    assert set(combinaciones_calculables(combinaciones, ["D", "L"])) == {"C1", "C2", "C3"}
    assert combinaciones_calculables(combinaciones, ["L"]) == {}


def test_la_casilla_se_guarda_y_los_archivos_antiguos_la_llevan_en_el_primer_caso_d(tmp_path: Path):
    p = proyecto_con()
    casos = {"D": CasoCarga("D", "Muerta", TipoCaso.D, peso_propio=False), "D2": CasoCarga("D2", "Otra", TipoCaso.D)}
    ruta = tmp_path / "p.json"
    guardar_proyecto(ruta, p, [], casos, {}, {}, {}, {})
    leidos = cargar_proyecto(ruta)[2]
    assert not leidos["D"].peso_propio and leidos["D2"].peso_propio
    # un archivo guardado antes de la casilla: el peso propio iba solo en el primer caso de carga muerta
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    for c in datos["casos"]:
        c.pop("peso_propio")
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    leidos = cargar_proyecto(ruta)[2]
    assert casos_con_peso_propio(leidos) == ["D"]


@requiere_ccx
def test_sin_la_casilla_el_caso_de_carga_muerta_no_lleva_peso_propio(tmp_path: Path):
    s = caja("S1", 0, 0, 0, 1.0, 0.1, 0.1)
    p = proyecto_con(s)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="S1", cara=cara_hacia(s, (-1, 0, 0)),
                            ux=True, uy=True, uz=True)
    casos = {"D": CasoCarga("D", "Muerta", TipoCaso.D), "SD": CasoCarga("SD", "Sobreimpuesta", TipoCaso.D, peso_propio=False)}
    r = ejecutar_analisis_solido(p, casos, {}, [apoyo], {}, {}, {}, tmp_path, tamano=0.05)
    assert np.abs(r.desplazamientos_por_caso["D"]).max() > 0  # peso propio: el voladizo baja
    assert np.abs(r.desplazamientos_por_caso["SD"]).max() < 1e-12  # sin cargas ni peso propio


def test_el_formulario_de_casos_tiene_la_casilla_solo_para_carga_muerta():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ifc_structural.gui.dialogo_casos import DialogoCasos

    QApplication.instance() or QApplication([])
    casos, combinaciones = casos_y_combinaciones_por_defecto()
    d = DialogoCasos(casos, combinaciones, casos_con_cargas=set())
    d._pedir_nuevo_caso()
    d.edit_caso_id.setText("SD")
    d.combo_caso_tipo.setCurrentIndex(d.combo_caso_tipo.findText(TipoCaso.L.value))
    assert not d.check_peso_propio.isEnabled()
    d.combo_caso_tipo.setCurrentIndex(d.combo_caso_tipo.findText(TipoCaso.D.value))
    assert d.check_peso_propio.isEnabled() and d.check_peso_propio.isChecked()  # marcada por defecto
    d.check_peso_propio.setChecked(False)
    assert d._guardar_caso()
    nuevos, _ = d.resultado()
    assert nuevos["SD"].tipo == TipoCaso.D and not nuevos["SD"].peso_propio and nuevos["D"].peso_propio
    assert "peso propio" in d.lista_casos.item(0).text() and "peso propio" not in d.lista_casos.item(2).text()


def test_el_dialogo_de_analizar_elige_casos_y_dice_que_combinaciones_salen():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ifc_structural.gui.dialogo_analizar import DialogoAnalizar

    QApplication.instance() or QApplication([])
    casos, combinaciones = casos_y_combinaciones_por_defecto()
    d = DialogoAnalizar(casos, combinaciones, texto_malla="Malla: prueba", elegidos=["D"])
    assert d.casos_elegidos() == ["D"] and set(d.combinaciones_elegidas()) == {"COMB1"}
    assert "Sin calcular" in d.combinaciones.text() and "COMB2" in d.combinaciones.text()
    d._marcar_todos(Qt.Unchecked)
    assert d.casos_elegidos() == [] and not d.botones.button(d.botones.StandardButton.Ok).isEnabled()
    d._marcar_todos(Qt.Checked)
    assert set(d.combinaciones_elegidas()) == {"COMB1", "COMB2"}
