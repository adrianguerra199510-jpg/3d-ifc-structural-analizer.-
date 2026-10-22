"""Vista de resultados: colorear por desplazamiento (como la deformada del reporte) y ancho de la ventana."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_el_panel_ofrece_colorear_por_desplazamiento_en_mm(app):
    from ifc_structural.gui.panel_resultados import DESPLAZAMIENTO, PanelEsfuerzos

    panel = PanelEsfuerzos()
    panel.establecer_resultados([("D", "Muerta")], [])
    assert panel.unidad == "MPa"
    panel.combo_magnitud.setCurrentIndex(panel.combo_magnitud.findData(DESPLAZAMIENTO))
    assert panel.parametros().magnitud == DESPLAZAMIENTO and panel.unidad == "mm"
    assert panel.etiqueta_maximo.text() == "Máximo [mm]:" and "[mm]" in panel.tabla.horizontalHeaderItem(2).text()


def test_desplazamiento_total_de_un_caso_y_de_la_envolvente():
    from ifc_structural.gui.panel_resultados import ENVOLVENTE
    from ifc_structural.gui.ventana_principal import VentanaPrincipal

    u1 = np.array([[0.001, 0, 0], [0, 0, 0], [0, 0.0003, 0.0004]])
    u2 = np.array([[0, 0, 0], [0, 0.002, 0], [0, 0, 0]])
    r = SimpleNamespace(desplazamientos_por_caso={"D": u1}, desplazamientos_por_combinacion={"C1": u1, "C2": u2})
    valores, combinacion, _ = VentanaPrincipal._desplazamiento_total(r, "D")
    assert valores == pytest.approx([1.0, 0.0, 0.5]) and combinacion is None
    valores, combinacion, _ = VentanaPrincipal._desplazamiento_total(r, ENVOLVENTE)
    assert valores == pytest.approx([1.0, 2.0, 0.5]) and list(combinacion) == ["C1", "C2", "C1"]
    assert VentanaPrincipal._desplazamiento_total(r, "X")[0] is None


def test_la_barra_de_vista_no_ensancha_la_ventana(app):
    from ifc_structural.gui.ventana_principal import VentanaPrincipal

    v = VentanaPrincipal()
    v.show()
    v.paginas.setCurrentWidget(v.contenedor_visor)
    for d in v._docks:
        d.show()
    app.processEvents()
    # con los dos paneles abiertos cabe en una pantalla de 1366 px (antes pedía 1955 px por los nombres de los botones)
    assert v.minimumSizeHint().width() <= 1366
    v.resize(1366, 800)
    for _ in range(5):
        app.processEvents()
    assert v.width() == 1366 and v.barra_vista.compacta
    v.close()
