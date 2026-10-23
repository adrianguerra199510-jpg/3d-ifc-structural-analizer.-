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


def test_el_plano_de_la_seccion_recorre_el_modelo():
    from ifc_structural.gui.visor_3d import Seccion

    puntos = np.array([[0.0, 0, 0], [2.0, 1, 3]])
    n, o = Seccion(activa=True, eje="X", posicion=0.25).plano(puntos)
    assert np.allclose(n, [1, 0, 0]) and o[0] == pytest.approx(0.5)
    n, o = Seccion(activa=True, eje="Z", posicion=1.0).plano(puntos)
    assert o[2] == pytest.approx(3.0)
    libre = Seccion(activa=True, eje="LIBRE", normal_libre=(1.0, 1.0, 0.0))
    assert np.allclose(libre.normal, np.array([1, 1, 0]) / np.sqrt(2))
    assert Seccion.posicion_de(puntos, (0, 0, 1), (0, 0, 1.5)) == pytest.approx(0.5)


def test_la_seccion_corta_la_malla_y_colorea_la_cara_con_valores_interpolados(app):
    import pyvista as pv

    from ifc_structural.gui.panel_resultados import PanelEsfuerzos
    from ifc_structural.gui.ventana_principal import VentanaPrincipal

    v = VentanaPrincipal()
    visor = v.visor
    grid = pv.ImageData(dimensions=(11, 11, 11), spacing=(0.1, 0.1, 0.1)).cast_to_unstructured_grid()
    grid.point_data["esfuerzo"] = grid.points[:, 2] * 10.0  # campo lineal en z
    visor._puntos_sin_deformar = np.asarray(grid.points)
    panel = PanelEsfuerzos()
    panel.check_seccion.setChecked(True)
    panel.combo_eje.setCurrentIndex(panel.combo_eje.findData("Z"))
    panel.slider_seccion.setValue(250)
    visor._seccion = panel.seccion()
    cortado = visor._recortar(grid)
    zs = np.asarray(cortado.points)[:, 2]
    # queda el lado opuesto a la normal (la cara del corte mira a la cámara en la vista isométrica), cortado en z = 0.25
    assert zs.min() == pytest.approx(0.0) and zs.max() == pytest.approx(0.25)
    panel.boton_invertir.setChecked(True)
    visor._seccion = panel.seccion()
    zs = np.asarray(visor._recortar(grid).points)[:, 2]
    assert zs.min() == pytest.approx(0.25) and zs.max() == pytest.approx(1.0)
    panel.combo_modo_seccion.setCurrentIndex(1)  # solo la cara: valores interpolados en el plano
    visor._seccion = panel.seccion()
    cara = visor._recortar(grid)
    assert np.allclose(cara.points[:, 2], 0.25) and np.allclose(cara.point_data["esfuerzo"], 2.5)
    panel.check_seccion.setChecked(False)
    visor._seccion = panel.seccion()
    assert visor._recortar(grid) is grid
    v.close()
