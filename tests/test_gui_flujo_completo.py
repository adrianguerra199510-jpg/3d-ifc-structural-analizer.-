"""Prueba de humo (smoke test) de la GUI: construye la ventana principal y
ejercita el flujo completo (importar IFC sintético, apoyos, cargas, casos,
análisis) llamando directamente a los métodos de la ventana, sin simular
clics de mouse (eso requeriría una sesión interactiva real). Se omite si
PySide6/pyvista no están instalados, o si no hay un ejecutable `ccx`
disponible (el análisis real requiere CalculiX).
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from ifc_structural.core.cargas import Apoyo, CargaPuntual
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ifc_structural.ifc.importador import importar_ifc

from .ayuda_ifc import crear_ifc_prueba

pytestmark = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_flujo_completo_importar_cargar_analizar(qapp, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from ifc_structural.gui.ventana_principal import VentanaPrincipal

    # QMessageBox.information/critical/question son modales: en una
    # ejecución automatizada (sin alguien para hacer clic en "OK") se
    # quedarían esperando para siempre. Se reemplazan por no-ops.
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))

    ruta_ifc = tmp_path / "prueba.ifc"
    crear_ifc_prueba(str(ruta_ifc))

    ventana = VentanaPrincipal()
    ventana.configuracion.directorio_trabajo = str(tmp_path / "trabajo_ccx")

    ventana.proyecto = importar_ifc(str(ruta_ifc))
    ventana.visor.mostrar_proyecto(ventana.proyecto)
    ventana.panel_elementos.actualizar(ventana.proyecto)

    columna = next(iter(ventana.proyecto.columnas.values()))
    viga = next(iter(ventana.proyecto.vigas.values()))
    losa = next(iter(ventana.proyecto.losas.values()))

    # El importador no infiere automáticamente qué viga soporta cada lado
    # de la losa (requiere asignación del usuario); para esta prueba de
    # humo del flujo completo, se asigna la única viga del modelo al
    # primer lado del contorno.
    losa.vigas_soporte[0] = viga.id

    # Apoyo empotrado en la base de la columna (nodo con menor Z).
    nodo_base = min(
        (columna.nodo_inicio, columna.nodo_fin),
        key=lambda nid: ventana.proyecto.nodos[nid].z,
    )
    ventana.apoyos.append(Apoyo(id="AP1", nodo_id=nodo_base))

    ventana.casos = {
        "D": CasoCarga(id="D", nombre="Carga muerta", tipo=TipoCaso.D),
        "L": CasoCarga(id="L", nombre="Carga viva", tipo=TipoCaso.L),
    }
    ventana.combinaciones = {
        "C1": Combinacion(id="C1", nombre="1.2D+1.6L", factores={"D": 1.2, "L": 1.6}, norma="ACI 318"),
    }

    nodo_libre_viga = viga.nodo_fin
    ventana.cargas_puntuales_por_caso["L"] = [
        CargaPuntual(id="CP1", caso_id="L", nodo_id=nodo_libre_viga, fz=-10.0)
    ]

    ventana._ejecutar_analisis()

    assert "D" in ventana.resultados_por_caso
    assert "L" in ventana.resultados_por_caso
    assert "C1" in ventana.resultados_por_combinacion

    # El caso D debe producir algún desplazamiento vertical no nulo en la
    # viga (por su propio peso), lo que confirma que el peso propio y la
    # transferencia de la losa realmente se aplicaron.
    nodo_libre_ccx = ventana.malla.nodo_ccx_de_nodo_original[nodo_libre_viga]
    desplazamiento_z = ventana.resultados_por_caso["D"].desplazamientos[nodo_libre_ccx][2]
    assert desplazamiento_z < 0
