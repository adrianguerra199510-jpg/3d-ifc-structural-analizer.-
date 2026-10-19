"""Guion que ejercita la ventana principal con clics de mouse reales.

No es una prueba de pytest: lo lanza ``test_gui_flujo_completo.py`` en un
subproceso bajo ``xvfb-run`` (con OpenGL por software), porque el visor 3D
necesita un contexto OpenGL que la plataforma Qt «offscreen» no ofrece.
Uso: ``python -m tests.guion_gui_xvfb <directorio_salida>``. Termina con
código 0 si todo salió como se esperaba.
"""

import sys
import time
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
from vtkmodules.vtkRenderingCore import vtkCoordinate

from ifc_structural.core.asignacion import Objetivo

from .ayuda_ifc import crear_ifc_portico


def verificar(condicion: bool, mensaje: str) -> None:
    if not condicion:
        print(f"FALLO: {mensaje}")
        sys.exit(1)


def main(salida: Path) -> None:
    app = QApplication([])
    from ifc_structural.gui.estilo import aplicar_estilo

    aplicar_estilo(app)
    mensajes = []
    for nombre in ("information", "warning", "critical"):
        setattr(QMessageBox, nombre, staticmethod(lambda *a, _n=nombre, **k: mensajes.append((_n, a[1:3])) or QMessageBox.Ok))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QDialog.exec = lambda self: 1  # los diálogos se aceptan con sus valores por defecto

    from ifc_structural.gui.ventana_principal import VentanaPrincipal

    def bombear(n=15):
        for _ in range(n):
            app.processEvents()

    ruta_ifc = salida / "portico.ifc"
    crear_ifc_portico(str(ruta_ifc))
    ventana = VentanaPrincipal()
    ventana.resize(1400, 850)
    ventana.show()
    bombear()
    verificar(ventana.importar_archivo(str(ruta_ifc)), "importar el IFC")
    visor = ventana.visor
    visor.render()
    bombear()

    def a_pantalla(x, y, z):
        c = vtkCoordinate()
        c.SetCoordinateSystemToWorld()
        c.SetValue(x, y, z)
        dx, dy = c.GetComputedDisplayValue(visor.renderer)
        return int(dx), int(dy)

    def clic(x, y, modificador=Qt.NoModifier):
        QTest.mouseClick(visor, Qt.LeftButton, modificador, QPoint(x, visor.height() - 1 - y))
        bombear()

    # 1) Clic sobre una columna en modo «Elemento»
    x, y = a_pantalla(4.0, -0.2, 1.5)
    clic(x, y)
    verificar(visor.seleccion == [Objetivo("COLUMNA", "C2")], f"selección de columna: {visor.seleccion}")

    # 2) Apoyos: vista inferior, modo «Cara», Ctrl+clic en las cuatro bases
    ventana.accion_modo_cara.trigger()
    visor.set_visibilidad("LOSA", False)
    visor.view_xy(negative=True)
    visor.reset_camera()
    bombear()
    visor.render()
    bombear()
    for i, (px, py) in enumerate([(0, 0), (4, 0), (4, 6), (0, 6)]):
        x, y = a_pantalla(px, py, 0.0)
        clic(x, y, Qt.NoModifier if i == 0 else Qt.ControlModifier)
    verificar(len(visor.seleccion) == 4 and all(o.cara == 0 for o in visor.seleccion), f"bases: {visor.seleccion}")
    ventana.asignar_apoyo()
    verificar(len(ventana.apoyos) == 4, "cuatro apoyos creados")
    verificar(all(a.grados_restringidos() == [1, 2, 3, 4, 5, 6] for a in ventana.apoyos), "apoyos empotrados")

    # 3) Carga de área sobre la cara superior de la losa
    visor.set_visibilidad("LOSA", True)
    visor.vista_isometrica()
    visor.reset_camera()
    bombear()
    visor.render()
    bombear()
    x, y = a_pantalla(2.0, 3.0, 3.0)
    clic(x, y)
    verificar(visor.seleccion == [Objetivo("LOSA", "L1", 1)], f"cara superior de la losa: {visor.seleccion}")
    ventana.asignar_carga()
    cargas = ventana.cargas_area_por_caso.get("D", [])
    verificar(len(cargas) == 1 and cargas[0].direccion[2] < 0 and cargas[0].presion > 0, f"carga de área: {cargas}")
    salida_asignaciones = salida / "asignado.png"
    ventana.grab().save(str(salida_asignaciones))

    # 4) Botón ANALIZAR
    verificar(ventana.accion_analizar.isEnabled(), "el botón ANALIZAR está habilitado")
    mensajes.clear()
    ventana.accion_analizar.trigger()
    inicio = time.time()
    while ventana._worker.isRunning() and time.time() - inicio < 150:
        bombear(5)
        time.sleep(0.05)
    bombear(30)
    verificar(ventana.resultado is not None, f"el análisis terminó ({mensajes})")
    verificar(not [m for m in mensajes if m[0] == "critical"], f"sin errores: {mensajes}")

    # 5) Se muestran esfuerzos (no fuerzas) y la envolvente da valores razonables
    verificar(ventana.visor.vista == "ESFUERZOS", "el visor muestra esfuerzos")
    maximo = float(ventana.visor._valores_esfuerzos.max())
    verificar(0.5 < maximo < 50.0, f"von Mises máximo plausible en MPa: {maximo}")
    ventana.panel_esfuerzos.combo_magnitud.setCurrentIndex(1)  # σ1
    bombear()
    ventana.grab().save(str(salida / "esfuerzos.png"))
    ventana.volver_al_modelo()
    verificar(ventana.visor.vista != "ESFUERZOS", "se vuelve a la vista del modelo")
    print("OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
