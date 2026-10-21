"""Guion de la GUI con una conexión metálica (perfiles en I) y cargas en las
caras extremas de los elementos, con clics de mouse reales.

Lo lanza ``test_gui_flujo_completo.py`` bajo ``xvfb-run``. Uso:
``python -m tests.guion_gui_conexion_xvfb <directorio_salida>``. Termina con
código 0 e imprime ``OK`` si los esfuerzos coinciden con el cálculo manual.
"""

import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
from vtkmodules.vtkRenderingCore import vtkCoordinate

from ifc_structural.core.asignacion import Objetivo

from .ayuda_ifc import crear_ifc_conexion_metalica


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
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)  # «¿analizar de todos modos?» (los sólidos no entran en el análisis de barras)
    QDialog.exec = lambda self: 1

    import ifc_structural.gui.ventana_principal as vp
    from ifc_structural.combinaciones import esfuerzos
    from ifc_structural.gui.dialogo_cargas import DialogoCargas

    def bombear(n=15):
        for _ in range(n):
            app.processEvents()

    ruta_ifc = salida / "conexion_metalica.ifc"
    crear_ifc_conexion_metalica(str(ruta_ifc))
    ventana = vp.VentanaPrincipal()
    ventana.resize(1500, 900)
    ventana.show()
    bombear()
    verificar(ventana.importar_archivo(str(ruta_ifc)), "importar el IFC de la conexión metálica")
    verificar(any("sólidos" in str(m) and "Plate" in str(m) for m in mensajes), f"aviso de elementos importados como sólidos: {mensajes}")
    proyecto = ventana.proyecto
    verificar(len(proyecto.vigas) == 2 and len(proyecto.columnas) == 2, "columna partida en la unión")
    visor = ventana.visor

    def a_pantalla(x, y, z):
        c = vtkCoordinate()
        c.SetCoordinateSystemToWorld()
        c.SetValue(x, y, z)
        dx, dy = c.GetComputedDisplayValue(visor.renderer)
        return int(dx), int(dy)

    def vista(nombre, **k):
        getattr(visor, nombre)(**k)
        visor.reset_camera()
        bombear()
        visor.render()
        bombear()

    def clic(x, y):
        QTest.mouseClick(visor, Qt.LeftButton, Qt.NoModifier, QPoint(x, visor.height() - 1 - y))
        bombear()

    def fuerza(valor):
        class Dialogo(DialogoCargas):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.combo_tipo.setCurrentIndex(self.combo_tipo.findText("Fuerza puntual (kN)"))
                self.combo_direccion.setCurrentIndex(self.combo_direccion.findText("Gravedad (−Z)"))
                self.spin_valor.setValue(valor)

        vp.DialogoCargas = Dialogo
        ventana.asignar_carga()

    # Apoyo: empotramiento en la base de la columna (cara inferior)
    ventana.accion_modo_cara.trigger()
    vista("view_xy", negative=True)
    clic(*a_pantalla(0.0, 0.0, 0.0))
    verificar(visor.seleccion == [Objetivo("COLUMNA", "C1", 0)], f"base de la columna: {visor.seleccion}")
    ventana.asignar_apoyo()
    verificar(len(ventana.apoyos) == 1, "apoyo creado")

    # Cargas en las caras extremas de las vigas y de la columna
    vista("view_yz")
    clic(*a_pantalla(2.6, 0.0, 3.0))
    verificar(visor.seleccion == [Objetivo("VIGA", "B1", 1)], f"extremo de B1: {visor.seleccion}")
    fuerza(25.0)
    vista("view_yz", negative=True)
    clic(*a_pantalla(-2.6, 0.0, 3.0))
    verificar(visor.seleccion == [Objetivo("VIGA", "B2", 0)], f"extremo de B2: {visor.seleccion}")
    fuerza(15.0)
    vista("view_xy")
    clic(*a_pantalla(0.0, 0.0, 4.0))
    verificar(visor.seleccion == [Objetivo("COLUMNA", "C2", 1)], f"tope de la columna: {visor.seleccion}")
    fuerza(100.0)
    fuerzas = sorted(-c.fz for c in ventana.cargas_puntuales_por_caso["D"])
    verificar(fuerzas == [15.0, 25.0, 100.0], f"cargas creadas: {fuerzas}")
    visor.seleccionar([])
    vista("view_isometric")
    ventana.grab().save(str(salida / "conexion_cargas.png"))

    # Análisis
    mensajes.clear()
    ventana.analizar(modo="BARRAS")
    inicio = time.time()
    while ventana._worker.isRunning() and time.time() - inicio < 150:
        bombear(5)
        time.sleep(0.05)
    bombear(30)
    resultado = ventana.resultado
    verificar(resultado is not None, f"el análisis terminó ({mensajes})")
    verificar(not [m for m in mensajes if m[0] == "critical"], f"sin errores: {mensajes}")
    verificar(ventana.visor.vista == "ESFUERZOS", "el visor muestra esfuerzos")
    ventana.grab().save(str(salida / "conexion_esfuerzos.png"))

    # Comparación con el cálculo manual (con el peso propio: +1,35 kN·m en la raíz de B1)
    t = resultado.tensiones_por_caso["D"]
    x = resultado.malla_tensiones.coordenadas
    barras = np.array(resultado.barra_de_cada_celda())
    conectividad = resultado.malla_tensiones.conectividad

    def nodos(b):
        return np.unique(conectividad[barras == b])

    ipe, heb = proyecto.perfiles["IPE300"], proyecto.perfiles["HEB200"]
    vm = esfuerzos.von_mises(t)
    n = nodos("B1")
    raiz = n[np.isclose(x[n, 0], 0.0, atol=0.01)]
    esperado = (25.0 * 2.6 + ipe.area * 77.0 * 2.6**2 / 2) * (ipe.altura / 2) / ipe.inercia_z
    verificar(abs(vm[raiz].max() - esperado) < 0.02 * esperado, f"σ en la raíz de B1: {vm[raiz].max()} vs {esperado}")
    n = nodos("C2")
    medio = n[np.isclose(x[n, 2], 3.5, atol=0.01)]
    verificar(abs(t[medio, 2].mean() + 100.0 / heb.area - 0) < 0.03 * 100.0 / heb.area, "axial en la cabeza de la columna")

    # Análisis sólido (3D) del mismo modelo: botón ANALIZAR 3D
    from ifc_structural.gui.dialogo_analisis_3d import DialogoAnalisis3D

    class Dialogo3D(DialogoAnalisis3D):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            verificar("elementos" in self.estimacion.text(), "el diálogo 3D estima el número de elementos")
            self.spin_tamano.setValue(25)

    vp.DialogoAnalisis3D = Dialogo3D
    verificar(ventana.accion_analizar.isEnabled(), "el botón ANALIZAR está habilitado")
    verificar(ventana._detectar_modo_de_analisis()[0] == "SOLIDO", "la conexión tiene placas y pernos: el análisis automático es 3D")
    ventana.volver_al_modelo()
    mensajes.clear()
    ventana.accion_analizar.trigger()
    inicio = time.time()
    while ventana._worker.isRunning() and time.time() - inicio < 200:
        bombear(5)
        time.sleep(0.05)
    bombear(30)
    solido = ventana.resultado
    verificar(solido is not None and solido.es_solido, f"el análisis 3D terminó ({mensajes})")
    verificar(not [m for m in mensajes if m[0] == "critical"], f"sin errores en 3D: {mensajes}")
    verificar(ventana.visor.vista == "ESFUERZOS", "el visor muestra los esfuerzos del modelo sólido")
    ventana.grab().save(str(salida / "conexion_solido.png"))
    t3 = solido.tensiones_por_caso["D"]
    x3 = solido.malla_tensiones.coordenadas
    b3 = np.array(solido.barra_de_cada_celda())
    n = np.unique(solido.malla_tensiones.conectividad[b3 == "B1"])
    m = n[np.isclose(x3[n, 0], 1.3, atol=0.03)]
    esperado = (25.0 * 1.3 + ipe.area * 77.0 * 1.3**2 / 2) * (ipe.altura / 2) / ipe.inercia_z
    verificar(abs(np.abs(t3[m, 0]).max() - esperado) < 0.05 * esperado, f"σ en B1 (3D): {np.abs(t3[m, 0]).max()} vs {esperado}")

    # Deformada: se muestra, se anima y se graba en video (MP4 si hay codificador, y GIF)
    from PySide6.QtWidgets import QFileDialog

    panel = ventana.panel_esfuerzos
    escala_auto = panel.spin_escala_deformada.value()
    verificar(escala_auto >= 1.0, f"escala automática de la deformada: {escala_auto}")
    panel.combo_resultado.setCurrentIndex(panel.combo_resultado.findData("L"))
    panel.check_deformada.setChecked(True)
    bombear(30)
    verificar(ventana.visor.hay_deformada, "se dibuja la deformada del modelo sólido")
    panel.boton_animar.setChecked(True)
    for _ in range(12):
        bombear(3)
        time.sleep(0.05)
    verificar(ventana._timer_animacion.isActive() and ventana._paso_animacion > 0, "la animación avanza")
    panel.boton_animar.setChecked(False)
    verificar(not ventana._timer_animacion.isActive(), "la animación se detiene")
    formatos = ["gif"]
    try:
        import imageio_ffmpeg  # noqa: F401

        formatos.insert(0, "mp4")
    except ImportError:
        pass
    for formato in formatos:
        ruta_video = salida / f"deformada.{formato}"
        QFileDialog.getSaveFileName = staticmethod(lambda *a, _r=str(ruta_video), **k: (_r, ""))
        mensajes.clear()
        ventana._guardar_video()
        bombear(10)
        verificar(ruta_video.exists() and ruta_video.stat().st_size > 5000, f"video {formato} guardado ({mensajes})")
    print("OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
