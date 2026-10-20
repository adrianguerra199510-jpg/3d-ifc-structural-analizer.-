"""Guion de la GUI con geometría de forma libre: se importa un IFC cuyos elementos no son barras,
losas ni muros, se eligen superficies con clics reales (planas y curva), se les asignan apoyos y
cargas y se analiza en 3D.

Lo lanza ``test_gui_flujo_completo.py`` bajo ``xvfb-run``. Uso: ``python -m tests.guion_gui_libre_xvfb <directorio>``.
Termina con código 0 e imprime ``OK`` si todo coincide con lo esperado.
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

from .ayuda_ifc import crear_ifc_geometria_libre


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
    QDialog.exec = lambda self: 1

    import ifc_structural.gui.ventana_principal as vp
    from ifc_structural.combinaciones import esfuerzos
    from ifc_structural.gui.dialogo_superficie import DialogoApoyoSuperficie, DialogoCargaSuperficie

    def bombear(n=15):
        for _ in range(n):
            app.processEvents()

    ruta_ifc = salida / "geometria_libre.ifc"
    crear_ifc_geometria_libre(str(ruta_ifc))
    ventana = vp.VentanaPrincipal()
    ventana.resize(1500, 900)
    ventana.show()
    bombear()
    verificar(ventana.importar_archivo(str(ruta_ifc)), "importar el IFC de geometría libre")
    proyecto = ventana.proyecto
    verificar(len(proyecto.solidos) == 4 and not proyecto.vigas and not proyecto.columnas,
              f"se importan 4 sólidos de forma libre: {list(proyecto.solidos)}")
    ids = {s.nombre: s.id for s in proyecto.solidos.values()}
    verificar(len(proyecto.solidos[ids["POSTE"]].caras) == 3, "el poste cilíndrico tiene 3 superficies (2 tapas y la curva)")
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

    def cara_con_normal(nombre, normal):
        s = proyecto.solidos[ids[nombre]]
        return max(range(len(s.caras)), key=lambda i: float(np.dot(s.caras[i].normal, normal)))

    vista("view_isometric")
    ventana.grab().save(str(salida / "01_importado.png"))

    # Apoyo: cara inferior de la placa base (vista desde abajo)
    ventana.accion_modo_cara.trigger()
    vista("view_xy", negative=True)
    clic(*a_pantalla(0.2, 0.15, 0.0))
    verificar(visor.seleccion == [Objetivo("SOLIDO", ids["PLACA_BASE"], cara_con_normal("PLACA_BASE", (0, 0, -1)))],
              f"cara inferior de la placa base: {visor.seleccion}")
    ventana.grab().save(str(salida / "02_superficie_seleccionada.png"))
    ventana.asignar_apoyo()
    verificar(len(ventana.apoyos) == 1 and ventana.apoyos[0].grados_restringidos() == [1, 2, 3], f"apoyo creado: {ventana.apoyos}")

    # Carga: presión sobre la cara superior de la placa en L
    vista("view_xy")
    clic(*a_pantalla(0.1, 0.15, 0.55))
    verificar(visor.seleccion == [Objetivo("SOLIDO", ids["PLACA_L"], cara_con_normal("PLACA_L", (0, 0, 1)))],
              f"cara superior de la placa en L: {visor.seleccion}")

    class DialogoPresion(DialogoCargaSuperficie):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.combo_tipo.setCurrentIndex(self.combo_tipo.findText("Presión (kPa) sobre toda la superficie"))
            self.spin_valor.setValue(2000.0)

    vp.DialogoCargaSuperficie = DialogoPresion
    ventana.asignar_carga()
    todas = [c for lista in ventana.cargas_superficie_por_caso.values() for c in lista]
    verificar(len(todas) == 1 and todas[0].tipo == "PRESION", f"presión asignada: {todas}")

    # Carga: fuerza horizontal sobre la superficie curva del poste
    visor.seleccionar([])
    vista("view_isometric")
    clic(*a_pantalla(0.2, 0.10, 0.25))  # centro de la cara -Y del poste cilíndrico
    verificar(visor.seleccion == [Objetivo("SOLIDO", ids["POSTE"], cara_con_normal("POSTE", (0, 1, 0)) if False else visor.seleccion[0].cara)]
              and proyecto.solidos[ids["POSTE"]].caras[visor.seleccion[0].cara].tipo == "CURVA",
              f"superficie curva del poste: {visor.seleccion}")

    class DialogoFuerza(DialogoCargaSuperficie):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.combo_tipo.setCurrentIndex(self.combo_tipo.findText("Fuerza total (kN) repartida en la superficie"))
            self.spin_valor.setValue(3.0)
            self.combo_direccion.setCurrentIndex(self.combo_direccion.findText("+Y"))

    vp.DialogoCargaSuperficie = DialogoFuerza
    ventana.asignar_carga()
    todas = [c for lista in ventana.cargas_superficie_por_caso.values() for c in lista]
    verificar(len(todas) == 2 and {c.tipo for c in todas} == {"PRESION", "FUERZA"}, f"fuerza asignada: {todas}")
    visor.seleccionar([])
    vista("view_isometric")
    ventana.grab().save(str(salida / "03_apoyo_y_cargas.png"))

    # Flechas de las cargas: se ocultan, se muestran y se escalan desde la barra de vista
    barra = ventana.barra_vista
    verificar("ASIG_CARGAS" in visor.actors and "ASIG_SUPERFICIE_CARGA" in visor.actors, "las flechas de las cargas se dibujan")
    barra.boton_flechas.setChecked(False)
    bombear()
    verificar("ASIG_CARGAS" not in visor.actors and "ASIG_SUPERFICIE_CARGA" not in visor.actors, "las flechas se ocultan")
    verificar(any(n.startswith("ASIG_APOYOS") for n in visor.actors), "los apoyos siguen visibles")
    ventana.grab().save(str(salida / "03b_flechas_ocultas.png"))
    barra.boton_flechas.setChecked(True)
    bombear()
    largo_normal = visor.actors["ASIG_CARGAS"].GetMapper().GetInput().GetBounds()
    barra.deslizador_flechas.setValue(250)
    bombear()
    largo_grande = visor.actors["ASIG_CARGAS"].GetMapper().GetInput().GetBounds()
    verificar((largo_grande[5] - largo_grande[4]) > 1.15 * (largo_normal[5] - largo_normal[4]), "el deslizador agranda las flechas")
    barra.deslizador_flechas.setValue(100)
    bombear()

    # Análisis 3D
    from ifc_structural.gui.dialogo_analisis_3d import DialogoAnalisis3D

    class Dialogo3D(DialogoAnalisis3D):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.spin_tamano.setValue(20)

    vp.DialogoAnalisis3D = Dialogo3D
    verificar(ventana.accion_analizar_3d.isEnabled(), "el botón ANALIZAR 3D está habilitado")
    mensajes.clear()
    ventana.accion_analizar_3d.trigger()
    inicio = time.time()
    while ventana._worker.isRunning() and time.time() - inicio < 250:
        bombear(5)
        time.sleep(0.05)
    bombear(30)
    resultado = ventana.resultado
    verificar(resultado is not None and resultado.es_solido, f"el análisis 3D terminó ({mensajes})")
    verificar(not [m for m in mensajes if m[0] == "critical"], f"sin errores: {mensajes}")
    verificar(ventana.visor.vista == "ESFUERZOS", "el visor muestra esfuerzos")
    ventana.grab().save(str(salida / "04_esfuerzos.png"))
    vm = esfuerzos.von_mises(resultado.tensiones_por_caso["D"])
    verificar(np.isfinite(vm).all() and vm.max() > 0, "esfuerzos finitos")
    x = resultado.malla_tensiones.coordenadas
    u = resultado.desplazamientos_por_caso["D"]
    # equilibrio: la placa en L (arriba) se mueve hacia abajo y el poste se flexiona hacia +Y
    verificar(u[np.isclose(x[:, 2], 0.55, atol=0.01), 2].mean() < 0, "la placa superior baja con la presión")
    verificar(u[:, 1].max() > 0, "el poste se desplaza hacia +Y con la fuerza horizontal")
    verificar(np.abs(u[np.isclose(x[:, 2], 0.0, atol=1e-6)]).max() < 1e-9, "la base apoyada no se mueve")
    ventana.panel_esfuerzos.check_deformada.setChecked(True)
    bombear(30)
    ventana.grab().save(str(salida / "05_deformada.png"))
    print("OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
