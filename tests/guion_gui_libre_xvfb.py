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

    def proyecto_actual():
        return ventana.proyecto

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
    clic(*a_pantalla(0.3, 0.1, 0.55))
    verificar(visor.seleccion == [Objetivo("SOLIDO", ids["PLACA_L"], cara_con_normal("PLACA_L", (0, 0, 1)))],
              f"cara superior de la placa en L: {visor.seleccion}")

    class DialogoPresion(DialogoCargaSuperficie):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.combo_tipo.setCurrentIndex(self.combo_tipo.findText("Presión (kPa) sobre toda la superficie"))
            self.spin_valor.setValue(200.0)

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
    def diagonal(b):
        return float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))

    verificar(diagonal(largo_grande) > 1.05 * diagonal(largo_normal), "el deslizador agranda las flechas")
    barra.deslizador_flechas.setValue(100)
    bombear()

    # Selección por zona: arrastrar un rectángulo (modo Cara: izquierda→derecha, solo caras que quedan dentro)
    def arrastrar(x0, y0, x1, y1, modificador=Qt.NoModifier):
        alto = visor.height() - 1
        QTest.mousePress(visor, Qt.LeftButton, modificador, QPoint(x0, alto - y0))
        QTest.mouseMove(visor, QPoint((x0 + x1) // 2, alto - (y0 + y1) // 2))
        QTest.mouseMove(visor, QPoint(x1, alto - y1))
        QTest.mouseRelease(visor, Qt.LeftButton, modificador, QPoint(x1, alto - y1))
        bombear()

    ventana.accion_modo_cara.trigger()
    vista("view_xy")
    xa, ya = a_pantalla(0.0, 0.0, 0.55)
    xb, yb = a_pantalla(0.4, 0.3, 0.55)
    margen = 40
    # ventana grande que contiene la placa base y todo lo demás: de izquierda a derecha solo caras completas
    arrastrar(min(xa, xb) - margen, min(ya, yb) - margen, max(xa, xb) + margen, max(ya, yb) + margen, Qt.ShiftModifier)
    caras_sel = {(o.tipo, o.id, o.cara) for o in visor.seleccion}
    verificar((("SOLIDO", ids["PLACA_L"], cara_con_normal("PLACA_L", (0, 0, 1)))) in caras_sel,
              f"la cara superior de la placa en L queda dentro de la ventana: {caras_sel}")
    verificar(all(o.cara is not None for o in visor.seleccion) and len(visor.seleccion) >= 2, "se eligen superficies")
    ventana.grab().save(str(salida / "03d_seleccion_zona.png"))
    # rectángulo pequeño de derecha a izquierda (toca): solo la placa en L, que está arriba, en elemento
    ventana.accion_modo_elemento.trigger()
    vista("view_xy")
    cx, cy = a_pantalla(0.3, 0.1, 0.55)
    arrastrar(cx + 6, cy + 6, cx - 6, cy - 6)
    verificar(Objetivo("SOLIDO", ids["PLACA_L"]) in visor.seleccion, f"zona que toca a la placa en L: {visor.seleccion}")
    # modo zona activado desde la barra: arrastrar sin teclas ya no gira la vista
    arrastrar(min(xa, xb) - margen, min(ya, yb) - margen, max(xa, xb) + margen, max(ya, yb) + margen)  # sin teclas: ya es selección
    verificar(len(visor.seleccion) == len(proyecto.solidos), f"la ventana grande elige los 4 sólidos: {visor.seleccion}")
    visor.seleccionar([])
    vista("view_isometric")

    # Navegación estándar: arrastrar con el botón izquierdo NO mueve la cámara; central desplaza; Ctrl+central gira; rueda = zoom
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent

    camara = visor.renderer.GetActiveCamera()

    def estado():
        return np.array(camara.GetPosition()), np.array(camara.GetFocalPoint())

    def arrastrar_boton(boton, x0, y0, x1, y1, modificador=Qt.NoModifier):
        alto = visor.height() - 1
        QTest.mousePress(visor, boton, modificador, QPoint(x0, alto - y0))
        QTest.mouseMove(visor, QPoint((x0 + x1) // 2, alto - (y0 + y1) // 2))
        QTest.mouseMove(visor, QPoint(x1, alto - y1))
        QTest.mouseRelease(visor, boton, modificador, QPoint(x1, alto - y1))
        bombear()

    cx0, cy0 = a_pantalla(0.2, 0.15, 0.2)
    antes = estado()
    arrastrar_boton(Qt.LeftButton, cx0 - 60, cy0 - 40, cx0 + 80, cy0 + 50)
    despues = estado()
    verificar(np.allclose(antes[0], despues[0]) and np.allclose(antes[1], despues[1]), "el arrastre izquierdo no mueve la cámara")
    arrastrar_boton(Qt.MiddleButton, cx0, cy0, cx0 + 70, cy0 - 30)
    despues = estado()
    verificar(not np.allclose(antes[1], despues[1]), "el botón central desplaza la vista")
    verificar(abs(np.linalg.norm(despues[0] - despues[1]) - np.linalg.norm(antes[0] - antes[1])) < 1e-6, "desplazar no cambia la distancia")
    antes = estado()
    arrastrar_boton(Qt.MiddleButton, cx0, cy0, cx0 + 90, cy0, Qt.ControlModifier)
    despues = estado()
    verificar(not np.allclose(antes[0], despues[0]) and np.allclose(antes[1], despues[1]), "Ctrl + central gira alrededor del foco")
    verificar(abs(np.array(camara.GetViewUp())[2]) > 0.5, "el giro mantiene el eje Z vertical")
    distancia = np.linalg.norm(np.array(camara.GetPosition()) - np.array(camara.GetFocalPoint()))
    angulo = camara.GetViewAngle()
    visor_centro = QPointF(visor.width() / 2, visor.height() / 2)
    from PySide6.QtCore import QPoint as _P

    QApplication.sendEvent(visor, QWheelEvent(visor_centro, visor.mapToGlobal(visor_centro), _P(0, 0), _P(0, 120), Qt.NoButton,
                                              Qt.NoModifier, Qt.NoScrollPhase, False))
    bombear()
    verificar(camara.GetViewAngle() < angulo - 1e-6, "la rueda hace zoom")
    # con «Rotar» activo, arrastrar con el botón izquierdo gira la vista
    ventana.barra_vista.botones_navegacion["ROTAR"].setChecked(True)
    antes = estado()
    arrastrar_boton(Qt.LeftButton, cx0, cy0, cx0 + 60, cy0 + 20)
    verificar(not np.allclose(antes[0], estado()[0]), "con el modo Rotar el arrastre izquierdo gira")
    ventana.barra_vista.botones_navegacion["SELECCION"].setChecked(True)
    vista("view_isometric")

    # Constraint entre dos elementos enteros: se unen sus caras más cercanas (tapa del poste y cara inferior de la placa en L)
    ventana.accion_modo_elemento.trigger()
    visor.seleccionar([Objetivo("SOLIDO", ids["PLACA_L"]), Objetivo("SOLIDO", ids["POSTE"])])
    bombear()
    ventana.asignar_union_superficie()
    verificar(len(proyecto.uniones) == 1 and proyecto.uniones[0].tipo == "SOLDADURA", f"constraint creado: {proyecto.uniones}")
    u = proyecto.uniones[0]
    caras = {(u.a_id, u.a_cara), (u.b_id, u.b_cara)}
    verificar(caras == {(ids["PLACA_L"], cara_con_normal("PLACA_L", (0, 0, -1))), (ids["POSTE"], cara_con_normal("POSTE", (0, 0, 1)))},
              f"caras más cercanas elegidas: {caras}")
    visor.seleccionar([])
    vista("view_isometric")
    verificar("ASIG_UNIONES" in visor.actors, "el constraint se dibuja")
    ventana.grab().save(str(salida / "03c_constraint.png"))

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
    # Eliminar elementos y deshacer / rehacer
    ventana.volver_al_modelo()
    ventana.accion_modo_elemento.trigger()
    n_cargas = sum(len(v) for v in ventana.cargas_superficie_por_caso.values())
    visor.seleccionar([Objetivo("SOLIDO", ids["PLACA_L"])])
    ventana.eliminar_seleccion()
    verificar(ids["PLACA_L"] not in proyecto_actual().solidos, "la placa en L se elimina")
    verificar(not proyecto_actual().uniones, "se elimina también su constraint")
    verificar(sum(len(v) for v in ventana.cargas_superficie_por_caso.values()) == n_cargas - 1, "y su presión")
    verificar(ventana.accion_deshacer.isEnabled() and not ventana.accion_rehacer.isEnabled(), "se puede deshacer")
    ventana.accion_deshacer.trigger()
    bombear()
    verificar(ids["PLACA_L"] in proyecto_actual().solidos and len(proyecto_actual().uniones) == 1, "deshacer restaura la placa y el constraint")
    verificar(sum(len(v) for v in ventana.cargas_superficie_por_caso.values()) == n_cargas, "y la presión")
    verificar(ventana.accion_rehacer.isEnabled(), "se puede rehacer")
    ventana.accion_rehacer.trigger()
    bombear()
    verificar(ids["PLACA_L"] not in proyecto_actual().solidos, "rehacer vuelve a eliminarla")
    ventana.accion_deshacer.trigger()
    bombear()
    ventana.grab().save(str(salida / "06_deshacer.png"))
    print("OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
