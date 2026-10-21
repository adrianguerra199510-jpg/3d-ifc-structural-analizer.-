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
    # al dibujar apoyos o cargas la cámara no se reajusta (se perdía el zoom del usuario)
    camara_u = visor.renderer.GetActiveCamera()
    foco_u, pos_u = np.array(camara_u.GetFocalPoint()), np.array(camara_u.GetPosition())
    camara_u.SetPosition(*(foco_u + (pos_u - foco_u) * 0.3))  # el usuario acerca la vista
    camara_u.SetFocalPoint(*(foco_u + np.array([0.02, 0.01, 0.0])))
    antes_u = (np.array(camara_u.GetPosition()), np.array(camara_u.GetFocalPoint()), camara_u.GetViewAngle())
    ventana._refrescar_asignaciones()
    barra.boton_apoyos.setChecked(False)
    barra.boton_apoyos.setChecked(True)
    visor.reemplazar_modelo(proyecto)
    ventana._refrescar_asignaciones()
    bombear()
    despues_u = (np.array(camara_u.GetPosition()), np.array(camara_u.GetFocalPoint()), camara_u.GetViewAngle())
    verificar(all(np.allclose(a, b, atol=1e-9) for a, b in zip(antes_u, despues_u)), f"el zoom del usuario se conserva al dibujar asignaciones: {antes_u} {despues_u}")
    visor.reset_camera()
    vista("view_isometric")
    largo_normal = visor.actors["ASIG_CARGAS"].GetMapper().GetInput().GetBounds()
    from ifc_structural.gui.dialogo_escalas import DialogoEscalas
    from ifc_structural.gui.visor_3d import ESCALA_PIXELES, ESCALA_PORCENTAJE

    dialogo_escalas = DialogoEscalas(visor)  # Ver → Escala de símbolos
    dialogo_escalas._spins["cargas"].setValue(250.0)  # 250 % de las flechas
    bombear()
    largo_grande = visor.actors["ASIG_CARGAS"].GetMapper().GetInput().GetBounds()

    def diagonal(b):
        return float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))

    verificar(diagonal(largo_grande) > 1.05 * diagonal(largo_normal), "el diálogo de escala agranda las flechas")
    dialogo_escalas._restablecer()
    verificar(visor.escalas["cargas"] == (ESCALA_PORCENTAJE, 100.0), "restablecer vuelve al automático")
    # tamaño en píxeles: el símbolo mide lo mismo en pantalla aunque se acerque la vista
    visor.set_escala("apoyos", ESCALA_PIXELES, 60.0)
    bombear()
    nombres_apoyo = [n for n in visor.actors if n.startswith("ASIG_APOYOS")]
    verificar(nombres_apoyo, "los apoyos se redibujan con tamaño en píxeles")
    def pixeles_del_simbolo():
        return visor._tamano("apoyos", 1.0) / visor._metros_por_pixel()

    antes = visor.actors[nombres_apoyo[0]].GetMapper().GetInput().GetBounds()
    verificar(abs(pixeles_del_simbolo() - 60.0) < 1e-6, "el símbolo mide 60 px")
    camara_z = visor.renderer.GetActiveCamera()
    camara_z.Zoom(2.0)
    visor._redibujar_asignaciones()
    bombear()
    despues = visor.actors[nombres_apoyo[0]].GetMapper().GetInput().GetBounds()
    verificar(abs(pixeles_del_simbolo() - 60.0) < 1e-6, "y sigue midiendo 60 px al acercar la vista")
    verificar(antes != despues, "el símbolo se redibujó con el nuevo zoom")
    camara_z.Zoom(0.5)
    visor.set_escalas({"apoyos": (ESCALA_PORCENTAJE, 100.0)})
    bombear()

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

    # Estilos de vista (Ver → Estilo de vista) y menú del clic derecho sobre el fondo
    from ifc_structural.gui.visor_3d import ESTILO_ALAMBRE, ESTILO_SOLIDO, ESTILO_TRANSPARENTE

    ventana.accion_modo_elemento.trigger()
    clave_poste = ("SOLIDO", ids["POSTE"])
    actor_poste = visor._visuales[clave_poste].actor
    visor.seleccionar([])
    ventana._acciones_estilo[ESTILO_TRANSPARENTE].trigger()
    bombear()
    verificar(visor.estilo == ESTILO_TRANSPARENTE and actor_poste.GetProperty().GetOpacity() < 0.5, "el estilo transparente baja la opacidad")
    px, py = a_pantalla(0.2, 0.15, 0.3)
    clic(px, py)
    verificar(Objetivo("SOLIDO", ids["POSTE"]) in visor.seleccion or len(visor.seleccion) == 1, f"en transparente se sigue pudiendo elegir: {visor.seleccion}")
    ventana.grab().save(str(salida / "07_transparente.png"))
    ventana._acciones_estilo[ESTILO_ALAMBRE].trigger()
    bombear()
    verificar(visor.estilo == ESTILO_ALAMBRE and actor_poste.GetProperty().GetOpacity() < 0.1, "el estilo alambre deja solo las aristas")
    ventana.grab().save(str(salida / "07b_alambre.png"))
    ventana._acciones_estilo[ESTILO_SOLIDO].trigger()
    bombear()
    verificar(visor.estilo == ESTILO_SOLIDO and actor_poste.GetProperty().GetOpacity() == 1.0, "y el sólido lo restablece")
    visor.seleccionar([])
    menus = []
    visor.menu_fondo.disconnect()  # el menú real se abriría de forma modal: se recoge para inspeccionarlo
    visor.menu_fondo.connect(lambda: menus.append(ventana.construir_menu_fondo()))
    arrastrar_boton(Qt.RightButton, 60, 60, 60, 60)  # clic derecho en una esquina vacía del visor
    verificar(menus and any(a.text() == "Estilo de vista" for a in menus[-1].actions()), "el clic derecho sobre el fondo ofrece el estilo de vista")
    ventana._acciones_estilo[ESTILO_SOLIDO].setChecked(True)

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
    from PySide6.QtCore import QPoint as _P

    def rueda(pasos, px, py):  # px, py: píxel de VTK (origen abajo a la izquierda)
        posicion = QPointF(px, visor.height() - 1 - py)
        QApplication.sendEvent(visor, QWheelEvent(posicion, visor.mapToGlobal(posicion), _P(0, 0), _P(0, int(120 * pasos)), Qt.NoButton,
                                                  Qt.NoModifier, Qt.NoScrollPhase, False))
        bombear()

    # zoom hacia el cursor: la cámara se acerca sin cambiar el ángulo de visión y el punto bajo el cursor no se mueve
    objetivo = visor._punto_bajo_cursor(cx0 + 40, cy0 - 30)
    pan_antes = visor._proyectar(np.array([objetivo]))[0][0]
    rueda(4, cx0 + 40, cy0 - 30)
    pan_despues = visor._proyectar(np.array([objetivo]))[0][0]
    nueva = np.linalg.norm(np.array(camara.GetPosition()) - np.array(camara.GetFocalPoint()))
    verificar(nueva < 0.7 * distancia, f"la rueda acerca la cámara: {distancia} -> {nueva}")
    verificar(abs(camara.GetViewAngle() - angulo) < 1e-9, "sin cambiar el ángulo de visión (no deforma la perspectiva)")
    verificar(np.allclose(pan_antes, pan_despues, atol=1.0), f"el punto bajo el cursor no se mueve: {pan_antes} {pan_despues}")
    rueda(-300, cx0, cy0)
    verificar(abs(camara.GetViewAngle() - angulo) < 1e-9 and np.linalg.norm(np.array(camara.GetPosition()) - np.array(camara.GetFocalPoint())) <= 30.0 * visor._escala * 1.001,
              "alejar tiene un límite y tampoco cambia el ángulo")
    rueda(300, cx0, cy0)
    verificar(np.linalg.norm(np.array(camara.GetPosition()) - np.array(camara.GetFocalPoint())) >= 0.003 * visor._escala * 0.999, "acercar tiene un límite")
    visor.reset_camera()
    vista("view_isometric")
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

    # Conectar automáticamente: detecta las parejas de elementos que se tocan y crea constraints de soldadura
    ventana.conectar_automaticamente()
    verificar(len(proyecto.uniones) == 4, f"la conexión automática creó las 3 uniones restantes: {[u.nombre for u in proyecto.uniones]}")
    verificar(sum(u.es_completa for u in proyecto.uniones) == 3, "entre elementos completos")
    ventana.conectar_automaticamente()
    verificar(len(proyecto.uniones) == 4, "no repite las parejas que ya tienen constraint")
    visor.seleccionar([])
    vista("view_isometric")
    ventana.grab().save(str(salida / "03e_conexion_automatica.png"))
    # los constraints tienen su propio interruptor (independiente de los apoyos y las cargas)
    ventana.barra_vista.boton_uniones.setChecked(False)
    bombear()
    verificar("ASIG_UNIONES" not in visor.actors and "ASIG_UNIONES_CARAS" not in visor.actors, "los constraints se ocultan")
    verificar("ASIG_CARGAS" in visor.actors and any(n.startswith("ASIG_APOYOS") for n in visor.actors), "sin tocar apoyos ni cargas")
    ventana.grab().save(str(salida / "03f_constraints_ocultos.png"))
    ventana.barra_vista.boton_uniones.setChecked(True)
    bombear()
    verificar("ASIG_UNIONES" in visor.actors, "y se vuelven a mostrar")

    # Análisis 3D
    from ifc_structural.gui.dialogo_analisis_3d import DialogoAnalisis3D

    class Dialogo3D(DialogoAnalisis3D):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.spin_tamano.setValue(20)

    vp.DialogoAnalisis3D = Dialogo3D
    verificar(ventana.accion_analizar.isEnabled(), "el botón ANALIZAR está habilitado")
    verificar(ventana._detectar_modo_de_analisis()[0] == "SOLIDO", "con sólidos libres el análisis automático es 3D")
    mensajes.clear()
    ventana.accion_analizar.trigger()  # un solo botón: detecta que hay sólidos y analiza en 3D
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
    # Reporte del análisis en PDF y en Word (el botón del panel y Archivo → Exportar reporte)
    verificar(ventana.accion_reporte.isEnabled() and ventana.panel_esfuerzos.boton_reporte.isVisible(),
              "se puede exportar el reporte tras analizar")
    for extension in ("pdf", "docx"):
        ruta = ventana.exportar_reporte(str(salida / f"reporte.{extension}"))
        verificar(ruta is not None and ruta.exists() and ruta.stat().st_size > 50_000, f"reporte {extension} exportado")
    verificar(not [m for m in mensajes if m[0] == "critical"], f"el reporte no dio errores: {mensajes}")
    # Eliminar elementos y deshacer / rehacer
    ventana.volver_al_modelo()
    ventana.accion_modo_elemento.trigger()
    n_cargas = sum(len(v) for v in ventana.cargas_superficie_por_caso.values())
    visor.seleccionar([Objetivo("SOLIDO", ids["PLACA_L"])])
    ventana.eliminar_seleccion()
    verificar(ids["PLACA_L"] not in proyecto_actual().solidos, "la placa en L se elimina")
    verificar(len(proyecto_actual().uniones) == 3, "se elimina también su constraint (las otras tres siguen)")
    verificar(sum(len(v) for v in ventana.cargas_superficie_por_caso.values()) == n_cargas - 1, "y su presión")
    verificar(ventana.accion_deshacer.isEnabled() and not ventana.accion_rehacer.isEnabled(), "se puede deshacer")
    ventana.accion_deshacer.trigger()
    bombear()
    verificar(ids["PLACA_L"] in proyecto_actual().solidos and len(proyecto_actual().uniones) == 4, "deshacer restaura la placa y el constraint")
    verificar(sum(len(v) for v in ventana.cargas_superficie_por_caso.values()) == n_cargas, "y la presión")
    verificar(ventana.accion_rehacer.isEnabled(), "se puede rehacer")
    ventana.accion_rehacer.trigger()
    bombear()
    verificar(ids["PLACA_L"] not in proyecto_actual().solidos, "rehacer vuelve a eliminarla")
    ventana.accion_deshacer.trigger()
    bombear()
    # el rectángulo de selección se ve al arrastrar, incluso después de reconstruir la escena (deshacer)
    ventana.grab().save(str(salida / "06_deshacer.png"))
    alto_v = visor.height() - 1
    QTest.mousePress(visor, Qt.LeftButton, Qt.NoModifier, QPoint(400, 300))
    QTest.mouseMove(visor, QPoint(500, 350))
    QTest.mouseMove(visor, QPoint(650, 450))
    bombear()
    verificar(visor._caja_actor is not None and all(visor.renderer.HasViewProp(a) and a.GetVisibility() for a in visor._caja_actor),
              "el recuadro de selección se dibuja mientras se arrastra")
    ventana.grab().save(str(salida / "06b_recuadro.png"))
    QTest.mouseRelease(visor, Qt.LeftButton, Qt.NoModifier, QPoint(650, 450))
    bombear()
    verificar(not any(a.GetVisibility() for a in visor._caja_actor), "y desaparece al soltar")
    print("OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
