"""Ventana principal: visor 3D con selección por mouse, asignación de
apoyos y cargas sobre lo seleccionado, botón ANALIZAR y mapa de esfuerzos.

Flujo de uso:
1. Importar un IFC.
2. Seleccionar con el mouse la cara extrema de las columnas (vista inferior
   para ver las bases) y asignar apoyos.
3. Seleccionar losas, vigas o caras y asignar cargas (al caso activo).
4. Pulsar ANALIZAR y ver los esfuerzos por caso, combinación o envolvente.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QWidget,
)

from ..combinaciones import esfuerzos
from ..combinaciones.motor_superposicion import calcular_envolvente
from ..config import Configuracion, cargar_configuracion
from ..core.asignacion import ErrorAsignacion, Objetivo, cargas_para_objetivo, descripcion
from ..core.cargas import Apoyo, CargaArea, CargaLineal, CargaPuntual
from ..core.casos_carga import CasoCarga, Combinacion, casos_y_combinaciones_por_defecto
from ..core.modelo import Proyecto
from ..core.persistencia import cargar_proyecto, guardar_proyecto
from ..ifc.importador import importar_ifc
from ..solver.analisis import ResultadoAnalisis, ejecutar_analisis, validar_modelo
from .dialogo_apoyos import DialogoApoyo
from .dialogo_cargas import DialogoCargas
from .dialogo_casos import DialogoCasos
from .panel_asignaciones import PanelAsignaciones
from .panel_elementos import PanelElementos
from .panel_resultados import ENVOLVENTE, PanelEsfuerzos
from .trabajador_analisis import TrabajadorAnalisis
from .visor_3d import MODO_CARA, MODO_ELEMENTO, Visor3D

logger = logging.getLogger(__name__)

ESTILO_BOTON_ANALIZAR = """
QToolButton { background-color: #2e8b57; color: white; font-weight: bold; font-size: 14px;
              padding: 6px 20px; border-radius: 4px; }
QToolButton:hover { background-color: #3cb371; }
QToolButton:disabled { background-color: #9aa; }
"""

AYUDA = (
    "<b>Cómo se usa</b><ol>"
    "<li><b>Archivo → Importar IFC</b>.</li>"
    "<li>Elija el modo <b>Cara</b>, use <b>Ver → Vista inferior</b> y haga clic en la cara de la base de "
    "cada columna; pulse <b>Asignar apoyo</b> (Ctrl o Mayús + clic para elegir varias).</li>"
    "<li>Elija el <b>caso activo</b> (D, L...). Haga clic en una losa (o en su cara superior) y pulse "
    "<b>Asignar carga</b>. También puede cargar vigas completas o sus caras.</li>"
    "<li>Pulse <b>▶ ANALIZAR</b> (F5).</li>"
    "<li>Elija en el panel <b>Esfuerzos</b> el caso, la combinación o la envolvente, y el tipo de esfuerzo. "
    "Haga clic en el visor para leer el valor en un punto.</li></ol>"
    "Con el botón central del mouse se desplaza la vista; con la rueda se acerca o aleja; arrastrando con "
    "el botón izquierdo se gira."
)


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador Estructural 3D (IFC + CalculiX)")
        self.resize(1500, 920)

        self.configuracion: Configuracion = cargar_configuracion()
        self.proyecto = Proyecto()
        self.apoyos: List[Apoyo] = []
        self.casos: Dict[str, CasoCarga]
        self.combinaciones: Dict[str, Combinacion]
        self.casos, self.combinaciones = casos_y_combinaciones_por_defecto()
        self.cargas_area_por_caso: Dict[str, List[CargaArea]] = {}
        self.cargas_lineales_por_caso: Dict[str, List[CargaLineal]] = {}
        self.cargas_puntuales_por_caso: Dict[str, List[CargaPuntual]] = {}

        self.resultado: Optional[ResultadoAnalisis] = None
        self._resultado_desactualizado = False
        self._worker: Optional[TrabajadorAnalisis] = None
        self._dialogo_progreso: Optional[QProgressDialog] = None

        self.visor = Visor3D(self)
        self.setCentralWidget(self.visor)

        self.panel_elementos = PanelElementos(self)
        dock_elementos = QDockWidget("Elementos", self)
        dock_elementos.setWidget(self.panel_elementos)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_elementos)

        self.panel_asignaciones = PanelAsignaciones(self)
        self.panel_esfuerzos = PanelEsfuerzos(self)
        self.pestanas = QTabWidget()
        self.pestanas.addTab(self.panel_asignaciones, "Asignaciones")
        self.pestanas.addTab(self.panel_esfuerzos, "Esfuerzos")
        dock_derecho = QDockWidget("Panel", self)
        dock_derecho.setWidget(self.pestanas)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_derecho)

        self._etiqueta_seleccion = QLabel("Sin selección")
        self.statusBar().addPermanentWidget(self._etiqueta_seleccion)

        self._crear_acciones()
        self._crear_menus()
        self._crear_barra_de_herramientas()
        self._conectar_senales()

        self._actualizar_combo_caso()
        self._actualizar_estado_de_acciones()
        self.statusBar().showMessage("Importe un modelo IFC (Archivo → Importar IFC). Ayuda → Cómo se usa.")

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _crear_acciones(self) -> None:
        self.accion_importar = QAction("Importar IFC…", self, shortcut="Ctrl+I", triggered=self._importar_ifc)
        self.accion_abrir = QAction("Abrir proyecto…", self, shortcut="Ctrl+O", triggered=self._abrir_proyecto)
        self.accion_guardar = QAction("Guardar proyecto como…", self, shortcut="Ctrl+S", triggered=self._guardar_proyecto)
        self.accion_salir = QAction("Salir", self, triggered=self.close)

        self.accion_modo_elemento = QAction("Elemento", self, checkable=True, checked=True)
        self.accion_modo_elemento.setToolTip("Seleccionar elementos completos con el mouse")
        self.accion_modo_cara = QAction("Cara", self, checkable=True)
        self.accion_modo_cara.setToolTip("Seleccionar caras de los elementos con el mouse (para apoyos y cargas)")
        grupo_modo = QActionGroup(self)
        grupo_modo.setExclusive(True)
        grupo_modo.addAction(self.accion_modo_elemento)
        grupo_modo.addAction(self.accion_modo_cara)

        self.accion_apoyo = QAction("Asignar apoyo", self, triggered=self.asignar_apoyo)
        self.accion_apoyo.setToolTip("Asigna un apoyo en la cara extrema seleccionada de una viga o columna")
        self.accion_carga = QAction("Asignar carga", self, triggered=self.asignar_carga)
        self.accion_carga.setToolTip("Asigna una carga a las losas, vigas, columnas o caras seleccionadas")
        self.accion_casos = QAction("Casos y combinaciones…", self, triggered=self._gestionar_casos)
        self.accion_analizar = QAction("Analizar", self, shortcut="F5", triggered=self.analizar)

        self.accion_ver_modelo = QAction("Modelo", self, checkable=True, checked=True,
                                         triggered=lambda: self.volver_al_modelo())
        self.accion_ver_esfuerzos = QAction("Esfuerzos", self, checkable=True,
                                            triggered=lambda: self.mostrar_esfuerzos())
        grupo_vista = QActionGroup(self)
        grupo_vista.setExclusive(True)
        grupo_vista.addAction(self.accion_ver_modelo)
        grupo_vista.addAction(self.accion_ver_esfuerzos)

        self.accion_ayuda = QAction("Cómo se usa", self, triggered=lambda: QMessageBox.information(self, "Ayuda", AYUDA))

    def _crear_menus(self) -> None:
        menu = self.menuBar().addMenu("&Archivo")
        menu.addActions([self.accion_importar, self.accion_abrir, self.accion_guardar])
        menu.addSeparator()
        menu.addAction(self.accion_salir)

        menu = self.menuBar().addMenu("A&signar")
        menu.addActions([self.accion_apoyo, self.accion_carga])
        menu.addSeparator()
        menu.addAction(self.accion_casos)

        menu = self.menuBar().addMenu("A&nalizar")
        menu.addAction(self.accion_analizar)

        self.menu_ver = QMenu("&Ver", self)
        self.menuBar().addMenu(self.menu_ver)
        self.menu_ver.addActions([self.accion_ver_modelo, self.accion_ver_esfuerzos])
        self.menu_ver.addSeparator()
        for tipo, texto in (("LOSA", "losas"), ("MURO", "muros"), ("VIGA", "vigas"), ("COLUMNA", "columnas")):
            accion = QAction(f"Mostrar {texto}", self, checkable=True, checked=True)
            accion.toggled.connect(lambda visible, t=tipo: self.visor.set_visibilidad(t, visible))
            self.menu_ver.addAction(accion)
        self.menu_ver.addSeparator()
        for texto, funcion in (
            ("Vista isométrica", lambda: self.visor.vista_isometrica()),
            ("Vista en planta", lambda: self.visor.vista_planta()),
            ("Vista inferior (bases de columnas)", lambda: self._vista_inferior()),
            ("Vista frontal", lambda: self.visor.vista_frontal()),
            ("Vista lateral", lambda: self.visor.vista_lateral()),
        ):
            self.menu_ver.addAction(QAction(texto, self, triggered=funcion))

        menu = self.menuBar().addMenu("A&yuda")
        menu.addAction(self.accion_ayuda)

    def _crear_barra_de_herramientas(self) -> None:
        barra = self.addToolBar("Principal")
        barra.setMovable(False)
        barra.addAction(self.accion_importar)
        barra.addSeparator()
        barra.addWidget(QLabel(" Seleccionar: "))
        barra.addAction(self.accion_modo_elemento)
        barra.addAction(self.accion_modo_cara)
        barra.addSeparator()
        barra.addWidget(QLabel(" Caso activo: "))
        self.combo_caso = QComboBox()
        self.combo_caso.setMinimumWidth(110)
        barra.addWidget(self.combo_caso)
        barra.addAction(self.accion_apoyo)
        barra.addAction(self.accion_carga)
        barra.addAction(self.accion_casos)
        barra.addSeparator()
        barra.addWidget(QLabel(" Ver: "))
        barra.addAction(self.accion_ver_modelo)
        barra.addAction(self.accion_ver_esfuerzos)
        boton_vistas = QToolButton()
        boton_vistas.setText("Vistas ▾")
        boton_vistas.setMenu(self.menu_ver)
        boton_vistas.setPopupMode(QToolButton.InstantPopup)
        barra.addWidget(boton_vistas)

        espacio = QWidget()
        espacio.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        barra.addWidget(espacio)

        self.boton_analizar = QToolButton()
        self.boton_analizar.setDefaultAction(self.accion_analizar)
        self.boton_analizar.setText("▶  ANALIZAR")
        self.boton_analizar.setToolTip("Resuelve todos los casos con CalculiX y muestra los esfuerzos (F5)")
        self.boton_analizar.setStyleSheet(ESTILO_BOTON_ANALIZAR)
        barra.addWidget(self.boton_analizar)
        self.accion_analizar.setText("▶  ANALIZAR")

    def _conectar_senales(self) -> None:
        self.accion_modo_elemento.toggled.connect(lambda marcado: marcado and self.visor.set_modo(MODO_ELEMENTO))
        self.accion_modo_cara.toggled.connect(lambda marcado: marcado and self.visor.set_modo(MODO_CARA))
        self.combo_caso.currentIndexChanged.connect(lambda _: self._refrescar_asignaciones())
        self.visor.seleccion_cambiada.connect(self._al_cambiar_seleccion)
        self.visor.punto_sondeado.connect(lambda texto: self.statusBar().showMessage(texto))
        self.panel_elementos.elemento_seleccionado.connect(lambda tipo, id_: self.visor.seleccionar([Objetivo(tipo, id_)]))
        self.panel_asignaciones.eliminar_solicitado.connect(self.eliminar_asignaciones)
        self.panel_esfuerzos.cambio.connect(self.mostrar_esfuerzos)
        self.panel_esfuerzos.volver_al_modelo.connect(self.volver_al_modelo)

    # ------------------------------------------------------------------
    # Estado derivado
    # ------------------------------------------------------------------
    def caso_activo(self) -> str:
        return self.combo_caso.currentData() or ""

    def _actualizar_combo_caso(self) -> None:
        actual = self.caso_activo()
        self.combo_caso.blockSignals(True)
        self.combo_caso.clear()
        for caso in self.casos.values():
            self.combo_caso.addItem(f"{caso.id} — {caso.nombre}", caso.id)
        indice = self.combo_caso.findData(actual)
        self.combo_caso.setCurrentIndex(max(indice, 0))
        self.combo_caso.blockSignals(False)

    def _actualizar_estado_de_acciones(self) -> None:
        hay_modelo = bool(self.proyecto.todas_las_barras())
        ocupado = self._worker is not None and self._worker.isRunning()
        self.accion_analizar.setEnabled(hay_modelo and not ocupado)
        self.accion_apoyo.setEnabled(hay_modelo)
        self.accion_carga.setEnabled(hay_modelo or bool(self.proyecto.losas))
        self.accion_ver_esfuerzos.setEnabled(self.resultado is not None)

    def _al_cambiar_seleccion(self, objetivos: List[Objetivo]) -> None:
        if not objetivos:
            self._etiqueta_seleccion.setText("Sin selección")
            return
        textos = [descripcion(self.proyecto, o) for o in objetivos[:3]]
        if len(objetivos) > 3:
            textos.append(f"y {len(objetivos) - 3} más")
        self._etiqueta_seleccion.setText("Selección: " + "; ".join(textos))

    def _vista_inferior(self) -> None:
        self.visor.view_xy(negative=True)
        self.visor.reset_camera()
        self.visor.render()

    # ------------------------------------------------------------------
    # Archivo
    # ------------------------------------------------------------------
    def _importar_ifc(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(self, "Importar IFC", "", "Archivos IFC (*.ifc)")
        if ruta:
            self.importar_archivo(ruta)

    def importar_archivo(self, ruta: str) -> bool:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            proyecto = importar_ifc(ruta)
        except Exception as exc:
            logger.exception("Error al importar el IFC")
            QMessageBox.critical(self, "Error al importar", str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()

        casos, combinaciones = casos_y_combinaciones_por_defecto()
        self._establecer_proyecto(proyecto, [], casos, combinaciones, {}, {}, {})

        sin_soporte = [l.id for l in proyecto.losas.values() if not any(l.vigas_soporte)]
        texto = (
            f"Vigas: {len(proyecto.vigas)}\nColumnas: {len(proyecto.columnas)}\n"
            f"Losas: {len(proyecto.losas)}\nMuros: {len(proyecto.muros)}"
        )
        if sin_soporte:
            texto += (
                f"\n\nNo se detectaron vigas de soporte para las losas: {', '.join(sin_soporte)}. Su peso propio "
                "y sus cargas de área no se transferirán a las vigas."
            )
        QMessageBox.information(self, "Importación completa", texto)
        self.statusBar().showMessage(
            "Modelo importado. Ahora asigne los apoyos: modo Cara + Ver → Vista inferior, clic en la base de cada columna."
        )
        return True

    def _establecer_proyecto(self, proyecto, apoyos, casos, combinaciones, cargas_area, cargas_lineales, cargas_puntuales) -> None:
        self.proyecto = proyecto
        self.apoyos = apoyos
        self.casos = casos or casos_y_combinaciones_por_defecto()[0]
        self.combinaciones = combinaciones
        self.cargas_area_por_caso = cargas_area
        self.cargas_lineales_por_caso = cargas_lineales
        self.cargas_puntuales_por_caso = cargas_puntuales
        self.resultado = None
        self._resultado_desactualizado = False
        self.panel_esfuerzos.establecer_resultados([], [])
        self.panel_esfuerzos.mostrar_resumen("Ejecute el análisis para ver los esfuerzos.", {})

        self.visor.mostrar_modelo(proyecto)
        self.panel_elementos.actualizar(proyecto)
        self._actualizar_combo_caso()
        self.accion_ver_modelo.setChecked(True)
        self._refrescar_asignaciones()
        self._actualizar_estado_de_acciones()

    def _guardar_proyecto(self) -> None:
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar proyecto", "", "Proyecto (*.json)")
        if not ruta:
            return
        guardar_proyecto(
            Path(ruta), self.proyecto, self.apoyos, self.casos, self.combinaciones,
            self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso,
        )
        self.statusBar().showMessage(f"Proyecto guardado en {ruta}")

    def _abrir_proyecto(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir proyecto", "", "Proyecto (*.json)")
        if not ruta:
            return
        try:
            datos = cargar_proyecto(Path(ruta))
        except Exception as exc:
            logger.exception("Error al abrir el proyecto")
            QMessageBox.critical(self, "Error al abrir", str(exc))
            return
        self._establecer_proyecto(*datos)
        self.statusBar().showMessage(f"Proyecto cargado desde {ruta}")

    # ------------------------------------------------------------------
    # Asignaciones
    # ------------------------------------------------------------------
    def _ids_en_uso(self) -> set:
        usados = {a.id for a in self.apoyos}
        for por_caso in (self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso):
            usados.update(c.id for lista in por_caso.values() for c in lista)
        return usados

    def _siguiente_id(self, prefijo: str) -> str:
        usados = self._ids_en_uso()
        n = 1
        while f"{prefijo}{n}" in usados:
            n += 1
        return f"{prefijo}{n}"

    def _marcar_resultados_desactualizados(self) -> None:
        if self.resultado is not None and not self._resultado_desactualizado:
            self._resultado_desactualizado = True
            self.statusBar().showMessage(
                "Los esfuerzos mostrados son anteriores a los últimos cambios: vuelva a pulsar ANALIZAR."
            )

    def asignar_apoyo(self) -> None:
        dialogo = DialogoApoyo(self.proyecto, self.visor.seleccion, self)
        if not dialogo.exec():
            return
        restricciones = dialogo.restricciones()
        nodos = dialogo.nodos_elegidos()
        if not nodos:
            return
        ux, uy, uz, rx, ry, rz = restricciones
        for nodo in nodos:
            self.apoyos = [a for a in self.apoyos if a.nodo_id != nodo]  # un solo apoyo por nodo
            self.apoyos.append(
                Apoyo(id=self._siguiente_id("AP"), nodo_id=nodo, ux=ux, uy=uy, uz=uz, rx=rx, ry=ry, rz=rz,
                      nombre=f"Apoyo en el nodo {nodo}")
            )
        self._marcar_resultados_desactualizados()
        self._refrescar_asignaciones()
        self.statusBar().showMessage(f"Apoyo asignado en: {', '.join(nodos)}")

    def asignar_carga(self) -> None:
        dialogo = DialogoCargas(self.proyecto, self.visor.seleccion, self.casos, self.caso_activo(), self)
        if not dialogo.exec():
            return
        caso_id, presion, w, direccion = dialogo.parametros()
        creadas, errores = 0, []
        for objetivo in self.visor.seleccion:
            try:
                cargas = cargas_para_objetivo(
                    self.proyecto, objetivo, caso_id, self._siguiente_id("CARGA"), presion, w, direccion
                )
            except ErrorAsignacion as exc:
                errores.append(str(exc))
                continue
            for carga in cargas:
                carga.id = self._siguiente_id("CARGA")
                self._guardar_carga(carga)
                creadas += 1
        if errores:
            QMessageBox.warning(self, "Algunas cargas no se asignaron", "\n\n".join(errores))
        if creadas:
            indice = self.combo_caso.findData(caso_id)
            if indice >= 0:
                self.combo_caso.setCurrentIndex(indice)  # muestra el caso al que se asignó
            self._marcar_resultados_desactualizados()
            self._refrescar_asignaciones()
            self.statusBar().showMessage(f"{creadas} carga(s) asignada(s) al caso {caso_id}.")

    def _guardar_carga(self, carga) -> None:
        if isinstance(carga, CargaArea):
            destino = self.cargas_area_por_caso
        elif isinstance(carga, CargaLineal):
            destino = self.cargas_lineales_por_caso
        else:
            destino = self.cargas_puntuales_por_caso
        destino.setdefault(carga.caso_id, []).append(carga)

    def eliminar_asignaciones(self, claves: Iterable[Tuple[str, str]]) -> None:
        claves = set(claves)
        self.apoyos = [a for a in self.apoyos if ("Apoyo", a.id) not in claves]
        for tipo, por_caso in (
            ("Carga de área", self.cargas_area_por_caso),
            ("Carga lineal", self.cargas_lineales_por_caso),
            ("Carga puntual", self.cargas_puntuales_por_caso),
        ):
            for caso_id in list(por_caso):
                por_caso[caso_id] = [c for c in por_caso[caso_id] if (tipo, c.id) not in claves]
        self._marcar_resultados_desactualizados()
        self._refrescar_asignaciones()

    def _refrescar_asignaciones(self) -> None:
        caso = self.caso_activo()
        self.visor.mostrar_asignaciones(
            self.apoyos,
            self.cargas_area_por_caso.get(caso, []),
            self.cargas_lineales_por_caso.get(caso, []),
            self.cargas_puntuales_por_caso.get(caso, []),
        )
        filas = [("Apoyo", a.id, f"{a.nombre} [gdl restringidos: {', '.join(map(str, a.grados_restringidos()))}]", "todos")
                 for a in self.apoyos]
        for tipo, por_caso in (
            ("Carga de área", self.cargas_area_por_caso),
            ("Carga lineal", self.cargas_lineales_por_caso),
            ("Carga puntual", self.cargas_puntuales_por_caso),
        ):
            filas += [(tipo, c.id, c.nombre, caso_id) for caso_id, lista in por_caso.items() for c in lista]
        self.panel_asignaciones.mostrar(filas)

    def _gestionar_casos(self) -> None:
        dialogo = DialogoCasos(self.casos, self.combinaciones, self)
        if dialogo.exec():
            self.casos, self.combinaciones = dialogo.resultado()
            self._actualizar_combo_caso()
            self._refrescar_asignaciones()
            self._marcar_resultados_desactualizados()

    # ------------------------------------------------------------------
    # Análisis
    # ------------------------------------------------------------------
    def analizar(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        errores, advertencias = validar_modelo(self.proyecto, self.casos, self.combinaciones, self.apoyos)
        if errores:
            QMessageBox.warning(self, "Todavía no se puede analizar", "\n\n".join(f"• {e}" for e in errores))
            return
        if advertencias:
            respuesta = QMessageBox.question(
                self, "Advertencias antes de analizar",
                "\n\n".join(f"• {a}" for a in advertencias) + "\n\n¿Analizar de todos modos?",
            )
            if respuesta != QMessageBox.Yes:
                return
        self._iniciar_hilo()

    def _iniciar_hilo(self) -> None:
        proyecto, casos, combinaciones, apoyos = self.proyecto, dict(self.casos), dict(self.combinaciones), list(self.apoyos)
        area = {k: list(v) for k, v in self.cargas_area_por_caso.items()}
        lineales = {k: list(v) for k, v in self.cargas_lineales_por_caso.items()}
        puntuales = {k: list(v) for k, v in self.cargas_puntuales_por_caso.items()}
        configuracion = self.configuracion

        def trabajo(avanzar):
            return ejecutar_analisis(
                proyecto, casos, combinaciones, apoyos, area, lineales, puntuales,
                configuracion.directorio_trabajo_absoluto, configuracion.ruta_ccx, avanzar,
            )

        self._worker = TrabajadorAnalisis(trabajo, self)
        self._worker.terminado.connect(self._al_terminar_analisis)
        self._worker.fallido.connect(self._al_fallar_analisis)
        self._worker.finished.connect(self._al_finalizar_hilo)

        self._dialogo_progreso = QProgressDialog("Preparando el análisis...", "", 0, 0, self)
        self._dialogo_progreso.setWindowTitle("Analizando")
        self._dialogo_progreso.setCancelButton(None)
        self._dialogo_progreso.setWindowModality(Qt.WindowModal)
        self._dialogo_progreso.setMinimumDuration(0)
        self._worker.progreso.connect(self._dialogo_progreso.setLabelText)
        self._dialogo_progreso.show()

        self._actualizar_estado_de_acciones()
        self._worker.start()

    def _cerrar_dialogo_progreso(self) -> None:
        if self._dialogo_progreso is not None:
            self._dialogo_progreso.close()
            self._dialogo_progreso = None

    def _al_finalizar_hilo(self) -> None:
        self._actualizar_estado_de_acciones()

    def _al_fallar_analisis(self, mensaje: str) -> None:
        self._cerrar_dialogo_progreso()
        QMessageBox.critical(self, "El análisis no pudo completarse", mensaje)
        self.statusBar().showMessage("El análisis falló.")

    def _al_terminar_analisis(self, resultado: ResultadoAnalisis) -> None:
        self._cerrar_dialogo_progreso()
        self.resultado = resultado
        self._resultado_desactualizado = False
        self.panel_esfuerzos.establecer_resultados(
            [(c.id, c.nombre) for c in self.casos.values() if c.id in resultado.tensiones_por_caso],
            [(c.id, c.nombre) for c in self.combinaciones.values() if c.id in resultado.tensiones_por_combinacion],
        )
        self._actualizar_estado_de_acciones()
        self.accion_ver_esfuerzos.setChecked(True)
        self.pestanas.setCurrentWidget(self.panel_esfuerzos)
        self.mostrar_esfuerzos()
        if resultado.advertencias:
            QMessageBox.warning(self, "Análisis completo, con advertencias", "\n\n".join(f"• {a}" for a in resultado.advertencias))

    # ------------------------------------------------------------------
    # Vista de esfuerzos
    # ------------------------------------------------------------------
    def volver_al_modelo(self) -> None:
        self.visor.volver_al_modelo()
        self.accion_ver_modelo.setChecked(True)
        self.statusBar().showMessage("Vista del modelo.")

    def mostrar_esfuerzos(self) -> None:
        resultado = self.resultado
        parametros = self.panel_esfuerzos.parametros()
        if resultado is None or parametros is None:
            return
        self.accion_ver_esfuerzos.setChecked(True)

        magnitud = esfuerzos.MAGNITUDES[parametros.magnitud]
        if parametros.resultado == ENVOLVENTE:
            envolvente = calcular_envolvente(resultado.tensiones_por_combinacion, parametros.magnitud)
            valores_kpa, origen = envolvente.valores, "Envolvente de combinaciones"
        else:
            valores_kpa = esfuerzos.calcular(parametros.magnitud, resultado.resultados_disponibles()[parametros.resultado])
            envolvente, origen = None, self.panel_esfuerzos.combo_resultado.currentText()
        valores = valores_kpa * esfuerzos.FACTOR_KPA_A_MPA

        if parametros.escala_automatica:
            minimo, maximo = float(valores.min()), float(valores.max())
            if maximo - minimo < 1e-9:
                minimo, maximo = minimo - 0.5, maximo + 0.5
            self.panel_esfuerzos.fijar_escala(minimo, maximo)
        else:
            minimo, maximo = parametros.minimo, parametros.maximo
            if maximo <= minimo:
                maximo = minimo + 1e-6

        i_critico = int(np.argmin(valores) if magnitud.envolvente == "min" else np.argmax(valores))
        valor_critico = float(valores[i_critico])
        celda_critica = next(
            (c for c, conectividad in enumerate(resultado.malla_tensiones.conectividad) if i_critico in conectividad), 0
        )
        barra_critica = resultado.barra_de_cada_celda()[celda_critica]
        resumen = (
            f"{magnitud.nombre} — {origen}\n"
            f"Valor más desfavorable: {valor_critico:.3f} MPa en la barra {barra_critica}"
            + (f" (combinación {envolvente.combinacion_por_nodo[i_critico]})" if envolvente is not None else "")
            + ".\nHaga clic en el visor para leer el valor en un punto."
        )
        if self._resultado_desactualizado:
            resumen = "⚠ Resultados anteriores a los últimos cambios.\n" + resumen

        self.visor.mostrar_esfuerzos(
            resultado.malla_tensiones, valores, resultado.barra_de_cada_celda(),
            titulo=f"{magnitud.nombre} [MPa]", rango=(minimo, maximo), cmap=parametros.colormap,
            texto=f"{origen}\n{magnitud.nombre}",
        )
        self.panel_esfuerzos.mostrar_resumen(resumen, resultado.resumen_por_barra(valores))

    # ------------------------------------------------------------------
    def closeEvent(self, evento) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        self.visor.close()
        super().closeEvent(evento)
