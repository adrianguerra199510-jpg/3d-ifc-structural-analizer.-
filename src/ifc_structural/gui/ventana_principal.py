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
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..combinaciones import esfuerzos
from ..combinaciones.motor_superposicion import calcular_envolvente
from ..config import Configuracion, cargar_configuracion
from ..core.asignacion import (
    ErrorAsignacion,
    Objetivo,
    apoyo_para_objetivo,
    asignar_material,
    cargas_desde_parametros,
    descripcion,
)
from ..core.cargas import Apoyo, ApoyoArea, CargaArea, CargaLineal, CargaPuntual, Conexion
from ..core.casos_carga import CasoCarga, Combinacion, casos_y_combinaciones_por_defecto
from ..core.materiales import descripcion_material
from ..core.modelo import Proyecto
from ..core.persistencia import cargar_proyecto, guardar_proyecto
from ..ifc.importador import importar_ifc
from ..solver.analisis import ResultadoAnalisis, ejecutar_analisis, validar_modelo
from .barra_vista import BarraVista
from .dialogo_acerca import DialogoAcerca
from .dialogo_apoyos import DialogoApoyo
from .dialogo_cargas import DialogoCargas
from .dialogo_casos import DialogoCasos
from .dialogo_conexion import DialogoConexion
from .dialogo_materiales import DialogoAsignarMaterial, DialogoMateriales
from .estilo import etiqueta_logo
from .iconos import icono, icono_aplicacion, icono_blanco
from .pagina_bienvenida import PaginaBienvenida
from .panel_asignaciones import PanelAsignaciones
from .panel_elementos import PanelElementos
from .panel_resultados import ENVOLVENTE, PanelEsfuerzos
from .trabajador_analisis import TrabajadorAnalisis
from .visor_3d import MODO_CARA, MODO_ELEMENTO, Visor3D

logger = logging.getLogger(__name__)

AYUDA = (
    "<b>Cómo se usa</b><ol>"
    "<li><b>Archivo → Importar IFC</b>.</li>"
    "<li>Haga <b>clic</b> en un elemento para seleccionarlo (modo <b>Elemento</b>) o en una de sus caras "
    "(modo <b>Cara</b>); Ctrl o Mayús + clic suma más. Con <b>clic derecho</b> sobre la selección elige qué "
    "asignar: carga, apoyo, material o una conexión entre dos elementos.</li>"
    "<li><b>Apoyos</b>: puntuales en la cara extrema de una columna (use <b>Ver → Vista inferior</b>) o de "
    "área a lo largo de una barra o de las vigas de una losa; usted elige cuántos grados de libertad se restringen.</li>"
    "<li><b>Cargas</b>: elija el caso activo, el tipo (presión, distribuida, puntual, momento), su forma "
    "(uniforme, triangular, trapezoidal) y la dirección de aplicación.</li>"
    "<li><b>Materiales</b>: cree los suyos (o use las plantillas) y asígnelos a los elementos.</li>"
    "<li><b>Conectar elementos</b>: articulación o enlace rígido entre dos elementos seleccionados.</li>"
    "<li>Pulse <b>▶ ANALIZAR</b> (F5) y revise los esfuerzos en el panel <b>Esfuerzos</b>.</li></ol>"
    "Con el botón central del mouse se desplaza la vista; con la rueda se acerca o aleja; arrastrando con "
    "el botón izquierdo se gira."
)


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador Estructural 3D (IFC + CalculiX)")
        self.resize(1440, 880)

        self.configuracion: Configuracion = cargar_configuracion()
        self.proyecto = Proyecto()
        self.apoyos: list = []  # Apoyo (nodales) y ApoyoArea (continuos)
        self.conexiones: List[Conexion] = []
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

        self.setWindowIcon(icono_aplicacion())
        self.visor = Visor3D(self)
        self.barra_vista = BarraVista(self)
        self.pagina_bienvenida = PaginaBienvenida(self)
        self.contenedor_visor = QWidget(self)
        capa = QVBoxLayout(self.contenedor_visor)
        capa.setContentsMargins(0, 0, 0, 0)
        capa.setSpacing(0)
        capa.addWidget(self.barra_vista)
        capa.addWidget(self.visor, 1)
        self.paginas = QStackedWidget(self)
        self.paginas.addWidget(self.pagina_bienvenida)
        self.paginas.addWidget(self.contenedor_visor)
        self.setCentralWidget(self.paginas)

        self.panel_elementos = PanelElementos(self)
        dock_elementos = QDockWidget("Elementos del modelo", self)
        self._docks = []
        dock_elementos.setWidget(self.panel_elementos)
        dock_elementos.setMinimumWidth(255)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_elementos)
        self._docks.append(dock_elementos)

        self.panel_asignaciones = PanelAsignaciones(self)
        self.panel_esfuerzos = PanelEsfuerzos(self)
        self.pestanas = QTabWidget()
        self.pestanas.addTab(self.panel_asignaciones, "Asignaciones")
        self.pestanas.addTab(self.panel_esfuerzos, "Esfuerzos")
        dock_derecho = QDockWidget("Asignaciones y resultados", self)
        dock_derecho.setMinimumWidth(320)
        dock_derecho.setWidget(self.pestanas)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_derecho)
        self._docks.append(dock_derecho)
        for dock in self._docks:
            dock.hide()  # los paneles aparecen al cargar un modelo (la pantalla de inicio los oculta)

        self._etiqueta_seleccion = QLabel("Sin selección")
        self._etiqueta_seleccion.setObjectName("etiquetaEstado")
        self._etiqueta_modo = QLabel("Modo: elemento")
        self._etiqueta_modo.setObjectName("etiquetaEstado")
        self._etiqueta_caso = QLabel("Caso: —")
        self._etiqueta_caso.setObjectName("etiquetaEstado")
        for etiqueta in (self._etiqueta_modo, self._etiqueta_caso, self._etiqueta_seleccion):
            self.statusBar().addPermanentWidget(etiqueta)

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
        def accion(texto, funcion=None, icono_=None, atajo=None, ayuda=None, **kwargs):
            a = QAction(texto, self, **kwargs)
            if icono_:
                a.setIcon(icono(icono_))
            if atajo:
                a.setShortcut(atajo)
            if ayuda:
                a.setToolTip(ayuda)
                a.setStatusTip(ayuda)
            if funcion:
                a.triggered.connect(funcion)
            return a

        self.accion_importar = accion("Importar", self._importar_ifc, "importar", "Ctrl+I", "Importar un modelo IFC (Ctrl+I)")
        self.accion_abrir = accion("Abrir", self._abrir_proyecto, "abrir", "Ctrl+O", "Abrir un proyecto guardado (Ctrl+O)")
        self.accion_guardar = accion("Guardar", self._guardar_proyecto, "guardar", "Ctrl+S", "Guardar el proyecto (Ctrl+S)")
        self.accion_salir = accion("Salir", self.close)

        self.accion_modo_elemento = accion(
            "Elemento", icono_="sel_elemento", ayuda="Seleccionar elementos completos con el mouse",
            checkable=True, checked=True,
        )
        self.accion_modo_cara = accion(
            "Cara", icono_="sel_cara", ayuda="Seleccionar caras de los elementos con el mouse (para apoyos y cargas)",
            checkable=True,
        )
        grupo_modo = QActionGroup(self)
        grupo_modo.setExclusive(True)
        grupo_modo.addAction(self.accion_modo_elemento)
        grupo_modo.addAction(self.accion_modo_cara)

        self.accion_apoyo = accion("Apoyo", self.asignar_apoyo, "apoyo", ayuda="Asignar un apoyo (puntual o de área) a la selección")
        self.accion_carga = accion("Carga", self.asignar_carga, "carga_dist", ayuda="Asignar cargas (presión, distribuida, puntual) a la selección")
        self.accion_material = accion("Material", self.asignar_material, "material", ayuda="Asignar un material a los elementos seleccionados")
        self.accion_conexion = accion("Conectar", self.asignar_conexion, "conectar",
                                      ayuda="Articular o unir dos elementos seleccionados (constraint)")
        self.accion_materiales = accion("Materiales", self.gestionar_materiales, "materiales", ayuda="Crear y editar materiales")
        self.accion_casos = accion("Casos", self._gestionar_casos, "casos", ayuda="Casos de carga y combinaciones")
        self.accion_analizar = accion("ANALIZAR", self.analizar, None, "F5", "Resolver todos los casos con CalculiX y mostrar los esfuerzos (F5)")
        self.accion_analizar.setIcon(icono_blanco("analizar"))

        self.accion_ver_modelo = accion("Modelo", lambda: self.volver_al_modelo(), "modelo",
                                        ayuda="Volver a la vista del modelo", checkable=True, checked=True)
        self.accion_ver_esfuerzos = accion("Esfuerzos", lambda: self.mostrar_esfuerzos(), "esfuerzos",
                                           ayuda="Ver el mapa de esfuerzos del último análisis", checkable=True)
        grupo_vista = QActionGroup(self)
        grupo_vista.setExclusive(True)
        grupo_vista.addAction(self.accion_ver_modelo)
        grupo_vista.addAction(self.accion_ver_esfuerzos)

        self.accion_ayuda = accion("Cómo se usa", lambda: QMessageBox.information(self, "Ayuda", AYUDA), "ayuda", "F1")
        self.accion_acerca = accion("Acerca de…", lambda: DialogoAcerca(self).exec(), "info")

    def _crear_menus(self) -> None:
        menu = self.menuBar().addMenu("&Archivo")
        menu.addActions([self.accion_importar, self.accion_abrir, self.accion_guardar])
        menu.addSeparator()
        menu.addAction(self.accion_salir)

        menu = self.menuBar().addMenu("A&signar")
        menu.addActions([self.accion_apoyo, self.accion_carga, self.accion_material, self.accion_conexion])
        menu.addSeparator()
        menu.addActions([self.accion_materiales, self.accion_casos])

        menu = self.menuBar().addMenu("A&nalizar")
        menu.addAction(self.accion_analizar)

        self.menu_ver = QMenu("&Ver", self)
        self.menuBar().addMenu(self.menu_ver)
        self.menu_ver.addActions([self.accion_ver_modelo, self.accion_ver_esfuerzos])
        self.menu_ver.addSeparator()
        for dock in self._docks:  # reabrir los paneles si se cerraron
            self.menu_ver.addAction(dock.toggleViewAction())
        self.menu_ver.addSeparator()
        self._acciones_visibilidad = {}
        for tipo, texto, nombre_icono in (
            ("LOSA", "losas", "losa"), ("MURO", "muros", "muro"), ("VIGA", "vigas", "viga"), ("COLUMNA", "columnas", "columna"),
        ):
            accion = QAction(icono(nombre_icono), f"Mostrar {texto}", self, checkable=True, checked=True)
            accion.toggled.connect(lambda visible, t=tipo: self._cambiar_visibilidad(t, visible, desde_menu=True))
            self.menu_ver.addAction(accion)
            self._acciones_visibilidad[tipo] = accion
        self.menu_ver.addSeparator()
        for clave, texto in (
            ("vista_iso", "Vista isométrica"), ("vista_planta", "Vista en planta"),
            ("vista_inferior", "Vista inferior (bases de columnas)"), ("vista_frontal", "Vista frontal"),
            ("vista_lateral", "Vista lateral"), ("ajustar", "Encuadrar el modelo"),
        ):
            self.menu_ver.addAction(QAction(icono(clave), texto, self, triggered=lambda _=False, c=clave: self._cambiar_vista(c)))

        menu = self.menuBar().addMenu("A&yuda")
        menu.addActions([self.accion_ayuda, self.accion_acerca])

    def _crear_barra_de_herramientas(self) -> None:
        barra = QToolBar("Principal", self)
        self.addToolBar(barra)
        barra.setMovable(False)
        barra.setIconSize(QSize(26, 26))
        barra.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.barra_principal = barra

        marca = QWidget()
        fila = QHBoxLayout(marca)
        fila.setContentsMargins(2, 0, 12, 0)
        fila.setSpacing(9)
        fila.addWidget(etiqueta_logo(34))
        textos = QVBoxLayout()
        textos.setSpacing(0)
        nombre = QLabel("Analizador Estructural")
        nombre.setStyleSheet("font-size: 13px; font-weight: 800; color: #0d2a50;")
        detalle = QLabel("IFC · CalculiX")
        detalle.setStyleSheet("font-size: 10px; color: #5b6b7c;")
        textos.addWidget(nombre)
        textos.addWidget(detalle)
        fila.addLayout(textos)
        marca.setToolTip("Analizador Estructural 3D")
        barra.addWidget(marca)
        barra.addSeparator()

        barra.addActions([self.accion_importar, self.accion_abrir, self.accion_guardar])
        for accion in (self.accion_importar, self.accion_abrir, self.accion_guardar):
            barra.widgetForAction(accion).setToolButtonStyle(Qt.ToolButtonIconOnly)  # más compacto
        barra.addSeparator()
        barra.addActions([self.accion_modo_elemento, self.accion_modo_cara])
        barra.addSeparator()

        contenedor_caso = QWidget()
        capa = QVBoxLayout(contenedor_caso)
        capa.setContentsMargins(4, 0, 4, 0)
        capa.setSpacing(2)
        capa.addWidget(QLabel("Caso activo"))
        self.combo_caso = QComboBox()
        self.combo_caso.setMinimumWidth(130)
        capa.addWidget(self.combo_caso)
        barra.addWidget(contenedor_caso)
        barra.addSeparator()

        barra.addActions([self.accion_apoyo, self.accion_carga, self.accion_material, self.accion_conexion])
        barra.addSeparator()
        barra.addActions([self.accion_materiales, self.accion_casos])

        self.barra_vista.agregar_accion(self.accion_ver_modelo)
        self.barra_vista.agregar_accion(self.accion_ver_esfuerzos)

        espacio = QWidget()
        espacio.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        barra.addWidget(espacio)

        self.boton_analizar = QToolButton()
        self.boton_analizar.setObjectName("botonAnalizar")
        self.boton_analizar.setDefaultAction(self.accion_analizar)
        self.boton_analizar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.boton_analizar.setIconSize(QSize(18, 18))
        self.boton_analizar.setCursor(Qt.PointingHandCursor)
        barra.addWidget(self.boton_analizar)

    def _conectar_senales(self) -> None:
        self.accion_modo_elemento.toggled.connect(lambda marcado: marcado and self._cambiar_modo(MODO_ELEMENTO))
        self.accion_modo_cara.toggled.connect(lambda marcado: marcado and self._cambiar_modo(MODO_CARA))
        self.combo_caso.currentIndexChanged.connect(lambda _: self._al_cambiar_caso())
        self.pagina_bienvenida.importar_solicitado.connect(self._importar_ifc)
        self.pagina_bienvenida.abrir_solicitado.connect(self._abrir_proyecto)
        self.barra_vista.vista_solicitada.connect(self._cambiar_vista)
        self.barra_vista.visibilidad_cambiada.connect(lambda t, v: self._cambiar_visibilidad(t, v, desde_menu=False))
        self.visor.seleccion_cambiada.connect(self._al_cambiar_seleccion)
        self.visor.menu_contextual.connect(self._mostrar_menu_contextual)
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
        caso_actual = self.casos.get(self.caso_activo())
        self._etiqueta_caso.setText(f"Caso: {caso_actual.id}" if caso_actual else "Caso: —")

    def _actualizar_estado_de_acciones(self) -> None:
        hay_modelo = bool(self.proyecto.todas_las_barras())
        ocupado = self._worker is not None and self._worker.isRunning()
        self.accion_analizar.setEnabled(hay_modelo and not ocupado)
        self.accion_apoyo.setEnabled(hay_modelo)
        self.accion_material.setEnabled(hay_modelo or bool(self.proyecto.losas))
        self.accion_conexion.setEnabled(len(self.proyecto.todas_las_barras()) >= 2)
        self.accion_materiales.setEnabled(True)
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

    def _cambiar_modo(self, modo: str) -> None:
        self.visor.set_modo(modo)
        self._etiqueta_modo.setText("Modo: cara" if modo == MODO_CARA else "Modo: elemento")

    def _al_cambiar_caso(self) -> None:
        caso = self.casos.get(self.caso_activo())
        self._etiqueta_caso.setText(f"Caso: {caso.id}" if caso else "Caso: —")
        self._refrescar_asignaciones()

    def _cambiar_vista(self, clave: str) -> None:
        if clave == "vista_inferior":
            self._vista_inferior()
            return
        {
            "vista_iso": self.visor.vista_isometrica, "vista_planta": self.visor.vista_planta,
            "vista_frontal": self.visor.vista_frontal, "vista_lateral": self.visor.vista_lateral,
        }.get(clave, lambda: None)()
        self.visor.reset_camera()
        self.visor.render()

    def _cambiar_visibilidad(self, tipo: str, visible: bool, desde_menu: bool) -> None:
        self.visor.set_visibilidad(tipo, visible)
        if desde_menu:
            self.barra_vista.fijar_visibilidad(tipo, visible)
        else:
            accion = self._acciones_visibilidad[tipo]
            accion.blockSignals(True)
            accion.setChecked(visible)
            accion.blockSignals(False)

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
        self._establecer_proyecto(proyecto, [], casos, combinaciones, {}, {}, {}, [])

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

    def _establecer_proyecto(
        self, proyecto, apoyos, casos, combinaciones, cargas_area, cargas_lineales, cargas_puntuales, conexiones=None
    ) -> None:
        self.proyecto = proyecto
        self.apoyos = apoyos
        self.conexiones = list(conexiones or [])
        self.casos = casos or casos_y_combinaciones_por_defecto()[0]
        self.combinaciones = combinaciones
        self.cargas_area_por_caso = cargas_area
        self.cargas_lineales_por_caso = cargas_lineales
        self.cargas_puntuales_por_caso = cargas_puntuales
        self.resultado = None
        self._resultado_desactualizado = False
        self.panel_esfuerzos.establecer_resultados([], [])
        self.panel_esfuerzos.mostrar_resumen("Ejecute el análisis para ver los esfuerzos.", {})

        self.paginas.setCurrentWidget(self.contenedor_visor)  # sale de la pantalla de inicio
        for dock in self._docks:
            dock.show()
        self.resizeDocks(self._docks, [300, 360], Qt.Horizontal)
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
            self.conexiones,
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
        usados = {a.id for a in self.apoyos} | {c.id for c in self.conexiones}
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
        creados, errores = [], []
        nodo_elegido = dialogo.nodo_elegido()
        if nodo_elegido is not None:  # sin selección: apoyo puntual en el nodo de la lista
            ux, uy, uz, rx, ry, rz = restricciones
            creados.append(Apoyo(id="", nodo_id=nodo_elegido, ux=ux, uy=uy, uz=uz, rx=rx, ry=ry, rz=rz,
                                 nombre=f"Apoyo en el nodo {nodo_elegido}"))
        desde, hasta = dialogo.tramo()
        for objetivo in self.visor.seleccion:
            try:
                creados.append(apoyo_para_objetivo(
                    self.proyecto, objetivo, "", restricciones, dialogo.tipo(), desde, hasta,
                ))
            except ErrorAsignacion as exc:
                errores.append(str(exc))
        for apoyo in creados:
            if isinstance(apoyo, Apoyo):  # un solo apoyo puntual por nodo
                self.apoyos = [a for a in self.apoyos if isinstance(a, ApoyoArea) or a.nodo_id != apoyo.nodo_id]
            apoyo.id = self._siguiente_id("AP")
            self.apoyos.append(apoyo)
        if errores:
            QMessageBox.warning(self, "Algunos apoyos no se asignaron", "\n\n".join(errores))
        if creados:
            self._marcar_resultados_desactualizados()
            self._refrescar_asignaciones()
            self.statusBar().showMessage(f"{len(creados)} apoyo(s) asignado(s).")

    def asignar_carga(self) -> None:
        dialogo = DialogoCargas(self.proyecto, self.visor.seleccion, self.casos, self.caso_activo(), self)
        if not dialogo.exec():
            return
        caso_id, parametros = dialogo.parametros()
        creadas, errores = 0, []
        for objetivo in self.visor.seleccion:
            try:
                cargas = cargas_desde_parametros(self.proyecto, objetivo, caso_id, "", parametros)
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

    def asignar_material(self) -> None:
        dialogo = DialogoAsignarMaterial(self.proyecto, self.visor.seleccion, self)
        if not dialogo.exec():
            return
        try:
            cambiados = asignar_material(self.proyecto, self.visor.seleccion, dialogo.material_elegido())
        except ErrorAsignacion as exc:
            QMessageBox.warning(self, "No se pudo asignar el material", str(exc))
            return
        self.panel_elementos.actualizar(self.proyecto)
        self._marcar_resultados_desactualizados()
        self.statusBar().showMessage(f"Material «{dialogo.material_elegido()}» asignado a {len(cambiados)} elemento(s).")

    def gestionar_materiales(self) -> None:
        dialogo = DialogoMateriales(self.proyecto, self)
        if dialogo.exec():
            self.proyecto.materiales = dialogo.resultado()
            self.panel_elementos.actualizar(self.proyecto)
            self._marcar_resultados_desactualizados()
            self.statusBar().showMessage(f"{len(self.proyecto.materiales)} material(es) definidos.")

    def asignar_conexion(self) -> None:
        barras = list(dict.fromkeys(o for o in self.visor.seleccion if o.tipo in ("VIGA", "COLUMNA")))
        distintas = {o.id for o in barras}
        if len(distintas) != 2:
            QMessageBox.information(
                self, "Conectar elementos",
                "Seleccione exactamente dos elementos (vigas o columnas; Ctrl + clic para sumar el segundo). "
                "Si no comparten un nudo, seleccione en modo «Cara» las caras extremas que quiere unir.",
            )
            return
        a = barras[0]
        b = next(o for o in barras if o.id != a.id)
        dialogo = DialogoConexion(self.proyecto, a, b, self)
        if not dialogo.exec():
            return
        try:
            conexion = dialogo.conexion(self._siguiente_id("CX"))
        except ErrorAsignacion as exc:
            QMessageBox.warning(self, "No se pudo conectar", str(exc))
            return
        self.conexiones = [
            c for c in self.conexiones
            if not (c.barra_id == conexion.barra_id and c.nodo_id == conexion.nodo_id and c.nodo_destino == conexion.nodo_destino)
        ]
        self.conexiones.append(conexion)
        self._marcar_resultados_desactualizados()
        self._refrescar_asignaciones()
        self.statusBar().showMessage(f"Conexión creada: {conexion.nombre}")

    # ------------------------------------------------------------------
    # Menú contextual (clic derecho sobre un elemento)
    # ------------------------------------------------------------------
    def construir_menu_contextual(self) -> QMenu:
        seleccion = self.visor.seleccion
        menu = QMenu(self)
        titulo = menu.addAction(
            descripcion(self.proyecto, seleccion[0]) if len(seleccion) == 1 else f"{len(seleccion)} elementos seleccionados"
        )
        titulo.setEnabled(False)
        menu.addSeparator()
        menu.addAction("Asignar carga…", self.asignar_carga)
        menu.addAction("Asignar apoyo…", self.asignar_apoyo)
        menu.addAction("Asignar material…", self.asignar_material)
        barras = {o.id for o in seleccion if o.tipo in ("VIGA", "COLUMNA")}
        accion = menu.addAction("Conectar elementos (constraint)…", self.asignar_conexion)
        accion.setEnabled(len(barras) == 2)
        if len(barras) != 2:
            accion.setToolTip("Seleccione dos vigas/columnas (Ctrl + clic)")
        menu.addSeparator()
        menu.addAction("Propiedades…", self.mostrar_propiedades)
        menu.addAction("Quitar la selección", lambda: self.visor.seleccionar([]))
        return menu

    def _mostrar_menu_contextual(self) -> None:
        if not self.visor.seleccion:
            return
        self.construir_menu_contextual().exec(QCursor.pos())

    def propiedades_texto(self, objetivo: Objetivo) -> str:
        p = self.proyecto
        lineas = [f"<b>{descripcion(p, Objetivo(objetivo.tipo, objetivo.id))}</b>"]
        if objetivo.tipo in ("VIGA", "COLUMNA"):
            barra = p.elemento_barra(objetivo.id)
            perfil = p.perfiles[barra.perfil]
            lineas += [
                f"Nodos: {barra.nodo_inicio} → {barra.nodo_fin}  (longitud {p.longitud_barra(barra.id):.3f} m)",
                f"Perfil: {perfil.nombre}  (área {perfil.area:.4g} m², {perfil.ancho or 0:.3g} × {perfil.altura or 0:.3g} m)",
            ]
            material = barra.material
        else:
            elemento = (p.losas if objetivo.tipo == "LOSA" else p.muros)[objetivo.id]
            lineas.append(f"Espesor: {elemento.espesor:.3f} m  —  {len(elemento.nodos_contorno)} nodos de contorno")
            material = elemento.material
        lineas.append("Material: " + (descripcion_material(p.materiales[material]) if material in p.materiales else material))
        return "<br>".join(lineas)

    def mostrar_propiedades(self) -> None:
        elementos = list(dict.fromkeys((o.tipo, o.id) for o in self.visor.seleccion))
        if elementos:
            QMessageBox.information(
                self, "Propiedades", "<br><br>".join(self.propiedades_texto(Objetivo(t, i)) for t, i in elementos[:6])
            )

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
        self.apoyos = [
            a for a in self.apoyos
            if (("Apoyo de área" if isinstance(a, ApoyoArea) else "Apoyo"), a.id) not in claves
        ]
        self.conexiones = [c for c in self.conexiones if ("Conexión", c.id) not in claves]
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
            self.conexiones,
        )
        nombres_gdl = {1: "Ux", 2: "Uy", 3: "Uz", 4: "Rx", 5: "Ry", 6: "Rz"}
        filas = [
            ("Apoyo de área" if isinstance(a, ApoyoArea) else "Apoyo", a.id,
             f"{a.nombre} [{', '.join(nombres_gdl[g] for g in a.grados_restringidos())}]", "todos")
            for a in self.apoyos
        ]
        filas += [
            ("Conexión", c.id, f"{c.nombre} [{c.tipo.lower()}: {', '.join(nombres_gdl[g] for g in c.grados_transmitidos())}]", "todos")
            for c in self.conexiones
        ]
        for tipo, por_caso in (
            ("Carga de área", self.cargas_area_por_caso),
            ("Carga lineal", self.cargas_lineales_por_caso),
            ("Carga puntual", self.cargas_puntuales_por_caso),
        ):
            filas += [(tipo, c.id, c.nombre, caso_id) for caso_id, lista in por_caso.items() for c in lista]
        self.panel_asignaciones.mostrar(filas)

    def _gestionar_casos(self) -> None:
        con_cargas = {
            caso_id
            for por_caso in (self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso)
            for caso_id, lista in por_caso.items() if lista
        }
        dialogo = DialogoCasos(self.casos, self.combinaciones, con_cargas, self)
        if not dialogo.exec():
            return
        self.casos, self.combinaciones = dialogo.resultado()
        for caso_id in dialogo.casos_eliminados():
            for por_caso in (self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso):
                por_caso.pop(caso_id, None)
        self._actualizar_combo_caso()
        self._refrescar_asignaciones()
        self._marcar_resultados_desactualizados()
        self.statusBar().showMessage(
            f"{len(self.casos)} caso(s) de carga y {len(self.combinaciones)} combinación(es) definidos."
        )

    # ------------------------------------------------------------------
    # Análisis
    # ------------------------------------------------------------------
    def analizar(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        errores, advertencias = validar_modelo(self.proyecto, self.casos, self.combinaciones, self.apoyos, self.conexiones)
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
        conexiones = list(self.conexiones)
        area = {k: list(v) for k, v in self.cargas_area_por_caso.items()}
        lineales = {k: list(v) for k, v in self.cargas_lineales_por_caso.items()}
        puntuales = {k: list(v) for k, v in self.cargas_puntuales_por_caso.items()}
        configuracion = self.configuracion

        def trabajo(avanzar):
            return ejecutar_analisis(
                proyecto, casos, combinaciones, apoyos, area, lineales, puntuales,
                configuracion.directorio_trabajo_absoluto, configuracion.ruta_ccx_efectiva, avanzar,
                conexiones=conexiones,
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
