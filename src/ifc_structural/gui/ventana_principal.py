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

import copy
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSizePolicy,
    QScrollArea,
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
from ..core.cargas import (
    Apoyo, ApoyoArea, ApoyoSuperficie, CargaArea, CargaLineal, CargaPuntual, CargaSuperficie, Conexion, UnionSuperficie,
)
from ..core.casos_carga import CasoCarga, Combinacion, casos_y_combinaciones_por_defecto
from ..core.edicion import eliminar_elementos
from ..core.materiales import descripcion_material
from ..core.modelo import Proyecto
from ..core.persistencia import cargar_cargas_superficie, cargar_proyecto, guardar_proyecto
from ..ifc.importador import importar_ifc
from ..solido.analisis_solido import (MallaPrevia, celdas_vtk_de_malla, ejecutar_analisis_solido, huella_geometria, mallar_modelo,
                                      tamano_sugerido)
from ..solver.analisis import ResultadoAnalisis, ejecutar_analisis, validar_modelo
from .barra_vista import BarraVista
from .dialogo_analisis_3d import DialogoAnalisis3D
from .dialogo_analizar import DialogoAnalizar
from .dialogo_acerca import DialogoAcerca
from .dialogo_apoyos import DialogoApoyo
from .dialogo_cargas import DialogoCargas
from .dialogo_superficie import DialogoApoyoSuperficie, DialogoCargaSuperficie, DialogoUnionSuperficie
from .dialogo_casos import DialogoCasos
from .dialogo_conexion import DialogoConexion
from .dialogo_conexion_automatica import DialogoConexionAutomatica
from .dialogo_escalas import DialogoEscalas
from .dialogo_materiales import DialogoAsignarMaterial, DialogoMateriales
from .estilo import etiqueta_logo
from .iconos import icono, icono_aplicacion, icono_blanco
from .pagina_bienvenida import PaginaBienvenida
from .panel_arbol import PanelArbol
from .panel_resultados import DESPLAZAMIENTO, ENVOLVENTE, PanelEsfuerzos
from .trabajador_analisis import TrabajadorAnalisis
from .visor_3d import ESTILO_ALAMBRE, ESTILO_SOLIDO, ESTILO_TRANSPARENTE, MODO_CARA, MODO_ELEMENTO, Visor3D

logger = logging.getLogger(__name__)

#: Fotogramas de un ciclo de la deformada animada (0 → 1 → 0).
PASOS_DE_ANIMACION = 48

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
    "<li>Si quiere revisar la malla antes de calcular, pulse <b>Mallar</b> (F7): se genera y se dibuja la malla de "
    "elementos finitos del análisis sólido, y ANALIZAR la vuelve a usar mientras no cambie el modelo.</li>"
    "<li>Pulse <b>▶ ANALIZAR</b> (F5) y revise los esfuerzos en el panel <b>Esfuerzos</b>. "
    "<b>ANALIZAR 3D</b> (F6) lo resuelve como modelo sólido, con la geometría real de cada pieza.</li></ol>"
    "Con el botón central del mouse se desplaza la vista; con la rueda se acerca o aleja; arrastrando con "
    "el botón izquierdo se gira."
)


def ruta_con_formato(ruta: str, filtro: str) -> str:
    """Ruta del reporte con la extensión del tipo elegido en el diálogo de guardar. El nombre propuesto termina en
    «.pdf» y al elegir «Word (*.docx)» el diálogo de Windows no lo cambia: manda el tipo elegido, no el nombre."""

    base, extension = ruta, ""
    if ruta.lower().endswith((".pdf", ".docx")):
        base, extension = ruta.rsplit(".", 1)
        extension = "." + extension.lower()
    if "docx" in (filtro or "").lower():
        return base + ".docx"
    if "pdf" in (filtro or "").lower():
        return base + ".pdf"
    return base + (extension or ".pdf")


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
        self.cargas_superficie_por_caso: Dict[str, List[CargaSuperficie]] = {}

        self.resultado: Optional[ResultadoAnalisis] = None
        self._resultado_desactualizado = False
        self._malla_previa: Optional[MallaPrevia] = None  # la del botón Mallar (ANALIZAR la reutiliza si sirve)
        self._casos_elegidos: Optional[set] = None  # casos marcados en el último ANALIZAR (los nuevos, marcados)
        self._casos_conocidos: set = set()
        self._barra_principal = None
        self.modo_analisis = "AUTO"  # AUTO, BARRAS o SOLIDO (Analizar → Modo de análisis)
        self._deshacer: list = []  # estados anteriores del modelo editable (el último es el más reciente)
        self._rehacer: list = []
        self._estado_actual = None
        self._restaurando = False
        self._desde_arbol = False  # la selección del visor la provocó el árbol (evita el bucle)
        self._worker: Optional[TrabajadorAnalisis] = None
        self._timer_animacion = QTimer(self)
        self._timer_animacion.setInterval(40)
        self._timer_animacion.timeout.connect(self._paso_de_animacion)
        self._paso_animacion = 0
        self._dialogo_progreso: Optional[QProgressDialog] = None

        self.setWindowIcon(icono_aplicacion())
        self.visor = Visor3D(self)
        self.barra_vista = BarraVista(self)
        self.pagina_bienvenida = PaginaBienvenida(self)
        self.contenedor_visor = QWidget(self)
        capa = QVBoxLayout(self.contenedor_visor)
        capa.setContentsMargins(0, 0, 0, 0)
        capa.setSpacing(0)
        # en una ventana estrecha la barra pierde los nombres de los botones y, si aun así no cabe, se desplaza: nunca
        # obliga a la ventana a ser más ancha que la pantalla
        desplazable = QScrollArea(self.contenedor_visor)
        desplazable.setWidget(self.barra_vista)
        desplazable.setWidgetResizable(True)
        desplazable.setFrameShape(QFrame.NoFrame)
        desplazable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        desplazable.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        desplazable.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        def ajustar_alto(_minimo=0, maximo=0):
            barra = desplazable.horizontalScrollBar()
            desplazable.setFixedHeight(self.barra_vista.sizeHint().height() + (barra.sizeHint().height() if maximo > 0 else 0))

        desplazable.horizontalScrollBar().rangeChanged.connect(ajustar_alto)
        ajustar_alto()
        self._barra_vista_desplazable = desplazable
        capa.addWidget(desplazable)
        capa.addWidget(self.visor, 1)
        self.paginas = QStackedWidget(self)
        self.paginas.addWidget(self.pagina_bienvenida)
        self.paginas.addWidget(self.contenedor_visor)
        self.setCentralWidget(self.paginas)

        self.panel_arbol = PanelArbol(self)
        dock_elementos = QDockWidget("Modelo", self)
        self._docks = []
        dock_elementos.setWidget(self.panel_arbol)
        dock_elementos.setMinimumWidth(300)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_elementos)
        self._docks.append(dock_elementos)

        self.panel_esfuerzos = PanelEsfuerzos(self)
        dock_derecho = QDockWidget("Resultados", self)
        dock_derecho.setMinimumWidth(320)
        dock_derecho.setWidget(self.panel_esfuerzos)
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
        self._cargar_escalas()

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
        self.accion_reporte = accion("Exportar reporte de análisis…", self.exportar_reporte, "esfuerzos", "Ctrl+R",
                                     "Reporte del último análisis (malla, Von Mises y deformada) en PDF o Word (Ctrl+R)")
        self.accion_deshacer = accion("Deshacer", self.deshacer, "deshacer", "Ctrl+Z", "Deshacer el último cambio (Ctrl+Z)")
        self.accion_rehacer = accion("Rehacer", self.rehacer, "rehacer", "Ctrl+Y", "Rehacer el cambio deshecho (Ctrl+Y o Ctrl+Mayús+Z)")
        self.accion_deshacer.setEnabled(False)
        self.accion_rehacer.setEnabled(False)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.rehacer)
        self.accion_eliminar = accion("Eliminar", self.eliminar_seleccion, "eliminar", None,
                                      "Eliminar del modelo los elementos seleccionados (Supr con el visor activo)")
        self.accion_eliminar.setShortcut(QKeySequence(Qt.Key_Delete))
        self.accion_eliminar.setShortcutContext(Qt.WidgetShortcut)  # solo con el foco en el visor (el árbol usa Supr para las asignaciones)
        self.visor.addAction(self.accion_eliminar)

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
        self.accion_conexion = accion("Conectar", self._mostrar_menu_conectar, "conectar",
                                      ayuda="Conectar elementos: manualmente (los que seleccione) o automáticamente (los que están unidos)")
        self.accion_materiales = accion("Materiales", self.gestionar_materiales, "materiales", ayuda="Crear y editar materiales")
        self.accion_casos = accion("Casos", self._gestionar_casos, "casos", ayuda="Casos de carga y combinaciones")
        self.accion_analizar = accion("ANALIZAR", self.analizar, None, "F5", "Analizar el modelo: el programa decide si lo resuelve como barras o como sólido 3D (F5)")
        self.accion_analizar.setIcon(icono_blanco("analizar"))
        self.accion_mallar = accion("Mallar", self.mallar, "malla", "F7",
                                    "Generar y ver la malla de elementos finitos antes de analizar; ANALIZAR la reutiliza (F7)")

        self.accion_ver_modelo = accion("Modelo", lambda: self.volver_al_modelo(), "modelo",
                                        ayuda="Volver a la vista del modelo", checkable=True, checked=True)
        self.accion_ver_esfuerzos = accion("Esfuerzos", lambda: self.mostrar_esfuerzos(), "esfuerzos",
                                           ayuda="Ver el mapa de esfuerzos del último análisis", checkable=True)
        self.accion_ver_malla = accion("Malla", lambda: self.mostrar_malla(), "malla",
                                       ayuda="Ver la malla de elementos finitos generada con Mallar", checkable=True)
        grupo_vista = QActionGroup(self)
        grupo_vista.setExclusive(True)
        grupo_vista.addAction(self.accion_ver_modelo)
        grupo_vista.addAction(self.accion_ver_malla)
        grupo_vista.addAction(self.accion_ver_esfuerzos)

        self.accion_ayuda = accion("Cómo se usa", lambda: QMessageBox.information(self, "Ayuda", AYUDA), "ayuda", "F1")
        self.accion_acerca = accion("Acerca de…", lambda: DialogoAcerca(self).exec(), "info")

    def _crear_menus(self) -> None:
        menu = self.menuBar().addMenu("&Archivo")
        menu.addActions([self.accion_importar, self.accion_abrir, self.accion_guardar])
        menu.addSeparator()
        menu.addAction(self.accion_reporte)
        menu.addSeparator()
        menu.addAction(self.accion_salir)

        menu = self.menuBar().addMenu("&Editar")
        menu.addActions([self.accion_deshacer, self.accion_rehacer])
        menu.addSeparator()
        menu.addAction(self.accion_eliminar)

        menu = self.menuBar().addMenu("A&signar")
        menu.addActions([self.accion_apoyo, self.accion_carga, self.accion_material, self.accion_conexion])
        submenu = menu.addMenu(icono("conectar"), "Conectar elementos")
        submenu.addAction("Conectar manualmente…", self.asignar_conexion)
        submenu.addAction("Conectar automáticamente…", self.conectar_automaticamente)
        menu.addSeparator()
        menu.addActions([self.accion_materiales, self.accion_casos])

        menu = self.menuBar().addMenu("A&nalizar")
        menu.addActions([self.accion_mallar, self.accion_analizar])
        menu.addSeparator()
        submenu = menu.addMenu("Modo de análisis")
        grupo_modo_analisis = QActionGroup(self)
        for texto, modo in (("Automático (según el modelo)", "AUTO"), ("Siempre como barras", "BARRAS"), ("Siempre como sólido 3D", "SOLIDO")):
            accion_modo = QAction(texto, self, checkable=True, checked=modo == "AUTO")
            accion_modo.triggered.connect(lambda _=False, m=modo: setattr(self, "modo_analisis", m))
            grupo_modo_analisis.addAction(accion_modo)
            submenu.addAction(accion_modo)

        self.menu_ver = QMenu("&Ver", self)
        self.menuBar().addMenu(self.menu_ver)
        self.menu_ver.addActions([self.accion_ver_modelo, self.accion_ver_malla, self.accion_ver_esfuerzos])
        self.menu_ver.addSeparator()
        # Estilo de vista: sólido, transparente o alambre (las mismas acciones salen al hacer clic derecho sobre el fondo)
        self._acciones_estilo = {}
        grupo_estilo = QActionGroup(self)
        for texto, estilo in (("Sólido", ESTILO_SOLIDO), ("Transparente", ESTILO_TRANSPARENTE), ("Alambre (solo aristas)", ESTILO_ALAMBRE)):
            accion_estilo = QAction(texto, self, checkable=True, checked=estilo == ESTILO_SOLIDO)
            accion_estilo.triggered.connect(lambda _=False, e=estilo: self._cambiar_estilo(e))
            grupo_estilo.addAction(accion_estilo)
            self._acciones_estilo[estilo] = accion_estilo
        submenu_estilo = self.menu_ver.addMenu("Estilo de vista")
        submenu_estilo.addActions(list(self._acciones_estilo.values()))
        self.menu_ver.addSeparator()
        for dock in self._docks:  # reabrir los paneles si se cerraron
            self.menu_ver.addAction(dock.toggleViewAction())
        self.menu_ver.addSeparator()
        self._acciones_mostrar = []
        for texto, boton in (("Mostrar apoyos", self.barra_vista.boton_apoyos), ("Mostrar cargas", self.barra_vista.boton_flechas),
                             ("Mostrar constraints", self.barra_vista.boton_uniones)):
            accion_vista = QAction(texto, self, checkable=True, checked=True)
            accion_vista.toggled.connect(boton.setChecked)
            boton.toggled.connect(accion_vista.setChecked)
            self.menu_ver.addAction(accion_vista)
            self._acciones_mostrar.append(accion_vista)
        self.menu_ver.addAction("Escala de símbolos…", self.cambiar_escalas)
        self.menu_ver.addSeparator()
        self._acciones_visibilidad = {}
        for tipo, texto, nombre_icono in (
            ("LOSA", "losas", "losa"), ("MURO", "muros", "muro"), ("VIGA", "vigas", "viga"), ("COLUMNA", "columnas", "columna"),
            ("SOLIDO", "sólidos", "solido"),
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
        self._barra_principal = barra
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
        barra.addActions([self.accion_deshacer, self.accion_rehacer, self.accion_eliminar])
        for accion in (self.accion_deshacer, self.accion_rehacer, self.accion_eliminar):
            barra.widgetForAction(accion).setToolButtonStyle(Qt.ToolButtonIconOnly)
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
        self.barra_vista.agregar_accion(self.accion_ver_malla)
        self.barra_vista.agregar_accion(self.accion_ver_esfuerzos)

        espacio = QWidget()
        espacio.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        barra.addWidget(espacio)

        self.boton_mallar = QToolButton()
        self.boton_mallar.setObjectName("botonMallar")
        self.boton_mallar.setDefaultAction(self.accion_mallar)
        self.boton_mallar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.boton_mallar.setIconSize(QSize(18, 18))
        self.boton_mallar.setCursor(Qt.PointingHandCursor)
        barra.addWidget(self.boton_mallar)

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
        self.barra_vista.navegacion_cambiada.connect(self.visor.set_navegacion)
        self.barra_vista.flechas_visibles.connect(self.visor.configurar_flechas)
        self.barra_vista.apoyos_visibles.connect(self.visor.configurar_apoyos)
        self.barra_vista.uniones_visibles.connect(self.visor.configurar_uniones)
        self.barra_vista.visibilidad_cambiada.connect(lambda t, v: self._cambiar_visibilidad(t, v, desde_menu=False))
        self.visor.seleccion_cambiada.connect(self._al_cambiar_seleccion)
        self.visor.menu_contextual.connect(self._mostrar_menu_contextual)
        self.visor.menu_fondo.connect(self._mostrar_menu_fondo)
        self.visor.punto_sondeado.connect(lambda texto: self.statusBar().showMessage(texto))
        self.panel_arbol.elementos_seleccionados.connect(self._seleccion_desde_arbol)
        self.panel_arbol.eliminar_solicitado.connect(self.eliminar_asignaciones)
        self.panel_arbol.menu_solicitado.connect(self._mostrar_menu_arbol)
        self.panel_arbol.resultado_solicitado.connect(self._mostrar_resultado_de_arbol)
        self.panel_arbol.editar_materiales.connect(self.gestionar_materiales)
        self.panel_arbol.editar_casos.connect(self._gestionar_casos)
        self.panel_arbol.propiedades_solicitadas.connect(self.mostrar_propiedades)
        self.panel_esfuerzos.cambio.connect(self.mostrar_esfuerzos)
        self.panel_esfuerzos.animacion_alternada.connect(self._alternar_animacion)
        self.panel_esfuerzos.video_solicitado.connect(self._guardar_video)
        self.panel_esfuerzos.reporte_solicitado.connect(self.exportar_reporte)
        self.panel_esfuerzos.escala_automatica_solicitada.connect(self._aplicar_escala_automatica)
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
        hay_solidos = bool(self.proyecto.solidos)
        ocupado = self._worker is not None and self._worker.isRunning()
        self.accion_analizar.setEnabled((hay_modelo or hay_solidos or bool(self.proyecto.losas or self.proyecto.muros)) and not ocupado)
        self.accion_mallar.setEnabled(self.accion_analizar.isEnabled())
        self.accion_ver_malla.setEnabled(self._malla_previa is not None)
        self.accion_apoyo.setEnabled(hay_modelo or hay_solidos)
        self.accion_material.setEnabled(hay_modelo or bool(self.proyecto.losas) or hay_solidos)
        self.accion_conexion.setEnabled(len(self.proyecto.todas_las_barras()) >= 2 or len(self.proyecto.solidos) + len(self.proyecto.losas) + len(self.proyecto.muros) >= 2)
        self.accion_materiales.setEnabled(True)
        self.accion_carga.setEnabled(hay_modelo or bool(self.proyecto.losas) or hay_solidos)
        self.accion_ver_esfuerzos.setEnabled(self.resultado is not None)
        self.accion_reporte.setEnabled(self.resultado is not None and not ocupado)

    def _al_cambiar_seleccion(self, objetivos: List[Objetivo]) -> None:
        if not self._desde_arbol:
            self.panel_arbol.resaltar(list(dict.fromkeys((o.tipo, o.id) for o in objetivos)))
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
        if proyecto.avisos:
            texto += "\n\nAvisos:\n- " + "\n- ".join(proyecto.avisos)
        QMessageBox.information(self, "Importación completa", texto)
        self.statusBar().showMessage(
            "Modelo importado. Ahora asigne los apoyos: modo Cara + Ver → Vista inferior, clic en la base de cada columna."
        )
        return True

    def _establecer_proyecto(
        self, proyecto, apoyos, casos, combinaciones, cargas_area, cargas_lineales, cargas_puntuales, conexiones=None,
        cargas_superficie=None,
    ) -> None:
        self.proyecto = proyecto
        self.apoyos = apoyos
        self.conexiones = list(conexiones or [])
        self.cargas_superficie_por_caso = dict(cargas_superficie or {})
        self.casos = casos or casos_y_combinaciones_por_defecto()[0]
        self.combinaciones = combinaciones
        self.cargas_area_por_caso = cargas_area
        self.cargas_lineales_por_caso = cargas_lineales
        self.cargas_puntuales_por_caso = cargas_puntuales
        self.resultado = None
        self._resultado_desactualizado = False
        self._malla_previa = None
        self.panel_esfuerzos.establecer_resultados([], [])
        self.panel_esfuerzos.mostrar_resumen("Ejecute el análisis para ver los esfuerzos.", {})

        self.paginas.setCurrentWidget(self.contenedor_visor)  # sale de la pantalla de inicio
        self._docks[0].show()  # el panel de resultados aparece al analizar
        self._docks[1].hide()
        self.resizeDocks(self._docks[:1], [380], Qt.Horizontal)
        self.visor.mostrar_modelo(proyecto)
        self._refrescar_arbol()
        self._actualizar_combo_caso()
        self.accion_ver_modelo.setChecked(True)
        self._refrescar_asignaciones()
        self._actualizar_estado_de_acciones()
        self._reiniciar_historial()

    def _guardar_proyecto(self) -> None:
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar proyecto", "", "Proyecto (*.json)")
        if not ruta:
            return
        guardar_proyecto(
            Path(ruta), self.proyecto, self.apoyos, self.casos, self.combinaciones,
            self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso,
            self.conexiones, self.cargas_superficie_por_caso,
        )
        self.statusBar().showMessage(f"Proyecto guardado en {ruta}")

    def _abrir_proyecto(self) -> None:
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir proyecto", "", "Proyecto (*.json)")
        if not ruta:
            return
        try:
            datos = cargar_proyecto(Path(ruta))
            cargas_superficie = cargar_cargas_superficie(Path(ruta))
        except Exception as exc:
            logger.exception("Error al abrir el proyecto")
            QMessageBox.critical(self, "Error al abrir", str(exc))
            return
        self._establecer_proyecto(*datos, cargas_superficie=cargas_superficie)
        self.statusBar().showMessage(f"Proyecto cargado desde {ruta}")

    # ------------------------------------------------------------------
    # Asignaciones
    # ------------------------------------------------------------------
    def _ids_en_uso(self) -> set:
        usados = {a.id for a in self.apoyos} | {c.id for c in self.conexiones} | {u.id for u in self.proyecto.uniones}
        for por_caso in (self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso,
                         self.cargas_superficie_por_caso):
            usados.update(c.id for lista in por_caso.values() for c in lista)
        return usados

    def _siguiente_id(self, prefijo: str) -> str:
        usados = self._ids_en_uso()
        n = 1
        while f"{prefijo}{n}" in usados:
            n += 1
        return f"{prefijo}{n}"

    def _marcar_resultados_desactualizados(self) -> None:
        self._registrar_cambio()
        if self.resultado is not None and not self._resultado_desactualizado:
            self._resultado_desactualizado = True
            self._refrescar_arbol()
            self.statusBar().showMessage(
                "Los esfuerzos mostrados son anteriores a los últimos cambios: vuelva a pulsar ANALIZAR."
            )

    # ------------------------------------------------------------------
    # Deshacer / rehacer y eliminar elementos
    # ------------------------------------------------------------------
    def _instantanea(self):
        """Copia del estado editable. Las mallas de los sólidos (grandes e inmutables) se comparten."""

        memo = {}
        for s in self.proyecto.solidos.values():
            for matriz in (s.vertices, s.triangulos, s.cara_de_triangulo):
                memo[id(matriz)] = matriz
            for cara in s.caras:
                for matriz in (cara.triangulos, cara.centro, cara.normal):
                    memo[id(matriz)] = matriz
        estado = (self.proyecto, self.apoyos, self.casos, self.combinaciones, self.cargas_area_por_caso,
                  self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso, self.conexiones,
                  self.cargas_superficie_por_caso)
        return copy.deepcopy(estado, memo)

    def _reiniciar_historial(self) -> None:
        self._deshacer.clear()
        self._rehacer.clear()
        self._estado_actual = self._instantanea()
        self._actualizar_acciones_historial()

    def _registrar_cambio(self) -> None:
        """Se llama justo después de cada cambio del modelo: el estado anterior pasa a la pila de deshacer."""

        if self._restaurando or self._estado_actual is None:
            return
        self._deshacer.append(self._estado_actual)
        del self._deshacer[:-60]  # los últimos 60 cambios
        self._rehacer.clear()
        self._estado_actual = self._instantanea()
        self._actualizar_acciones_historial()

    def _actualizar_acciones_historial(self) -> None:
        self.accion_deshacer.setEnabled(bool(self._deshacer))
        self.accion_rehacer.setEnabled(bool(self._rehacer))

    def _restaurar(self, estado) -> None:
        (self.proyecto, self.apoyos, self.casos, self.combinaciones, self.cargas_area_por_caso,
         self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso, self.conexiones,
         self.cargas_superficie_por_caso) = copy.deepcopy(estado, {id(m): m for m in self._matrices_de(estado[0])})
        self._restaurando = True
        try:
            self.visor.reemplazar_modelo(self.proyecto)
            self.accion_ver_modelo.setChecked(True)
            self._estado_actual = self._instantanea()
            self._marcar_resultados_desactualizados()
            self._actualizar_combo_caso()
            self._refrescar_asignaciones()
            self._actualizar_estado_de_acciones()
        finally:
            self._restaurando = False
        self._actualizar_acciones_historial()

    @staticmethod
    def _matrices_de(proyecto):
        for s in proyecto.solidos.values():
            yield from (s.vertices, s.triangulos, s.cara_de_triangulo)
            for cara in s.caras:
                yield from (cara.triangulos, cara.centro, cara.normal)

    def deshacer(self) -> None:
        if not self._deshacer or self._worker is not None and self._worker.isRunning():
            return
        self._rehacer.append(self._estado_actual)
        self._restaurar(self._deshacer.pop())
        self.statusBar().showMessage("Cambio deshecho.")

    def rehacer(self) -> None:
        if not self._rehacer or self._worker is not None and self._worker.isRunning():
            return
        self._deshacer.append(self._estado_actual)
        self._restaurar(self._rehacer.pop())
        self.statusBar().showMessage("Cambio rehecho.")

    def eliminar_seleccion(self) -> None:
        """Quita del modelo los elementos seleccionados (en modo Cara, el elemento al que pertenece cada cara)."""

        elementos = list(dict.fromkeys((o.tipo, o.id) for o in self.visor.seleccion))
        if not elementos:
            self.statusBar().showMessage("Seleccione los elementos que quiere eliminar.")
            return
        self.visor.seleccionar([])
        self.apoyos, self.conexiones, resultado = eliminar_elementos(
            self.proyecto, elementos, self.apoyos, self.conexiones, self.cargas_area_por_caso,
            self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso, self.cargas_superficie_por_caso,
        )
        self.visor.reemplazar_modelo(self.proyecto)
        self.accion_ver_modelo.setChecked(True)
        self._marcar_resultados_desactualizados()
        self._refrescar_asignaciones()
        self._actualizar_estado_de_acciones()
        self.statusBar().showMessage(f"Eliminado: {resultado.resumen()}. Ctrl+Z lo deshace.")

    def _hay_solidos_libres(self) -> bool:
        return any(o.tipo == "SOLIDO" for o in self.visor.seleccion)

    def asignar_apoyo_superficie(self) -> None:
        """Apoyo sobre las superficies seleccionadas (modo «Cara»): cualquier cara de cualquier elemento."""

        dialogo = DialogoApoyoSuperficie(self.proyecto, self.visor.seleccion, self)
        if not dialogo.exec():
            return
        creados, errores = dialogo.apoyos(lambda: self._siguiente_id("AP"))
        for apoyo in creados:
            apoyo.id = self._siguiente_id("AP")
            self.apoyos.append(apoyo)
        if errores:
            QMessageBox.warning(self, "Algunos apoyos no se asignaron", "\n\n".join(errores))
        if creados:
            self._marcar_resultados_desactualizados()
            self._refrescar_asignaciones()
            self.statusBar().showMessage(f"{len(creados)} apoyo(s) en superficie asignado(s). Se usan en ANALIZAR 3D.")

    def asignar_carga_superficie(self) -> None:
        """Carga (presión, fuerza o momento) sobre las superficies seleccionadas (modo «Cara»)."""

        dialogo = DialogoCargaSuperficie(self.proyecto, self.visor.seleccion, self.casos, self.caso_activo(), self)
        if not dialogo.exec():
            return
        creadas, errores = dialogo.cargas(lambda: "")
        for carga in creadas:
            carga.id = self._siguiente_id("CS")
            self.cargas_superficie_por_caso.setdefault(carga.caso_id, []).append(carga)
        if errores:
            QMessageBox.warning(self, "Algunas cargas no se asignaron", "\n\n".join(errores))
        if creadas:
            indice = self.combo_caso.findData(dialogo.caso_id())
            if indice >= 0:
                self.combo_caso.setCurrentIndex(indice)
            self._marcar_resultados_desactualizados()
            self._refrescar_asignaciones()
            self.statusBar().showMessage(f"{len(creadas)} carga(s) en superficie asignada(s). Se usan en ANALIZAR 3D.")

    def asignar_apoyo(self) -> None:
        if self._hay_solidos_libres():  # un sólido de forma libre solo tiene superficies
            self.asignar_apoyo_superficie()
            return
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
        if self._hay_solidos_libres():
            self.asignar_carga_superficie()
            return
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
        self._refrescar_arbol()
        self._marcar_resultados_desactualizados()
        self.statusBar().showMessage(f"Material «{dialogo.material_elegido()}» asignado a {len(cambiados)} elemento(s).")

    def gestionar_materiales(self) -> None:
        dialogo = DialogoMateriales(self.proyecto, self)
        if dialogo.exec():
            self.proyecto.materiales = dialogo.resultado()
            self._refrescar_arbol()
            self._marcar_resultados_desactualizados()
            self.statusBar().showMessage(f"{len(self.proyecto.materiales)} material(es) definidos.")

    def _cargar_escalas(self) -> None:
        """Recupera el tamaño de los símbolos de la sesión anterior."""

        try:
            ajustes = QSettings("AnalizadorEstructural", "Visor")
            escalas = {}
            for parametro in self.visor.escalas:
                modo = ajustes.value(f"escala/{parametro}/modo")
                valor = ajustes.value(f"escala/{parametro}/valor")
                if modo and valor is not None:
                    escalas[parametro] = (str(modo), float(valor))
            if escalas:
                self.visor.set_escalas(escalas)
            estilo = ajustes.value("vista/estilo")
            if estilo in self._acciones_estilo:
                self._cambiar_estilo(str(estilo))
        except Exception:  # preferencias ilegibles: se usa el tamaño automático
            logger.exception("No se pudieron leer las escalas guardadas")

    def cambiar_escalas(self) -> None:
        """Ver → Escala de símbolos: tamaño de apoyos, cargas, constraints y etiquetas (porcentaje o píxeles)."""

        dialogo = DialogoEscalas(self.visor, self)
        if not dialogo.exec():
            return
        try:
            ajustes = QSettings("AnalizadorEstructural", "Visor")
            for parametro, (modo, valor) in dialogo.escalas().items():
                ajustes.setValue(f"escala/{parametro}/modo", modo)
                ajustes.setValue(f"escala/{parametro}/valor", valor)
        except Exception:
            logger.exception("No se pudieron guardar las escalas")

    def _mostrar_menu_conectar(self) -> None:
        """El botón Conectar ofrece las dos formas: manual (lo seleccionado) y automática (lo que está unido)."""

        menu = QMenu(self)
        menu.addAction(icono("conectar"), "Conectar manualmente…", self.asignar_conexion)
        menu.addAction(icono("conectar"), "Conectar automáticamente…", self.conectar_automaticamente)
        posicion = QCursor.pos()
        if self._barra_principal is not None:
            widget = self._barra_principal.widgetForAction(self.accion_conexion)
            if widget is not None:
                posicion = widget.mapToGlobal(widget.rect().bottomLeft())
        menu.exec(posicion)

    def conectar_automaticamente(self) -> None:
        """Detecta los elementos que están unidos (se tocan o casi) y crea un constraint para cada pareja."""

        elementos = (len(self.proyecto.todas_las_barras()) + len(self.proyecto.losas) + len(self.proyecto.muros)
                     + len(self.proyecto.solidos))
        if elementos < 2:
            QMessageBox.information(self, "Conectar automáticamente", "El modelo necesita al menos dos elementos.")
            return
        dialogo = DialogoConexionAutomatica(self.proyecto, self)
        if not dialogo.exec():
            return
        usados = self._ids_en_uso()
        contador = [0]

        def siguiente() -> str:
            while True:
                contador[0] += 1
                if f"CT{contador[0]}" not in usados:
                    usados.add(f"CT{contador[0]}")
                    return f"CT{contador[0]}"

        nuevas = dialogo.uniones(siguiente)
        if not nuevas:
            self.statusBar().showMessage("No se creó ningún constraint.")
            return
        self.proyecto.uniones.extend(nuevas)
        self._marcar_resultados_desactualizados()
        self._refrescar_asignaciones()
        self.statusBar().showMessage(f"{len(nuevas)} constraint(s) creados automáticamente. Se aplican en el análisis 3D.")

    def asignar_union_superficie(self) -> None:
        """Constraint 3D entre dos superficies (o dos elementos: se toman las caras más cercanas)."""

        pares = list(dict.fromkeys((o.tipo, o.id, o.cara) for o in self.visor.seleccion))
        if len(pares) != 2:
            QMessageBox.information(
                self, "Constraint entre superficies",
                "Seleccione exactamente dos superficies (modo «Cara») o dos elementos (modo «Elemento»); Ctrl + clic suma el "
                "segundo. Con elementos enteros se unen sus caras más cercanas.",
            )
            return
        a, b = (Objetivo(*p) for p in pares)
        dialogo = DialogoUnionSuperficie(self.proyecto, a, b, self)
        if not dialogo.exec():
            return
        try:
            union = dialogo.union(self._siguiente_id("CT"))
        except ErrorAsignacion as exc:
            QMessageBox.warning(self, "No se pudo crear el constraint", str(exc))
            return
        self.proyecto.uniones.append(union)
        self._marcar_resultados_desactualizados()
        self._refrescar_asignaciones()
        self.statusBar().showMessage(f"Constraint creado: {union.nombre}. Se aplica en ANALIZAR 3D.")

    def asignar_conexion(self) -> None:
        """Conectar manualmente los dos elementos o superficies seleccionados."""

        seleccion = self.visor.seleccion
        solo_barras = bool(seleccion) and all(o.tipo in ("VIGA", "COLUMNA") and o.cara is None for o in seleccion)
        if solo_barras and len({o.id for o in seleccion}) == 2:
            # entre dos barras hay dos clases de conexión: la del modelo de barras y el constraint 3D
            caja = QMessageBox(self)
            caja.setWindowTitle("Conectar manualmente")
            caja.setText("¿Qué conexión quiere entre estas dos barras?")
            caja.setInformativeText("La conexión de barras (articulación o enlace rígido) se aplica en el análisis de barras; "
                                    "el constraint 3D, en el análisis como sólido (soldadura, cuerpo rígido, articulación).")
            barras = caja.addButton("Conexión de barras", QMessageBox.AcceptRole)
            caja.addButton("Constraint 3D", QMessageBox.AcceptRole).setDefault(False)
            caja.addButton(QMessageBox.Cancel)
            caja.exec()
            elegido = caja.clickedButton()
            if elegido is not None and elegido.text() == "Constraint 3D":
                self.asignar_union_superficie()
                return
            if elegido is not None and elegido is not barras:
                return
        elif any(o.tipo in ("SOLIDO", "LOSA", "MURO") or o.cara is not None for o in seleccion):
            self.asignar_union_superficie()  # entre sólidos libres, losas, muros o superficies solo hay constraints 3D
            return
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
        menu.addAction(icono("eliminar"), "Eliminar del modelo", self.eliminar_seleccion)
        hay_caras = all(o.cara is not None for o in seleccion)
        if not self._hay_solidos_libres():  # en un sólido libre las dos primeras ya son de superficie
            for texto, funcion in (("Asignar apoyo en la superficie… (análisis 3D)", self.asignar_apoyo_superficie),
                                   ("Asignar carga en la superficie… (análisis 3D)", self.asignar_carga_superficie)):
                accion_sup = menu.addAction(texto, funcion)
                accion_sup.setEnabled(hay_caras)
                if not hay_caras:
                    accion_sup.setToolTip("Seleccione una cara (modo «Cara»)")
        conectar = menu.addMenu(icono("conectar"), "Conectar")
        manual = conectar.addAction("Conectar manualmente…", self.asignar_conexion)
        manual.setEnabled(len(set((o.tipo, o.id, o.cara) for o in seleccion)) == 2)
        if not manual.isEnabled():
            manual.setToolTip("Seleccione dos superficies o dos elementos (Ctrl + clic)")
        conectar.addAction("Conectar automáticamente…", self.conectar_automaticamente)
        menu.addSeparator()
        menu.addAction("Propiedades…", self.mostrar_propiedades)
        menu.addAction("Quitar la selección", lambda: self.visor.seleccionar([]))
        return menu

    def _cambiar_estilo(self, estilo: str) -> None:
        self.visor.set_estilo(estilo)
        self._acciones_estilo[estilo].setChecked(True)
        try:
            QSettings("AnalizadorEstructural", "Visor").setValue("vista/estilo", estilo)
        except Exception:
            logger.exception("No se pudo guardar el estilo de vista")

    def construir_menu_fondo(self) -> QMenu:
        """Menú del clic derecho sobre el fondo: estilo de vista, vistas de cámara y qué se muestra."""

        menu = QMenu(self)
        estilo = menu.addMenu("Estilo de vista")
        estilo.addActions(list(self._acciones_estilo.values()))
        vistas = menu.addMenu("Vista")
        for clave, texto in (("vista_iso", "Isométrica"), ("vista_planta", "Planta"), ("vista_frontal", "Frontal"),
                             ("vista_lateral", "Lateral"), ("vista_inferior", "Inferior"), ("ajustar", "Encuadrar todo")):
            vistas.addAction(texto, lambda c=clave: self._cambiar_vista(c))
        menu.addSeparator()
        menu.addActions(self._acciones_mostrar)
        menu.addAction("Escala de símbolos…", self.cambiar_escalas)
        if self.visor.seleccion:
            menu.addSeparator()
            menu.addAction("Quitar la selección", lambda: self.visor.seleccionar([]))
        return menu

    def _mostrar_menu_fondo(self) -> None:
        self.construir_menu_fondo().exec(QCursor.pos())

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
        elif objetivo.tipo == "SOLIDO":
            solido = p.solidos[objetivo.id]
            e = solido.extension
            lineas += [
                f"Elemento IFC: {solido.tipo_ifc}  ({solido.nombre})",
                f"Dimensiones: {e[0]:.3f} × {e[1]:.3f} × {e[2]:.3f} m   ·   volumen {solido.volumen:.5g} m³",
                f"{len(solido.caras)} superficies, {len(solido.triangulos)} triángulos"
                + ("" if solido.cerrado else " — <b>malla abierta: no se puede mallar en 3D</b>"),
            ]
            material = solido.material
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
        elif isinstance(carga, CargaSuperficie):
            destino = self.cargas_superficie_por_caso
        elif isinstance(carga, CargaLineal):
            destino = self.cargas_lineales_por_caso
        else:
            destino = self.cargas_puntuales_por_caso
        destino.setdefault(carga.caso_id, []).append(carga)

    def eliminar_asignaciones(self, claves: Iterable[Tuple[str, str]]) -> None:
        claves = set(claves)
        self.apoyos = [
            a for a in self.apoyos
            if (("Apoyo de área" if isinstance(a, ApoyoArea)
                 else "Apoyo en superficie" if isinstance(a, ApoyoSuperficie) else "Apoyo"), a.id) not in claves
        ]
        self.conexiones = [c for c in self.conexiones if ("Conexión", c.id) not in claves]
        self.proyecto.uniones = [u for u in self.proyecto.uniones if ("Constraint", u.id) not in claves]
        for tipo, por_caso in (
            ("Carga en superficie", self.cargas_superficie_por_caso),
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
            self.cargas_superficie_por_caso.get(caso, []),
            self.proyecto.uniones,
        )
        self._refrescar_arbol()

    # ------------------------------------------------------------------
    # Árbol del modelo
    # ------------------------------------------------------------------
    def _refrescar_arbol(self) -> None:
        self.panel_arbol.actualizar(
            self.proyecto, self.apoyos, self.conexiones, self.casos, self.combinaciones,
            self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso,
            self.resultado, self._resultado_desactualizado, self.cargas_superficie_por_caso,
        )

    def _seleccion_desde_arbol(self, elementos) -> None:
        self._desde_arbol = True
        try:
            self.visor.seleccionar([Objetivo(tipo, id_) for tipo, id_ in elementos])
        finally:
            self._desde_arbol = False

    def _mostrar_resultado_de_arbol(self, id_: str) -> None:
        indice = self.panel_esfuerzos.combo_resultado.findData(id_)
        if indice < 0 or self.resultado is None:
            return
        self.panel_esfuerzos.combo_resultado.setCurrentIndex(indice)
        self._docks[1].show()
        self.mostrar_esfuerzos()

    def asignar_material_elegido(self, nombre: str) -> None:
        try:
            cambiados = asignar_material(self.proyecto, self.visor.seleccion, nombre)
        except ErrorAsignacion as exc:
            QMessageBox.warning(self, "No se pudo asignar el material", str(exc))
            return
        self._refrescar_arbol()
        self._marcar_resultados_desactualizados()
        self.statusBar().showMessage(f"Material «{nombre}» asignado a {len(cambiados)} elemento(s).")

    def construir_menu_arbol(self, claves) -> Optional[QMenu]:
        """Menú contextual del árbol según lo que se haya señalado."""

        if not claves:
            return None
        tipos = {c[0] for c in claves}
        grupos = {c[1] for c in claves if c[0] == "GRUPO"}
        if tipos & {"ELEMENTO", "ASIGNACION"} and self.visor.seleccion:
            menu = self.construir_menu_contextual()
            asignaciones = [(c[1], c[2]) for c in claves if c[0] == "ASIGNACION"]
            if asignaciones:
                menu.addSeparator()
                menu.addAction(icono("eliminar"), f"Eliminar {len(asignaciones)} asignación(es)",
                               lambda a=asignaciones: self.eliminar_asignaciones(a))
            return menu
        menu = QMenu(self)
        if "MATERIAL" in tipos:
            nombre = next(c[1] for c in claves if c[0] == "MATERIAL")
            accion = menu.addAction(icono("material"), f"Asignar «{nombre}» a la selección",
                                    lambda n=nombre: self.asignar_material_elegido(n))
            accion.setEnabled(bool(self.visor.seleccion))
            menu.addAction(icono("materiales"), "Crear / editar materiales…", self.gestionar_materiales)
        elif "RESULTADO" in tipos:
            clave = next(c[1] for c in claves if c[0] == "RESULTADO")
            menu.addAction(icono("esfuerzos"), "Mostrar en el visor", lambda c=clave: self._mostrar_resultado_de_arbol(c))
        elif tipos & {"CASO", "COMBINACION"} or grupos & {"Casos de carga", "Combinaciones", "Caso"}:
            menu.addAction(icono("casos"), "Casos de carga y combinaciones…", self._gestionar_casos)
        elif "Materiales" in grupos:
            menu.addAction(icono("materiales"), "Crear / editar materiales…", self.gestionar_materiales)
        elif "Apoyos" in grupos:
            menu.addAction(icono("apoyo"), "Asignar apoyo a la selección…", self.asignar_apoyo)
        elif "Cargas" in grupos:
            menu.addAction(icono("carga_dist"), "Asignar carga a la selección…", self.asignar_carga)
        elif "Conexiones" in grupos:
            menu.addAction(icono("conectar"), "Conectar elementos…", self.asignar_conexion)
        else:
            return None
        return menu

    def _mostrar_menu_arbol(self, claves) -> None:
        menu = self.construir_menu_arbol(claves)
        if menu is not None:
            menu.exec(QCursor.pos())

    def _gestionar_casos(self) -> None:
        con_cargas = {
            caso_id
            for por_caso in (self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso,
                             self.cargas_superficie_por_caso)
            for caso_id, lista in por_caso.items() if lista
        }
        dialogo = DialogoCasos(self.casos, self.combinaciones, con_cargas, self)
        if not dialogo.exec():
            return
        self.casos, self.combinaciones = dialogo.resultado()
        for caso_id in dialogo.casos_eliminados():
            for por_caso in (self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso,
                             self.cargas_superficie_por_caso):
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
    def _detectar_modo_de_analisis(self) -> Tuple[str, str]:
        """Qué análisis corresponde al modelo: ``("SOLIDO", motivo)`` si hay algo que solo se resuelve como sólido 3D
        (sólidos de forma libre, apoyos o cargas sobre superficies, constraints, o ninguna barra), si no ``("BARRAS", "")``."""

        p = self.proyecto
        if p.solidos:
            return "SOLIDO", f"el modelo tiene {len(p.solidos)} sólido(s) de forma libre"
        if any(isinstance(a, ApoyoSuperficie) for a in self.apoyos) or any(self.cargas_superficie_por_caso.values()):
            return "SOLIDO", "hay apoyos o cargas asignados a superficies"
        if p.uniones:
            return "SOLIDO", "hay constraints entre superficies o elementos"
        if not p.todas_las_barras():
            return "SOLIDO", "el modelo no tiene vigas ni columnas (solo losas o muros)"
        return "BARRAS", ""

    def analizar(self, modo: Optional[str] = None) -> None:
        """Botón ANALIZAR: elige solo el tipo de análisis (barras o sólido 3D) según lo que tenga el modelo y pregunta qué
        casos de carga se resuelven. ``modo`` (``"BARRAS"`` o ``"SOLIDO"``) lo fuerza; si no, se usa el modo elegido en
        *Analizar → Modo de análisis*. El tamaño de la malla del análisis sólido se elige con Mallar."""

        if self._worker is not None and self._worker.isRunning():
            return
        modo = modo if modo in ("BARRAS", "SOLIDO") else self.modo_analisis
        motivo = ""
        if modo == "AUTO":
            modo, motivo = self._detectar_modo_de_analisis()
        if modo == "SOLIDO":
            self._analizar_solido(motivo)
        else:
            self._analizar_barras()

    def _elegir_casos(self, motivo: str, texto_malla: str = "") -> Optional[Tuple[Dict[str, CasoCarga], Dict[str, Combinacion]]]:
        """Casos y combinaciones que se analizan (diálogo de ANALIZAR); ``None`` si se cancela."""

        if not self.casos:
            QMessageBox.warning(self, "Todavía no se puede analizar", "• No hay casos de carga definidos (botón Casos).")
            return None
        previos = None if self._casos_elegidos is None else [i for i in self.casos if i in self._casos_elegidos
                                                              or i not in self._casos_conocidos]
        dialogo = DialogoAnalizar(self.casos, self.combinaciones, self, motivo, texto_malla, previos)
        if not dialogo.exec():
            return None
        elegidos = dialogo.casos_elegidos()
        self._casos_elegidos, self._casos_conocidos = set(elegidos), set(self.casos)
        return {i: self.casos[i] for i in elegidos}, dialogo.combinaciones_elegidas()

    def _analizar_barras(self) -> None:
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
        elegidos = self._elegir_casos("Se analiza como <b>modelo de barras</b>.")
        if elegidos is not None:
            self._iniciar_hilo(casos=elegidos[0], combinaciones=elegidos[1])

    def _analizar_solido(self, motivo: str = "") -> None:
        if not self.proyecto.todas_las_barras() and not self.proyecto.losas and not self.proyecto.muros and not self.proyecto.solidos:
            QMessageBox.information(self, "Analizar", "El modelo no tiene elementos que analizar.")
            return
        if not self.apoyos:
            QMessageBox.warning(self, "Todavía no se puede analizar",
                                "• No hay apoyos asignados: seleccione una superficie (modo «Cara») o la cara extrema de una columna o viga y asigne un apoyo.")
            return
        # la malla: la del botón Mallar si la hay; si no, con el tamaño sugerido
        previa = self._malla_previa
        if previa is not None:
            tamano, barridos = previa.tamano, previa.barridos
            if previa.huella == huella_geometria(self.proyecto):
                texto_malla = f"<b>Malla:</b> la generada con Mallar ({previa.resumen()})."
            else:
                texto_malla = (f"<b>Malla:</b> el modelo cambió después de mallar; se volverá a mallar con elementos de "
                               f"{tamano * 1000:.0f} mm.")
        else:
            tamano, barridos = tamano_sugerido(self.proyecto), True
            texto_malla = (f"<b>Malla:</b> se mallará con elementos de {tamano * 1000:.0f} mm (tamaño sugerido). Para "
                           "revisarla o cambiar el tamaño de elemento, use <b>Mallar</b> antes de analizar.")
        texto_motivo = "Se analiza como <b>sólido 3D</b>" + (f" porque {motivo}." if motivo else ".")
        elegidos = self._elegir_casos(texto_motivo, texto_malla)
        if elegidos is not None:
            self._iniciar_hilo(solido=True, tamano=tamano, barridos=barridos, casos=elegidos[0], combinaciones=elegidos[1])

    def mallar(self) -> None:
        """Botón Mallar: genera la malla del análisis sólido sin resolverlo, para revisarla antes de ANALIZAR."""

        if self._worker is not None and self._worker.isRunning():
            return
        if not self.proyecto.todas_las_barras() and not self.proyecto.losas and not self.proyecto.muros and not self.proyecto.solidos:
            QMessageBox.information(self, "Mallar", "El modelo no tiene elementos que mallar.")
            return
        previa = self._malla_previa
        dialogo = DialogoAnalisis3D(self.proyecto, self, para_mallar=True, tamano_inicial=previa.tamano if previa else None,
                                    barridos=previa.barridos if previa else True)
        if not dialogo.exec():
            return
        proyecto, apoyos, tamano, elemento = self.proyecto, list(self.apoyos), dialogo.tamano(), dialogo.elemento()
        barridos = dialogo.barridos()
        puntuales = {k: list(v) for k, v in self.cargas_puntuales_por_caso.items()}
        lineales = {k: list(v) for k, v in self.cargas_lineales_por_caso.items()}

        def trabajo(avanzar):
            return mallar_modelo(proyecto, tamano, apoyos, puntuales, lineales, elemento, avanzar, barridos)

        self._lanzar_trabajo(trabajo, "Mallando", "Preparando la malla...", self._al_terminar_mallado,
                             "No se pudo generar la malla")

    def _al_terminar_mallado(self, previa: MallaPrevia) -> None:
        self._cerrar_dialogo_progreso()
        self._malla_previa = previa
        self._actualizar_estado_de_acciones()
        self._detener_animacion()
        self.mostrar_malla()
        mensaje = f"Malla: {previa.resumen()}. ANALIZAR la usará mientras no cambie el modelo."
        if self.modo_analisis == "BARRAS" or self.modo_analisis == "AUTO" and self._detectar_modo_de_analisis()[0] == "BARRAS":
            mensaje += " (El modelo se analiza como barras; elija Analizar → Modo de análisis → sólido 3D para usarla.)"
        self.statusBar().showMessage(mensaje)
        avisos = list(previa.advertencias) + list(previa.malla.avisos)
        if avisos:
            QMessageBox.warning(self, "Malla generada, con advertencias", "\n\n".join(f"• {a}" for a in avisos))

    def mostrar_malla(self) -> None:
        previa = self._malla_previa
        if previa is None:
            return
        celdas, tipos, es_lamina = celdas_vtk_de_malla(previa.malla)
        texto = f"Malla: {previa.resumen()}"
        if previa.huella != huella_geometria(self.proyecto):
            texto += "\nEl modelo cambió después de mallar: vuelva a pulsar Mallar"
        self.visor.mostrar_malla(celdas, tipos, previa.malla.coordenadas, es_lamina, texto)
        self.accion_ver_malla.setChecked(True)

    def _iniciar_hilo(self, solido: bool = False, tamano: float = 0.05, elemento_solido: Optional[str] = None,
                      barridos: bool = True, casos: Optional[Dict[str, CasoCarga]] = None,
                      combinaciones: Optional[Dict[str, Combinacion]] = None) -> None:
        casos = dict(self.casos if casos is None else casos)
        combinaciones = dict(self.combinaciones if combinaciones is None else combinaciones)
        proyecto, apoyos = self.proyecto, list(self.apoyos)
        conexiones = list(self.conexiones)
        area = {k: list(v) for k, v in self.cargas_area_por_caso.items()}
        lineales = {k: list(v) for k, v in self.cargas_lineales_por_caso.items()}
        puntuales = {k: list(v) for k, v in self.cargas_puntuales_por_caso.items()}
        superficiales = {k: list(v) for k, v in self.cargas_superficie_por_caso.items()}
        uniones = list(self.proyecto.uniones)
        configuracion = self.configuracion
        malla_previa = self._malla_previa

        def trabajo(avanzar):
            if solido:
                return ejecutar_analisis_solido(
                    proyecto, casos, combinaciones, apoyos, area, lineales, puntuales,
                    configuracion.directorio_trabajo_absoluto, configuracion.ruta_ccx_efectiva, avanzar,
                    tamano=tamano, conexiones=conexiones, cargas_superficie_por_caso=superficiales,
                    uniones_manuales=uniones, elemento_solido=elemento_solido, malla_previa=malla_previa,
                    barridos=barridos,
                )
            return ejecutar_analisis(
                proyecto, casos, combinaciones, apoyos, area, lineales, puntuales,
                configuracion.directorio_trabajo_absoluto, configuracion.ruta_ccx_efectiva, avanzar,
                conexiones=conexiones,
            )

        self._lanzar_trabajo(trabajo, "Analizando", "Preparando el análisis 3D..." if solido else "Preparando el análisis...",
                             self._al_terminar_analisis, "El análisis no pudo completarse")

    def _lanzar_trabajo(self, trabajo, titulo: str, texto: str, al_terminar, titulo_error: str) -> None:
        """Ejecuta ``trabajo`` en un hilo con un diálogo de progreso; ``al_terminar`` recibe lo que devuelva."""

        self._worker = TrabajadorAnalisis(trabajo, self)
        self._worker.terminado.connect(al_terminar)
        self._worker.fallido.connect(lambda mensaje: self._al_fallar_analisis(mensaje, titulo_error))
        self._worker.finished.connect(self._al_finalizar_hilo)

        self._dialogo_progreso = QProgressDialog(texto, "", 0, 0, self)
        self._dialogo_progreso.setWindowTitle(titulo)
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

    def _al_fallar_analisis(self, mensaje: str, titulo: str = "El análisis no pudo completarse") -> None:
        self._cerrar_dialogo_progreso()
        QMessageBox.critical(self, titulo, mensaje)
        self.statusBar().showMessage(f"{titulo}.")

    def _al_terminar_analisis(self, resultado: ResultadoAnalisis) -> None:
        self._cerrar_dialogo_progreso()
        self.resultado = resultado
        self._resultado_desactualizado = False
        self.panel_esfuerzos.establecer_resultados(
            [(c.id, c.nombre) for c in self.casos.values() if c.id in resultado.tensiones_por_caso],
            [(c.id, c.nombre) for c in self.combinaciones.values() if c.id in resultado.tensiones_por_combinacion],
        )
        self._actualizar_estado_de_acciones()
        self._detener_animacion()
        self.panel_esfuerzos.fijar_escala_deformada(self._escala_deformada_automatica(resultado))
        self.accion_ver_esfuerzos.setChecked(True)
        self._docks[1].show()
        self._refrescar_arbol()
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

        por_desplazamiento = parametros.magnitud == DESPLAZAMIENTO
        unidad = "mm" if por_desplazamiento else "MPa"
        if por_desplazamiento:
            nombre_magnitud, buscar_minimo = "Desplazamiento total", False
            valores, combinacion_por_nodo, origen = self._desplazamiento_total(resultado, parametros.resultado)
            if valores is None:
                self.panel_esfuerzos.mostrar_resumen("Este resultado no tiene desplazamientos.", {})
                return
            origen = origen or self.panel_esfuerzos.combo_resultado.currentText()
        else:
            magnitud = esfuerzos.MAGNITUDES[parametros.magnitud]
            nombre_magnitud, buscar_minimo = magnitud.nombre, magnitud.envolvente == "min"
            if parametros.resultado == ENVOLVENTE:
                envolvente = calcular_envolvente(resultado.tensiones_por_combinacion, parametros.magnitud)
                valores_kpa, origen = envolvente.valores, "Envolvente de combinaciones"
                combinacion_por_nodo = envolvente.combinacion_por_nodo
            else:
                valores_kpa = esfuerzos.calcular(parametros.magnitud, resultado.resultados_disponibles()[parametros.resultado])
                combinacion_por_nodo, origen = None, self.panel_esfuerzos.combo_resultado.currentText()
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

        i_critico = int(np.argmin(valores) if buscar_minimo else np.argmax(valores))
        valor_critico = float(valores[i_critico])
        celda_critica = next(
            (c for c, conectividad in enumerate(resultado.malla_tensiones.conectividad) if i_critico in conectividad), 0
        )
        barra_critica = resultado.barra_de_cada_celda()[celda_critica]
        resumen = (
            f"{nombre_magnitud} — {origen}\n"
            f"Valor más desfavorable: {valor_critico:.3f} {unidad} en la barra {barra_critica}"
            + (f" (combinación {combinacion_por_nodo[i_critico]})" if combinacion_por_nodo is not None else "")
            + ".\nHaga clic en el visor para leer el valor en un punto."
        )
        if self._resultado_desactualizado:
            resumen = "⚠ Resultados anteriores a los últimos cambios.\n" + resumen

        desplazamientos, origen_u = (None, "")
        if parametros.deformada:
            desplazamientos, origen_u = self._desplazamientos_de(resultado, parametros.resultado)
            if desplazamientos is not None:
                u_max = float(np.linalg.norm(desplazamientos, axis=1).max())
                resumen += (f"\nDeformada de {origen_u}: desplazamiento máximo {u_max * 1000:.3f} mm, "
                            f"dibujada ×{parametros.escala_deformada:g}.")
        self.visor.mostrar_esfuerzos(
            resultado.malla_tensiones, valores, resultado.barra_de_cada_celda(),
            titulo=f"{nombre_magnitud} [{unidad}]", rango=(minimo, maximo), cmap=parametros.colormap, unidad=unidad,
            texto=f"{origen}\n{nombre_magnitud}" + (" (modelo sólido 3D)" if resultado.es_solido else "")
            + (f"\nDeformada ×{parametros.escala_deformada:g}" if desplazamientos is not None else ""),
            ocultar_planos=resultado.es_solido,
            desplazamientos=desplazamientos, escala_deformada=parametros.escala_deformada,
            factor_deformada=self._factor_animacion(), mostrar_original=parametros.mostrar_original,
        )
        self.panel_esfuerzos.mostrar_resumen(resumen, resultado.resumen_por_barra(valores))

    # ------------------------------------------------------------------
    # Deformada, animación y video
    # ------------------------------------------------------------------
    @staticmethod
    def _desplazamiento_total(resultado: ResultadoAnalisis, id_resultado: str):
        """(|u| en mm por nodo, combinación de cada nodo o None, texto) del caso o combinación elegidos; con la
        envolvente, el mayor |u| de cada nodo entre todas las combinaciones."""

        if id_resultado == ENVOLVENTE:
            ids = list(resultado.desplazamientos_por_combinacion)
            if not ids:
                return None, None, ""
            normas = np.vstack([np.linalg.norm(resultado.desplazamientos_por_combinacion[i], axis=1) for i in ids])
            indice = np.argmax(normas, axis=0)
            return (normas[indice, np.arange(normas.shape[1])] * 1000.0, np.array(ids, dtype=object)[indice],
                    "Envolvente de combinaciones")
        u = resultado.desplazamientos_por_combinacion.get(id_resultado, resultado.desplazamientos_por_caso.get(id_resultado))
        if u is None:
            return None, None, ""
        return np.linalg.norm(np.asarray(u, dtype=float), axis=1) * 1000.0, None, ""

    @staticmethod
    def _desplazamientos_de(resultado: ResultadoAnalisis, id_resultado: str):
        """(desplazamientos (N, 3), texto) del caso o combinación elegidos; con la envolvente,
        los de la combinación que más desplaza la estructura."""

        if id_resultado == ENVOLVENTE:
            if not resultado.desplazamientos_por_combinacion:
                return None, ""
            mayor = max(resultado.desplazamientos_por_combinacion,
                        key=lambda c: float(np.linalg.norm(resultado.desplazamientos_por_combinacion[c], axis=1).max()))
            return resultado.desplazamientos_por_combinacion[mayor], f"la combinación {mayor} (la mayor)"
        if id_resultado in resultado.desplazamientos_por_combinacion:
            return resultado.desplazamientos_por_combinacion[id_resultado], f"la combinación {id_resultado}"
        if id_resultado in resultado.desplazamientos_por_caso:
            return resultado.desplazamientos_por_caso[id_resultado], f"el caso {id_resultado}"
        return None, ""

    @staticmethod
    def _escala_deformada_automatica(resultado: ResultadoAnalisis) -> float:
        """Ampliación con la que el mayor desplazamiento mide ~8 % del tamaño del modelo."""

        todos = list(resultado.desplazamientos_por_caso.values()) + list(resultado.desplazamientos_por_combinacion.values())
        if not todos:
            return 1.0
        u_max = max(float(np.linalg.norm(u, axis=1).max()) for u in todos)
        coordenadas = resultado.malla_tensiones.coordenadas
        tamano = float(np.linalg.norm(coordenadas.max(axis=0) - coordenadas.min(axis=0)))
        if u_max < 1e-12 or tamano < 1e-9:
            return 1.0
        bruta = 0.08 * tamano / u_max
        cifras = 10 ** (math.floor(math.log10(bruta)) - 1)  # dos cifras significativas
        return float(min(1e7, max(1.0, round(bruta / cifras) * cifras)))

    def _aplicar_escala_automatica(self) -> None:
        if self.resultado is not None:
            self.panel_esfuerzos.fijar_escala_deformada(self._escala_deformada_automatica(self.resultado))
            self.mostrar_esfuerzos()

    def _factor_animacion(self) -> float:
        if not self._timer_animacion.isActive():
            return 1.0
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * self._paso_animacion / PASOS_DE_ANIMACION))

    def _alternar_animacion(self, activa: bool) -> None:
        if not activa:
            self._detener_animacion()
            return
        self.mostrar_esfuerzos()  # asegura que la deformada está dibujada
        if not self.visor.hay_deformada:
            self.panel_esfuerzos.boton_animar.setChecked(False)
            QMessageBox.information(self, "Animar", "Este resultado no tiene desplazamientos que animar.")
            return
        self._paso_animacion = 0
        self._timer_animacion.start()

    def _detener_animacion(self) -> None:
        self._timer_animacion.stop()
        if self.panel_esfuerzos.boton_animar.isChecked():
            self.panel_esfuerzos.boton_animar.blockSignals(True)
            self.panel_esfuerzos.boton_animar.setChecked(False)
            self.panel_esfuerzos.boton_animar.setText("▶ Animar")
            self.panel_esfuerzos.boton_animar.blockSignals(False)
        self.visor.actualizar_deformada(1.0)

    def _paso_de_animacion(self) -> None:
        self._paso_animacion = (self._paso_animacion + 1) % PASOS_DE_ANIMACION
        self.visor.actualizar_deformada(self._factor_animacion())

    def _guardar_video(self) -> None:
        if self.resultado is None:
            return
        self._timer_animacion.stop()
        self.panel_esfuerzos.check_deformada.setChecked(True)
        self.mostrar_esfuerzos()
        if not self.visor.hay_deformada:
            QMessageBox.information(self, "Guardar video", "Este resultado no tiene desplazamientos que grabar.")
            return
        ruta, filtro = QFileDialog.getSaveFileName(
            self, "Guardar video de la deformada", "deformada.mp4", "Video MP4 (*.mp4);;GIF animado (*.gif)")
        if not ruta:
            return
        if not ruta.lower().endswith((".mp4", ".gif")):
            ruta += ".gif" if "gif" in filtro.lower() else ".mp4"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.visor.guardar_video(
                ruta, fotogramas=PASOS_DE_ANIMACION, fps=24,
                al_avanzar=lambda k, n: (self.statusBar().showMessage(f"Grabando el video: {k} de {n} fotogramas..."),
                                         QApplication.processEvents()),
            )
        except Exception as exc:  # falta el codificador, ruta sin permisos, etc.
            logger.exception("No se pudo guardar el video")
            QMessageBox.critical(self, "No se pudo guardar el video", f"{exc}\n\nPruebe a guardarlo como GIF.")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Video guardado en {ruta}")

    # ------------------------------------------------------------------
    # Reporte
    # ------------------------------------------------------------------
    def exportar_reporte(self, ruta: Optional[str] = None) -> Optional[Path]:
        """Reporte del último análisis en PDF o Word (según la extensión elegida)."""

        from ..reporte import DatosReporte, exportar_reporte

        if self.resultado is None:
            QMessageBox.information(self, "Exportar reporte", "Primero ejecute el análisis (ANALIZAR).")
            return None
        if not ruta:
            nombre = Path(self.proyecto.ruta_ifc).stem if getattr(self.proyecto, "ruta_ifc", None) else "reporte"
            ruta, filtro = QFileDialog.getSaveFileName(
                self, "Exportar reporte de análisis", f"{nombre}_reporte.pdf", "PDF (*.pdf);;Word (*.docx)")
            if not ruta:
                return None
            ruta = ruta_con_formato(ruta, filtro)
        parametros = self.panel_esfuerzos.parametros()
        cargas_por_caso: Dict[str, int] = {}
        for por_caso in (self.cargas_area_por_caso, self.cargas_lineales_por_caso, self.cargas_puntuales_por_caso,
                         self.cargas_superficie_por_caso):
            for caso, lista in por_caso.items():
                cargas_por_caso[caso] = cargas_por_caso.get(caso, 0) + len(lista)
        datos = DatosReporte(
            proyecto=self.proyecto, resultado=self.resultado,
            # solo los casos y combinaciones que se analizaron (ANALIZAR deja elegir los casos)
            casos={i: c for i, c in self.casos.items() if i in self.resultado.tensiones_por_caso},
            combinaciones={i: c for i, c in self.combinaciones.items() if i in self.resultado.tensiones_por_combinacion},
            apoyos=list(self.apoyos), cargas_por_caso=cargas_por_caso,
            escala_deformada=parametros.escala_deformada if parametros is not None else None,
            colormap=parametros.colormap if parametros is not None else "turbo")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            salida = exportar_reporte(ruta, datos, al_avanzar=lambda texto: (self.statusBar().showMessage(texto),
                                                                              QApplication.processEvents()))
        except Exception as exc:  # archivo abierto en Word, carpeta sin permisos...
            logger.exception("No se pudo exportar el reporte")
            QMessageBox.critical(self, "No se pudo exportar el reporte",
                                 f"{exc}\n\nSi el archivo está abierto en otro programa, ciérrelo y vuelva a intentarlo.")
            return None
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Reporte guardado en {salida}")
        return salida

    # ------------------------------------------------------------------
    def closeEvent(self, evento) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        self.visor.close()
        super().closeEvent(evento)
