"""Barra sobre el visor 3D: vistas de cámara, visibilidad por tipo de
elemento y leyenda de colores."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QToolButton, QWidget

from .iconos import icono
from .visor_3d import COLORES_BASE

VISTAS = (
    ("vista_iso", "Iso", "Vista isométrica"),
    ("vista_planta", "Planta", "Vista en planta (desde arriba)"),
    ("vista_frontal", "Frontal", "Vista frontal"),
    ("vista_lateral", "Lateral", "Vista lateral"),
    ("vista_inferior", "Inferior", "Vista inferior: permite hacer clic en la base de las columnas"),
    ("ajustar", "Ajustar", "Encuadrar todo el modelo"),
)
TIPOS = (("VIGA", "Vigas", "viga"), ("COLUMNA", "Columnas", "columna"), ("LOSA", "Losas", "losa"), ("MURO", "Muros", "muro"),
         ("SOLIDO", "Sólidos", "solido"))


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


class BarraVista(QWidget):
    vista_solicitada = Signal(str)  # "vista_iso", "vista_planta", ..., "ajustar"
    visibilidad_cambiada = Signal(str, bool)  # tipo, visible
    flechas_visibles = Signal(bool)  # mostrar u ocultar las flechas de las cargas
    apoyos_visibles = Signal(bool)  # mostrar u ocultar los apoyos
    uniones_visibles = Signal(bool)  # mostrar u ocultar los constraints
    navegacion_cambiada = Signal(str)  # qué hace arrastrar con el botón izquierdo: SELECCION, MOVER o ROTAR

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("barraVista")
        fila = QHBoxLayout(self)
        fila.setContentsMargins(10, 4, 10, 4)
        fila.setSpacing(2)

        self.botones_vista = {}
        for clave, texto, ayuda in VISTAS:
            boton = QToolButton()
            boton.setText(texto)
            boton.setIcon(icono(clave))
            boton.setToolTip(ayuda)
            boton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            boton.clicked.connect(lambda _=False, c=clave: self.vista_solicitada.emit(c))
            fila.addWidget(boton)
            self.botones_vista[clave] = boton

        separador = QFrame()
        separador.setFrameShape(QFrame.VLine)
        separador.setStyleSheet("color: #d3dae3;")
        fila.addSpacing(6)
        fila.addWidget(separador)
        fila.addSpacing(6)

        etiqueta = QLabel("Mostrar:")
        etiqueta.setObjectName("chipLeyenda")
        fila.addWidget(etiqueta)
        self.botones_tipo = {}
        for tipo, texto, nombre_icono in TIPOS:
            boton = QToolButton()
            boton.setCheckable(True)
            boton.setChecked(True)
            color = _hex(COLORES_BASE[tipo])  # el icono lleva el color del elemento: hace de leyenda
            boton.setIcon(icono(nombre_icono, color, color))
            boton.setToolTip(f"Mostrar u ocultar {texto.lower()}")
            boton.setToolButtonStyle(Qt.ToolButtonIconOnly)
            boton.toggled.connect(lambda visible, t=tipo: self.visibilidad_cambiada.emit(t, visible))
            fila.addWidget(boton)
            self.botones_tipo[tipo] = boton

        separador = QFrame()
        separador.setFrameShape(QFrame.VLine)
        separador.setStyleSheet("color: #d3dae3;")
        fila.addSpacing(6)
        fila.addWidget(separador)
        fila.addSpacing(6)
        etiqueta_nav = QLabel("Ratón:")
        etiqueta_nav.setObjectName("chipLeyenda")
        fila.addWidget(etiqueta_nav)
        self.grupo_navegacion = QButtonGroup(self)
        self.grupo_navegacion.setExclusive(True)
        self.botones_navegacion = {}
        for modo, texto, nombre_icono, ayuda in (
            ("SELECCION", "Seleccionar", "zona",
             "Arrastrar con el botón izquierdo selecciona: de izquierda a derecha lo que queda dentro del rectángulo, "
             "de derecha a izquierda todo lo que toca (Ctrl o Mayús suma a la selección)."),
            ("MOVER", "Mover", "mover", "Arrastrar con el botón izquierdo desplaza la vista (también: botón central)."),
            ("ROTAR", "Rotar", "rotar", "Arrastrar con el botón izquierdo gira la vista (también: Ctrl o Mayús + botón central)."),
        ):
            boton = QToolButton()
            boton.setIcon(icono(nombre_icono))
            boton.setCheckable(True)
            boton.setChecked(modo == "SELECCION")
            boton.setToolButtonStyle(Qt.ToolButtonIconOnly)
            boton.setToolTip(f"{texto}: {ayuda}\nRueda del ratón: zoom.")
            boton.toggled.connect(lambda activo, m=modo: activo and self.navegacion_cambiada.emit(m))
            self.grupo_navegacion.addButton(boton)
            fila.addWidget(boton)
            self.botones_navegacion[modo] = boton
        separador = QFrame()
        separador.setFrameShape(QFrame.VLine)
        separador.setStyleSheet("color: #d3dae3;")
        fila.addSpacing(6)
        fila.addWidget(separador)
        fila.addSpacing(6)
        # Interruptores solo con el símbolo (el nombre va en la ayuda); el tamaño se cambia en Ver → Escala de símbolos
        def interruptor(nombre_icono: str, ayuda: str, senal) -> QToolButton:
            boton = QToolButton()
            boton.setIcon(icono(nombre_icono))
            boton.setCheckable(True)
            boton.setChecked(True)
            boton.setToolButtonStyle(Qt.ToolButtonIconOnly)
            boton.setToolTip(ayuda)
            boton.toggled.connect(senal.emit)
            fila.addWidget(boton)
            return boton

        self.boton_apoyos = interruptor("apoyo", "Mostrar u ocultar los apoyos", self.apoyos_visibles)
        self.boton_flechas = interruptor("carga", "Mostrar u ocultar las cargas", self.flechas_visibles)
        self.boton_uniones = interruptor("conectar", "Mostrar u ocultar los constraints", self.uniones_visibles)
        fila.addStretch(1)
        self._fila = fila

    def agregar_accion(self, accion) -> None:
        """Añade, al extremo derecho, un botón ligado a una acción (p. ej. Modelo / Esfuerzos)."""

        boton = QToolButton()
        boton.setDefaultAction(accion)
        boton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._fila.addWidget(boton)

    def fijar_visibilidad(self, tipo: str, visible: bool) -> None:
        self.botones_tipo[tipo].blockSignals(True)
        self.botones_tipo[tipo].setChecked(visible)
        self.botones_tipo[tipo].blockSignals(False)
