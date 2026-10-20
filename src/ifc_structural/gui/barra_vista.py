"""Barra sobre el visor 3D: vistas de cámara, visibilidad por tipo de
elemento y leyenda de colores."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSlider, QToolButton, QWidget

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
    escala_flechas = Signal(float)  # tamaño de las flechas (1 = automático)
    zona_activada = Signal(bool)  # arrastrar con el botón izquierdo dibuja un rectángulo de selección

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
        self.boton_zona = QToolButton()
        self.boton_zona.setText("Zona")
        self.boton_zona.setIcon(icono("zona"))
        self.boton_zona.setCheckable(True)
        self.boton_zona.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.boton_zona.setToolTip(
            "Seleccionar arrastrando un rectángulo (también Ctrl o Mayús + arrastrar).\n"
            "De izquierda a derecha: solo lo que queda dentro. De derecha a izquierda: todo lo que toca."
        )
        self.boton_zona.toggled.connect(self.zona_activada.emit)
        fila.addWidget(self.boton_zona)
        self.boton_flechas = QToolButton()
        self.boton_flechas.setText("Cargas")
        self.boton_flechas.setIcon(icono("carga"))
        self.boton_flechas.setCheckable(True)
        self.boton_flechas.setChecked(True)
        self.boton_flechas.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.boton_flechas.setToolTip("Mostrar u ocultar las flechas de las cargas")
        self.boton_flechas.toggled.connect(self._al_cambiar_flechas)
        fila.addWidget(self.boton_flechas)
        self.deslizador_flechas = QSlider(Qt.Horizontal)
        self.deslizador_flechas.setRange(20, 400)  # % del tamaño automático
        self.deslizador_flechas.setValue(100)
        self.deslizador_flechas.setFixedWidth(90)
        self.deslizador_flechas.setToolTip("Tamaño de las flechas de las cargas (100 % = automático)")
        self.deslizador_flechas.valueChanged.connect(lambda v: self.escala_flechas.emit(v / 100.0))
        fila.addWidget(self.deslizador_flechas)
        fila.addStretch(1)
        self._fila = fila

    def _al_cambiar_flechas(self, visibles: bool) -> None:
        self.deslizador_flechas.setEnabled(visibles)
        self.flechas_visibles.emit(visibles)

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
