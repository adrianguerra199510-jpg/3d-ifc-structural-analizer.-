"""Barra sobre el visor 3D: vistas de cámara, visibilidad por tipo de
elemento y leyenda de colores."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QWidget

from .iconos import icono
from .visor_3d import COLORES_BASE

VISTAS = (
    ("vista_iso", "Isométrica", "Vista isométrica"),
    ("vista_planta", "Planta", "Vista en planta (desde arriba)"),
    ("vista_frontal", "Frontal", "Vista frontal"),
    ("vista_lateral", "Lateral", "Vista lateral"),
    ("vista_inferior", "Inferior", "Vista inferior: permite hacer clic en la base de las columnas"),
    ("ajustar", "Ajustar", "Encuadrar todo el modelo"),
)
TIPOS = (("VIGA", "Vigas", "viga"), ("COLUMNA", "Columnas", "columna"), ("LOSA", "Losas", "losa"), ("MURO", "Muros", "muro"))


def _cuadro_color(rgb) -> QPixmap:
    px = QPixmap(12, 12)
    px.fill(QColor(*rgb))
    return px


class BarraVista(QWidget):
    vista_solicitada = Signal(str)  # "vista_iso", "vista_planta", ..., "ajustar"
    visibilidad_cambiada = Signal(str, bool)  # tipo, visible

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
            boton.setText(texto)
            boton.setCheckable(True)
            boton.setChecked(True)
            boton.setIcon(icono(nombre_icono))
            boton.setToolTip(f"Mostrar u ocultar {texto.lower()}")
            boton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            boton.toggled.connect(lambda visible, t=tipo: self.visibilidad_cambiada.emit(t, visible))
            fila.addWidget(boton)
            self.botones_tipo[tipo] = boton
        fila.addStretch(1)

        # Leyenda de colores del modelo
        for tipo, texto, _ in TIPOS:
            cuadro = QLabel()
            cuadro.setPixmap(_cuadro_color(COLORES_BASE[tipo]))
            cuadro.setToolTip(texto)
            fila.addWidget(cuadro)
        etiqueta_leyenda = QLabel("  colores del modelo")
        etiqueta_leyenda.setObjectName("chipLeyenda")
        fila.addWidget(etiqueta_leyenda)

    def fijar_visibilidad(self, tipo: str, visible: bool) -> None:
        self.botones_tipo[tipo].blockSignals(True)
        self.botones_tipo[tipo].setChecked(visible)
        self.botones_tipo[tipo].blockSignals(False)
