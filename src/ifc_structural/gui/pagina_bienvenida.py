"""Pantalla de inicio: se muestra mientras no hay un modelo cargado."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import __version__
from .estilo import ACENTO, ACENTO_CLARO, BORDE, FONDO, SUPERFICIE, TEXTO, TEXTO_SUAVE
from .iconos import icono, pixmap_logo

PASOS = (
    ("importar", "1", "Importe el modelo", "Cargue el IFC exportado desde Revit: vigas, columnas, losas y muros."),
    ("sel_elemento", "2", "Seleccione y asigne",
     "Clic para elegir un elemento o una cara; clic derecho para asignar apoyos, cargas, materiales o conexiones."),
    ("analizar", "3", "Analice",
     "Pulse ANALIZAR y revise los esfuerzos (von Mises, σ₁, σ₃, τmax) por caso, combinación o envolvente."),
)


class _TarjetaPaso(QFrame):
    def __init__(self, nombre_icono: str, numero: str, titulo: str, texto: str):
        super().__init__()
        self.setObjectName("tarjetaPaso")
        self.setStyleSheet(
            f"QFrame#tarjetaPaso {{ background: {SUPERFICIE}; border: 1px solid {BORDE}; border-radius: 12px; }}"
            f"QLabel {{ background: transparent; }}"
        )
        self.setMinimumWidth(210)
        self.setMaximumWidth(250)
        capa = QVBoxLayout(self)
        capa.setContentsMargins(16, 16, 16, 16)
        capa.setSpacing(8)
        fila = QHBoxLayout()
        pastilla = QLabel()
        pastilla.setPixmap(icono(nombre_icono).pixmap(28, 28))
        pastilla.setStyleSheet(f"background: {ACENTO_CLARO}; border-radius: 9px; padding: 6px;")
        fila.addWidget(pastilla)
        fila.addStretch(1)
        n = QLabel(numero)
        n.setStyleSheet(f"font-size: 26px; font-weight: 800; color: {BORDE};")
        fila.addWidget(n)
        capa.addLayout(fila)
        t = QLabel(titulo)
        t.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {TEXTO};")
        capa.addWidget(t)
        d = QLabel(texto)
        d.setWordWrap(True)
        d.setStyleSheet(f"font-size: 11px; color: {TEXTO_SUAVE};")
        capa.addWidget(d)
        capa.addStretch(1)


class PaginaBienvenida(QWidget):
    importar_solicitado = Signal()
    abrir_solicitado = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"PaginaBienvenida {{ background: {FONDO}; }}")
        raiz = QVBoxLayout(self)
        raiz.setAlignment(Qt.AlignCenter)
        raiz.setSpacing(0)

        logo = QLabel()
        logo.setPixmap(pixmap_logo(112, 2.0))
        logo.setAlignment(Qt.AlignCenter)
        raiz.addWidget(logo)
        raiz.addSpacing(14)

        titulo = QLabel("Analizador Estructural 3D")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {TEXTO};")
        raiz.addWidget(titulo)
        subtitulo = QLabel("Modelos IFC · análisis lineal con CalculiX · resultados en esfuerzos")
        subtitulo.setAlignment(Qt.AlignCenter)
        subtitulo.setStyleSheet(f"font-size: 13px; color: {TEXTO_SUAVE};")
        raiz.addWidget(subtitulo)
        raiz.addSpacing(22)

        botones = QHBoxLayout()
        botones.setAlignment(Qt.AlignCenter)
        botones.setSpacing(12)
        self.boton_importar = QPushButton("  Importar modelo IFC")
        self.boton_importar.setObjectName("botonPrimario")
        self.boton_importar.setIcon(icono("importar", "#ffffff", "#ffffff"))
        self.boton_importar.setCursor(Qt.PointingHandCursor)
        self.boton_importar.clicked.connect(self.importar_solicitado)
        self.boton_abrir = QPushButton("  Abrir proyecto")
        self.boton_abrir.setObjectName("botonSecundario")
        self.boton_abrir.setIcon(icono("abrir"))
        self.boton_abrir.setCursor(Qt.PointingHandCursor)
        self.boton_abrir.clicked.connect(self.abrir_solicitado)
        botones.addWidget(self.boton_importar)
        botones.addWidget(self.boton_abrir)
        raiz.addLayout(botones)
        raiz.addSpacing(30)

        pasos = QHBoxLayout()
        pasos.setAlignment(Qt.AlignCenter)
        pasos.setSpacing(16)
        for paso in PASOS:
            pasos.addWidget(_TarjetaPaso(*paso))
        raiz.addLayout(pasos)
        raiz.addSpacing(26)

        pie = QLabel(f"Versión {__version__}  ·  CalculiX · ifcopenshell · PyVista · Qt")
        pie.setAlignment(Qt.AlignCenter)
        pie.setStyleSheet(f"font-size: 10px; color: {TEXTO_SUAVE};")
        raiz.addWidget(pie)
