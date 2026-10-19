"""Diálogo «Acerca de»: logo, versión y componentes de terceros."""

from __future__ import annotations

from importlib import metadata

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from .. import __version__
from .estilo import TEXTO, TEXTO_SUAVE
from .iconos import pixmap_logo


def _version(paquete: str) -> str:
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return "—"


class DialogoAcerca(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Acerca de")
        self.setFixedWidth(460)
        capa = QVBoxLayout(self)
        capa.setContentsMargins(28, 24, 28, 18)
        capa.setSpacing(6)
        capa.setAlignment(Qt.AlignHCenter)

        logo = QLabel()
        logo.setPixmap(pixmap_logo(104, 2.0))
        logo.setAlignment(Qt.AlignCenter)
        capa.addWidget(logo)
        titulo = QLabel("Analizador Estructural 3D")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {TEXTO};")
        capa.addWidget(titulo)
        version = QLabel(f"Versión {__version__}")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet(f"color: {TEXTO_SUAVE};")
        capa.addWidget(version)
        capa.addSpacing(8)

        descripcion = QLabel(
            "Análisis estructural lineal de modelos IFC con CalculiX: apoyos, cargas, combinaciones y "
            "resultados en esfuerzos."
        )
        descripcion.setWordWrap(True)
        descripcion.setAlignment(Qt.AlignCenter)
        capa.addWidget(descripcion)
        capa.addSpacing(8)

        componentes = QLabel(
            "<b>Componentes</b><br>"
            f"CalculiX 2.21 (solver, GPL) · ifcopenshell {_version('ifcopenshell')}<br>"
            f"PyVista {_version('pyvista')} · PySide6 {_version('PySide6')} · NumPy {_version('numpy')}"
        )
        componentes.setAlignment(Qt.AlignCenter)
        componentes.setStyleSheet(f"font-size: 11px; color: {TEXTO_SUAVE};")
        capa.addWidget(componentes)
        capa.addSpacing(10)

        botones = QDialogButtonBox(QDialogButtonBox.Close)
        botones.rejected.connect(self.reject)
        botones.accepted.connect(self.accept)
        capa.addWidget(botones)
