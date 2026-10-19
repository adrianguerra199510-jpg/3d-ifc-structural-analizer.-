"""Sistema visual de la aplicación: paleta, estilo Fusion y hoja de estilos.

Se usa el estilo Fusion de Qt (igual en todas las plataformas) con una
paleta propia, y una hoja de estilos solo para lo que Qt no resuelve bien
con la paleta: barras de herramientas, paneles acoplables, pestañas, tablas
y botones. Los cuadros combinados y los selectores numéricos conservan su
aspecto nativo de Fusion (flechas incluidas).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .iconos import icono, pixmap_logo

# -- paleta ---------------------------------------------------------------------------
NAVY = "#0d2a50"
FONDO = "#eef1f5"
SUPERFICIE = "#ffffff"
SUPERFICIE_SUAVE = "#f6f8fb"
BORDE = "#d3dae3"
BORDE_SUAVE = "#e4e9f0"
TEXTO = "#1d2b3a"
TEXTO_SUAVE = "#5b6b7c"
ACENTO = "#1f6feb"
ACENTO_OSCURO = "#1857bd"
ACENTO_CLARO = "#e6f0ff"
ACENTO_MEDIO = "#cfe2ff"
VERDE = "#2e8b57"
VERDE_CLARO = "#3cb371"
ROJO = "#d64545"
AMBAR = "#e0a100"

FUENTES = '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif'


def _paleta() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window, QColor(FONDO))
    p.setColor(QPalette.WindowText, QColor(TEXTO))
    p.setColor(QPalette.Base, QColor(SUPERFICIE))
    p.setColor(QPalette.AlternateBase, QColor(SUPERFICIE_SUAVE))
    p.setColor(QPalette.Text, QColor(TEXTO))
    p.setColor(QPalette.Button, QColor(SUPERFICIE))
    p.setColor(QPalette.ButtonText, QColor(TEXTO))
    p.setColor(QPalette.ToolTipBase, QColor(NAVY))
    p.setColor(QPalette.ToolTipText, QColor("#ffffff"))
    p.setColor(QPalette.Highlight, QColor(ACENTO))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.PlaceholderText, QColor("#8a97a6"))
    p.setColor(QPalette.Link, QColor(ACENTO))
    p.setColor(QPalette.Mid, QColor(BORDE))
    p.setColor(QPalette.Midlight, QColor(BORDE_SUAVE))
    p.setColor(QPalette.Dark, QColor("#9aa6b5"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#9aa6b5"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#9aa6b5"))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#9aa6b5"))
    return p


HOJA_DE_ESTILOS = f"""
* {{ font-family: {FUENTES}; }}
QMainWindow, QDialog {{ background: {FONDO}; }}
QToolTip {{ background: {NAVY}; color: white; border: none; padding: 5px 8px; border-radius: 4px; }}

/* ---- barras de herramientas ---- */
QToolBar {{ background: {SUPERFICIE}; border: none; border-bottom: 1px solid {BORDE}; spacing: 2px; padding: 4px 8px; }}
QToolBar::separator {{ background: {BORDE}; width: 1px; margin: 6px 7px; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 7px;
                       padding: 4px 8px; color: {TEXTO}; font-size: 11px; }}
QToolBar QToolButton:hover {{ background: {ACENTO_CLARO}; border-color: {ACENTO_MEDIO}; }}
QToolBar QToolButton:pressed {{ background: {ACENTO_MEDIO}; }}
QToolBar QToolButton:checked {{ background: {ACENTO_MEDIO}; border-color: {ACENTO}; color: {ACENTO_OSCURO}; font-weight: 600; }}
QToolBar QToolButton:disabled {{ color: #a3aebb; }}
QToolBar QToolButton::menu-indicator {{ image: none; }}
QToolBar QLabel {{ color: {TEXTO_SUAVE}; font-size: 10px; }}
QToolButton#botonAnalizar {{ background: {VERDE}; color: white; font-weight: 700; font-size: 13px;
                            padding: 7px 22px; border: 1px solid #25744a; border-radius: 9px; }}
QToolButton#botonAnalizar:hover {{ background: {VERDE_CLARO}; }}
QToolButton#botonAnalizar:pressed {{ background: #25744a; }}
QToolButton#botonAnalizar:disabled {{ background: #b9c4cf; border-color: #b9c4cf; color: #eef1f5; }}

/* ---- barra de vista sobre el visor ---- */
QWidget#barraVista {{ background: {SUPERFICIE}; border-bottom: 1px solid {BORDE}; }}
QWidget#barraVista QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 3px 6px;
                                  color: {TEXTO_SUAVE}; font-size: 11px; }}
QWidget#barraVista QToolButton:hover {{ background: {ACENTO_CLARO}; color: {TEXTO}; }}
QWidget#barraVista QToolButton:checked {{ background: {ACENTO_CLARO}; border-color: {ACENTO_MEDIO}; color: {ACENTO_OSCURO}; }}
QLabel#chipLeyenda {{ color: {TEXTO_SUAVE}; font-size: 11px; padding: 0 2px; }}

/* ---- paneles acoplables ---- */
QDockWidget {{ font-weight: 600; color: {TEXTO}; }}
QDockWidget::title {{ background: {SUPERFICIE_SUAVE}; padding: 8px 12px; border-bottom: 1px solid {BORDE};
                     border-top: 1px solid {BORDE}; text-align: left; }}
QDockWidget > QWidget {{ background: {SUPERFICIE}; }}

/* ---- pestañas ---- */
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDE}; background: {SUPERFICIE}; }}
QTabBar::tab {{ background: transparent; color: {TEXTO_SUAVE}; padding: 8px 14px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXTO}; background: {SUPERFICIE_SUAVE}; }}
QTabBar::tab:selected {{ color: {ACENTO}; border-bottom: 2px solid {ACENTO}; font-weight: 600; }}
QDialog QTabWidget::pane {{ border: 1px solid {BORDE}; border-radius: 6px; top: -1px; }}
QDialog QTabBar::tab {{ padding: 7px 16px; }}

/* ---- tablas y listas ---- */
QTableWidget, QListWidget, QPlainTextEdit {{ background: {SUPERFICIE}; alternate-background-color: {SUPERFICIE_SUAVE};
    border: 1px solid {BORDE}; border-radius: 6px; gridline-color: {BORDE_SUAVE}; selection-background-color: {ACENTO_MEDIO};
    selection-color: {TEXTO}; }}
QDockWidget QTableWidget {{ border: none; border-radius: 0; }}
QTableWidget::item {{ padding: 5px 6px; }}
QListWidget::item {{ padding: 7px 8px; border-radius: 5px; margin: 1px 3px; }}
QListWidget::item:selected {{ background: {ACENTO_MEDIO}; color: {TEXTO}; }}
QListWidget::item:hover:!selected {{ background: {SUPERFICIE_SUAVE}; }}
QHeaderView::section {{ background: {SUPERFICIE_SUAVE}; color: {TEXTO_SUAVE}; border: none; border-bottom: 1px solid {BORDE};
                       border-right: 1px solid {BORDE_SUAVE}; padding: 6px 8px; font-weight: 600; font-size: 11px; }}
QTableCornerButton::section {{ background: {SUPERFICIE_SUAVE}; border: none; }}

/* ---- botones ---- */
QPushButton {{ background: {SUPERFICIE}; border: 1px solid {BORDE}; border-radius: 7px; padding: 6px 16px; color: {TEXTO}; }}
QPushButton:hover {{ border-color: {ACENTO}; background: {ACENTO_CLARO}; }}
QPushButton:pressed {{ background: {ACENTO_MEDIO}; }}
QPushButton:default {{ background: {ACENTO}; border-color: {ACENTO_OSCURO}; color: white; font-weight: 600; }}
QPushButton:default:hover {{ background: {ACENTO_OSCURO}; }}
QPushButton:disabled {{ background: {SUPERFICIE_SUAVE}; color: #a3aebb; border-color: {BORDE_SUAVE}; }}
QPushButton#botonPrimario {{ background: {ACENTO}; border-color: {ACENTO_OSCURO}; color: white; font-weight: 600; padding: 10px 26px; font-size: 13px; }}
QPushButton#botonPrimario:hover {{ background: {ACENTO_OSCURO}; }}
QPushButton#botonSecundario {{ padding: 10px 22px; font-size: 13px; }}

/* ---- campos y grupos ---- */
QGroupBox {{ background: {SUPERFICIE}; border: 1px solid {BORDE}; border-radius: 9px; margin-top: 16px; padding: 14px 12px 10px 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px; color: {ACENTO_OSCURO}; background: {FONDO}; }}
QLineEdit {{ border: 1px solid {BORDE}; border-radius: 6px; padding: 5px 8px; background: {SUPERFICIE}; }}
QLineEdit:focus {{ border-color: {ACENTO}; }}
QLineEdit:disabled {{ background: {SUPERFICIE_SUAVE}; color: #8a97a6; }}
QCheckBox {{ spacing: 7px; }}
QStatusBar {{ background: {SUPERFICIE}; border-top: 1px solid {BORDE}; color: {TEXTO_SUAVE}; }}
QStatusBar::item {{ border: none; }}
QLabel#etiquetaEstado {{ color: {TEXTO}; background: {SUPERFICIE_SUAVE}; border: 1px solid {BORDE_SUAVE}; border-radius: 9px; padding: 2px 10px; font-size: 11px; }}
QMenuBar {{ background: {SUPERFICIE}; border-bottom: 1px solid {BORDE_SUAVE}; padding: 2px 6px; }}
QMenuBar::item {{ padding: 5px 10px; border-radius: 5px; background: transparent; }}
QMenuBar::item:selected {{ background: {ACENTO_CLARO}; }}
QMenu {{ background: {SUPERFICIE}; border: 1px solid {BORDE}; border-radius: 8px; padding: 5px; }}
QMenu::item {{ padding: 6px 26px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACENTO_CLARO}; color: {TEXTO}; }}
QMenu::item:disabled {{ color: #a3aebb; }}
QMenu::separator {{ height: 1px; background: {BORDE_SUAVE}; margin: 5px 8px; }}
QProgressDialog, QMessageBox {{ background: {SUPERFICIE}; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c4cdd8; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #a9b5c3; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #c4cdd8; border-radius: 4px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""


def aplicar_estilo(app: QApplication) -> None:
    """Estilo Fusion + paleta + hoja de estilos de la aplicación."""

    app.setStyle("Fusion")
    app.setPalette(_paleta())
    fuente = QFont()
    fuente.setPointSizeF(max(fuente.pointSizeF(), 9.5))
    app.setFont(fuente)
    app.setStyleSheet(HOJA_DE_ESTILOS)


class EncabezadoDialogo(QFrame):
    """Banda superior de un diálogo: icono, título y una línea de ayuda."""

    def __init__(self, nombre_icono: str, titulo: str, subtitulo: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(
            f"EncabezadoDialogo {{ background: {SUPERFICIE}; border: 1px solid {BORDE}; border-radius: 10px; }}"
        )
        fila = QHBoxLayout(self)
        fila.setContentsMargins(14, 10, 14, 10)
        fila.setSpacing(12)
        etiqueta_icono = QLabel()
        etiqueta_icono.setPixmap(icono(nombre_icono).pixmap(34, 34))
        etiqueta_icono.setStyleSheet(f"background: {ACENTO_CLARO}; border-radius: 10px; padding: 7px;")
        fila.addWidget(etiqueta_icono)
        textos = QVBoxLayout()
        textos.setSpacing(1)
        t = QLabel(titulo)
        t.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {TEXTO}; background: transparent;")
        textos.addWidget(t)
        if subtitulo:
            s = QLabel(subtitulo)
            s.setWordWrap(True)
            s.setStyleSheet(f"font-size: 11px; color: {TEXTO_SUAVE}; background: transparent;")
            textos.addWidget(s)
        fila.addLayout(textos, 1)


def etiqueta_logo(lado: int = 28) -> QLabel:
    """QLabel con el logo, nítido en pantallas de alta densidad."""

    etiqueta = QLabel()
    ratio = 2.0
    etiqueta.setPixmap(pixmap_logo(lado, ratio))
    etiqueta.setFixedSize(lado, lado)
    return etiqueta
