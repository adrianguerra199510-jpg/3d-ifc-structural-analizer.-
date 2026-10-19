"""Panel de esfuerzos: qué resultado y qué magnitud mostrar en el visor,
con escala configurable, y la tabla de esfuerzos máximos por viga/columna.

Solo se muestran esfuerzos (kPa del solver convertidos a MPa), no fuerzas
internas ni desplazamientos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..combinaciones.esfuerzos import MAGNITUDES

ENVOLVENTE = "__ENVOLVENTE__"
COLORMAPS = ["turbo", "jet", "viridis", "coolwarm", "rainbow", "plasma"]


@dataclass
class ParametrosVista:
    """Lo que el usuario eligió mostrar."""

    resultado: str  # id de caso/combinación o ENVOLVENTE
    magnitud: str  # clave de esfuerzos.MAGNITUDES
    escala_automatica: bool
    minimo: float  # MPa
    maximo: float  # MPa
    colormap: str


class PanelEsfuerzos(QWidget):
    """Controles de la vista de esfuerzos. Emite ``cambio`` cada vez que
    algo que afecte a lo que se dibuja cambia, o ``volver_al_modelo``."""

    cambio = Signal()
    volver_al_modelo = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.combo_resultado = QComboBox()
        self.combo_magnitud = QComboBox()
        for magnitud in MAGNITUDES.values():
            self.combo_magnitud.addItem(magnitud.nombre, magnitud.clave)

        self.check_escala_automatica = QCheckBox("Escala automática")
        self.check_escala_automatica.setChecked(True)
        self.spin_minimo = self._spin()
        self.spin_maximo = self._spin()
        self.spin_maximo.setValue(10.0)
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(COLORMAPS)

        self.boton_volver = QPushButton("Volver al modelo (selección)")
        self.etiqueta_resumen = QLabel("Ejecute el análisis para ver los esfuerzos.")
        self.etiqueta_resumen.setWordWrap(True)

        self.tabla = QTableWidget(0, 3)
        self.tabla.setHorizontalHeaderLabels(["Barra", "Mínimo [MPa]", "Máximo [MPa]"])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla.setSelectionBehavior(QTableWidget.SelectRows)

        formulario = QFormLayout()
        formulario.addRow("Resultado:", self.combo_resultado)
        formulario.addRow("Esfuerzo:", self.combo_magnitud)
        formulario.addRow(self.check_escala_automatica)
        formulario.addRow("Mínimo [MPa]:", self.spin_minimo)
        formulario.addRow("Máximo [MPa]:", self.spin_maximo)
        formulario.addRow("Colores:", self.combo_colormap)

        layout = QVBoxLayout(self)
        layout.addLayout(formulario)
        layout.addWidget(self.etiqueta_resumen)
        layout.addWidget(self.tabla)
        layout.addWidget(self.boton_volver)

        self._actualizar_escala_habilitada()
        for senal in (
            self.combo_resultado.currentIndexChanged,
            self.combo_magnitud.currentIndexChanged,
            self.combo_colormap.currentIndexChanged,
            self.spin_minimo.editingFinished,
            self.spin_maximo.editingFinished,
        ):
            senal.connect(self.cambio)
        self.check_escala_automatica.toggled.connect(self._al_cambiar_escala_automatica)
        self.boton_volver.clicked.connect(self.volver_al_modelo)

    @staticmethod
    def _spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(3)
        spin.setSingleStep(0.5)
        return spin

    def _al_cambiar_escala_automatica(self) -> None:
        self._actualizar_escala_habilitada()
        self.cambio.emit()

    def _actualizar_escala_habilitada(self) -> None:
        manual = not self.check_escala_automatica.isChecked()
        self.spin_minimo.setEnabled(manual)
        self.spin_maximo.setEnabled(manual)

    # ------------------------------------------------------------------
    def establecer_resultados(self, casos: List[Tuple[str, str]], combinaciones: List[Tuple[str, str]]) -> None:
        """Opciones del combo: [(id, texto)] de casos y de combinaciones;
        la envolvente solo se ofrece si hay combinaciones."""

        self.combo_resultado.blockSignals(True)
        self.combo_resultado.clear()
        if combinaciones:
            self.combo_resultado.addItem("ENVOLVENTE (todas las combinaciones)", ENVOLVENTE)
        for id_, texto in combinaciones:
            self.combo_resultado.addItem(f"Combinación {id_}: {texto}", id_)
        for id_, texto in casos:
            self.combo_resultado.addItem(f"Caso {id_}: {texto}", id_)
        self.combo_resultado.blockSignals(False)

    def parametros(self) -> Optional[ParametrosVista]:
        if self.combo_resultado.count() == 0:
            return None
        return ParametrosVista(
            resultado=self.combo_resultado.currentData(),
            magnitud=self.combo_magnitud.currentData(),
            escala_automatica=self.check_escala_automatica.isChecked(),
            minimo=self.spin_minimo.value(),
            maximo=self.spin_maximo.value(),
            colormap=self.combo_colormap.currentText(),
        )

    def fijar_escala(self, minimo: float, maximo: float) -> None:
        """Muestra la escala automática en los cuadros (sin disparar ``cambio``)."""

        for spin, valor in ((self.spin_minimo, minimo), (self.spin_maximo, maximo)):
            spin.blockSignals(True)
            spin.setValue(valor)
            spin.blockSignals(False)

    def mostrar_resumen(self, texto: str, filas: Dict[str, Tuple[float, float]]) -> None:
        self.etiqueta_resumen.setText(texto)
        self.tabla.setRowCount(0)
        for barra, (minimo, maximo) in sorted(filas.items()):
            fila = self.tabla.rowCount()
            self.tabla.insertRow(fila)
            self.tabla.setItem(fila, 0, QTableWidgetItem(barra))
            self.tabla.setItem(fila, 1, QTableWidgetItem(f"{minimo:.3f}"))
            self.tabla.setItem(fila, 2, QTableWidgetItem(f"{maximo:.3f}"))
