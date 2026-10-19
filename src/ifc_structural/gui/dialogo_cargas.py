"""Diálogo para asignar cargas a lo que se seleccionó en el visor.

Qué carga resulta depende de lo seleccionado (ver ``core.asignacion``):
losas → carga de área; vigas/columnas completas → carga lineal; caras
laterales de vigas/columnas → carga lineal (presión × ancho de la cara);
caras extremas → fuerza axial en el nodo. La vista previa muestra, para
cada elemento seleccionado, la carga que se creará o por qué no se puede.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from ..core.asignacion import DIRECCIONES, Objetivo, vista_previa_cargas
from ..core.casos_carga import CasoCarga
from ..core.modelo import Proyecto


def _spin(valor: float, sufijo: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-1e9, 1e9)
    spin.setDecimals(3)
    spin.setValue(valor)
    spin.setSuffix(sufijo)
    return spin


class DialogoCargas(QDialog):
    def __init__(self, proyecto: Proyecto, objetivos: List[Objetivo], casos: Dict[str, CasoCarga],
                 caso_activo: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar carga")
        self.resize(560, 420)
        self._proyecto = proyecto
        self._objetivos = objetivos

        self.combo_caso = QComboBox()
        for caso in casos.values():
            self.combo_caso.addItem(f"{caso.id} — {caso.nombre}", caso.id)
        indice = self.combo_caso.findData(caso_activo)
        if indice >= 0:
            self.combo_caso.setCurrentIndex(indice)

        self.spin_presion = _spin(5.0, " kPa")
        self.spin_lineal = _spin(10.0, " kN/m")
        self.combo_direccion = QComboBox()
        self.combo_direccion.addItems(list(DIRECCIONES))

        formulario = QFormLayout()
        formulario.addRow("Caso de carga:", self.combo_caso)
        formulario.addRow("Presión:", self.spin_presion)
        formulario.addRow("Carga lineal:", self.spin_lineal)
        formulario.addRow("Dirección:", self.combo_direccion)

        ayuda = QLabel(
            "La <b>presión</b> se usa en losas y en las caras de vigas/columnas (positiva = empuja hacia "
            "dentro de la cara). La <b>carga lineal</b> se usa al seleccionar la viga o columna completa. "
            "La <b>dirección</b> solo cuenta cuando no se eligió una cara (en una cara, la carga es "
            "perpendicular a ella)."
        )
        ayuda.setWordWrap(True)

        self.vista_previa = QPlainTextEdit()
        self.vista_previa.setReadOnly(True)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(formulario)
        layout.addWidget(ayuda)
        layout.addWidget(QLabel("<b>Se crearán estas cargas:</b>"))
        layout.addWidget(self.vista_previa)
        layout.addWidget(self.botones)

        for senal in (self.spin_presion.valueChanged, self.spin_lineal.valueChanged,
                      self.combo_direccion.currentIndexChanged):
            senal.connect(self._actualizar_vista_previa)
        self._actualizar_vista_previa()

    def _actualizar_vista_previa(self) -> None:
        if not self._objetivos:
            self.vista_previa.setPlainText(
                "No hay nada seleccionado. Seleccione elementos o caras en el visor antes de asignar cargas."
            )
            self.botones.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        vista = vista_previa_cargas(self._proyecto, self._objetivos, *self.parametros()[1:])
        self.vista_previa.setPlainText("\n".join(f"{'✓' if ok else '✗'} {texto}" for ok, texto in vista))
        self.botones.button(QDialogButtonBox.Ok).setEnabled(any(ok for ok, _ in vista))

    def parametros(self) -> Tuple[str, float, float, Tuple[float, float, float]]:
        """(caso_id, presión kPa, carga lineal kN/m, dirección global)."""

        return (
            self.combo_caso.currentData(),
            self.spin_presion.value(),
            self.spin_lineal.value(),
            DIRECCIONES[self.combo_direccion.currentText()],
        )
