"""Diálogo para asignar un apoyo a un nodo.

En la Fase 1 los apoyos se asignan sobre nodos (extremos de vigas/columnas);
el soporte de apoyos sobre caras de losas queda para la Fase 2 (modo
shell), donde una cara sí tiene grados de libertad propios que restringir.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
)

from ..core.cargas import Apoyo
from ..core.modelo import Proyecto


class DialogoApoyo(QDialog):
    def __init__(self, proyecto: Proyecto, nodo_sugerido: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar apoyo")
        self._proyecto = proyecto

        self.combo_nodo = QComboBox()
        for nodo_id in sorted(proyecto.nodos):
            self.combo_nodo.addItem(nodo_id)
        if nodo_sugerido and nodo_sugerido in proyecto.nodos:
            self.combo_nodo.setCurrentText(nodo_sugerido)

        self.check_ux = QCheckBox("Ux (traslación X)")
        self.check_uy = QCheckBox("Uy (traslación Y)")
        self.check_uz = QCheckBox("Uz (traslación Z)")
        self.check_rx = QCheckBox("Rx (rotación X)")
        self.check_ry = QCheckBox("Ry (rotación Y)")
        self.check_rz = QCheckBox("Rz (rotación Z)")
        for chk in (self.check_ux, self.check_uy, self.check_uz, self.check_rx, self.check_ry, self.check_rz):
            chk.setChecked(True)  # por defecto: empotramiento

        formulario = QFormLayout()
        formulario.addRow("Nodo:", self.combo_nodo)
        formulario.addRow(self.check_ux)
        formulario.addRow(self.check_uy)
        formulario.addRow(self.check_uz)
        formulario.addRow(self.check_rx)
        formulario.addRow(self.check_ry)
        formulario.addRow(self.check_rz)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(formulario)
        layout.addWidget(botones)

    def apoyo_resultante(self, id_: str) -> Apoyo:
        return Apoyo(
            id=id_,
            nodo_id=self.combo_nodo.currentText(),
            ux=self.check_ux.isChecked(),
            uy=self.check_uy.isChecked(),
            uz=self.check_uz.isChecked(),
            rx=self.check_rx.isChecked(),
            ry=self.check_ry.isChecked(),
            rz=self.check_rz.isChecked(),
        )
