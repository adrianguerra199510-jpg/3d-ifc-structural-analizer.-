"""Diálogo para asignar apoyos a lo que se seleccionó en el visor.

Un apoyo va en el nodo del extremo de una viga o columna: se elige
seleccionando su **cara extrema** (inicial o final) en modo «Cara». Para
ver las bases de las columnas conviene la vista inferior. Si no hay nada
seleccionado, el diálogo permite elegir el nodo de una lista.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ..core.asignacion import ErrorAsignacion, Objetivo, descripcion, nodo_para_apoyo
from ..core.modelo import Proyecto

PRESETS = {
    "Empotramiento (restringe los 6 grados de libertad)": (True, True, True, True, True, True),
    "Articulado (restringe solo las 3 traslaciones)": (True, True, True, False, False, False),
    "Personalizado": None,
}


class DialogoApoyo(QDialog):
    def __init__(self, proyecto: Proyecto, objetivos: List[Objetivo], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar apoyo")
        self._proyecto = proyecto
        self.nodos: List[str] = []  # nodos donde se asignará el apoyo

        layout = QVBoxLayout(self)
        formulario = QFormLayout()

        self.combo_nodo: Optional[QComboBox] = None
        if objetivos:
            lineas = []
            for objetivo in objetivos:
                try:
                    nodo = nodo_para_apoyo(proyecto, objetivo)
                    if nodo not in self.nodos:
                        self.nodos.append(nodo)
                    lineas.append(f"✓ {descripcion(proyecto, objetivo)} → nodo {nodo}")
                except ErrorAsignacion as exc:
                    lineas.append(f"✗ {exc}")
            etiqueta = QLabel("\n".join(lineas))
            etiqueta.setWordWrap(True)
            layout.addWidget(etiqueta)
        else:
            layout.addWidget(QLabel("No hay nada seleccionado en el visor. Elija el nodo:"))
            self.combo_nodo = QComboBox()
            nodos_de_barras = sorted(
                {n for b in proyecto.todas_las_barras().values() for n in (b.nodo_inicio, b.nodo_fin)}
            )
            self.combo_nodo.addItems(nodos_de_barras)
            formulario.addRow("Nodo:", self.combo_nodo)

        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS))
        formulario.addRow("Tipo de apoyo:", self.combo_preset)

        self.checks = []
        for texto in ("Ux (traslación X)", "Uy (traslación Y)", "Uz (traslación Z)",
                      "Rx (giro alrededor de X)", "Ry (giro alrededor de Y)", "Rz (giro alrededor de Z)"):
            check = QCheckBox(texto)
            check.toggled.connect(self._al_cambiar_check)
            self.checks.append(check)
            formulario.addRow(check)
        layout.addLayout(formulario)

        nota = QLabel(
            "La estructura debe quedar estable: los apoyos deben impedir las 3 traslaciones y los 3 giros "
            "como cuerpo rígido (por ejemplo, un empotramiento, o varios apoyos articulados bien repartidos "
            "más una restricción de giro/torsión)."
        )
        nota.setWordWrap(True)
        layout.addWidget(nota)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)

        self.combo_preset.currentIndexChanged.connect(self._al_cambiar_preset)
        self._al_cambiar_preset()
        self._actualizar_boton_aceptar()

    def _al_cambiar_preset(self) -> None:
        valores = PRESETS[self.combo_preset.currentText()]
        if valores is None:
            return
        for check, valor in zip(self.checks, valores):
            check.blockSignals(True)
            check.setChecked(valor)
            check.blockSignals(False)

    def _al_cambiar_check(self) -> None:
        self.combo_preset.blockSignals(True)
        actuales = tuple(c.isChecked() for c in self.checks)
        coincide = next((n for n, v in PRESETS.items() if v == actuales), "Personalizado")
        self.combo_preset.setCurrentText(coincide)
        self.combo_preset.blockSignals(False)

    def _actualizar_boton_aceptar(self) -> None:
        hay_nodos = bool(self.nodos) or self.combo_nodo is not None
        self.botones.button(QDialogButtonBox.Ok).setEnabled(hay_nodos)

    def restricciones(self) -> Tuple[bool, bool, bool, bool, bool, bool]:
        return tuple(c.isChecked() for c in self.checks)  # type: ignore[return-value]

    def nodos_elegidos(self) -> List[str]:
        if self.combo_nodo is not None:
            return [self.combo_nodo.currentText()]
        return list(self.nodos)
