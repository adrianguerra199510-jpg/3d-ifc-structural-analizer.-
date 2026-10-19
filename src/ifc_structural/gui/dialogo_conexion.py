"""Diálogo para conectar dos elementos (constraint entre sólidos).

Selecciona dos vigas/columnas (o sus caras extremas) y elige qué se
transmite entre ellas:

* **Articulación**: se transmiten las traslaciones y se liberan los giros. Si
  comparten un nudo, la viga (o la segunda seleccionada) se separa del nudo y
  se une a él con una rótula.
* **Rígida**: se transmite todo. Solo tiene sentido entre extremos que no
  comparten nudo (se crea un enlace rígido); entre barras que ya comparten un
  nudo la unión ya es rígida.
* **Personalizada**: elige qué traslaciones se transmiten (y, si quiere, los giros).
"""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from ..core.asignacion import ErrorAsignacion, Objetivo, Restricciones, conexion_para_objetivos, descripcion
from ..core.modelo import Proyecto
from .estilo import EncabezadoDialogo

PRESETS = {
    "Articulación (transmite traslaciones, libera los 3 giros)": (True, True, True, False, False, False),
    "Rígida (transmite traslaciones y giros)": (True, True, True, True, True, True),
    "Personalizada": None,
}


class DialogoConexion(QDialog):
    def __init__(self, proyecto: Proyecto, a: Objetivo, b: Objetivo, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conectar elementos")
        self.resize(600, 560)
        self._proyecto, self._a, self._b = proyecto, a, b

        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo("conectar", "Conectar elementos", "Articulación o enlace rígido entre dos elementos (constraint)."))
        layout.addWidget(QLabel(
            f"<b>Conexión entre</b> {descripcion(proyecto, a)} <b>y</b> {descripcion(proyecto, b)}"
        ))
        formulario = QFormLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS))
        formulario.addRow("Tipo de conexión:", self.combo_preset)

        self.checks_traslacion = []
        for texto in ("Ux (traslación en X)", "Uy (traslación en Y)", "Uz (traslación en Z)"):
            check = QCheckBox(texto + " — se transmite")
            check.toggled.connect(self._al_cambiar_check)
            self.checks_traslacion.append(check)
            formulario.addRow(check)
        self.check_giros = QCheckBox("Giros Rx, Ry, Rz — se transmiten (los tres juntos)")
        self.check_giros.toggled.connect(self._al_cambiar_check)
        formulario.addRow(self.check_giros)
        self.check_torsion = QCheckBox(
            "Impedir el giro alrededor del eje de la barra articulada (torsión)"
        )
        self.check_torsion.setChecked(True)
        self.check_torsion.setToolTip(
            "Evita que una barra articulada en ambos extremos pueda girar sobre su propio eje (mecanismo). "
            "Solo se aplica si la barra es paralela a un eje global."
        )
        self.check_torsion.toggled.connect(self._actualizar_vista_previa)
        formulario.addRow(self.check_torsion)
        layout.addLayout(formulario)

        layout.addWidget(QLabel("<b>Resultado:</b>"))
        self.vista_previa = QPlainTextEdit()
        self.vista_previa.setReadOnly(True)
        layout.addWidget(self.vista_previa)

        nota = QLabel(
            "Limitación de CalculiX: los giros de las vigas se transmiten todos o ninguno (no hay "
            "articulaciones parciales). Las articulaciones dejan libre la estructura si no hay suficientes "
            "apoyos: el análisis avisa de los mecanismos."
        )
        nota.setWordWrap(True)
        layout.addWidget(nota)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)

        self.combo_preset.currentIndexChanged.connect(self._al_cambiar_preset)
        self._al_cambiar_preset()

    def _al_cambiar_preset(self) -> None:
        valores = PRESETS[self.combo_preset.currentText()]
        if valores is not None:
            self._fijar(valores)
        self._actualizar_vista_previa()

    def _fijar(self, valores: Restricciones) -> None:
        for check, valor in zip(self.checks_traslacion, valores[:3]):
            check.blockSignals(True)
            check.setChecked(valor)
            check.blockSignals(False)
        self.check_giros.blockSignals(True)
        self.check_giros.setChecked(valores[3])
        self.check_giros.blockSignals(False)

    def _al_cambiar_check(self) -> None:
        actuales = self.transmitidos()
        coincide = next((n for n, v in PRESETS.items() if v == actuales), "Personalizada")
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentText(coincide)
        self.combo_preset.blockSignals(False)
        self._actualizar_vista_previa()

    def transmitidos(self) -> Restricciones:
        t = [c.isChecked() for c in self.checks_traslacion]
        g = self.check_giros.isChecked()
        return (t[0], t[1], t[2], g, g, g)

    def fijar_torsion(self) -> bool:
        return self.check_torsion.isChecked()

    def conexion(self, id_: str = "-"):
        return conexion_para_objetivos(self._proyecto, self._a, self._b, id_, self.transmitidos(), self.fijar_torsion())

    def _actualizar_vista_previa(self) -> None:
        self.check_torsion.setEnabled(not self.check_giros.isChecked())
        try:
            conexion = self.conexion()
            if conexion.tipo == "EXTREMO":
                texto = (
                    f"✓ {conexion.nombre}\n   El extremo de la barra {conexion.barra_id} se separa del nudo "
                    f"{conexion.nodo_id} y se une transmitiendo: "
                    f"{self._texto_gdl(conexion.grados_transmitidos())}."
                )
            else:
                texto = f"✓ {conexion.nombre}\n   Se transmite: {self._texto_gdl(conexion.grados_transmitidos())}."
            self.botones.button(QDialogButtonBox.Ok).setEnabled(True)
        except ErrorAsignacion as exc:
            texto = f"✗ {exc}"
            self.botones.button(QDialogButtonBox.Ok).setEnabled(False)
        self.vista_previa.setPlainText(texto)

    @staticmethod
    def _texto_gdl(grados: List[int]) -> str:
        nombres = {1: "Ux", 2: "Uy", 3: "Uz", 4: "Rx", 5: "Ry", 6: "Rz"}
        return ", ".join(nombres[g] for g in grados) or "nada"
