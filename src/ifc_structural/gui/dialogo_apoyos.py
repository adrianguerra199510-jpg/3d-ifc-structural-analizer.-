"""Diálogo para asignar apoyos a lo que se seleccionó en el visor.

Dos tipos de apoyo:

* **Puntual**: en el nodo de la cara extrema (inicial o final) de una viga o
  columna, p. ej. la base de una columna.
* **De área** (continuo): a lo largo de toda la barra o de un tramo (viga,
  columna o una de sus caras laterales), o a lo largo de las vigas de una
  losa (cara superior/inferior). Es el apoyo de un elemento sobre el terreno
  o sobre un muro.

En ambos casos se elige cuántos grados de libertad se restringen (las tres
traslaciones y los tres giros, uno por uno).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from ..core.asignacion import (
    ErrorAsignacion,
    Objetivo,
    Restricciones,
    apoyo_para_objetivo,
    cara_de,
    restriccion_solo_normal,
)
from ..core.modelo import Proyecto

PRESETS = {
    "Empotramiento (restringe los 6 grados de libertad)": (True, True, True, True, True, True),
    "Articulado (restringe solo las 3 traslaciones)": (True, True, True, False, False, False),
    "Solo la traslación vertical Uz (rodillo)": (False, False, True, False, False, False),
    "Solo la dirección normal a la cara (rodillo)": None,  # depende de la cara elegida
    "Personalizado": None,
}
PRESET_NORMAL = "Solo la dirección normal a la cara (rodillo)"
PRESET_PERSONALIZADO = "Personalizado"

TIPOS = {
    "Automático (puntual en caras extremas, de área en el resto)": "AUTO",
    "Puntual (en el nodo del extremo)": "PUNTUAL",
    "De área (continuo a lo largo del elemento)": "AREA",
}


class DialogoApoyo(QDialog):
    def __init__(self, proyecto: Proyecto, objetivos: List[Objetivo], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar apoyo")
        self.resize(600, 600)
        self._proyecto = proyecto
        self._objetivos = objetivos

        layout = QVBoxLayout(self)
        formulario = QFormLayout()

        self.combo_nodo: Optional[QComboBox] = None
        if not objetivos:
            layout.addWidget(QLabel("No hay nada seleccionado en el visor. Elija el nodo donde va el apoyo:"))
            self.combo_nodo = QComboBox()
            nodos_de_barras = sorted(
                {n for b in proyecto.todas_las_barras().values() for n in (b.nodo_inicio, b.nodo_fin)}
            )
            self.combo_nodo.addItems(nodos_de_barras)
            formulario.addRow("Nodo:", self.combo_nodo)

        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(list(TIPOS))
        self.combo_tipo.setEnabled(bool(objetivos))
        formulario.addRow("Tipo de apoyo:", self.combo_tipo)

        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS))
        formulario.addRow("Grados de libertad:", self.combo_preset)

        self.checks: List[QCheckBox] = []
        for texto in ("Ux (traslación en X)", "Uy (traslación en Y)", "Uz (traslación en Z)",
                      "Rx (giro alrededor de X)", "Ry (giro alrededor de Y)", "Rz (giro alrededor de Z)"):
            check = QCheckBox(texto)
            check.toggled.connect(self._al_cambiar_check)
            self.checks.append(check)
            formulario.addRow(check)

        self.spin_desde = self._spin(0.0)
        self.spin_hasta = self._spin(0.0)
        formulario.addRow("Tramo apoyado: desde", self.spin_desde)
        formulario.addRow("hasta", self.spin_hasta)
        layout.addLayout(formulario)

        layout.addWidget(QLabel("<b>Se crearán estos apoyos:</b>"))
        self.vista_previa = QPlainTextEdit()
        self.vista_previa.setReadOnly(True)
        layout.addWidget(self.vista_previa)

        nota = QLabel(
            "La estructura debe quedar estable: los apoyos deben impedir las 3 traslaciones y los 3 giros "
            "como cuerpo rígido. En un apoyo de área sobre una <b>losa</b>, la restricción se aplica a lo "
            "largo de las vigas que la sostienen (la losa no se malla en la Fase 1)."
        )
        nota.setWordWrap(True)
        layout.addWidget(nota)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)

        self._configurar_tramo()
        self.combo_preset.currentIndexChanged.connect(self._al_cambiar_preset)
        self.combo_tipo.currentIndexChanged.connect(self._actualizar_vista_previa)
        self.spin_desde.valueChanged.connect(self._actualizar_vista_previa)
        self.spin_hasta.valueChanged.connect(self._actualizar_vista_previa)
        self._elegir_preset_inicial()
        self._actualizar_vista_previa()

    @staticmethod
    def _spin(valor: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1e6)
        spin.setDecimals(3)
        spin.setSuffix(" m")
        spin.setValue(valor)
        return spin

    # -- tramo ---------------------------------------------------------------
    def _barra_unica(self) -> Optional[str]:
        barras = {o.id for o in self._objetivos if o.tipo in ("VIGA", "COLUMNA")}
        return next(iter(barras)) if len(barras) == 1 and all(o.tipo in ("VIGA", "COLUMNA") for o in self._objetivos) else None

    def _configurar_tramo(self) -> None:
        barra = self._barra_unica()
        habilitar = barra is not None
        for spin in (self.spin_desde, self.spin_hasta):
            spin.setEnabled(habilitar)
        if habilitar:
            longitud = self._proyecto.longitud_barra(barra)
            self.spin_desde.setMaximum(longitud)
            self.spin_hasta.setMaximum(longitud)
            self.spin_hasta.setValue(longitud)

    # -- grados de libertad ----------------------------------------------------
    def _restriccion_normal(self) -> Optional[Restricciones]:
        for objetivo in self._objetivos:
            cara = cara_de(self._proyecto, objetivo)
            restriccion = restriccion_solo_normal(cara)
            if restriccion is not None:
                return restriccion
        return None

    def _elegir_preset_inicial(self) -> None:
        # Un apoyo de área con cara plana suele ser un rodillo; el resto, un empotramiento.
        self.combo_preset.setCurrentIndex(0)
        self._al_cambiar_preset()

    def _al_cambiar_preset(self) -> None:
        nombre = self.combo_preset.currentText()
        valores = PRESETS[nombre]
        if nombre == PRESET_NORMAL:
            valores = self._restriccion_normal()
            if valores is None:
                self.combo_preset.setCurrentText(PRESET_PERSONALIZADO)
                return
        if valores is None:
            return
        for check, valor in zip(self.checks, valores):
            check.blockSignals(True)
            check.setChecked(valor)
            check.blockSignals(False)
        self._actualizar_vista_previa()

    def _al_cambiar_check(self) -> None:
        actuales = tuple(c.isChecked() for c in self.checks)
        coincide = next((n for n, v in PRESETS.items() if v == actuales), PRESET_PERSONALIZADO)
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentText(coincide)
        self.combo_preset.blockSignals(False)
        self._actualizar_vista_previa()

    # -- vista previa ------------------------------------------------------------
    def _construir(self, id_: str = "-") -> List[Tuple[bool, str, object]]:
        """[(válido, texto, apoyo)] para cada objetivo con la configuración actual."""

        resultado = []
        for objetivo in self._objetivos:
            try:
                apoyo = apoyo_para_objetivo(
                    self._proyecto, objetivo, id_, self.restricciones(), self.tipo(),
                    self.spin_desde.value(), self.spin_hasta.value() if self._barra_unica() else None,
                )
                resultado.append((True, apoyo.nombre, apoyo))
            except ErrorAsignacion as exc:
                resultado.append((False, str(exc), None))
        return resultado

    def _actualizar_vista_previa(self) -> None:
        if not self._objetivos:
            self.vista_previa.setPlainText("Se creará un apoyo puntual en el nodo elegido.")
            self.botones.button(QDialogButtonBox.Ok).setEnabled(any(self.restricciones()))
            return
        vista = self._construir()
        self.vista_previa.setPlainText("\n".join(f"{'✓' if ok else '✗'} {texto}" for ok, texto, _ in vista))
        self.botones.button(QDialogButtonBox.Ok).setEnabled(any(ok for ok, _, _ in vista))

    # -- resultado ------------------------------------------------------------------
    def tipo(self) -> str:
        return TIPOS[self.combo_tipo.currentText()]

    def restricciones(self) -> Restricciones:
        return tuple(c.isChecked() for c in self.checks)  # type: ignore[return-value]

    def tramo(self) -> Tuple[float, Optional[float]]:
        """(desde, hasta) en metros; ``hasta`` None si no aplica un tramo parcial."""

        return (self.spin_desde.value(), self.spin_hasta.value()) if self._barra_unica() else (0.0, None)

    def nodo_elegido(self) -> Optional[str]:
        """Nodo elegido de la lista (solo cuando no había nada seleccionado)."""

        return self.combo_nodo.currentText() if self.combo_nodo is not None else None
