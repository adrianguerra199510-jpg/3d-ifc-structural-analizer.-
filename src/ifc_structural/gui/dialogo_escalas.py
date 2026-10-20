"""Tamaño de los símbolos del visor (apoyos, flechas de cargas, constraints y etiquetas): cada parámetro
se define como porcentaje del tamaño automático o como tamaño en píxeles de pantalla."""

from __future__ import annotations

from typing import Dict, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .estilo import EncabezadoDialogo
from .visor_3d import ESCALA_PIXELES, ESCALA_PORCENTAJE, PARAMETROS_ESCALA, TAMANO_ETIQUETA_AUTOMATICO

PARAMETROS = {
    "apoyos": ("Apoyos", "Tamaño de los símbolos de apoyo (longitud de referencia del símbolo)."),
    "cargas": ("Flechas de las cargas", "Longitud de referencia de las flechas de las cargas."),
    "constraints": ("Constraints", "Tamaño de las marcas y los enlaces de los constraints."),
    "etiquetas": ("Texto de las etiquetas", "Tamaño del texto («200 kPa», «empotrada»...). En píxeles es el alto del texto."),
}
MODOS = {"Porcentaje del automático": ESCALA_PORCENTAJE, "Píxeles en pantalla": ESCALA_PIXELES}
PIXELES_POR_DEFECTO = {"apoyos": 40.0, "cargas": 50.0, "constraints": 14.0, "etiquetas": float(TAMANO_ETIQUETA_AUTOMATICO) * 4 / 3}
AUTOMATICO = {p: (ESCALA_PORCENTAJE, 100.0) for p in PARAMETROS_ESCALA}


class DialogoEscalas(QDialog):
    """Cambia el tamaño de los símbolos del visor al instante; *Restablecer* vuelve al automático."""

    def __init__(self, visor, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Escala de símbolos")
        self._visor = visor
        self._inicial = visor.escalas
        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo(
            "modelo", "Escala de símbolos",
            "Tamaño de apoyos, cargas, constraints y etiquetas: en porcentaje del tamaño automático o en píxeles."))
        cuadricula = QGridLayout()
        cuadricula.addWidget(QLabel("<b>Parámetro</b>"), 0, 0)
        cuadricula.addWidget(QLabel("<b>Definir como</b>"), 0, 1)
        cuadricula.addWidget(QLabel("<b>Valor</b>"), 0, 2)
        self._combos: Dict[str, QComboBox] = {}
        self._spins: Dict[str, QDoubleSpinBox] = {}
        for fila, parametro in enumerate(PARAMETROS_ESCALA, start=1):
            texto, ayuda = PARAMETROS[parametro]
            etiqueta = QLabel(texto)
            etiqueta.setToolTip(ayuda)
            combo = QComboBox()
            combo.addItems(list(MODOS))
            spin = QDoubleSpinBox()
            spin.setDecimals(0)
            spin.setToolTip(ayuda)
            cuadricula.addWidget(etiqueta, fila, 0)
            cuadricula.addWidget(combo, fila, 1)
            cuadricula.addWidget(spin, fila, 2)
            self._combos[parametro], self._spins[parametro] = combo, spin
            modo, valor = self._inicial[parametro]
            combo.setCurrentText(next(t for t, m in MODOS.items() if m == modo))
            self._ajustar_spin(parametro, valor)
            combo.currentIndexChanged.connect(lambda _=0, p=parametro: self._al_cambiar_modo(p))
            spin.valueChanged.connect(self._aplicar)
        layout.addLayout(cuadricula)
        nota = QLabel("En píxeles el tamaño se mantiene al acercar o alejar la vista; en porcentaje, 100 % es el tamaño "
                      "automático (según el modelo) y crece o se achica con el zoom como el modelo.")
        nota.setWordWrap(True)
        layout.addWidget(nota)
        restablecer = QPushButton("Restablecer (automático)")
        restablecer.clicked.connect(self._restablecer)
        layout.addWidget(restablecer)
        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self._cancelar)
        layout.addWidget(botones)

    def _modo(self, parametro: str) -> str:
        return MODOS[self._combos[parametro].currentText()]

    def _ajustar_spin(self, parametro: str, valor: float) -> None:
        spin = self._spins[parametro]
        spin.blockSignals(True)
        if self._modo(parametro) == ESCALA_PIXELES:
            spin.setRange(2.0, 600.0)
            spin.setSuffix(" px")
        else:
            spin.setRange(5.0, 1000.0)
            spin.setSuffix(" %")
        spin.setValue(valor)
        spin.blockSignals(False)

    def _al_cambiar_modo(self, parametro: str) -> None:
        # al pasar de un modo al otro se parte de un valor equivalente razonable
        self._ajustar_spin(parametro, PIXELES_POR_DEFECTO[parametro] if self._modo(parametro) == ESCALA_PIXELES else 100.0)
        self._aplicar()

    def escalas(self) -> Dict[str, Tuple[str, float]]:
        return {p: (self._modo(p), self._spins[p].value()) for p in PARAMETROS_ESCALA}

    def _aplicar(self, *_args) -> None:
        self._visor.set_escalas(self.escalas())

    def _restablecer(self) -> None:
        for parametro in PARAMETROS_ESCALA:
            self._combos[parametro].blockSignals(True)
            self._combos[parametro].setCurrentText(next(t for t, m in MODOS.items() if m == ESCALA_PORCENTAJE))
            self._combos[parametro].blockSignals(False)
            self._ajustar_spin(parametro, 100.0)
        self._aplicar()

    def _cancelar(self) -> None:
        self._visor.set_escalas(self._inicial)
        self.reject()
