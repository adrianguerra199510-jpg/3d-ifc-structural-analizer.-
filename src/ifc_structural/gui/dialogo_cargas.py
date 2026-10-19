"""Diálogo para asignar cargas de área (sobre losas), lineales (sobre
vigas/columnas) y puntuales (sobre nodos o sobre un punto de una
viga/columna)."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.cargas import CargaArea, CargaLineal, CargaPuntual
from ..core.modelo import Proyecto


def _spin(minimo=-1e9, maximo=1e9, decimales=3, valor=0.0) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimo, maximo)
    spin.setDecimals(decimales)
    spin.setValue(valor)
    return spin


class DialogoCargas(QDialog):
    """Reúne las tres formas de carga de la Fase 1 en un solo diálogo con
    pestañas; ``resultado()`` devuelve la carga creada en la pestaña activa
    al momento de aceptar."""

    def __init__(self, proyecto: Proyecto, casos_disponibles: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar carga")
        self._proyecto = proyecto
        self._casos = casos_disponibles

        self.pestanas = QTabWidget()
        self.pestanas.addTab(self._crear_pestana_area(), "Carga de área (losa)")
        self.pestanas.addTab(self._crear_pestana_lineal(), "Carga lineal (viga/columna)")
        self.pestanas.addTab(self._crear_pestana_puntual(), "Carga puntual")

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.pestanas)
        layout.addWidget(botones)

    def _combo_casos(self) -> QComboBox:
        combo = QComboBox()
        combo.addItems(self._casos)
        return combo

    def _crear_pestana_area(self) -> QWidget:
        widget = QWidget()
        self.combo_caso_area = self._combo_casos()
        self.combo_losa = QComboBox()
        self.combo_losa.addItems(sorted(self._proyecto.losas))
        self.spin_presion = _spin(valor=5.0)
        self.spin_presion.setSuffix(" kPa")

        formulario = QFormLayout(widget)
        formulario.addRow("Caso de carga:", self.combo_caso_area)
        formulario.addRow("Losa:", self.combo_losa)
        formulario.addRow("Presión:", self.spin_presion)
        return widget

    def _crear_pestana_lineal(self) -> QWidget:
        widget = QWidget()
        self.combo_caso_lineal = self._combo_casos()
        self.combo_barra_lineal = QComboBox()
        self.combo_barra_lineal.addItems(sorted({**self._proyecto.vigas, **self._proyecto.columnas}))
        self.spin_w = _spin(valor=-10.0)
        self.spin_w.setSuffix(" kN/m")
        self.combo_direccion_lineal = QComboBox()
        self.combo_direccion_lineal.addItems(["Global Z (gravedad)", "Global X", "Global Y"])

        formulario = QFormLayout(widget)
        formulario.addRow("Caso de carga:", self.combo_caso_lineal)
        formulario.addRow("Viga/columna:", self.combo_barra_lineal)
        formulario.addRow("Intensidad uniforme:", self.spin_w)
        formulario.addRow("Dirección:", self.combo_direccion_lineal)
        return widget

    def _crear_pestana_puntual(self) -> QWidget:
        widget = QWidget()
        self.combo_caso_puntual = self._combo_casos()
        self.combo_nodo_puntual = QComboBox()
        self.combo_nodo_puntual.addItems(sorted(self._proyecto.nodos))
        self.spin_fx = _spin(valor=0.0)
        self.spin_fy = _spin(valor=0.0)
        self.spin_fz = _spin(valor=-10.0)
        for s in (self.spin_fx, self.spin_fy, self.spin_fz):
            s.setSuffix(" kN")

        formulario = QFormLayout(widget)
        formulario.addRow("Caso de carga:", self.combo_caso_puntual)
        formulario.addRow("Nodo:", self.combo_nodo_puntual)
        formulario.addRow("Fx:", self.spin_fx)
        formulario.addRow("Fy:", self.spin_fy)
        formulario.addRow("Fz:", self.spin_fz)
        return widget

    def _direccion_vector(self) -> tuple:
        indice = self.combo_direccion_lineal.currentIndex()
        return {0: (0.0, 0.0, -1.0), 1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0)}[indice]

    def resultado(self, id_: str):
        indice = self.pestanas.currentIndex()
        if indice == 0:
            return CargaArea(
                id=id_,
                caso_id=self.combo_caso_area.currentText(),
                presion=self.spin_presion.value(),
                losa_id=self.combo_losa.currentText(),
            )
        if indice == 1:
            w = self.spin_w.value()
            return CargaLineal(
                id=id_,
                caso_id=self.combo_caso_lineal.currentText(),
                elemento_id=self.combo_barra_lineal.currentText(),
                w_inicio=w,
                w_fin=w,
                direccion=self._direccion_vector(),
            )
        return CargaPuntual(
            id=id_,
            caso_id=self.combo_caso_puntual.currentText(),
            nodo_id=self.combo_nodo_puntual.currentText(),
            fx=self.spin_fx.value(),
            fy=self.spin_fy.value(),
            fz=self.spin_fz.value(),
        )
