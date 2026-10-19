"""Panel de resultados: tabla de fuerzas internas por elemento y diagrama
N/V/M de la viga/columna seleccionada, para el caso o combinación elegido.
"""

from __future__ import annotations

from typing import Dict, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..combinaciones.motor_superposicion import COMPONENTES_FUERZA
from ..solver.generador_inp import MallaAnalisis
from ..solver.lector_resultados import ResultadosCaso


def _widget_con_etiqueta(texto: str, widget: QWidget) -> QWidget:
    contenedor = QWidget()
    layout = QHBoxLayout(contenedor)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(texto))
    layout.addWidget(widget)
    return contenedor


class PanelResultados(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._malla: Optional[MallaAnalisis] = None
        self._resultados: Dict[str, ResultadosCaso] = {}

        self.combo_caso = QComboBox()
        self.combo_caso.currentTextChanged.connect(self._actualizar_tabla)
        self.combo_barra = QComboBox()
        self.combo_barra.currentTextChanged.connect(self._actualizar_diagrama)

        fila_superior = QHBoxLayout()
        fila_superior.addWidget(QWidgetConEtiqueta("Caso / combinación:", self.combo_caso))
        fila_superior.addWidget(QWidgetConEtiqueta("Viga/columna:", self.combo_barra))

        self.tabla = QTableWidget(0, 7)
        self.tabla.setHorizontalHeaderLabels(["Elemento"] + list(COMPONENTES_FUERZA))
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.figura = Figure(figsize=(5, 6))
        self.canvas = FigureCanvasQTAgg(self.figura)
        self._ejes = self.figura.subplots(3, 1, sharex=True)
        self._ejes[0].set_ylabel("N [kN]")
        self._ejes[1].set_ylabel("V [kN]")
        self._ejes[2].set_ylabel("M [kN·m]")
        self._ejes[2].set_xlabel("Posición sobre el elemento [m]")

        divisor = QSplitter()
        divisor.addWidget(self.tabla)
        divisor.addWidget(self.canvas)

        layout = QVBoxLayout(self)
        layout.addLayout(fila_superior)
        layout.addWidget(divisor)

    def establecer_malla(self, malla: MallaAnalisis) -> None:
        self._malla = malla
        self.combo_barra.clear()
        self.combo_barra.addItems(sorted(malla.elementos_por_barra))

    def actualizar_resultados(self, resultados_por_caso: Dict[str, ResultadosCaso]) -> None:
        self._resultados = resultados_por_caso
        self.combo_caso.clear()
        self.combo_caso.addItems(sorted(resultados_por_caso))

    def _actualizar_tabla(self) -> None:
        if self._malla is None:
            return
        caso = self.combo_caso.currentText()
        resultados = self._resultados.get(caso)
        self.tabla.setRowCount(0)
        if resultados is None:
            return
        for barra_id, nodos in sorted(self._malla.orden_nodos_por_barra.items()):
            for i, nodo in enumerate(nodos):
                valores = resultados.fuerzas_seccion.get(nodo)
                if valores is None:
                    continue
                fila = self.tabla.rowCount()
                self.tabla.insertRow(fila)
                etiqueta = f"{barra_id} (nodo {i + 1}/{len(nodos)})"
                self.tabla.setItem(fila, 0, QTableWidgetItem(etiqueta))
                for columna, valor in enumerate(valores, start=1):
                    self.tabla.setItem(fila, columna, QTableWidgetItem(f"{valor:.3g}"))
        self._actualizar_diagrama()

    def _actualizar_diagrama(self) -> None:
        if self._malla is None:
            return
        caso = self.combo_caso.currentText()
        barra_id = self.combo_barra.currentText()
        resultados = self._resultados.get(caso)
        for eje in self._ejes:
            eje.clear()
        if resultados is None or not barra_id:
            self.canvas.draw_idle()
            return

        nodos = self._malla.orden_nodos_por_barra.get(barra_id, [])
        xs = list(self._malla.posiciones_s_por_barra.get(barra_id, []))

        valores_n, valores_v, valores_m = [], [], []
        for nodo in nodos:
            v = resultados.fuerzas_seccion.get(nodo, (0.0,) * 6)
            valores_n.append(v[0])
            valores_v.append((v[1] ** 2 + v[2] ** 2) ** 0.5)
            valores_m.append((v[4] ** 2 + v[5] ** 2) ** 0.5)

        self._ejes[0].plot(xs, valores_n, "-o", color="steelblue", markersize=3)
        self._ejes[0].set_ylabel("N [kN]")
        self._ejes[1].plot(xs, valores_v, "-o", color="darkorange", markersize=3)
        self._ejes[1].set_ylabel("V [kN]")
        self._ejes[2].plot(xs, valores_m, "-o", color="seagreen", markersize=3)
        self._ejes[2].set_ylabel("M [kN·m]")
        self._ejes[2].set_xlabel("Posición relativa sobre el elemento (0 a 1)")
        self.figura.tight_layout()
        self.canvas.draw_idle()


def QWidgetConEtiqueta(texto: str, widget: QWidget) -> QWidget:
    from PySide6.QtWidgets import QLabel

    contenedor = QWidget()
    layout = QHBoxLayout(contenedor)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(texto))
    layout.addWidget(widget)
    return contenedor
