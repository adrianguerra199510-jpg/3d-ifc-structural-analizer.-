"""Panel con las asignaciones hechas: apoyos y cargas, con opción de
eliminarlas."""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class PanelAsignaciones(QWidget):
    """Tabla (tipo, descripción, caso). Cada fila guarda una clave que la
    ventana principal entiende para saber qué eliminar."""

    eliminar_solicitado = Signal(list)  # list[(tipo, id)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabla = QTableWidget(0, 3)
        self.tabla.setHorizontalHeaderLabels(["Tipo", "Descripción", "Caso"])
        self.tabla.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla.setSelectionBehavior(QTableWidget.SelectRows)
        self.tabla.setSelectionMode(QTableWidget.ExtendedSelection)

        self.boton_eliminar = QPushButton("Eliminar las asignaciones seleccionadas")
        self.boton_eliminar.clicked.connect(self._al_eliminar)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabla)
        layout.addWidget(self.boton_eliminar)

    def mostrar(self, filas: List[Tuple[str, str, str, str]]) -> None:
        """filas: [(tipo, id, descripción, caso)]. Se muestran tipo,
        descripción y caso; (tipo, id) queda guardado en la fila."""

        self.tabla.setRowCount(0)
        for tipo, id_, descripcion, caso in filas:
            fila = self.tabla.rowCount()
            self.tabla.insertRow(fila)
            item_tipo = QTableWidgetItem(tipo)
            item_tipo.setData(Qt.UserRole, (tipo, id_))
            self.tabla.setItem(fila, 0, item_tipo)
            self.tabla.setItem(fila, 1, QTableWidgetItem(descripcion))
            self.tabla.setItem(fila, 2, QTableWidgetItem(caso))

    def _al_eliminar(self) -> None:
        filas = sorted({i.row() for i in self.tabla.selectedItems()})
        claves = [self.tabla.item(f, 0).data(Qt.UserRole) for f in filas]
        if claves:
            self.eliminar_solicitado.emit(claves)
