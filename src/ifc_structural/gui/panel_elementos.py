"""Panel lateral con la lista de elementos del proyecto (vigas, columnas,
losas y muros), organizados en pestañas. Al seleccionar una fila se emite
``elemento_seleccionado`` para sincronizar con el visor 3D."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QTabWidget, QWidget

from ..core.modelo import Proyecto


class PanelElementos(QTabWidget):
    elemento_seleccionado = Signal(str, str)  # (tipo, id)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._tablas = {}
        self._tablas["VIGA"] = self._crear_tabla(["ID", "Nombre", "Material", "Perfil"])
        self._tablas["COLUMNA"] = self._crear_tabla(["ID", "Nombre", "Material", "Perfil"])
        self._tablas["LOSA"] = self._crear_tabla(["ID", "Nombre", "Espesor (m)", "Material"])
        self._tablas["MURO"] = self._crear_tabla(["ID", "Nombre", "Espesor (m)", "Material"])

        self.addTab(self._tablas["VIGA"], "Vigas")
        self.addTab(self._tablas["COLUMNA"], "Columnas")
        self.addTab(self._tablas["LOSA"], "Losas")
        self.addTab(self._tablas["MURO"], "Muros")

    def _crear_tabla(self, columnas: list) -> QTableWidget:
        tabla = QTableWidget(0, len(columnas))
        tabla.setHorizontalHeaderLabels(columnas)
        tabla.setSelectionBehavior(QTableWidget.SelectRows)
        tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        tabla.itemSelectionChanged.connect(lambda t=tabla: self._al_seleccionar(t))
        return tabla

    def _al_seleccionar(self, tabla: QTableWidget) -> None:
        filas = tabla.selectedItems()
        if not filas:
            return
        fila = filas[0].row()
        tipo = tabla.item(fila, 0).data(256)  # Qt.UserRole
        id_ = tabla.item(fila, 0).text()
        self.elemento_seleccionado.emit(tipo, id_)

    def actualizar(self, proyecto: Proyecto) -> None:
        self._llenar_barras(self._tablas["VIGA"], proyecto.vigas.values(), proyecto, "VIGA")
        self._llenar_barras(self._tablas["COLUMNA"], proyecto.columnas.values(), proyecto, "COLUMNA")
        self._llenar_superficies(self._tablas["LOSA"], proyecto.losas.values(), proyecto, "LOSA")
        self._llenar_superficies(self._tablas["MURO"], proyecto.muros.values(), proyecto, "MURO")

    def _llenar_barras(self, tabla: QTableWidget, barras, proyecto: Proyecto, tipo: str) -> None:
        tabla.setRowCount(0)
        for barra in barras:
            fila = tabla.rowCount()
            tabla.insertRow(fila)
            item_id = QTableWidgetItem(barra.id)
            item_id.setData(256, tipo)
            tabla.setItem(fila, 0, item_id)
            tabla.setItem(fila, 1, QTableWidgetItem(barra.nombre))
            tabla.setItem(fila, 2, QTableWidgetItem(barra.material))
            tabla.setItem(fila, 3, QTableWidgetItem(barra.perfil))

    def _llenar_superficies(self, tabla: QTableWidget, elementos, proyecto: Proyecto, tipo: str) -> None:
        tabla.setRowCount(0)
        for elemento in elementos:
            fila = tabla.rowCount()
            tabla.insertRow(fila)
            item_id = QTableWidgetItem(elemento.id)
            item_id.setData(256, tipo)
            tabla.setItem(fila, 0, item_id)
            tabla.setItem(fila, 1, QTableWidgetItem(elemento.nombre))
            tabla.setItem(fila, 2, QTableWidgetItem(f"{elemento.espesor:.3f}"))
            tabla.setItem(fila, 3, QTableWidgetItem(elemento.material))
