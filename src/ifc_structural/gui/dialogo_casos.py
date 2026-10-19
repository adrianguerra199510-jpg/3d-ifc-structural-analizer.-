"""Diálogo para gestionar casos de carga y combinaciones.

Las combinaciones se definen como una tabla editable: nombre, factor por
cada caso de carga existente, y norma de referencia (texto libre, p. ej.
"ACI 318: 1.2D+1.6L"). Incluye plantillas de combinaciones frecuentes de
ACI 318 / AASHTO LRFD como punto de partida editable.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _etiqueta_seccion(texto: str) -> QLabel:
    return QLabel(f"<b>{texto}</b>")

from ..core.casos_carga import PLANTILLAS_COMBINACIONES, CasoCarga, Combinacion, TipoCaso


class DialogoCasos(QDialog):
    def __init__(self, casos: dict, combinaciones: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Casos de carga y combinaciones")
        self.resize(640, 480)
        self._casos = dict(casos)
        self._combinaciones = dict(combinaciones)

        layout = QVBoxLayout(self)

        layout.addWidget(_etiqueta_seccion("Casos de carga"))
        self.tabla_casos = QTableWidget(0, 2)
        self.tabla_casos.setHorizontalHeaderLabels(["ID", "Tipo"])
        self.tabla_casos.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tabla_casos)

        fila_botones_casos = QHBoxLayout()
        self.combo_nuevo_tipo = QComboBox()
        self.combo_nuevo_tipo.addItems([t.value for t in TipoCaso])
        self.edit_nuevo_id = QLineEdit()
        self.edit_nuevo_id.setPlaceholderText("Ej.: D, L, W, E, SCP...")
        boton_agregar_caso = QPushButton("Agregar caso")
        boton_agregar_caso.clicked.connect(self._agregar_caso)
        boton_quitar_caso = QPushButton("Quitar caso seleccionado")
        boton_quitar_caso.clicked.connect(self._quitar_caso)
        fila_botones_casos.addWidget(self.edit_nuevo_id)
        fila_botones_casos.addWidget(self.combo_nuevo_tipo)
        fila_botones_casos.addWidget(boton_agregar_caso)
        fila_botones_casos.addWidget(boton_quitar_caso)
        layout.addLayout(fila_botones_casos)

        layout.addWidget(_etiqueta_seccion("Combinaciones (factor por caso de carga)"))
        self.tabla_combinaciones = QTableWidget(0, 1)
        layout.addWidget(self.tabla_combinaciones)

        fila_botones_comb = QHBoxLayout()
        self.combo_plantilla = QComboBox()
        self.combo_plantilla.addItem("(combinación vacía)")
        self.combo_plantilla.addItems(sorted(PLANTILLAS_COMBINACIONES))
        boton_agregar_comb = QPushButton("Agregar combinación")
        boton_agregar_comb.clicked.connect(self._agregar_combinacion)
        boton_quitar_comb = QPushButton("Quitar combinación seleccionada")
        boton_quitar_comb.clicked.connect(self._quitar_combinacion)
        fila_botones_comb.addWidget(self.combo_plantilla)
        fila_botones_comb.addWidget(boton_agregar_comb)
        fila_botones_comb.addWidget(boton_quitar_comb)
        layout.addLayout(fila_botones_comb)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

        self._redibujar_tabla_casos()
        self._redibujar_tabla_combinaciones()

    # -- Casos --------------------------------------------------------
    def _redibujar_tabla_casos(self) -> None:
        self.tabla_casos.setRowCount(0)
        for caso in self._casos.values():
            fila = self.tabla_casos.rowCount()
            self.tabla_casos.insertRow(fila)
            self.tabla_casos.setItem(fila, 0, QTableWidgetItem(caso.id))
            self.tabla_casos.setItem(fila, 1, QTableWidgetItem(caso.tipo.value))
        self._redibujar_tabla_combinaciones()

    def _agregar_caso(self) -> None:
        id_ = self.edit_nuevo_id.text().strip()
        if not id_ or id_ in self._casos:
            return
        self._casos[id_] = CasoCarga(id=id_, nombre=id_, tipo=TipoCaso(self.combo_nuevo_tipo.currentText()))
        self.edit_nuevo_id.clear()
        self._redibujar_tabla_casos()

    def _quitar_caso(self) -> None:
        filas = self.tabla_casos.selectedItems()
        if not filas:
            return
        id_ = self.tabla_casos.item(filas[0].row(), 0).text()
        self._casos.pop(id_, None)
        for combinacion in self._combinaciones.values():
            combinacion.factores.pop(id_, None)
        self._redibujar_tabla_casos()

    # -- Combinaciones --------------------------------------------------
    def _redibujar_tabla_combinaciones(self) -> None:
        ids_casos = sorted(self._casos)
        self.tabla_combinaciones.setColumnCount(2 + len(ids_casos))
        self.tabla_combinaciones.setHorizontalHeaderLabels(["ID", "Norma"] + ids_casos)
        self.tabla_combinaciones.setRowCount(0)
        for combinacion in self._combinaciones.values():
            fila = self.tabla_combinaciones.rowCount()
            self.tabla_combinaciones.insertRow(fila)
            self.tabla_combinaciones.setItem(fila, 0, QTableWidgetItem(combinacion.id))
            self.tabla_combinaciones.setItem(fila, 1, QTableWidgetItem(combinacion.norma))
            for columna, id_caso in enumerate(ids_casos, start=2):
                valor = combinacion.factores.get(id_caso, 0.0)
                self.tabla_combinaciones.setItem(fila, columna, QTableWidgetItem(str(valor)))

    def _agregar_combinacion(self) -> None:
        plantilla_nombre = self.combo_plantilla.currentText()
        factores = dict(PLANTILLAS_COMBINACIONES.get(plantilla_nombre, {}))
        contador = len(self._combinaciones) + 1
        id_ = f"COMB{contador}"
        while id_ in self._combinaciones:
            contador += 1
            id_ = f"COMB{contador}"
        self._combinaciones[id_] = Combinacion(id=id_, nombre=plantilla_nombre, factores=factores, norma=plantilla_nombre)
        self._redibujar_tabla_combinaciones()

    def _quitar_combinacion(self) -> None:
        filas = self.tabla_combinaciones.selectedItems()
        if not filas:
            return
        id_ = self.tabla_combinaciones.item(filas[0].row(), 0).text()
        self._combinaciones.pop(id_, None)
        self._redibujar_tabla_combinaciones()

    def _leer_combinaciones_de_tabla(self) -> None:
        """Vuelca lo editado en la tabla de combinaciones (factores,
        norma) de vuelta a los objetos Combinacion antes de cerrar."""

        ids_casos = sorted(self._casos)
        for fila in range(self.tabla_combinaciones.rowCount()):
            id_ = self.tabla_combinaciones.item(fila, 0).text()
            combinacion = self._combinaciones.get(id_)
            if combinacion is None:
                continue
            combinacion.norma = self.tabla_combinaciones.item(fila, 1).text()
            for columna, id_caso in enumerate(ids_casos, start=2):
                item = self.tabla_combinaciones.item(fila, columna)
                try:
                    combinacion.factores[id_caso] = float(item.text()) if item else 0.0
                except ValueError:
                    combinacion.factores[id_caso] = 0.0

    def accept(self) -> None:
        self._leer_combinaciones_de_tabla()
        super().accept()

    def resultado(self):
        return self._casos, self._combinaciones
