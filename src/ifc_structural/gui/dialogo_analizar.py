"""Diálogo del botón ANALIZAR: qué casos de carga se resuelven (y con qué malla, en el análisis sólido)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QVBoxLayout)

from ..core.casos_carga import CasoCarga, Combinacion, combinaciones_calculables
from .estilo import EncabezadoDialogo


class DialogoAnalizar(QDialog):
    """Lista de los casos de carga con una casilla cada uno; debajo, las combinaciones que se podrán calcular con los
    casos marcados (una combinación necesita todos sus casos) y, en el análisis sólido, la malla que se usará."""

    def __init__(self, casos: Dict[str, CasoCarga], combinaciones: Dict[str, Combinacion], parent=None,
                 motivo: str = "", texto_malla: str = "", elegidos: Optional[Iterable[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Analizar")
        self.resize(560, 520)
        self._casos, self._combinaciones = casos, combinaciones
        elegidos = set(casos) if elegidos is None else set(elegidos)

        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo("analizar", "Analizar", "Elija los casos de carga que se resuelven con CalculiX."))
        if motivo:
            aviso = QLabel(motivo)
            aviso.setWordWrap(True)
            layout.addWidget(aviso)

        layout.addWidget(QLabel("<b>Casos de carga</b>"))
        self.lista = QListWidget()
        for caso in casos.values():
            texto = f"{caso.id} — {caso.nombre}  [{caso.tipo.value}]" + ("  + peso propio" if caso.incluye_peso_propio else "")
            item = QListWidgetItem(texto)
            item.setData(Qt.UserRole, caso.id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if caso.id in elegidos else Qt.Unchecked)
            self.lista.addItem(item)
        layout.addWidget(self.lista, 1)
        botones_lista = QHBoxLayout()
        for texto, estado in (("Todos", Qt.Checked), ("Ninguno", Qt.Unchecked)):
            boton = QPushButton(texto)
            boton.clicked.connect(lambda _=False, e=estado: self._marcar_todos(e))
            botones_lista.addWidget(boton)
        botones_lista.addStretch(1)
        layout.addLayout(botones_lista)

        self.combinaciones = QLabel()
        self.combinaciones.setWordWrap(True)
        layout.addWidget(self.combinaciones)
        if texto_malla:
            malla = QLabel(texto_malla)
            malla.setWordWrap(True)
            layout.addWidget(malla)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.button(QDialogButtonBox.Ok).setText("Analizar")
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)

        self.lista.itemChanged.connect(lambda _item: self._actualizar())
        self._actualizar()

    def _marcar_todos(self, estado) -> None:
        for i in range(self.lista.count()):
            self.lista.item(i).setCheckState(estado)

    def casos_elegidos(self) -> List[str]:
        return [self.lista.item(i).data(Qt.UserRole) for i in range(self.lista.count())
                if self.lista.item(i).checkState() == Qt.Checked]

    def combinaciones_elegidas(self) -> Dict[str, Combinacion]:
        return combinaciones_calculables(self._combinaciones, self.casos_elegidos())

    def _actualizar(self) -> None:
        elegidos = self.casos_elegidos()
        calculables = self.combinaciones_elegidas()
        faltan = [c for i, c in self._combinaciones.items() if i not in calculables]
        lineas = []
        if calculables:
            lineas.append("Combinaciones que se calcularán: " + ", ".join(f"{c.id} ({c.nombre})" for c in calculables.values()) + ".")
        else:
            lineas.append("No se calculará ninguna combinación." if self._combinaciones else "No hay combinaciones definidas.")
        if faltan:
            lineas.append("<span style='color:#9a6700'>Sin calcular (les falta algún caso): "
                          + ", ".join(c.id for c in faltan) + ".</span>")
        con_peso = [i for i in elegidos if self._casos[i].incluye_peso_propio]
        if len(con_peso) > 1:
            lineas.append(f"<span style='color:#b3261e'>Los casos {', '.join(con_peso)} incluyen el peso propio: en una "
                          "combinación que los sume se cuenta dos veces.</span>")
        self.combinaciones.setText("<br>".join(lineas))
        self.botones.button(QDialogButtonBox.Ok).setEnabled(bool(elegidos))
