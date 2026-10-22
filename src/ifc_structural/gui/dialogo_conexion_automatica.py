"""Conexión automática: detecta los elementos que están unidos (se tocan o casi) y les crea un constraint."""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from ..core.asignacion import ErrorAsignacion, Objetivo
from ..core.asignacion_superficie import union_para
from ..core.cargas import UnionSuperficie
from ..core.modelo import Proyecto
from ..solido.deteccion import Contacto, detectar_contactos
from .dialogo_superficie import TIPOS_UNION_TEXTO
from .estilo import EncabezadoDialogo

TIPOS_AUTOMATICOS = {t: v for t, v in TIPOS_UNION_TEXTO.items() if v in ("SOLDADURA", "CUERPO_RIGIDO")}


class DialogoConexionAutomatica(QDialog):
    def __init__(self, proyecto: Proyecto, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conectar automáticamente")
        self.resize(680, 600)
        self._proyecto = proyecto
        self._contactos: List[Contacto] = []

        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo(
            "conectar", "Conectar automáticamente",
            "Detecta los elementos que están unidos (se tocan o casi) y les crea un constraint."))
        formulario = QFormLayout()
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(list(TIPOS_AUTOMATICOS))
        formulario.addRow("Tipo de unión:", self.combo_tipo)
        self.aviso_rigido = QLabel("<span style='color:#9a6700'>Con «cuerpo rígido», todas las piezas que queden conectadas "
                                   "entre sí forman un único cuerpo rígido: no se deforman ni dan esfuerzos.</span>")
        self.aviso_rigido.setWordWrap(True)
        formulario.addRow(self.aviso_rigido)
        self.combo_tipo.currentIndexChanged.connect(
            lambda _i: self.aviso_rigido.setVisible(TIPOS_AUTOMATICOS[self.combo_tipo.currentText()] == "CUERPO_RIGIDO"))
        self.spin_tolerancia = QDoubleSpinBox()
        self.spin_tolerancia.setRange(0.0, 500.0)
        self.spin_tolerancia.setDecimals(1)
        self.spin_tolerancia.setSuffix(" mm")
        self.spin_tolerancia.setValue(10.0)
        self.spin_tolerancia.setToolTip("Dos elementos se consideran unidos si sus caras que se miran están a menos de esta separación.")
        fila = QHBoxLayout()
        fila.addWidget(self.spin_tolerancia)
        self.boton_buscar = QPushButton("Buscar")
        self.boton_buscar.clicked.connect(self.buscar)
        fila.addWidget(self.boton_buscar)
        formulario.addRow("Separación máxima:", fila)
        layout.addLayout(formulario)

        self.resumen = QLabel()
        self.resumen.setWordWrap(True)
        layout.addWidget(self.resumen)
        self.lista = QListWidget()
        self.lista.itemChanged.connect(self._actualizar_boton)
        layout.addWidget(self.lista)
        marcar = QHBoxLayout()
        for texto, valor in (("Marcar todos", Qt.Checked), ("Desmarcar todos", Qt.Unchecked)):
            boton = QPushButton(texto)
            boton.clicked.connect(lambda _=False, v=valor: self._marcar(v))
            marcar.addWidget(boton)
        marcar.addStretch(1)
        layout.addLayout(marcar)
        nota = QLabel(
            "Se mide sobre la geometría real de cada elemento (sus superficies): se unen las partes que se tocan, se solapan "
            "o están a menos de la separación máxima. Las parejas que ya tienen un constraint no se repiten. En el análisis "
            "3D, dos elementos que se tocan ya se unen solos; el constraint hace la unión visible y editable, y la extiende a "
            "elementos con una pequeña holgura."
        )
        nota.setWordWrap(True)
        layout.addWidget(nota)
        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)
        self.aviso_rigido.setVisible(TIPOS_AUTOMATICOS[self.combo_tipo.currentText()] == "CUERPO_RIGIDO")
        self.buscar()

    def buscar(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            contactos = detectar_contactos(self._proyecto, self.spin_tolerancia.value() / 1000.0)
        except Exception as exc:  # el mallado puede fallar con geometrías raras: se informa, no se cae
            contactos = []
            import logging
            from .trabajador_analisis import ubicacion_del_error

            logging.getLogger(__name__).exception("No se pudo detectar contactos")
            self.resumen.setText(f"No se pudo analizar el modelo: {exc}\n(en {ubicacion_del_error(exc)})")
        finally:
            QApplication.restoreOverrideCursor()
        existentes = {frozenset(((u.a_tipo, u.a_id), (u.b_tipo, u.b_id))) for u in self._proyecto.uniones}
        self._contactos = [c for c in contactos if frozenset(((c.a_tipo, c.a_id), (c.b_tipo, c.b_id))) not in existentes
                           and c.area > 5e-5]  # menos de 0,5 cm²: contactos de arista, no uniones
        self.lista.blockSignals(True)
        self.lista.clear()
        for c in self._contactos:
            item = QListWidgetItem(c.descripcion())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.lista.addItem(item)
        self.lista.blockSignals(False)
        if self._contactos:
            self.resumen.setText(f"Se encontraron <b>{len(self._contactos)}</b> parejas de elementos unidos. "
                                 "Desmarque las que no quiera conectar.")
        elif not self.resumen.text().startswith("No se pudo"):
            self.resumen.setText("No se encontraron elementos nuevos que se toquen. Pruebe con una separación máxima mayor.")
        self._actualizar_boton()

    def _marcar(self, estado) -> None:
        self.lista.blockSignals(True)
        for i in range(self.lista.count()):
            self.lista.item(i).setCheckState(estado)
        self.lista.blockSignals(False)
        self._actualizar_boton()

    def _actualizar_boton(self, *_args) -> None:
        n = len(self.elegidos())
        boton = self.botones.button(QDialogButtonBox.Ok)
        boton.setEnabled(n > 0)
        boton.setText(f"Crear {n} constraint(s)" if n else "Crear")

    def elegidos(self) -> List[Contacto]:
        return [c for i, c in enumerate(self._contactos) if self.lista.item(i).checkState() == Qt.Checked]

    def uniones(self, siguiente_id) -> List[UnionSuperficie]:
        """Constraints de los contactos marcados (``siguiente_id()`` da cada identificador nuevo)."""

        tipo = TIPOS_AUTOMATICOS[self.combo_tipo.currentText()]
        tolerancia = self.spin_tolerancia.value() / 1000.0
        creadas = []
        for c in self.elegidos():
            try:
                u = union_para(self._proyecto, Objetivo(c.a_tipo, c.a_id), Objetivo(c.b_tipo, c.b_id), siguiente_id(), tipo,
                               (True,) * 6, completo=True, tolerancia=max(tolerancia, c.separacion * 1.2))
            except ErrorAsignacion:
                continue
            creadas.append(u)
        return creadas
