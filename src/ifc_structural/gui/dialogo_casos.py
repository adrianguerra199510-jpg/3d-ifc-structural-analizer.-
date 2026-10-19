"""Formulario para crear y editar casos de carga y combinaciones.

Dos pestañas con la misma disposición: a la izquierda la lista de lo que ya
existe y a la derecha el formulario del elemento elegido (o de uno nuevo).

* Caso de carga: identificador, nombre y tipo.
* Combinación: identificador, nombre, norma/descripción y el **factor de
  cada caso de carga** (un campo numérico por caso; 0 = no participa). Se
  puede partir de una plantilla de ACI 318 / AASHTO LRFD y editarla.

Los cambios solo se aplican al pulsar «Aceptar» en el diálogo; «Cancelar» no
modifica nada. Dentro de cada pestaña, «Guardar» confirma el formulario en la
lista.
"""

from __future__ import annotations

import copy
import re
from typing import Dict, Iterable, Optional, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.casos_carga import PLANTILLAS_COMBINACIONES, CasoCarga, Combinacion, TipoCaso
from .estilo import EncabezadoDialogo

_PATRON_ID = re.compile(r"[A-Za-z0-9_]+")
_SIN_PLANTILLA = "(en blanco)"


def formula(combinacion: Combinacion, casos: Dict[str, CasoCarga]) -> str:
    """Texto legible de una combinación: «1.2·D + 1.6·L»."""

    partes = ""
    for id_caso in casos:
        factor = combinacion.factores.get(id_caso, 0.0)
        if abs(factor) < 1e-12:
            continue
        termino = f"{abs(factor):g}·{id_caso}"
        if not partes:
            partes = termino if factor > 0 else f"−{termino}"
        else:
            partes += f" {'+' if factor > 0 else '−'} {termino}"
    return partes or "(sin factores)"


class DialogoCasos(QDialog):
    def __init__(
        self,
        casos: Dict[str, CasoCarga],
        combinaciones: Dict[str, Combinacion],
        casos_con_cargas: Iterable[str] = (),
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Casos de carga y combinaciones")
        self.resize(820, 620)
        self._casos = copy.deepcopy(casos)
        self._combinaciones = copy.deepcopy(combinaciones)
        self._casos_con_cargas: Set[str] = set(casos_con_cargas)
        self._casos_eliminados: Set[str] = set()
        self._id_caso_actual: Optional[str] = None  # None = formulario de un caso nuevo
        self._id_comb_actual: Optional[str] = None
        self._cargando = False
        self._sucio_caso = False
        self._sucio_comb = False

        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo("casos", "Casos de carga y combinaciones", "Defina los casos (D, L, W…) y combine sus factores."))
        self.pestanas = QTabWidget()
        self.pestanas.addTab(self._crear_pestana_casos(), "Casos de carga")
        self.pestanas.addTab(self._crear_pestana_combinaciones(), "Combinaciones")
        self.pestanas.currentChanged.connect(self._al_cambiar_pestana)
        layout.addWidget(self.pestanas)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

        self._refrescar_lista_casos()
        self._refrescar_lista_combinaciones()
        self._nuevo_caso()
        self._nueva_combinacion()
        if self._casos:
            self.lista_casos.setCurrentRow(0)
        if self._combinaciones:
            self.lista_combinaciones.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Construcción de las pestañas
    # ------------------------------------------------------------------
    def _crear_pestana_casos(self) -> QWidget:
        pagina = QWidget()
        fila = QHBoxLayout(pagina)

        izquierda = QVBoxLayout()
        izquierda.addWidget(QLabel("Casos existentes"))
        self.lista_casos = QListWidget()
        self.lista_casos.currentRowChanged.connect(self._al_elegir_caso)
        izquierda.addWidget(self.lista_casos)
        self.boton_nuevo_caso = QPushButton("＋ Nuevo caso")
        self.boton_nuevo_caso.clicked.connect(self._pedir_nuevo_caso)
        izquierda.addWidget(self.boton_nuevo_caso)
        fila.addLayout(izquierda, 1)

        self.grupo_caso = QGroupBox()
        formulario = QFormLayout(self.grupo_caso)
        self.edit_caso_id = QLineEdit()
        self.edit_caso_id.setPlaceholderText("Ej.: D, L, W, E, SCP")
        self.edit_caso_nombre = QLineEdit()
        self.edit_caso_nombre.setPlaceholderText("Ej.: Carga muerta")
        self.combo_caso_tipo = QComboBox()
        for tipo in TipoCaso:
            self.combo_caso_tipo.addItem(tipo.value)
        formulario.addRow("Identificador:", self.edit_caso_id)
        formulario.addRow("Nombre:", self.edit_caso_nombre)
        formulario.addRow("Tipo:", self.combo_caso_tipo)
        self.etiqueta_ayuda_caso = QLabel(
            "El identificador se usa en las combinaciones y en los archivos de cálculo "
            "(solo letras, números y guion bajo). No se puede cambiar después de guardar."
        )
        self.etiqueta_ayuda_caso.setWordWrap(True)
        formulario.addRow(self.etiqueta_ayuda_caso)
        botones = QHBoxLayout()
        self.boton_guardar_caso = QPushButton("Guardar caso")
        self.boton_guardar_caso.setDefault(True)
        self.boton_guardar_caso.clicked.connect(self._guardar_caso)
        self.boton_eliminar_caso = QPushButton("Eliminar caso")
        self.boton_eliminar_caso.clicked.connect(self._eliminar_caso)
        botones.addWidget(self.boton_guardar_caso)
        botones.addWidget(self.boton_eliminar_caso)
        formulario.addRow(botones)
        fila.addWidget(self.grupo_caso, 2)

        for editor in (self.edit_caso_id, self.edit_caso_nombre):
            editor.textChanged.connect(self._marcar_sucio_caso)
        self.combo_caso_tipo.currentIndexChanged.connect(self._marcar_sucio_caso)
        return pagina

    def _crear_pestana_combinaciones(self) -> QWidget:
        pagina = QWidget()
        fila = QHBoxLayout(pagina)

        izquierda = QVBoxLayout()
        izquierda.addWidget(QLabel("Combinaciones existentes"))
        self.lista_combinaciones = QListWidget()
        self.lista_combinaciones.currentRowChanged.connect(self._al_elegir_combinacion)
        izquierda.addWidget(self.lista_combinaciones)
        self.boton_nueva_comb = QPushButton("＋ Nueva combinación")
        self.boton_nueva_comb.clicked.connect(self._pedir_nueva_combinacion)
        izquierda.addWidget(self.boton_nueva_comb)
        fila.addLayout(izquierda, 1)

        self.grupo_comb = QGroupBox()
        derecha = QVBoxLayout(self.grupo_comb)
        formulario = QFormLayout()
        self.edit_comb_id = QLineEdit()
        self.edit_comb_id.setPlaceholderText("Ej.: COMB3")
        self.edit_comb_nombre = QLineEdit()
        self.edit_comb_nombre.setPlaceholderText("Ej.: Resistencia con viento")
        self.edit_comb_norma = QLineEdit()
        self.edit_comb_norma.setPlaceholderText("Ej.: ACI 318-19 §5.3.1")
        formulario.addRow("Identificador:", self.edit_comb_id)
        formulario.addRow("Nombre:", self.edit_comb_nombre)
        formulario.addRow("Norma / nota:", self.edit_comb_norma)
        derecha.addLayout(formulario)

        derecha.addWidget(QLabel("<b>Factor de cada caso de carga</b> (0 = no participa)"))
        self.tabla_factores = QTableWidget(0, 2)
        self.tabla_factores.setHorizontalHeaderLabels(["Caso de carga", "Factor"])
        self.tabla_factores.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tabla_factores.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tabla_factores.verticalHeader().setVisible(False)
        self.tabla_factores.setSelectionMode(QTableWidget.NoSelection)
        derecha.addWidget(self.tabla_factores)
        self._spins: Dict[str, QDoubleSpinBox] = {}

        self.etiqueta_formula = QLabel()
        self.etiqueta_formula.setTextFormat(Qt.PlainText)
        derecha.addWidget(self.etiqueta_formula)

        fila_plantilla = QHBoxLayout()
        fila_plantilla.addWidget(QLabel("Cargar plantilla:"))
        self.combo_plantilla = QComboBox()
        self.combo_plantilla.addItem(_SIN_PLANTILLA)
        self.combo_plantilla.addItems(sorted(PLANTILLAS_COMBINACIONES))
        fila_plantilla.addWidget(self.combo_plantilla, 1)
        self.boton_plantilla = QPushButton("Aplicar factores")
        self.boton_plantilla.clicked.connect(self._aplicar_plantilla)
        fila_plantilla.addWidget(self.boton_plantilla)
        derecha.addLayout(fila_plantilla)

        botones = QHBoxLayout()
        self.boton_guardar_comb = QPushButton("Guardar combinación")
        self.boton_guardar_comb.clicked.connect(self._guardar_combinacion)
        self.boton_eliminar_comb = QPushButton("Eliminar combinación")
        self.boton_eliminar_comb.clicked.connect(self._eliminar_combinacion)
        botones.addWidget(self.boton_guardar_comb)
        botones.addWidget(self.boton_eliminar_comb)
        derecha.addLayout(botones)
        fila.addWidget(self.grupo_comb, 2)

        for editor in (self.edit_comb_id, self.edit_comb_nombre, self.edit_comb_norma):
            editor.textChanged.connect(self._marcar_sucio_comb)
        return pagina

    # ------------------------------------------------------------------
    # Casos
    # ------------------------------------------------------------------
    def _refrescar_lista_casos(self, elegir: Optional[str] = None) -> None:
        self._cargando = True
        self.lista_casos.clear()
        for caso in self._casos.values():
            self.lista_casos.addItem(f"{caso.id} — {caso.nombre}  [{caso.tipo.value}]")
        if elegir in self._casos:
            self.lista_casos.setCurrentRow(list(self._casos).index(elegir))
        self._cargando = False

    def _marcar_sucio_caso(self, *_args) -> None:
        if not self._cargando:
            self._sucio_caso = True
            self._actualizar_titulo_caso()

    def _actualizar_titulo_caso(self) -> None:
        base = "Nuevo caso de carga" if self._id_caso_actual is None else f"Caso {self._id_caso_actual}"
        self.grupo_caso.setTitle(base + (" (sin guardar)" if self._sucio_caso else ""))

    def _nuevo_caso(self) -> None:
        self._cargando = True
        self._id_caso_actual = None
        self.edit_caso_id.clear()
        self.edit_caso_id.setEnabled(True)
        self.edit_caso_nombre.clear()
        self.combo_caso_tipo.setCurrentIndex(self.combo_caso_tipo.findText(TipoCaso.OTRO.value))
        self.boton_eliminar_caso.setEnabled(False)
        self.lista_casos.setCurrentRow(-1)
        self._sucio_caso = False
        self._cargando = False
        self._actualizar_titulo_caso()

    def _pedir_nuevo_caso(self) -> None:
        if self._resolver_pendiente_caso():
            self._nuevo_caso()
            self.edit_caso_id.setFocus()

    def _cargar_caso(self, id_: str) -> None:
        caso = self._casos[id_]
        self._cargando = True
        self._id_caso_actual = id_
        self.edit_caso_id.setText(caso.id)
        self.edit_caso_id.setEnabled(False)
        self.edit_caso_nombre.setText(caso.nombre)
        self.combo_caso_tipo.setCurrentIndex(self.combo_caso_tipo.findText(caso.tipo.value))
        self.boton_eliminar_caso.setEnabled(True)
        self._sucio_caso = False
        self._cargando = False
        self._actualizar_titulo_caso()

    def _al_elegir_caso(self, fila: int) -> None:
        if self._cargando or fila < 0:
            return
        id_nuevo = list(self._casos)[fila]
        if id_nuevo == self._id_caso_actual:
            return
        if not self._resolver_pendiente_caso():
            self._refrescar_lista_casos(self._id_caso_actual)  # vuelve a la selección anterior
            return
        self._cargar_caso(id_nuevo)

    def _guardar_caso(self) -> bool:
        id_ = self._id_caso_actual or self.edit_caso_id.text().strip()
        if not _PATRON_ID.fullmatch(id_):
            self._error("Identificador no válido",
                        "El identificador solo puede tener letras, números y guion bajo (se usa como nombre de archivo).")
            return False
        if self._id_caso_actual is None and id_ in self._casos:
            self._error("Identificador repetido", f"Ya existe un caso de carga «{id_}».")
            return False
        caso = CasoCarga(
            id=id_, nombre=self.edit_caso_nombre.text().strip(), tipo=TipoCaso(self.combo_caso_tipo.currentText())
        )
        self._casos[id_] = caso
        self._casos_eliminados.discard(id_)
        self._refrescar_lista_casos(id_)
        self._cargar_caso(id_)
        self._refrescar_tabla_factores()
        return True

    def _eliminar_caso(self) -> None:
        id_ = self._id_caso_actual
        if id_ is None:
            return
        aviso = f"¿Eliminar el caso «{id_}»? También se quita de todas las combinaciones."
        if id_ in self._casos_con_cargas:
            aviso += "\n\nSe borrarán las cargas asignadas a este caso."
        if QMessageBox.question(self, "Eliminar caso", aviso) != QMessageBox.Yes:
            return
        del self._casos[id_]
        self._casos_eliminados.add(id_)
        for combinacion in self._combinaciones.values():
            combinacion.factores.pop(id_, None)
        self._refrescar_lista_casos()
        self._nuevo_caso()
        self._refrescar_tabla_factores()
        self._refrescar_lista_combinaciones(self._id_comb_actual)

    def _resolver_pendiente_caso(self) -> bool:
        """True si se puede seguir (sin cambios pendientes, guardados o descartados)."""

        if not self._sucio_caso:
            return True
        if self._id_caso_actual is None and not self.edit_caso_id.text().strip() and not self.edit_caso_nombre.text().strip():
            return True
        respuesta = QMessageBox.question(self, "Cambios sin guardar", "Hay cambios sin guardar en el caso de carga. ¿Guardarlos?")
        if respuesta == QMessageBox.Yes:
            return self._guardar_caso()
        self._sucio_caso = False
        return True

    # ------------------------------------------------------------------
    # Combinaciones
    # ------------------------------------------------------------------
    def _refrescar_lista_combinaciones(self, elegir: Optional[str] = None) -> None:
        self._cargando = True
        self.lista_combinaciones.clear()
        for combinacion in self._combinaciones.values():
            self.lista_combinaciones.addItem(f"{combinacion.id} — {formula(combinacion, self._casos)}")
        if elegir in self._combinaciones:
            self.lista_combinaciones.setCurrentRow(list(self._combinaciones).index(elegir))
        self._cargando = False

    def _refrescar_tabla_factores(self, factores: Optional[Dict[str, float]] = None) -> None:
        """Reconstruye la tabla con un renglón por caso de carga existente,
        conservando los factores que ya estaban escritos en el formulario."""

        if factores is None:
            factores = {i: s.value() for i, s in self._spins.items()}
        self._cargando = True
        self.tabla_factores.setRowCount(0)
        self._spins = {}
        for id_caso, caso in self._casos.items():
            fila = self.tabla_factores.rowCount()
            self.tabla_factores.insertRow(fila)
            etiqueta = QLabel(f"  {id_caso} — {caso.nombre}")
            self.tabla_factores.setCellWidget(fila, 0, etiqueta)
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(float(factores.get(id_caso, 0.0)))
            spin.valueChanged.connect(self._al_cambiar_factor)
            self.tabla_factores.setCellWidget(fila, 1, spin)
            self._spins[id_caso] = spin
        self._cargando = False
        self._actualizar_formula()

    def _factores_del_formulario(self) -> Dict[str, float]:
        return {i: s.value() for i, s in self._spins.items() if abs(s.value()) > 1e-12}

    def _al_cambiar_factor(self, *_args) -> None:
        self._actualizar_formula()
        self._marcar_sucio_comb()

    def _actualizar_formula(self) -> None:
        temporal = Combinacion(id="_", nombre="_", factores=self._factores_del_formulario())
        self.etiqueta_formula.setText(f"Resulta: {formula(temporal, self._casos)}")

    def _marcar_sucio_comb(self, *_args) -> None:
        if not self._cargando:
            self._sucio_comb = True
            self._actualizar_titulo_comb()

    def _actualizar_titulo_comb(self) -> None:
        base = "Nueva combinación" if self._id_comb_actual is None else f"Combinación {self._id_comb_actual}"
        self.grupo_comb.setTitle(base + (" (sin guardar)" if self._sucio_comb else ""))

    def _id_combinacion_libre(self) -> str:
        n = len(self._combinaciones) + 1
        while f"COMB{n}" in self._combinaciones:
            n += 1
        return f"COMB{n}"

    def _nueva_combinacion(self) -> None:
        self._cargando = True
        self._id_comb_actual = None
        self.edit_comb_id.setText(self._id_combinacion_libre())
        self.edit_comb_id.setEnabled(True)
        self.edit_comb_nombre.clear()
        self.edit_comb_norma.clear()
        self.combo_plantilla.setCurrentIndex(0)
        self.boton_eliminar_comb.setEnabled(False)
        self.lista_combinaciones.setCurrentRow(-1)
        self._cargando = False
        self._refrescar_tabla_factores({})
        self._sucio_comb = False
        self._actualizar_titulo_comb()

    def _pedir_nueva_combinacion(self) -> None:
        if self._resolver_pendiente_comb():
            self._nueva_combinacion()
            self.edit_comb_nombre.setFocus()

    def _cargar_combinacion(self, id_: str) -> None:
        combinacion = self._combinaciones[id_]
        self._cargando = True
        self._id_comb_actual = id_
        self.edit_comb_id.setText(combinacion.id)
        self.edit_comb_id.setEnabled(False)
        self.edit_comb_nombre.setText(combinacion.nombre)
        self.edit_comb_norma.setText(combinacion.norma)
        self.boton_eliminar_comb.setEnabled(True)
        self._cargando = False
        self._refrescar_tabla_factores(dict(combinacion.factores))
        self._sucio_comb = False
        self._actualizar_titulo_comb()

    def _al_elegir_combinacion(self, fila: int) -> None:
        if self._cargando or fila < 0:
            return
        id_nuevo = list(self._combinaciones)[fila]
        if id_nuevo == self._id_comb_actual:
            return
        if not self._resolver_pendiente_comb():
            self._refrescar_lista_combinaciones(self._id_comb_actual)
            return
        self._cargar_combinacion(id_nuevo)

    def _aplicar_plantilla(self) -> None:
        nombre = self.combo_plantilla.currentText()
        if nombre == _SIN_PLANTILLA:
            return
        plantilla = PLANTILLAS_COMBINACIONES[nombre]
        ausentes = [i for i in plantilla if i not in self._casos]
        self._refrescar_tabla_factores({i: f for i, f in plantilla.items() if i in self._casos})
        if not self.edit_comb_norma.text().strip():
            self.edit_comb_norma.setText(nombre)
        self._marcar_sucio_comb()
        if ausentes:
            QMessageBox.information(
                self, "Casos que faltan",
                "La plantilla usa casos de carga que no existen: " + ", ".join(ausentes)
                + ".\nCréelos en la pestaña «Casos de carga» y vuelva a aplicar la plantilla.",
            )

    def _guardar_combinacion(self) -> bool:
        id_ = self._id_comb_actual or self.edit_comb_id.text().strip()
        if not _PATRON_ID.fullmatch(id_):
            self._error("Identificador no válido", "El identificador solo puede tener letras, números y guion bajo.")
            return False
        if self._id_comb_actual is None and id_ in self._combinaciones:
            self._error("Identificador repetido", f"Ya existe una combinación «{id_}».")
            return False
        factores = self._factores_del_formulario()
        if not factores:
            self._error("Combinación vacía", "Indique al menos un factor distinto de cero.")
            return False
        self._combinaciones[id_] = Combinacion(
            id=id_, nombre=self.edit_comb_nombre.text().strip(), factores=factores,
            norma=self.edit_comb_norma.text().strip(),
        )
        self._refrescar_lista_combinaciones(id_)
        self._cargar_combinacion(id_)
        return True

    def _eliminar_combinacion(self) -> None:
        id_ = self._id_comb_actual
        if id_ is None:
            return
        if QMessageBox.question(self, "Eliminar combinación", f"¿Eliminar la combinación «{id_}»?") != QMessageBox.Yes:
            return
        del self._combinaciones[id_]
        self._refrescar_lista_combinaciones()
        self._nueva_combinacion()

    def _resolver_pendiente_comb(self) -> bool:
        if not self._sucio_comb:
            return True
        if self._id_comb_actual is None and not self._factores_del_formulario() and not self.edit_comb_nombre.text().strip():
            return True
        respuesta = QMessageBox.question(self, "Cambios sin guardar", "Hay cambios sin guardar en la combinación. ¿Guardarlos?")
        if respuesta == QMessageBox.Yes:
            return self._guardar_combinacion()
        self._sucio_comb = False
        return True

    # ------------------------------------------------------------------
    def _al_cambiar_pestana(self, indice: int) -> None:
        if indice == 1:
            if not self._resolver_pendiente_caso():
                self.pestanas.setCurrentIndex(0)
                return
            self._refrescar_tabla_factores()
            self._refrescar_lista_combinaciones(self._id_comb_actual)

    def _error(self, titulo: str, texto: str) -> None:
        QMessageBox.warning(self, titulo, texto)

    def accept(self) -> None:
        if self._resolver_pendiente_caso() and self._resolver_pendiente_comb():
            super().accept()

    def resultado(self):
        return self._casos, self._combinaciones

    def casos_eliminados(self) -> Set[str]:
        """Casos que existían al abrir el diálogo y se quitaron."""

        return set(self._casos_eliminados)
