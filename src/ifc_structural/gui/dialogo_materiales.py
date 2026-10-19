"""Creación y edición de materiales, y asignación de un material a los
elementos seleccionados.

``DialogoMateriales``: lista de materiales a la izquierda y formulario a la
derecha (nombre, módulo de elasticidad en MPa, coeficiente de Poisson y peso
específico en kN/m³), con plantillas de materiales habituales. Los cambios se
aplican al aceptar.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ..core.asignacion import Objetivo, descripcion
from ..core.materiales import (
    PLANTILLAS_MATERIALES,
    descripcion_material,
    material_desde_valores,
    modulo_en_mpa,
    usos_de_material,
)
from ..core.modelo import Material, Proyecto

_SIN_PLANTILLA = "(elegir plantilla…)"


class DialogoMateriales(QDialog):
    def __init__(self, proyecto: Proyecto, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Materiales")
        self.resize(720, 460)
        self._proyecto = proyecto
        self._materiales: Dict[str, Material] = copy.deepcopy(proyecto.materiales)
        self._nombre_actual: Optional[str] = None  # None = material nuevo
        self._cargando = False
        self._sucio = False

        fila = QHBoxLayout()
        izquierda = QVBoxLayout()
        izquierda.addWidget(QLabel("Materiales del proyecto"))
        self.lista = QListWidget()
        self.lista.currentRowChanged.connect(self._al_elegir)
        izquierda.addWidget(self.lista)
        self.boton_nuevo = QPushButton("＋ Nuevo material")
        self.boton_nuevo.clicked.connect(self._pedir_nuevo)
        izquierda.addWidget(self.boton_nuevo)
        fila.addLayout(izquierda, 1)

        self.grupo = QGroupBox()
        formulario = QFormLayout(self.grupo)
        self.edit_nombre = QLineEdit()
        self.edit_nombre.setPlaceholderText("Ej.: HORMIGON_28, ACERO_A36")
        self.spin_e = self._spin(1.0, 1e7, 0, " MPa")
        self.spin_poisson = self._spin(0.0, 0.499, 3, "")
        self.spin_peso = self._spin(0.0, 1000.0, 2, " kN/m³")
        self.combo_plantilla = QComboBox()
        self.combo_plantilla.addItem(_SIN_PLANTILLA)
        self.combo_plantilla.addItems(list(PLANTILLAS_MATERIALES))
        self.combo_plantilla.activated.connect(self._aplicar_plantilla)
        formulario.addRow("Plantilla:", self.combo_plantilla)
        formulario.addRow("Nombre:", self.edit_nombre)
        formulario.addRow("Módulo de elasticidad E:", self.spin_e)
        formulario.addRow("Coeficiente de Poisson ν:", self.spin_poisson)
        formulario.addRow("Peso específico γ:", self.spin_peso)
        self.etiqueta_uso = QLabel()
        self.etiqueta_uso.setWordWrap(True)
        formulario.addRow(self.etiqueta_uso)
        botones = QHBoxLayout()
        self.boton_guardar = QPushButton("Guardar material")
        self.boton_guardar.setDefault(True)
        self.boton_guardar.clicked.connect(self._guardar)
        self.boton_eliminar = QPushButton("Eliminar material")
        self.boton_eliminar.clicked.connect(self._eliminar)
        botones.addWidget(self.boton_guardar)
        botones.addWidget(self.boton_eliminar)
        formulario.addRow(botones)
        fila.addWidget(self.grupo, 2)

        layout = QVBoxLayout(self)
        layout.addLayout(fila)
        caja = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        caja.accepted.connect(self.accept)
        caja.rejected.connect(self.reject)
        layout.addWidget(caja)

        for editor in (self.edit_nombre,):
            editor.textChanged.connect(self._marcar_sucio)
        for spin in (self.spin_e, self.spin_poisson, self.spin_peso):
            spin.valueChanged.connect(self._marcar_sucio)

        self._refrescar_lista()
        self._nuevo()
        if self._materiales:
            self.lista.setCurrentRow(0)

    @staticmethod
    def _spin(minimo: float, maximo: float, decimales: int, sufijo: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimo, maximo)
        spin.setDecimals(decimales)
        spin.setSuffix(sufijo)
        spin.setGroupSeparatorShown(True)
        return spin

    # -- estado -----------------------------------------------------------
    def _refrescar_lista(self, elegir: Optional[str] = None) -> None:
        self._cargando = True
        self.lista.clear()
        for material in self._materiales.values():
            self.lista.addItem(f"{material.nombre}   (E = {modulo_en_mpa(material):,.0f} MPa)")
        if elegir in self._materiales:
            self.lista.setCurrentRow(list(self._materiales).index(elegir))
        self._cargando = False

    def _marcar_sucio(self, *_args) -> None:
        if not self._cargando:
            self._sucio = True
            self._titulo()

    def _titulo(self) -> None:
        base = "Nuevo material" if self._nombre_actual is None else f"Material {self._nombre_actual}"
        self.grupo.setTitle(base + (" (sin guardar)" if self._sucio else ""))

    def _nuevo(self) -> None:
        self._cargando = True
        self._nombre_actual = None
        self.edit_nombre.clear()
        self.edit_nombre.setEnabled(True)
        self.spin_e.setValue(25000.0)
        self.spin_poisson.setValue(0.2)
        self.spin_peso.setValue(24.0)
        self.combo_plantilla.setCurrentIndex(0)
        self.boton_eliminar.setEnabled(False)
        self.etiqueta_uso.clear()
        self.lista.setCurrentRow(-1)
        self._sucio = False
        self._cargando = False
        self._titulo()

    def _pedir_nuevo(self) -> None:
        if self._resolver_pendiente():
            self._nuevo()
            self.edit_nombre.setFocus()

    def _cargar(self, nombre: str) -> None:
        material = self._materiales[nombre]
        self._cargando = True
        self._nombre_actual = nombre
        self.edit_nombre.setText(material.nombre)
        self.edit_nombre.setEnabled(False)  # los elementos lo referencian por nombre
        self.spin_e.setValue(modulo_en_mpa(material))
        self.spin_poisson.setValue(material.modulo_poisson)
        self.spin_peso.setValue(material.densidad)
        self.combo_plantilla.setCurrentIndex(0)
        self.boton_eliminar.setEnabled(True)
        usos = usos_de_material(self._proyecto, nombre)
        self.etiqueta_uso.setText(
            f"Usado por {len(usos)} elemento(s): {', '.join(usos[:8])}{'…' if len(usos) > 8 else ''}"
            if usos else "No lo usa ningún elemento."
        )
        self._sucio = False
        self._cargando = False
        self._titulo()

    def _al_elegir(self, fila: int) -> None:
        if self._cargando or fila < 0:
            return
        nuevo = list(self._materiales)[fila]
        if nuevo == self._nombre_actual:
            return
        if not self._resolver_pendiente():
            self._refrescar_lista(self._nombre_actual)
            return
        self._cargar(nuevo)

    def _aplicar_plantilla(self, indice: int) -> None:
        nombre = self.combo_plantilla.currentText()
        if nombre == _SIN_PLANTILLA:
            return
        e, nu, peso = PLANTILLAS_MATERIALES[nombre]
        self.spin_e.setValue(e)
        self.spin_poisson.setValue(nu)
        self.spin_peso.setValue(peso)
        if self._nombre_actual is None and not self.edit_nombre.text().strip():
            self.edit_nombre.setText("".join(c if c.isalnum() else "_" for c in nombre).strip("_").upper())

    def _guardar(self) -> bool:
        nombre = self._nombre_actual or self.edit_nombre.text().strip()
        if not nombre:
            QMessageBox.warning(self, "Falta el nombre", "Escriba un nombre para el material.")
            return False
        if self._nombre_actual is None and nombre in self._materiales:
            QMessageBox.warning(self, "Nombre repetido", f"Ya existe un material llamado «{nombre}».")
            return False
        self._materiales[nombre] = material_desde_valores(
            nombre, self.spin_e.value(), self.spin_poisson.value(), self.spin_peso.value()
        )
        self._refrescar_lista(nombre)
        self._cargar(nombre)
        return True

    def _eliminar(self) -> None:
        nombre = self._nombre_actual
        if nombre is None:
            return
        usos = usos_de_material(self._proyecto, nombre)
        if usos:
            QMessageBox.warning(
                self, "Material en uso",
                f"«{nombre}» lo usan {len(usos)} elemento(s) ({', '.join(usos[:8])}). Asigne otro material a esos "
                "elementos antes de eliminarlo.",
            )
            return
        if QMessageBox.question(self, "Eliminar material", f"¿Eliminar el material «{nombre}»?") != QMessageBox.Yes:
            return
        del self._materiales[nombre]
        self._refrescar_lista()
        self._nuevo()

    def _resolver_pendiente(self) -> bool:
        if not self._sucio:
            return True
        if self._nombre_actual is None and not self.edit_nombre.text().strip():
            return True
        respuesta = QMessageBox.question(self, "Cambios sin guardar", "Hay cambios sin guardar en el material. ¿Guardarlos?")
        if respuesta == QMessageBox.Yes:
            return self._guardar()
        self._sucio = False
        return True

    def accept(self) -> None:
        if self._resolver_pendiente():
            super().accept()

    def resultado(self) -> Dict[str, Material]:
        return self._materiales


class DialogoAsignarMaterial(QDialog):
    """Elige el material de los elementos seleccionados (con acceso a crear
    o editar materiales)."""

    def __init__(self, proyecto: Proyecto, objetivos: List[Objetivo], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar material")
        self.resize(520, 360)
        self._proyecto = proyecto

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Elementos seleccionados</b> (se asigna al elemento completo, aunque se haya elegido una cara):"))
        elementos = list(dict.fromkeys((o.tipo, o.id) for o in objetivos))
        resumen = QPlainTextEdit()
        resumen.setReadOnly(True)
        resumen.setPlainText(
            "\n".join(
                f"{descripcion(proyecto, Objetivo(t, i))} — material actual: "
                f"{({'VIGA': proyecto.vigas, 'COLUMNA': proyecto.columnas, 'LOSA': proyecto.losas, 'MURO': proyecto.muros}[t][i]).material}"
                for t, i in elementos
            ) or "No hay nada seleccionado."
        )
        resumen.setMaximumHeight(110)
        layout.addWidget(resumen)

        fila = QHBoxLayout()
        fila.addWidget(QLabel("Material:"))
        self.combo = QComboBox()
        fila.addWidget(self.combo, 1)
        self.boton_materiales = QPushButton("Crear / editar materiales…")
        self.boton_materiales.clicked.connect(self._gestionar)
        fila.addWidget(self.boton_materiales)
        layout.addLayout(fila)

        self.detalle = QLabel()
        self.detalle.setWordWrap(True)
        layout.addWidget(self.detalle)
        self.combo.currentIndexChanged.connect(self._actualizar_detalle)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)
        self._objetivos_vacios = not elementos
        self._llenar()

    def _llenar(self, elegir: Optional[str] = None) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        for nombre in self._proyecto.materiales:
            self.combo.addItem(nombre, nombre)
        if elegir:
            self.combo.setCurrentIndex(max(self.combo.findData(elegir), 0))
        self.combo.blockSignals(False)
        self._actualizar_detalle()

    def _actualizar_detalle(self) -> None:
        nombre = self.combo.currentData()
        material = self._proyecto.materiales.get(nombre)
        self.detalle.setText(descripcion_material(material) if material else "No hay materiales: cree uno.")
        self.botones.button(QDialogButtonBox.Ok).setEnabled(material is not None and not self._objetivos_vacios)

    def _gestionar(self) -> None:
        dialogo = DialogoMateriales(self._proyecto, self)
        if dialogo.exec():
            self._proyecto.materiales = dialogo.resultado()
            self._llenar(self.combo.currentData())

    def material_elegido(self) -> str:
        return self.combo.currentData()
