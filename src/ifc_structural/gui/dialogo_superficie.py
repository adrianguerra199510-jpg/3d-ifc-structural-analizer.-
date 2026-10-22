"""Diálogos para asignar apoyos y cargas a una superficie cualquiera (la cara de un sólido de
forma libre o de una barra, losa o muro). Se aplican en el análisis 3D."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from ..core.asignacion import DIRECCIONES, ErrorAsignacion, Objetivo, descripcion
from ..core.asignacion_superficie import (
    Superficie, apoyo_superficie_para, carga_superficie_para, superficie_de, union_para,
)
from ..core.casos_carga import CasoCarga
from ..core.modelo import Proyecto
from .estilo import EncabezadoDialogo

PRESETS_APOYO = {
    "Empotrada (restringe Ux, Uy y Uz)": (True, True, True, False, False, False),
    "Deslizante en Z (solo Uz)": (False, False, True, False, False, False),
    "Deslizante en X (solo Ux)": (True, False, False, False, False, False),
    "Deslizante en Y (solo Uy)": (False, True, False, False, False, False),
    "Cuerpo rígido con giros impedidos (los 6)": (True, True, True, True, True, True),
    "Personalizado": None,
}
#: Preset del pasador: fija las tres traslaciones del centro de la superficie (no las de cada nodo)
PRESET_PASADOR = "Pasador en un agujero (centro fijo, giros libres)"
AYUDA_PASADOR = (
    "<b>Pasador</b>: para el agujero de una oreja o de una cartela por donde pasa un perno. No se fija cada nodo de la "
    "superficie (eso pega el agujero a un perno infinitamente rígido y crea picos de esfuerzo): se fija el <b>centro</b> "
    "de la superficie, el promedio de sus desplazamientos ponderado por el área (acople distribuido, como el RBE3). El "
    "agujero se deforma y la pieza gira libremente alrededor del perno; la reacción se reparte por toda la superficie. "
    "Elija la superficie completa del agujero. Con un solo pasador, o con todos alineados, el modelo puede girar "
    "alrededor de ellos: añada otro apoyo que lo impida.")
PRESET_PERSONALIZADO = "Personalizado"
NOMBRES_GDL = ("Ux (traslación en X)", "Uy (traslación en Y)", "Uz (traslación en Z)",
               "Rx (giro alrededor de X)", "Ry (giro alrededor de Y)", "Rz (giro alrededor de Z)")

TIPOS_CARGA = {
    "Presión (kPa) sobre toda la superficie": "PRESION",
    "Fuerza total (kN) repartida en la superficie": "FUERZA",
    "Momento total (kN·m) respecto del centro de la superficie": "MOMENTO",
}
UNIDADES = {"PRESION": " kPa", "FUERZA": " kN", "MOMENTO": " kN·m"}
DIRECCION_NORMAL = "Normal a la superficie, hacia dentro"
DIRECCION_PERSONALIZADA = "Vector personalizado…"


def _superficies(proyecto: Proyecto, objetivos: List[Objetivo]) -> Tuple[List[Superficie], List[str]]:
    buenas, errores = [], []
    for o in objetivos:
        try:
            buenas.append(superficie_de(proyecto, o))
        except ErrorAsignacion as exc:
            errores.append(str(exc))
    return buenas, errores


class DialogoApoyoSuperficie(QDialog):
    def __init__(self, proyecto: Proyecto, objetivos: List[Objetivo], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar apoyo a una superficie")
        self.resize(620, 640)
        self._proyecto = proyecto
        self._objetivos = objetivos
        self._superficies, self._errores = _superficies(proyecto, objetivos)

        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo(
            "apoyo", "Apoyo en una superficie",
            "Restringe el movimiento de todos los nodos de la superficie (análisis 3D)."))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS_APOYO))
        self.combo_preset.insertItem(1, PRESET_PASADOR)
        formulario = QFormLayout()
        formulario.addRow("Tipo de apoyo:", self.combo_preset)
        self.checks: List[QCheckBox] = []
        for texto in NOMBRES_GDL:
            check = QCheckBox(texto)
            check.toggled.connect(self._al_cambiar_check)
            self.checks.append(check)
            formulario.addRow(check)
        layout.addLayout(formulario)

        nota = QLabel(
            "Con las tres traslaciones restringidas la superficie queda <b>empotrada</b> (no se mueve). Con una sola "
            "(p. ej. Uz) es un apoyo <b>deslizante</b>, como una pieza sobre el terreno. Si marca giros, la superficie "
            "se mueve como cuerpo rígido con esos giros impedidos. El modelo debe quedar estable."
        )
        nota.setWordWrap(True)
        layout.addWidget(nota)
        self.nota = nota
        layout.addWidget(QLabel("<b>Se crearán estos apoyos:</b>"))
        self.vista_previa = QPlainTextEdit()
        self.vista_previa.setReadOnly(True)
        layout.addWidget(self.vista_previa)
        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)

        self._nota_general = nota.text()
        self.combo_preset.currentIndexChanged.connect(self._al_cambiar_preset)
        self._al_cambiar_preset()

    @property
    def pasador(self) -> bool:
        return self.combo_preset.currentText() == PRESET_PASADOR

    def _al_cambiar_preset(self) -> None:
        pasador = self.pasador
        self.nota.setText(AYUDA_PASADOR if pasador else self._nota_general)
        for check in self.checks[3:]:  # el pasador deja los giros libres
            check.setEnabled(not pasador)
        valores = (True, True, True, False, False, False) if pasador else PRESETS_APOYO[self.combo_preset.currentText()]
        if valores is None:
            self._actualizar_vista_previa()
            return
        for check, valor in zip(self.checks, valores):
            check.blockSignals(True)
            check.setChecked(valor)
            check.blockSignals(False)
        self._actualizar_vista_previa()

    def _al_cambiar_check(self) -> None:
        if self.pasador:  # elegir qué traslaciones del centro se fijan no cambia el tipo de apoyo
            self._actualizar_vista_previa()
            return
        actuales = tuple(c.isChecked() for c in self.checks)
        nombre = next((n for n, v in PRESETS_APOYO.items() if v == actuales), PRESET_PERSONALIZADO)
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentText(nombre)
        self.combo_preset.blockSignals(False)
        self._actualizar_vista_previa()

    def restricciones(self) -> Tuple[bool, ...]:
        valores = tuple(c.isChecked() for c in self.checks)
        return valores[:3] + (False, False, False) if self.pasador else valores

    def _actualizar_vista_previa(self) -> None:
        marcados = any(self.restricciones())
        tipo = " — pasador" if self.pasador else ""
        lineas = [f"{'✓' if marcados else '✗'} {s.nombre} (área {s.area:.4g} m²){tipo}" for s in self._superficies]
        lineas += [f"✗ {e}" for e in self._errores]
        if not marcados:
            lineas.append("Marque al menos un grado de libertad.")
        self.vista_previa.setPlainText("\n".join(lineas))
        self.botones.button(QDialogButtonBox.Ok).setEnabled(marcados and bool(self._superficies))

    def apoyos(self, siguiente_id) -> Tuple[list, List[str]]:
        """Los apoyos a crear y los errores. ``siguiente_id`` da un id nuevo en cada llamada."""

        creados, errores = [], list(self._errores)
        for objetivo in self._objetivos:
            try:
                creados.append(apoyo_superficie_para(self._proyecto, objetivo, siguiente_id(), self.restricciones(),
                                                     pasador=self.pasador))
            except ErrorAsignacion as exc:
                if str(exc) not in errores:
                    errores.append(str(exc))
        return creados, errores


class DialogoCargaSuperficie(QDialog):
    def __init__(self, proyecto: Proyecto, objetivos: List[Objetivo], casos: Dict[str, CasoCarga], caso_activo: str,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar carga a una superficie")
        self.resize(660, 700)
        self._proyecto = proyecto
        self._objetivos = objetivos
        self._superficies, self._errores = _superficies(proyecto, objetivos)

        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo(
            "carga_dist", "Carga en una superficie",
            "Presión, fuerza o momento repartidos por toda la superficie, en la dirección que elija (análisis 3D)."))
        self.combo_caso = QComboBox()
        for caso in casos.values():
            self.combo_caso.addItem(f"{caso.id} — {caso.nombre}", caso.id)
        indice = self.combo_caso.findData(caso_activo)
        if indice >= 0:
            self.combo_caso.setCurrentIndex(indice)
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(list(TIPOS_CARGA))
        self.spin_valor = QDoubleSpinBox()
        self.spin_valor.setRange(-1e9, 1e9)
        self.spin_valor.setDecimals(4)
        self.spin_valor.setValue(10.0)
        self.combo_direccion = QComboBox()
        self.spin_v = []
        formulario = QFormLayout()
        formulario.addRow("Caso de carga:", self.combo_caso)
        formulario.addRow("Tipo de carga:", self.combo_tipo)
        formulario.addRow("Valor:", self.spin_valor)
        formulario.addRow("Dirección:", self.combo_direccion)
        for nombre, valor in (("Vector X:", 0.0), ("Vector Y:", 0.0), ("Vector Z:", -1.0)):
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(3)
            spin.setValue(valor)
            self.spin_v.append(spin)
            formulario.addRow(nombre, spin)
        self.formulario = formulario
        layout.addLayout(formulario)

        self.ayuda = QLabel()
        self.ayuda.setWordWrap(True)
        layout.addWidget(self.ayuda)
        layout.addWidget(QLabel("<b>Se crearán estas cargas:</b>"))
        self.vista_previa = QPlainTextEdit()
        self.vista_previa.setReadOnly(True)
        layout.addWidget(self.vista_previa)
        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)

        self.combo_tipo.currentIndexChanged.connect(self._al_cambiar_tipo)
        self.combo_direccion.currentIndexChanged.connect(self._actualizar)
        self.spin_valor.valueChanged.connect(self._actualizar)
        for spin in self.spin_v:
            spin.valueChanged.connect(self._actualizar)
        self._al_cambiar_tipo()

    def tipo(self) -> str:
        return TIPOS_CARGA[self.combo_tipo.currentText()]

    def _al_cambiar_tipo(self) -> None:
        tipo = self.tipo()
        self.combo_direccion.blockSignals(True)
        self.combo_direccion.clear()
        self.combo_direccion.addItem(DIRECCION_NORMAL)
        self.combo_direccion.addItems(list(DIRECCIONES))
        self.combo_direccion.addItem(DIRECCION_PERSONALIZADA)
        self.combo_direccion.blockSignals(False)
        self.ayuda.setText({
            "PRESION": "La <b>presión</b> actúa sobre toda la superficie. Por defecto es normal a ella y hacia dentro "
                       "(comprime); con una dirección elegida es una tracción en ese sentido (peso, viento, empuje).",
            "FUERZA": "La <b>fuerza total</b> se reparte uniformemente en el área de la superficie, en la dirección elegida.",
            "MOMENTO": "El <b>momento total</b> actúa alrededor del eje elegido (regla de la mano derecha), "
                       "respecto del centro de la superficie.",
        }[tipo])
        self.spin_valor.setSuffix(UNIDADES[tipo])
        self._actualizar()

    def _direccion(self, superficie: Superficie) -> Optional[Tuple[float, float, float]]:
        texto = self.combo_direccion.currentText()
        if texto == DIRECCION_NORMAL:
            if self.tipo() == "PRESION":
                return None  # la presión normal se calcula con la normal de cada superficie
            return tuple(-c for c in superficie.normal)
        if texto == DIRECCION_PERSONALIZADA:
            return tuple(s.value() for s in self.spin_v)
        return DIRECCIONES[texto]

    def _actualizar(self, *_args) -> None:
        personalizada = self.combo_direccion.currentText() == DIRECCION_PERSONALIZADA
        for fila, spin in enumerate(self.spin_v):
            self.formulario.setRowVisible(spin, personalizada)
        lineas, validas = [], 0
        for s, objetivo in zip(self._superficies, [o for o in self._objetivos if o.cara is not None]):
            try:
                carga = carga_superficie_para(self._proyecto, objetivo, "-", "-", self.tipo(), self.spin_valor.value(),
                                              self._direccion(s))
                total = self.spin_valor.value() * s.area if carga.tipo == "PRESION" else self.spin_valor.value()
                extra = f" = {total:.4g} kN en total" if carga.tipo == "PRESION" else ""
                lineas.append(f"✓ {carga.nombre}{extra}")
                validas += 1
            except ErrorAsignacion as exc:
                lineas.append(f"✗ {exc}")
        lineas += [f"✗ {e}" for e in self._errores]
        self.vista_previa.setPlainText("\n".join(lineas))
        self.botones.button(QDialogButtonBox.Ok).setEnabled(validas > 0)

    def caso_id(self) -> str:
        return self.combo_caso.currentData()

    def cargas(self, siguiente_id) -> Tuple[list, List[str]]:
        creadas, errores = [], list(self._errores)
        for objetivo in self._objetivos:
            if objetivo.cara is None:
                continue
            s = superficie_de(self._proyecto, objetivo)
            try:
                creadas.append(carga_superficie_para(
                    self._proyecto, objetivo, self.caso_id(), siguiente_id(), self.tipo(), self.spin_valor.value(),
                    self._direccion(s)))
            except ErrorAsignacion as exc:
                if str(exc) not in errores:
                    errores.append(str(exc))
        return creadas, errores


TIPOS_UNION_TEXTO = {
    "Soldadura: une las dos superficies punto a punto (cierra la holgura)": "SOLDADURA",
    "Cuerpo rígido: las dos superficies se mueven juntas como un cuerpo rígido": "CUERPO_RIGIDO",
    "Igualar movimientos: los puntos siguen al más cercano en los grados marcados": "IGUALAR",
    "Articulación: superficies rígidas con centros unidos; giros marcados igualados": "ARTICULACION",
}
ALCANCE_SUPERFICIES = "Las superficies más cercanas (o las elegidas)"
ALCANCE_COMPLETOS = "Los elementos 3D completos"
AYUDA_UNION = {
    "SOLDADURA": "Las superficies quedan soldadas en todos sus puntos (<code>*TIE</code>). Si hay holgura entre ellas, "
                 "se cierra. Entre elementos completos se sueldan por todas las superficies en que se tocan (o están a "
                 "menos de la separación máxima). Sustituye a la unión automática entre esas dos piezas.",
    "CUERPO_RIGIDO": "Los nodos de las dos superficies (o de los dos elementos completos) forman un único cuerpo rígido: "
                     "no se deforman y se mueven juntos (como el constraint «Body» de SAP). Si otro cuerpo rígido comparte "
                     "nodos con este, los dos se funden en uno.",
    "IGUALAR": "Cada nodo de la superficie más fina iguala, en las traslaciones marcadas, el movimiento del punto más "
               "cercano de la otra (interpolado en su cara, como una soldadura); las demás quedan libres (deslizamiento). "
               "Como el constraint «Equal» de SAP.",
    "ARTICULACION": "Cada superficie (o elemento completo) se vuelve rígida y los centros de las dos se igualan en las tres "
                    "traslaciones. Marque los giros que se igualan (los demás quedan libres): una rótula, una bisagra o una unión con un solo giro.",
}


class DialogoUnionSuperficie(QDialog):
    """Constraint entre dos superficies (o entre dos elementos: se toman las caras más cercanas)."""

    def __init__(self, proyecto: Proyecto, a: Objetivo, b: Objetivo, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Constraint entre superficies")
        self.resize(640, 560)
        self._proyecto, self._a, self._b = proyecto, a, b
        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo("conectar", "Constraint entre superficies",
                                           "Une dos superficies o elementos en el análisis 3D (como los joint constraints de SAP)."))
        self.etiqueta = QLabel()
        self.etiqueta.setWordWrap(True)
        layout.addWidget(self.etiqueta)
        formulario = QFormLayout()
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(list(TIPOS_UNION_TEXTO))
        formulario.addRow("Tipo:", self.combo_tipo)
        self.combo_alcance = QComboBox()
        self.combo_alcance.addItems([ALCANCE_SUPERFICIES, ALCANCE_COMPLETOS])
        self._con_caras = a.cara is not None and b.cara is not None  # se eligieron superficies concretas
        if self._con_caras:
            self.combo_alcance.setEnabled(False)
            self.combo_alcance.setToolTip("Se eligieron superficies concretas (modo «Cara»).")
        formulario.addRow("Se une:", self.combo_alcance)
        self.spin_tolerancia = QDoubleSpinBox()
        self.spin_tolerancia.setRange(0.0, 500.0)
        self.spin_tolerancia.setDecimals(1)
        self.spin_tolerancia.setSuffix(" mm")
        self.spin_tolerancia.setValue(10.0)
        self.spin_tolerancia.setToolTip("Separación máxima con la que la soldadura une dos elementos completos.")
        formulario.addRow("Separación máxima:", self.spin_tolerancia)
        self.checks = []
        for texto in NOMBRES_GDL:
            check = QCheckBox(texto + " — se iguala")
            check.toggled.connect(self._actualizar)
            self.checks.append(check)
            formulario.addRow(check)
        layout.addLayout(formulario)
        self.ayuda = QLabel()
        self.ayuda.setWordWrap(True)
        layout.addWidget(self.ayuda)
        layout.addWidget(QLabel("<b>Resultado:</b>"))
        self.vista_previa = QPlainTextEdit()
        self.vista_previa.setReadOnly(True)
        layout.addWidget(self.vista_previa)
        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)
        self.combo_tipo.currentIndexChanged.connect(self._al_cambiar_tipo)
        self.combo_alcance.currentIndexChanged.connect(self._actualizar)
        self.spin_tolerancia.valueChanged.connect(self._actualizar)
        self._al_cambiar_tipo()

    def tipo(self) -> str:
        return TIPOS_UNION_TEXTO[self.combo_tipo.currentText()]

    def completo(self) -> bool:
        return not self._con_caras and self.combo_alcance.currentText() == ALCANCE_COMPLETOS

    def _al_cambiar_tipo(self) -> None:
        tipo = self.tipo()
        for i, check in enumerate(self.checks):
            check.blockSignals(True)
            if tipo == "IGUALAR":
                check.setChecked(i < 3)
            elif tipo == "ARTICULACION":
                check.setChecked(i < 3)
            else:
                check.setChecked(True)
            check.setEnabled(tipo == "IGUALAR" and i < 3 or tipo == "ARTICULACION" and i >= 3)
            check.blockSignals(False)
        self.ayuda.setText(AYUDA_UNION[tipo])
        if not self._con_caras:  # entre elementos enteros, lo normal para el cuerpo rígido y la articulación es unir los elementos completos
            self.combo_alcance.blockSignals(True)
            self.combo_alcance.setCurrentText(ALCANCE_COMPLETOS if tipo in ("CUERPO_RIGIDO", "ARTICULACION") else ALCANCE_SUPERFICIES)
            self.combo_alcance.blockSignals(False)
        self._actualizar()

    def grados(self) -> Tuple[bool, ...]:
        return tuple(c.isChecked() for c in self.checks)

    def union(self, id_: str = "-"):
        return union_para(self._proyecto, self._a, self._b, id_, self.tipo(), self.grados(), completo=self.completo(),
                          tolerancia=self.spin_tolerancia.value() / 1000.0)

    def _actualizar(self, *_args) -> None:
        try:
            u = self.union()
            iguales = ", ".join(n for n, g in zip(("Ux", "Uy", "Uz", "Rx", "Ry", "Rz"), (u.ux, u.uy, u.uz, u.rx, u.ry, u.rz)) if g)
            texto = f"✓ {u.nombre}\n   Grados de libertad que se transmiten: {iguales}."
            self.spin_tolerancia.setEnabled(u.tipo == "SOLDADURA" and u.es_completa)
            self.etiqueta.setText(f"<b>A:</b> {descripcion(self._proyecto, Objetivo(u.a_tipo, u.a_id, None if u.a_cara < 0 else u.a_cara))}<br>"
                                  f"<b>B:</b> {descripcion(self._proyecto, Objetivo(u.b_tipo, u.b_id, None if u.b_cara < 0 else u.b_cara))}")
            self.botones.button(QDialogButtonBox.Ok).setEnabled(True)
        except ErrorAsignacion as exc:
            texto = f"✗ {exc}"
            self.botones.button(QDialogButtonBox.Ok).setEnabled(False)
        self.vista_previa.setPlainText(texto)
