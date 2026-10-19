"""Diálogo para asignar cargas a lo que se seleccionó en el visor.

Tipos de carga:

* **Presión (kPa)** sobre una losa o sobre una cara de una viga/columna.
  En losas puede ser uniforme o variar linealmente a lo largo de un eje
  (carga triangular o trapezoidal); en una cara lateral se reparte a lo largo
  de la barra con la forma elegida.
* **Carga distribuida (kN/m)** sobre una viga o columna: uniforme,
  triangular (creciente, decreciente o simétrica) o trapezoidal, en todo el
  largo o en un tramo.
* **Fuerza puntual (kN)** y **momento puntual (kN·m)** en un punto de la
  barra o en su extremo.

La **dirección** de aplicación se elige siempre: automática (normal a la cara
elegida, o gravedad), un eje global o un vector cualquiera.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from ..core.asignacion import (
    DIRECCIONES,
    FORMAS_CARGA,
    FormaCarga,
    Objetivo,
    ParametrosCarga,
    vista_previa_cargas,
)
from ..core.cargas import PerfilPresion
from ..core.casos_carga import CasoCarga
from ..core.modelo import Proyecto

TIPOS = {
    "Presión sobre cara o losa (kPa)": "PRESION",
    "Carga distribuida sobre la barra (kN/m)": "LINEAL",
    "Fuerza puntual (kN)": "PUNTUAL",
    "Momento puntual (kN·m)": "MOMENTO",
}
UNIDADES = {"PRESION": " kPa", "LINEAL": " kN/m", "PUNTUAL": " kN", "MOMENTO": " kN·m"}
ETIQUETA_VALOR = {
    "PRESION": "Presión:", "LINEAL": "Carga w:", "PUNTUAL": "Fuerza:", "MOMENTO": "Momento:",
}
DIRECCION_AUTOMATICA = "Automática (normal a la cara, hacia dentro; o gravedad)"
DIRECCION_PERSONALIZADA = "Vector personalizado…"
EJES = {"Eje X": 0, "Eje Y": 1, "Eje Z": 2}
SIN_VARIACION = "Uniforme (sin variación)"


def _spin(valor: float, sufijo: str = "", decimales: int = 3, minimo: float = -1e9, maximo: float = 1e9) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimo, maximo)
    spin.setDecimals(decimales)
    spin.setValue(valor)
    spin.setSuffix(sufijo)
    return spin


class DialogoCargas(QDialog):
    def __init__(self, proyecto: Proyecto, objetivos: List[Objetivo], casos: Dict[str, CasoCarga],
                 caso_activo: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Asignar carga")
        self.resize(640, 780)
        self._proyecto = proyecto
        self._objetivos = objetivos

        self.combo_caso = QComboBox()
        for caso in casos.values():
            self.combo_caso.addItem(f"{caso.id} — {caso.nombre}", caso.id)
        indice = self.combo_caso.findData(caso_activo)
        if indice >= 0:
            self.combo_caso.setCurrentIndex(indice)

        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(list(TIPOS))

        self.spin_valor = _spin(5.0, " kPa", 3)

        # -- forma a lo largo de la barra ------------------------------------------
        self.combo_forma = QComboBox()
        for clave, texto in FORMAS_CARGA.items():
            self.combo_forma.addItem(texto, clave)
        self.spin_w_fin = _spin(0.0, "", 3)
        self.spin_desde = _spin(0.0, " m", 3, 0.0, 1e6)
        self.spin_hasta = _spin(0.0, " m", 3, 0.0, 1e6)
        self.spin_rampa = _spin(1.0, " m", 3, 0.0, 1e6)

        # -- variación de la presión en losas -----------------------------------------
        self.combo_variacion = QComboBox()
        self.combo_variacion.addItem(SIN_VARIACION)
        for texto in EJES:
            self.combo_variacion.addItem(f"Varía linealmente según {texto.lower()} (triangular / trapezoidal)")
        self.spin_valor_a = _spin(0.0, " kPa", 3)
        self.spin_coord_a = _spin(0.0, " m", 3)
        self.spin_valor_b = _spin(5.0, " kPa", 3)
        self.spin_coord_b = _spin(1.0, " m", 3)

        # -- posición de cargas puntuales -----------------------------------------------
        self.spin_posicion = _spin(0.0, " m", 3, 0.0, 1e6)

        # -- dirección ------------------------------------------------------------------------
        self.combo_direccion = QComboBox()
        self.combo_direccion.addItem(DIRECCION_AUTOMATICA)
        self.combo_direccion.addItems(list(DIRECCIONES))
        self.combo_direccion.addItem(DIRECCION_PERSONALIZADA)
        self.spin_vx = _spin(0.0, "", 3)
        self.spin_vy = _spin(0.0, "", 3)
        self.spin_vz = _spin(-1.0, "", 3)

        self.formulario = QFormLayout()
        f = self.formulario
        f.addRow("Caso de carga:", self.combo_caso)
        f.addRow("Tipo de carga:", self.combo_tipo)
        f.addRow(ETIQUETA_VALOR["PRESION"], self.spin_valor)
        self._fila_valor = f.rowCount() - 1
        f.addRow("Forma a lo largo de la barra:", self.combo_forma)
        f.addRow("Valor final (w₂):", self.spin_w_fin)
        f.addRow("Tramo cargado: desde", self.spin_desde)
        f.addRow("hasta", self.spin_hasta)
        f.addRow("Largo de cada rampa:", self.spin_rampa)
        f.addRow("Variación de la presión en la losa:", self.combo_variacion)
        f.addRow("Presión en el punto A:", self.spin_valor_a)
        f.addRow("Coordenada del punto A:", self.spin_coord_a)
        f.addRow("Presión en el punto B:", self.spin_valor_b)
        f.addRow("Coordenada del punto B:", self.spin_coord_b)
        f.addRow("Posición desde el inicio de la barra:", self.spin_posicion)
        f.addRow("Dirección de aplicación:", self.combo_direccion)
        f.addRow("Vector X:", self.spin_vx)
        f.addRow("Vector Y:", self.spin_vy)
        f.addRow("Vector Z:", self.spin_vz)

        self.ayuda = QLabel()
        self.ayuda.setWordWrap(True)

        self.vista_previa = QPlainTextEdit()
        self.vista_previa.setReadOnly(True)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(self.formulario)
        layout.addWidget(self.ayuda)
        layout.addWidget(QLabel("<b>Se crearán estas cargas:</b>"))
        layout.addWidget(self.vista_previa)
        layout.addWidget(self.botones)

        self._inicializar_valores()
        for senal in (
            self.combo_tipo.currentIndexChanged, self.combo_forma.currentIndexChanged,
            self.combo_variacion.currentIndexChanged, self.combo_direccion.currentIndexChanged,
        ):
            senal.connect(self._al_cambiar_modo)
        self.combo_variacion.currentIndexChanged.connect(self._ajustar_coordenadas_a_la_losa)
        for spin in (
            self.spin_valor, self.spin_w_fin, self.spin_desde, self.spin_hasta, self.spin_rampa,
            self.spin_valor_a, self.spin_coord_a, self.spin_valor_b, self.spin_coord_b,
            self.spin_posicion, self.spin_vx, self.spin_vy, self.spin_vz,
        ):
            spin.valueChanged.connect(self._actualizar_vista_previa)
        self._al_cambiar_modo()

    # ------------------------------------------------------------------
    # Valores iniciales según lo seleccionado
    # ------------------------------------------------------------------
    def _barras_seleccionadas(self) -> List[str]:
        return [o.id for o in self._objetivos if o.tipo in ("VIGA", "COLUMNA")]

    def _hay_losas(self) -> bool:
        return any(o.tipo == "LOSA" for o in self._objetivos)

    def _inicializar_valores(self) -> None:
        # Tipo inicial: presión si hay losas o caras; carga lineal si solo hay barras completas.
        solo_barras_completas = bool(self._objetivos) and all(
            o.tipo in ("VIGA", "COLUMNA") and o.cara is None for o in self._objetivos
        )
        self.combo_tipo.setCurrentIndex(1 if solo_barras_completas else 0)
        barras = self._barras_seleccionadas()
        if barras:
            longitud = min(self._proyecto.longitud_barra(b) for b in barras)
            for spin in (self.spin_desde, self.spin_hasta, self.spin_posicion):
                spin.setMaximum(longitud)
            self.spin_hasta.setValue(longitud)
            self.spin_posicion.setValue(longitud / 2.0)
            self.spin_rampa.setValue(max(longitud / 4.0, 0.01))
        self._ajustar_coordenadas_a_la_losa()

    def _ajustar_coordenadas_a_la_losa(self) -> None:
        """Pone en A y B los extremos de la primera losa seleccionada según el eje elegido."""

        eje = self.combo_variacion.currentIndex() - 1
        losa = next((self._proyecto.losas[o.id] for o in self._objetivos if o.tipo == "LOSA"), None)
        if eje < 0 or losa is None:
            return
        valores = [self._proyecto.nodos[n].coordenadas[eje] for n in losa.nodos_contorno]
        self.spin_coord_a.setValue(min(valores))
        self.spin_coord_b.setValue(max(valores) if max(valores) > min(valores) else min(valores) + 1.0)

    # ------------------------------------------------------------------
    # Visibilidad de los campos según el tipo
    # ------------------------------------------------------------------
    def _fila_visible(self, widget, visible: bool) -> None:
        self.formulario.setRowVisible(widget, visible)

    def _al_cambiar_modo(self) -> None:
        tipo = self.tipo()
        self.spin_valor.setSuffix(UNIDADES[tipo])
        etiqueta = self.formulario.labelForField(self.spin_valor)
        if etiqueta is not None:
            etiqueta.setText(ETIQUETA_VALOR[tipo])
        if tipo == "PRESION":
            self.spin_valor.setValue(self.spin_valor.value() if self.spin_valor.value() else 5.0)
        forma = self.combo_forma.currentData()
        con_variacion = self.combo_variacion.currentIndex() > 0

        usa_forma = tipo == "LINEAL" or (tipo == "PRESION" and self._hay_caras_laterales())
        self._fila_visible(self.combo_forma, usa_forma)
        self._fila_visible(self.spin_w_fin, usa_forma and forma == "TRAPEZOIDAL")
        self._fila_visible(self.spin_desde, usa_forma)
        self._fila_visible(self.spin_hasta, usa_forma)
        self._fila_visible(self.spin_rampa, usa_forma and forma == "TRAPEZOIDAL_SIMETRICA")

        presion_losa = tipo == "PRESION" and self._hay_losas()
        self._fila_visible(self.combo_variacion, presion_losa)
        mostrar_perfil = presion_losa and con_variacion
        for spin in (self.spin_valor_a, self.spin_coord_a, self.spin_valor_b, self.spin_coord_b):
            self._fila_visible(spin, mostrar_perfil)
        self._fila_visible(self.spin_valor, not mostrar_perfil)

        self._fila_visible(self.spin_posicion, tipo in ("PUNTUAL", "MOMENTO"))
        personalizada = self.combo_direccion.currentText() == DIRECCION_PERSONALIZADA
        for spin in (self.spin_vx, self.spin_vy, self.spin_vz):
            self._fila_visible(spin, personalizada)

        self.ayuda.setText(self._texto_ayuda(tipo))
        self._actualizar_vista_previa()

    def _hay_caras_laterales(self) -> bool:
        from ..core.asignacion import cara_de

        for o in self._objetivos:
            if o.tipo in ("VIGA", "COLUMNA") and o.cara is not None:
                cara = cara_de(self._proyecto, o)
                if cara is not None and cara.tipo == "LATERAL":
                    return True
        return False

    @staticmethod
    def _texto_ayuda(tipo: str) -> str:
        return {
            "PRESION": "La <b>presión</b> actúa sobre la losa o la cara elegida (positiva = hacia dentro de la cara). "
                       "En una cara lateral de viga/columna se convierte en carga lineal (presión × ancho de la cara); "
                       "en una cara extrema, en una fuerza axial.",
            "LINEAL": "La <b>carga distribuida</b> (kN/m) se aplica sobre la viga o columna completa o sobre un tramo. "
                      "Triangular: 0 en un extremo y w en el otro. Trapezoidal: w inicial y w final distintos.",
            "PUNTUAL": "La <b>fuerza puntual</b> se aplica en un punto de la barra (posición medida desde su nodo inicial) "
                       "o en el nodo de la cara extrema elegida.",
            "MOMENTO": "El <b>momento puntual</b> se aplica como un vector en la dirección elegida (regla de la mano derecha).",
        }[tipo]

    # ------------------------------------------------------------------
    # Parámetros y vista previa
    # ------------------------------------------------------------------
    def tipo(self) -> str:
        return TIPOS[self.combo_tipo.currentText()]

    def _direccion(self) -> Optional[Tuple[float, float, float]]:
        texto = self.combo_direccion.currentText()
        if texto == DIRECCION_AUTOMATICA:
            return None
        if texto == DIRECCION_PERSONALIZADA:
            return (self.spin_vx.value(), self.spin_vy.value(), self.spin_vz.value())
        return DIRECCIONES[texto]

    def _perfil(self) -> Optional[PerfilPresion]:
        eje = self.combo_variacion.currentIndex() - 1
        if eje < 0 or self.tipo() != "PRESION" or not self._hay_losas():
            return None
        return PerfilPresion(
            eje=eje, coord_a=self.spin_coord_a.value(), valor_a=self.spin_valor_a.value(),
            coord_b=self.spin_coord_b.value(), valor_b=self.spin_valor_b.value(),
        )

    def _hasta(self) -> Optional[float]:
        """Fin del tramo cargado; ``None`` = hasta el extremo de cada barra."""

        valor = self.spin_hasta.value()
        return None if valor <= 0.0 or valor >= self.spin_hasta.maximum() - 1e-9 else valor

    def parametros_carga(self) -> ParametrosCarga:
        tipo = self.tipo()
        valor = self.spin_valor.value()
        return ParametrosCarga(
            tipo=tipo,
            presion=valor if tipo == "PRESION" else 0.0,
            w=valor if tipo == "LINEAL" else 0.0,
            fuerza=valor if tipo == "PUNTUAL" else 0.0,
            momento=valor if tipo == "MOMENTO" else 0.0,
            posicion=self.spin_posicion.value(),
            direccion=self._direccion(),
            forma=FormaCarga(
                tipo=self.combo_forma.currentData(), w_fin=self.spin_w_fin.value(),
                desde=self.spin_desde.value(), hasta=self._hasta(), rampa=self.spin_rampa.value(),
            ),
            perfil_presion=self._perfil(),
        )

    def parametros(self) -> Tuple[str, ParametrosCarga]:
        """(caso_id, parámetros de la carga)."""

        return self.combo_caso.currentData(), self.parametros_carga()

    def _actualizar_vista_previa(self, *_args) -> None:
        if not self._objetivos:
            self.vista_previa.setPlainText(
                "No hay nada seleccionado. Seleccione elementos o caras en el visor antes de asignar cargas."
            )
            self.botones.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        vista = vista_previa_cargas(self._proyecto, self._objetivos, self.parametros_carga())
        self.vista_previa.setPlainText("\n".join(f"{'✓' if ok else '✗'} {texto}" for ok, texto in vista))
        self.botones.button(QDialogButtonBox.Ok).setEnabled(any(ok for ok, _ in vista))
