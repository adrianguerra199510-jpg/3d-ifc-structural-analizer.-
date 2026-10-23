"""Panel de esfuerzos: qué resultado y qué magnitud mostrar en el visor,
con escala configurable, y la tabla de esfuerzos máximos por viga/columna.

Se muestran esfuerzos (kPa del solver convertidos a MPa), no fuerzas internas.
Opcionalmente la deformada (desplazamientos ampliados), animada y exportable a video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..combinaciones.esfuerzos import MAGNITUDES
from .visor_3d import Seccion

ENVOLVENTE = "__ENVOLVENTE__"
#: Clave del combo «Esfuerzo» para colorear por el desplazamiento total |u| (mm), como la deformada del reporte.
DESPLAZAMIENTO = "__DESPLAZAMIENTO__"
#: Clave del combo para la deformación plástica equivalente PEEQ (%), solo con resultados del análisis elastoplástico.
DEFORMACION_PLASTICA = "__PEEQ__"
TEXTO_PLASTICA = "Deformación plástica (PEEQ)"
COLORMAPS = ["turbo", "jet", "viridis", "coolwarm", "rainbow", "plasma"]


@dataclass
class ParametrosVista:
    """Lo que el usuario eligió mostrar."""

    resultado: str  # id de caso/combinación o ENVOLVENTE
    magnitud: str  # clave de esfuerzos.MAGNITUDES
    escala_automatica: bool
    minimo: float  # MPa (mm con DESPLAZAMIENTO)
    maximo: float  # MPa (mm con DESPLAZAMIENTO)
    colormap: str
    deformada: bool = False
    escala_deformada: float = 1.0  # factor de ampliación de los desplazamientos
    mostrar_original: bool = True


class PanelEsfuerzos(QWidget):
    """Controles de la vista de esfuerzos. Emite ``cambio`` cada vez que
    algo que afecte a lo que se dibuja cambia, o ``volver_al_modelo``."""

    cambio = Signal()
    volver_al_modelo = Signal()
    animacion_alternada = Signal(bool)
    video_solicitado = Signal()
    reporte_solicitado = Signal()
    escala_automatica_solicitada = Signal()
    seccion_cambiada = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.combo_resultado = QComboBox()
        self.combo_magnitud = QComboBox()
        for magnitud in MAGNITUDES.values():
            self.combo_magnitud.addItem(magnitud.nombre, magnitud.clave)
        self.combo_magnitud.addItem("Desplazamiento total", DESPLAZAMIENTO)

        self.check_escala_automatica = QCheckBox("Escala automática")
        self.check_escala_automatica.setChecked(True)
        self.spin_minimo = self._spin()
        self.spin_maximo = self._spin()
        self.spin_maximo.setValue(10.0)
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(COLORMAPS)

        self.check_deformada = QCheckBox("Mostrar deformada")
        self.spin_escala_deformada = QDoubleSpinBox()
        self.spin_escala_deformada.setRange(0.1, 1e7)
        self.spin_escala_deformada.setDecimals(1)
        self.spin_escala_deformada.setValue(100.0)
        self.spin_escala_deformada.setPrefix("× ")
        self.boton_escala_auto = QPushButton("Auto")
        self.boton_escala_auto.setToolTip("Ampliación que hace visible la deformada (≈ 8 % del tamaño del modelo)")
        self.check_original = QCheckBox("Mostrar el contorno sin deformar")
        self.check_original.setChecked(True)
        self.boton_animar = QPushButton("▶ Animar")
        self.boton_animar.setCheckable(True)
        self.boton_video = QPushButton("Guardar video…")
        self.boton_video.setToolTip("Graba la deformada animada en un video MP4 (o GIF) con la vista actual")

        # Sección: un plano que corta el modelo para ver los esfuerzos por dentro
        self.check_seccion = QCheckBox("Cortar con un plano")
        self.check_seccion.setToolTip("Corta la vista de resultados con un plano: la cara del corte se colorea con los "
                                      "valores dentro del sólido (p. ej. una zapata por la mitad, para ver cómo baja la carga)")
        self.combo_eje = QComboBox()
        for texto, eje in (("Normal X (plano YZ)", "X"), ("Normal Y (plano XZ)", "Y"), ("Normal Z (plano XY, horizontal)", "Z"),
                           ("Libre (movido con el ratón)", "LIBRE")):
            self.combo_eje.addItem(texto, eje)
        self.slider_seccion = QSlider(Qt.Horizontal)
        self.slider_seccion.setRange(0, 1000)
        self.slider_seccion.setValue(500)
        self.slider_seccion.setToolTip("Posición del plano a lo largo del modelo")
        self.etiqueta_seccion = QLabel("")
        self.etiqueta_seccion.setWordWrap(True)
        self.boton_invertir = QPushButton("Invertir lado")
        self.boton_invertir.setCheckable(True)
        self.boton_invertir.setToolTip("Muestra la otra mitad del modelo")
        self.combo_modo_seccion = QComboBox()
        self.combo_modo_seccion.addItem("Modelo cortado (con la cara del corte)", False)
        self.combo_modo_seccion.addItem("Solo la cara del corte", True)
        self.check_seccion_raton = QCheckBox("Mover y girar el plano con el ratón")
        self.check_seccion_raton.setToolTip("Muestra el plano en el visor: arrastre su centro para moverlo y su normal "
                                            "(la flecha) para girarlo a cualquier orientación")

        self.boton_reporte = QPushButton("Exportar reporte (PDF / Word)…")
        self.boton_reporte.setToolTip("Reporte del análisis con la malla, los esfuerzos de Von Mises y la deformada de cada caso "
                                      "y combinación, en PDF o en Word")
        self.boton_volver = QPushButton("Volver al modelo (selección)")
        self.etiqueta_resumen = QLabel("Ejecute el análisis para ver los esfuerzos.")
        self.etiqueta_resumen.setWordWrap(True)

        self.tabla = QTableWidget(0, 3)
        self.tabla.setHorizontalHeaderLabels(["Barra", "Mínimo [MPa]", "Máximo [MPa]"])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla.setSelectionBehavior(QTableWidget.SelectRows)

        formulario = QFormLayout()
        formulario.addRow("Resultado:", self.combo_resultado)
        formulario.addRow("Colorear por:", self.combo_magnitud)
        formulario.addRow(self.check_escala_automatica)
        self.etiqueta_minimo, self.etiqueta_maximo = QLabel("Mínimo [MPa]:"), QLabel("Máximo [MPa]:")
        formulario.addRow(self.etiqueta_minimo, self.spin_minimo)
        formulario.addRow(self.etiqueta_maximo, self.spin_maximo)
        formulario.addRow("Colores:", self.combo_colormap)

        grupo = QGroupBox("Deformada")
        capa = QVBoxLayout(grupo)
        capa.addWidget(self.check_deformada)
        fila = QHBoxLayout()
        fila.addWidget(self.spin_escala_deformada, 1)
        fila.addWidget(self.boton_escala_auto)
        capa.addLayout(fila)
        capa.addWidget(self.check_original)
        botones = QHBoxLayout()
        botones.addWidget(self.boton_animar)
        botones.addWidget(self.boton_video)
        capa.addLayout(botones)

        grupo_seccion = QGroupBox("Sección")
        capa_s = QVBoxLayout(grupo_seccion)
        capa_s.addWidget(self.check_seccion)
        capa_s.addWidget(self.combo_eje)
        capa_s.addWidget(self.slider_seccion)
        capa_s.addWidget(self.etiqueta_seccion)
        fila_s = QHBoxLayout()
        fila_s.addWidget(self.combo_modo_seccion, 1)
        fila_s.addWidget(self.boton_invertir)
        capa_s.addLayout(fila_s)
        capa_s.addWidget(self.check_seccion_raton)
        self.grupo_seccion = grupo_seccion

        layout = QVBoxLayout(self)
        layout.addLayout(formulario)
        layout.addWidget(grupo)
        layout.addWidget(grupo_seccion)
        layout.addWidget(self.etiqueta_resumen)
        layout.addWidget(self.tabla)
        layout.addWidget(self.boton_reporte)
        layout.addWidget(self.boton_volver)

        self._actualizar_escala_habilitada()
        for senal in (
            self.combo_resultado.currentIndexChanged,
            self.combo_magnitud.currentIndexChanged,
            self.combo_colormap.currentIndexChanged,
            self.spin_minimo.editingFinished,
            self.spin_maximo.editingFinished,
        ):
            senal.connect(self.cambio)
        self.combo_magnitud.currentIndexChanged.connect(self._actualizar_unidades)
        self.check_escala_automatica.toggled.connect(self._al_cambiar_escala_automatica)
        self.boton_volver.clicked.connect(self.volver_al_modelo)
        self.check_deformada.toggled.connect(self._al_cambiar_deformada)
        self.check_original.toggled.connect(self.cambio)
        self.spin_escala_deformada.editingFinished.connect(self.cambio)
        self.boton_escala_auto.clicked.connect(self.escala_automatica_solicitada)
        self.boton_animar.toggled.connect(self._al_alternar_animacion)
        self.boton_video.clicked.connect(self.video_solicitado)
        self.boton_reporte.clicked.connect(self.reporte_solicitado)
        self._actualizar_deformada_habilitada()
        for senal in (self.check_seccion.toggled, self.combo_eje.currentIndexChanged, self.slider_seccion.valueChanged,
                      self.boton_invertir.toggled, self.combo_modo_seccion.currentIndexChanged, self.check_seccion_raton.toggled):
            senal.connect(self._al_cambiar_seccion)
        self._normal_libre = (1.0, 0.0, 0.0)
        self._actualizar_seccion_habilitada()

    @staticmethod
    def _spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(3)
        spin.setSingleStep(0.5)
        return spin

    def _al_cambiar_escala_automatica(self) -> None:
        self._actualizar_escala_habilitada()
        self.cambio.emit()

    def _al_cambiar_deformada(self) -> None:
        self._actualizar_deformada_habilitada()
        if not self.check_deformada.isChecked() and self.boton_animar.isChecked():
            self.boton_animar.setChecked(False)
        self.cambio.emit()

    def _al_alternar_animacion(self, activa: bool) -> None:
        self.boton_animar.setText("■ Detener" if activa else "▶ Animar")
        if activa and not self.check_deformada.isChecked():
            self.check_deformada.setChecked(True)  # muestra la deformada antes de animarla
        self.animacion_alternada.emit(activa)

    def _actualizar_deformada_habilitada(self) -> None:
        activa = self.check_deformada.isChecked()
        for w in (self.spin_escala_deformada, self.boton_escala_auto, self.check_original):
            w.setEnabled(activa)

    # -- sección -----------------------------------------------------------
    def _actualizar_seccion_habilitada(self) -> None:
        activa = self.check_seccion.isChecked()
        for w in (self.combo_eje, self.slider_seccion, self.boton_invertir, self.combo_modo_seccion, self.check_seccion_raton,
                  self.etiqueta_seccion):
            w.setEnabled(activa)

    def _al_cambiar_seccion(self, *_args) -> None:
        self._actualizar_seccion_habilitada()
        self.seccion_cambiada.emit()

    def seccion(self) -> Seccion:
        return Seccion(activa=self.check_seccion.isChecked(), eje=self.combo_eje.currentData(),
                       posicion=self.slider_seccion.value() / 1000.0, invertir=self.boton_invertir.isChecked(),
                       solo_cara=bool(self.combo_modo_seccion.currentData()), con_raton=self.check_seccion_raton.isChecked(),
                       normal_libre=self._normal_libre)

    def fijar_seccion(self, seccion: Seccion) -> None:
        """Muestra en los controles el plano que se movió con el ratón (sin emitir ``seccion_cambiada``)."""

        self._normal_libre = tuple(seccion.normal_libre)
        controles = (self.check_seccion, self.combo_eje, self.slider_seccion, self.boton_invertir, self.combo_modo_seccion,
                     self.check_seccion_raton)
        for w in controles:
            w.blockSignals(True)
        self.check_seccion.setChecked(seccion.activa)
        self.combo_eje.setCurrentIndex(max(0, self.combo_eje.findData(seccion.eje)))
        self.slider_seccion.setValue(int(round(seccion.posicion * 1000)))
        for w in controles:
            w.blockSignals(False)
        self._actualizar_seccion_habilitada()

    def mostrar_plano(self, texto: str) -> None:
        self.etiqueta_seccion.setText(texto)

    def fijar_escala_deformada(self, valor: float) -> None:
        self.spin_escala_deformada.blockSignals(True)
        self.spin_escala_deformada.setValue(valor)
        self.spin_escala_deformada.blockSignals(False)

    @property
    def unidad(self) -> str:
        clave = self.combo_magnitud.currentData()
        return "mm" if clave == DESPLAZAMIENTO else "%" if clave == DEFORMACION_PLASTICA else "MPa"

    def _actualizar_unidades(self) -> None:
        unidad = self.unidad
        self.etiqueta_minimo.setText(f"Mínimo [{unidad}]:")
        self.etiqueta_maximo.setText(f"Máximo [{unidad}]:")
        self.tabla.setHorizontalHeaderLabels(["Barra", f"Mínimo [{unidad}]", f"Máximo [{unidad}]"])

    def _actualizar_escala_habilitada(self) -> None:
        manual = not self.check_escala_automatica.isChecked()
        self.spin_minimo.setEnabled(manual)
        self.spin_maximo.setEnabled(manual)

    # ------------------------------------------------------------------
    def establecer_resultados(self, casos: List[Tuple[str, str]], combinaciones: List[Tuple[str, str]],
                              plastica: bool = False) -> None:
        """Opciones del combo: [(id, texto)] de casos y de combinaciones;
        la envolvente solo se ofrece si hay combinaciones. ``plastica``: hay deformaciones plásticas que mostrar."""

        indice = self.combo_magnitud.findData(DEFORMACION_PLASTICA)
        if plastica and indice < 0:
            self.combo_magnitud.addItem(TEXTO_PLASTICA, DEFORMACION_PLASTICA)
        elif not plastica and indice >= 0:
            self.combo_magnitud.blockSignals(True)
            if self.combo_magnitud.currentIndex() == indice:
                self.combo_magnitud.setCurrentIndex(0)
            self.combo_magnitud.removeItem(indice)
            self.combo_magnitud.blockSignals(False)
            self._actualizar_unidades()

        self.combo_resultado.blockSignals(True)
        self.combo_resultado.clear()
        if combinaciones:
            self.combo_resultado.addItem("ENVOLVENTE (todas las combinaciones)", ENVOLVENTE)
        for id_, texto in combinaciones:
            self.combo_resultado.addItem(f"Combinación {id_}: {texto}", id_)
        for id_, texto in casos:
            self.combo_resultado.addItem(f"Caso {id_}: {texto}", id_)
        self.combo_resultado.blockSignals(False)

    def parametros(self) -> Optional[ParametrosVista]:
        if self.combo_resultado.count() == 0:
            return None
        return ParametrosVista(
            resultado=self.combo_resultado.currentData(),
            magnitud=self.combo_magnitud.currentData(),
            escala_automatica=self.check_escala_automatica.isChecked(),
            minimo=self.spin_minimo.value(),
            maximo=self.spin_maximo.value(),
            colormap=self.combo_colormap.currentText(),
            deformada=self.check_deformada.isChecked(),
            escala_deformada=self.spin_escala_deformada.value(),
            mostrar_original=self.check_original.isChecked(),
        )

    def fijar_escala(self, minimo: float, maximo: float) -> None:
        """Muestra la escala automática en los cuadros (sin disparar ``cambio``)."""

        for spin, valor in ((self.spin_minimo, minimo), (self.spin_maximo, maximo)):
            spin.blockSignals(True)
            spin.setValue(valor)
            spin.blockSignals(False)

    def mostrar_resumen(self, texto: str, filas: Dict[str, Tuple[float, float]]) -> None:
        self.etiqueta_resumen.setText(texto)
        self.tabla.setRowCount(0)
        for barra, (minimo, maximo) in sorted(filas.items()):
            fila = self.tabla.rowCount()
            self.tabla.insertRow(fila)
            self.tabla.setItem(fila, 0, QTableWidgetItem(barra))
            self.tabla.setItem(fila, 1, QTableWidgetItem(f"{minimo:.3f}"))
            self.tabla.setItem(fila, 2, QTableWidgetItem(f"{maximo:.3f}"))
