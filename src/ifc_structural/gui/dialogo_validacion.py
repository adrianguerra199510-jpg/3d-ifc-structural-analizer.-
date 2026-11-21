"""Informe de la validación del modelo (botón Validar, y antes de ANALIZAR si hay errores)."""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout)

from ..core.asignacion import Objetivo
from ..solido.validacion_modelo import AVISO, CORRECTO, ERROR, InformeValidacion
from .estilo import EncabezadoDialogo
from .iconos import icono

ICONO_NIVEL = {ERROR: "quitar", AVISO: "advertencia", CORRECTO: "correcto"}
COLOR_NIVEL = {ERROR: "#b3261e", AVISO: "#9a6700", CORRECTO: "#1e7a3c"}
ORDEN_SECCIONES = ("Datos", "Materiales", "Geometría", "Conectividad", "Estabilidad", "Cargas", "Malla", "CalculiX",
                   "Modelo de barras")

#: Lo que el usuario eligió al cerrar el informe.
CERRAR, PRUEBA, ANALIZAR = "CERRAR", "PRUEBA", "ANALIZAR"


class DialogoValidacion(QDialog):
    """Hallazgos agrupados por sección (✗ errores, ⚠ avisos, ✓ correcto). Doble clic en un hallazgo que nombra
    elementos los selecciona en el visor (``seleccion_pedida``)."""

    seleccion_pedida = Signal(list)  # [Objetivo]

    def __init__(self, informe: InformeValidacion, parent=None, antes_de_analizar: bool = False,
                 ofrecer_prueba: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Validación del modelo")
        self.resize(780, 560)
        self.eleccion = CERRAR
        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo(
            "correcto", "Validación del modelo",
            "Comprueba, antes de analizar, si el análisis va a correr y a converger."))

        resumen = QLabel(informe.resumen())
        resumen.setWordWrap(True)
        nivel = ERROR if informe.errores else AVISO if informe.avisos else CORRECTO
        resumen.setStyleSheet(f"font-weight: 700; color: {COLOR_NIVEL[nivel]};")
        layout.addWidget(resumen)

        self.arbol = QTreeWidget()
        self.arbol.setHeaderHidden(True)
        self.arbol.setWordWrap(True)
        self.arbol.setUniformRowHeights(False)
        secciones = sorted({h.seccion for h in informe.hallazgos},
                           key=lambda s: ORDEN_SECCIONES.index(s) if s in ORDEN_SECCIONES else 99)
        for seccion in secciones:
            hallazgos = [h for h in informe.hallazgos if h.seccion == seccion]
            peor = ERROR if any(h.nivel == ERROR for h in hallazgos) else AVISO if any(h.nivel == AVISO for h in hallazgos) else CORRECTO
            raiz = QTreeWidgetItem([seccion])
            raiz.setIcon(0, icono(ICONO_NIVEL[peor]))
            fuente = raiz.font(0)
            fuente.setBold(True)
            raiz.setFont(0, fuente)
            self.arbol.addTopLevelItem(raiz)
            for h in sorted(hallazgos, key=lambda h: (ERROR, AVISO, CORRECTO).index(h.nivel)):
                item = QTreeWidgetItem([h.texto + ("   (doble clic: seleccionar)" if h.objetivos else "")])
                item.setIcon(0, icono(ICONO_NIVEL[h.nivel]))
                item.setData(0, Qt.UserRole, h.objetivos)
                item.setToolTip(0, h.texto)
                raiz.addChild(item)
            raiz.setExpanded(peor != CORRECTO or len(secciones) <= 4)
        self.arbol.itemDoubleClicked.connect(self._al_doble_clic)
        layout.addWidget(self.arbol, 1)

        if not informe.prueba_ccx and not informe.errores:
            nota = QLabel("La <b>prueba con CalculiX</b> hace el análisis elástico completo: confirma que el modelo se "
                          "resuelve y, con materiales con fluencia, estima si el análisis elastoplástico convergerá. "
                          "Tarda lo mismo que un análisis elástico.")
            nota.setWordWrap(True)
            layout.addWidget(nota)

        botones = QDialogButtonBox()
        self.boton_prueba = None
        if ofrecer_prueba and not informe.prueba_ccx and not informe.errores:
            self.boton_prueba = QPushButton("Prueba con CalculiX")
            botones.addButton(self.boton_prueba, QDialogButtonBox.ActionRole)
            self.boton_prueba.clicked.connect(lambda: self._terminar(PRUEBA))
        texto_analizar = ("Analizar de todos modos" if informe.errores else "Analizar") if antes_de_analizar else (
            None if informe.errores else "Analizar")
        self.boton_analizar = None
        if texto_analizar:
            self.boton_analizar = QPushButton(texto_analizar)
            botones.addButton(self.boton_analizar, QDialogButtonBox.AcceptRole)
            self.boton_analizar.clicked.connect(lambda: self._terminar(ANALIZAR))
        cerrar = botones.addButton("Cerrar" if not antes_de_analizar else "Cancelar", QDialogButtonBox.RejectRole)
        cerrar.clicked.connect(lambda: self._terminar(CERRAR))
        layout.addWidget(botones)

    def _terminar(self, eleccion: str) -> None:
        self.eleccion = eleccion
        (self.accept if eleccion != CERRAR else self.reject)()

    def _al_doble_clic(self, item: QTreeWidgetItem, _columna: int) -> None:
        objetivos: List[Objetivo] = item.data(0, Qt.UserRole) or []
        if objetivos:
            self.seleccion_pedida.emit(list(objetivos))
