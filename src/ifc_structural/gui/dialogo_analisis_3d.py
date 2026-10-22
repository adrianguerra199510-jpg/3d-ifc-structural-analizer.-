"""Diálogo del análisis como modelo sólido (3D) y del botón Mallar: tamaño de elemento y estimación."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout

from ..core.modelo import Proyecto
from ..solido.analisis_solido import ELEMENTOS_CALCULO_LARGO, estimar, memoria_disponible, memoria_solver, tamano_sugerido
from .estilo import EncabezadoDialogo


class DialogoAnalisis3D(QDialog):
    def __init__(self, proyecto: Proyecto, parent=None, motivo: str = "", para_mallar: bool = False,
                 tamano_inicial: Optional[float] = None):
        super().__init__(parent)
        self.setWindowTitle("Mallar el modelo sólido (3D)" if para_mallar else "Analizar como modelo sólido (3D)")
        self.resize(560, 420)
        self._proyecto = proyecto

        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo(
            "malla" if para_mallar else "solido", "Malla del modelo sólido (3D)" if para_mallar else "Análisis sólido (3D)",
            "Cada pieza se malla con su geometría real; ANALIZAR usará esta misma malla mientras no cambie el modelo."
            if para_mallar else "Cada pieza se malla con su geometría real y se resuelve con CalculiX."))
        if motivo:
            aviso = QLabel(f"Se analiza como <b>sólido 3D</b> porque {motivo}.")
            aviso.setWordWrap(True)
            layout.addWidget(aviso)
        explicacion = QLabel(
            "• Elementos: <b>tetraedros cuadráticos C3D10</b> en todas las piezas macizas (también vigas, columnas, losas "
            "y muros gruesos) y <b>láminas S6</b> (cáscaras en la superficie media, con su espesor) en las piezas que son "
            "láminas: espesor ≤ 1/20 de su tamaño en planta (chapas, paredes de depósitos, losas y muros delgados).<br>"
            "• Las vigas y columnas se mallan con su forma real (un perfil en I tiene alas y alma); las piezas se unen "
            "por las superficies que se tocan, como una soldadura continua.<br>"
            "• Los apoyos y las cargas asignados se aplican sobre la <b>cara o la sección</b> elegida.<br>"
            "• No incluye pretensión de pernos ni contacto con separación o fricción."
        )
        explicacion.setWordWrap(True)
        layout.addWidget(explicacion)

        formulario = QFormLayout()
        self.spin_tamano = QDoubleSpinBox()
        self.spin_tamano.setRange(5.0, 500.0)
        self.spin_tamano.setDecimals(0)
        self.spin_tamano.setSuffix(" mm")
        if tamano_inicial is None:
            tamano_inicial = tamano_sugerido(proyecto, elemento_solido=self.elemento())
        self.spin_tamano.setValue(round(tamano_inicial * 1000.0))
        formulario.addRow("Tamaño de elemento:", self.spin_tamano)
        layout.addLayout(formulario)

        self.estimacion = QLabel()
        self.estimacion.setWordWrap(True)
        layout.addWidget(self.estimacion)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.button(QDialogButtonBox.Ok).setText("Mallar" if para_mallar else "Analizar en 3D")
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)

        self._libre = memoria_disponible()
        self.spin_tamano.valueChanged.connect(self._actualizar)
        self._actualizar()

    def elemento(self) -> str:
        """Elemento de los sólidos: siempre tetraedros C3D10 (las láminas, S6)."""

        return "TETRA"

    def tamano(self) -> float:
        """Tamaño de elemento en metros."""

        return self.spin_tamano.value() / 1000.0

    def _actualizar(self) -> None:
        n = estimar(self._proyecto, self.tamano(), self.elemento())
        memoria = memoria_solver(n)
        texto = f"Unos <b>{n:,}</b> elementos (estimación aproximada), unos {memoria / 1e9:.1f} GB de memoria".replace(",", " ")
        if self._libre is not None and memoria > 0.8 * self._libre:
            texto += (f": hay {self._libre / 1e9:.1f} GB libres, CalculiX puede quedarse sin memoria o ir muy lento. Cierre otros "
                      "programas o aumente el tamaño de elemento.")
            color = "#b3261e"
        elif n > ELEMENTOS_CALCULO_LARGO:
            texto += ": el cálculo puede tardar varios minutos."
            color = "#9a6700"
        else:
            texto += ": el cálculo tardará de segundos a pocos minutos."
            color = "#1a7f4b"
        self.estimacion.setText(f'<span style="color:{color}">{texto}</span>')
