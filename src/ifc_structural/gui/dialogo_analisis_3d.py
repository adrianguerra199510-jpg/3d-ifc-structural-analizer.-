"""Diálogo del análisis como modelo sólido (3D): tamaño de elemento y estimación."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout

from ..core.modelo import Proyecto
from ..solido.analisis_solido import MAXIMO_HEXAEDROS, elemento_solido_por_defecto, estimar, tamano_sugerido
from .estilo import EncabezadoDialogo


class DialogoAnalisis3D(QDialog):
    def __init__(self, proyecto: Proyecto, parent=None, motivo: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Analizar como modelo sólido (3D)")
        self.resize(560, 420)
        self._proyecto = proyecto

        layout = QVBoxLayout(self)
        layout.addWidget(EncabezadoDialogo(
            "solido", "Análisis sólido (3D)",
            "Cada pieza se malla con su geometría real y se resuelve con CalculiX."))
        if motivo:
            aviso = QLabel(f"Se analiza como <b>sólido 3D</b> porque {motivo}.")
            aviso.setWordWrap(True)
            layout.addWidget(aviso)
        explicacion = QLabel(
            "• Las vigas, columnas, losas y muros se mallan como <b>sólidos</b> con su forma real (un perfil en I "
            "tiene alas y alma), y se unen por las superficies que se tocan, como una soldadura continua.<br>"
            "• Los apoyos y las cargas asignados se aplican sobre la <b>cara o la sección</b> elegida.<br>"
            "• Las placas, pernos y demás piezas del IFC se mallan como sólidos con su geometría real.<br>"
            "• Se ven esfuerzos locales (aletas, alma, uniones) que el modelo de barras no muestra. "
            "No incluye pretensión de pernos ni contacto con separación o fricción."
        )
        explicacion.setWordWrap(True)
        layout.addWidget(explicacion)

        formulario = QFormLayout()
        self.combo_elemento = QComboBox()
        self.combo_elemento.addItem("Tetraedros cuadráticos C3D10 (Gmsh)", "TETRA")
        self.combo_elemento.addItem("Hexaedros C3D8I (voxelización)", "HEXA")
        self.combo_elemento.setCurrentIndex(self.combo_elemento.findData(elemento_solido_por_defecto()))
        self.combo_elemento.setToolTip("Elemento finito de los sólidos libres (las barras, losas y muros llevan hexaedros C3D8I). "
                                       "Los tetraedros cuadráticos siguen las superficies curvas e inclinadas.")
        formulario.addRow("Elemento de los sólidos:", self.combo_elemento)
        self.spin_tamano = QDoubleSpinBox()
        self.spin_tamano.setRange(5.0, 500.0)
        self.spin_tamano.setDecimals(0)
        self.spin_tamano.setSuffix(" mm")
        self.spin_tamano.setValue(round(tamano_sugerido(proyecto, elemento_solido=self.elemento()) * 1000.0))
        formulario.addRow("Tamaño de elemento:", self.spin_tamano)
        layout.addLayout(formulario)

        self.estimacion = QLabel()
        self.estimacion.setWordWrap(True)
        layout.addWidget(self.estimacion)

        self.botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.botones.button(QDialogButtonBox.Ok).setText("Analizar en 3D")
        self.botones.accepted.connect(self.accept)
        self.botones.rejected.connect(self.reject)
        layout.addWidget(self.botones)

        self.spin_tamano.valueChanged.connect(self._actualizar)
        self.combo_elemento.currentIndexChanged.connect(self._actualizar)
        self._actualizar()

    def elemento(self) -> str:
        """``"TETRA"`` (C3D10) o ``"HEXA"`` (C3D8I): elemento de los sólidos libres."""

        return self.combo_elemento.currentData()

    def tamano(self) -> float:
        """Tamaño de elemento en metros."""

        return self.spin_tamano.value() / 1000.0

    def _actualizar(self) -> None:
        n = estimar(self._proyecto, self.tamano(), self.elemento())
        texto = f"Unos <b>{n:,}</b> elementos (estimación aproximada)".replace(",", " ")
        if n > MAXIMO_HEXAEDROS:
            texto += f" — demasiados (máximo {MAXIMO_HEXAEDROS:,}). Aumente el tamaño.".replace(",", " ")
            color = "#b3261e"
        elif n > 150_000:
            texto += ": el cálculo puede tardar varios minutos y usar varios GB de memoria."
            color = "#9a6700"
        else:
            texto += ": el cálculo tardará de segundos a pocos minutos."
            color = "#1a7f4b"
        self.estimacion.setText(f'<span style="color:{color}">{texto}</span>')
        self.botones.button(QDialogButtonBox.Ok).setEnabled(n <= MAXIMO_HEXAEDROS)
