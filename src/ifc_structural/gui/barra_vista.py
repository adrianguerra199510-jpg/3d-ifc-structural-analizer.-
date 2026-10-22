"""Barra sobre el visor 3D: vistas de cámara, visibilidad por tipo de
elemento y leyenda de colores."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QToolButton, QWidget

from .iconos import icono
from .visor_3d import COLORES_BASE

VISTAS = (
    ("vista_iso", "Iso", "Vista isométrica"),
    ("vista_planta", "Planta", "Vista en planta (desde arriba)"),
    ("vista_frontal", "Frontal", "Vista frontal"),
    ("vista_lateral", "Lateral", "Vista lateral"),
    ("vista_inferior", "Inferior", "Vista inferior: permite hacer clic en la base de las columnas"),
    ("ajustar", "Ajustar", "Encuadrar todo el modelo"),
)
TIPOS = (("VIGA", "Vigas", "viga"), ("COLUMNA", "Columnas", "columna"), ("LOSA", "Losas", "losa"), ("MURO", "Muros", "muro"),
         ("SOLIDO", "Sólidos", "solido"))


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


class BarraVista(QWidget):
    vista_solicitada = Signal(str)  # "vista_iso", "vista_planta", ..., "ajustar"
    visibilidad_cambiada = Signal(str, bool)  # tipo, visible
    flechas_visibles = Signal(bool)  # mostrar u ocultar las flechas de las cargas
    apoyos_visibles = Signal(bool)  # mostrar u ocultar los apoyos
    uniones_visibles = Signal(bool)  # mostrar u ocultar los constraints
    navegacion_cambiada = Signal(str)  # qué hace arrastrar con el botón izquierdo: SELECCION, MOVER o ROTAR

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("barraVista")
        fila = QHBoxLayout(self)
        fila.setContentsMargins(10, 4, 10, 4)
        fila.setSpacing(2)

        self.botones_vista = {}
        self._con_texto = []  # botones con nombre: lo pierden (queda el icono y la ayuda) si la barra no cabe
        self._anchos = None  # (con texto, solo iconos)
        for clave, texto, ayuda in VISTAS:
            boton = QToolButton()
            boton.setText(texto)
            boton.setIcon(icono(clave))
            boton.setToolTip(ayuda)
            boton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            boton.clicked.connect(lambda _=False, c=clave: self.vista_solicitada.emit(c))
            fila.addWidget(boton)
            self.botones_vista[clave] = boton
            self._con_texto.append(boton)

        separador = QFrame()
        separador.setFrameShape(QFrame.VLine)
        separador.setStyleSheet("color: #d3dae3;")
        fila.addSpacing(6)
        fila.addWidget(separador)
        fila.addSpacing(6)

        etiqueta = QLabel("Mostrar:")
        etiqueta.setObjectName("chipLeyenda")
        fila.addWidget(etiqueta)
        self.botones_tipo = {}
        for tipo, texto, nombre_icono in TIPOS:
            boton = QToolButton()
            boton.setCheckable(True)
            boton.setChecked(True)
            color = _hex(COLORES_BASE[tipo])  # el icono lleva el color del elemento: hace de leyenda
            boton.setIcon(icono(nombre_icono, color, color))
            boton.setToolTip(f"Mostrar u ocultar {texto.lower()}")
            boton.setToolButtonStyle(Qt.ToolButtonIconOnly)
            boton.toggled.connect(lambda visible, t=tipo: self.visibilidad_cambiada.emit(t, visible))
            fila.addWidget(boton)
            self.botones_tipo[tipo] = boton

        separador = QFrame()
        separador.setFrameShape(QFrame.VLine)
        separador.setStyleSheet("color: #d3dae3;")
        fila.addSpacing(6)
        fila.addWidget(separador)
        fila.addSpacing(6)
        etiqueta_nav = QLabel("Ratón:")
        etiqueta_nav.setObjectName("chipLeyenda")
        fila.addWidget(etiqueta_nav)
        self.grupo_navegacion = QButtonGroup(self)
        self.grupo_navegacion.setExclusive(True)
        self.botones_navegacion = {}
        for modo, texto, nombre_icono, ayuda in (
            ("SELECCION", "Seleccionar", "zona",
             "Arrastrar con el botón izquierdo selecciona: de izquierda a derecha lo que queda dentro del rectángulo, "
             "de derecha a izquierda todo lo que toca (Ctrl o Mayús suma a la selección)."),
            ("MOVER", "Mover", "mover", "Arrastrar con el botón izquierdo desplaza la vista (también: botón central)."),
            ("ROTAR", "Rotar", "rotar", "Arrastrar con el botón izquierdo gira la vista (también: Ctrl o Mayús + botón central)."),
        ):
            boton = QToolButton()
            boton.setIcon(icono(nombre_icono))
            boton.setCheckable(True)
            boton.setChecked(modo == "SELECCION")
            boton.setToolButtonStyle(Qt.ToolButtonIconOnly)
            boton.setToolTip(f"{texto}: {ayuda}\nRueda del ratón: zoom.")
            boton.toggled.connect(lambda activo, m=modo: activo and self.navegacion_cambiada.emit(m))
            self.grupo_navegacion.addButton(boton)
            fila.addWidget(boton)
            self.botones_navegacion[modo] = boton
        separador = QFrame()
        separador.setFrameShape(QFrame.VLine)
        separador.setStyleSheet("color: #d3dae3;")
        fila.addSpacing(6)
        fila.addWidget(separador)
        fila.addSpacing(6)
        # Interruptores solo con el símbolo (el nombre va en la ayuda); el tamaño se cambia en Ver → Escala de símbolos
        def interruptor(nombre_icono: str, ayuda: str, senal) -> QToolButton:
            boton = QToolButton()
            boton.setIcon(icono(nombre_icono))
            boton.setCheckable(True)
            boton.setChecked(True)
            boton.setToolButtonStyle(Qt.ToolButtonIconOnly)
            boton.setToolTip(ayuda)
            boton.toggled.connect(senal.emit)
            fila.addWidget(boton)
            return boton

        self.boton_apoyos = interruptor("apoyo", "Mostrar u ocultar los apoyos", self.apoyos_visibles)
        self.boton_flechas = interruptor("carga", "Mostrar u ocultar las cargas", self.flechas_visibles)
        self.boton_uniones = interruptor("conectar", "Mostrar u ocultar los constraints", self.uniones_visibles)
        fila.addStretch(1)
        self._fila = fila

    def agregar_accion(self, accion) -> None:
        """Añade, al extremo derecho, un botón ligado a una acción (p. ej. Modelo / Esfuerzos)."""

        boton = QToolButton()
        boton.setDefaultAction(accion)
        boton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._fila.addWidget(boton)
        self._con_texto.append(boton)
        self._anchos = None

    # La barra no debe imponer su ancho a la ventana: con los paneles laterales abiertos, los nombres de los botones
    # la hacían más ancha que una pantalla de 1920 px (la ventana maximizada quedaba recortada a la derecha).
    def _estilo(self, compacto: bool) -> None:
        estilo = Qt.ToolButtonIconOnly if compacto else Qt.ToolButtonTextBesideIcon
        for boton in self._con_texto:
            if boton.toolButtonStyle() != estilo:
                boton.setToolButtonStyle(estilo)

    def _medir(self):
        if self._anchos is None:
            actual = bool(self._con_texto) and self._con_texto[0].toolButtonStyle() == Qt.ToolButtonIconOnly
            anchos = []
            for compacto in (False, True):
                self._estilo(compacto)
                self._fila.invalidate()
                anchos.append(self._fila.sizeHint().width())
            self._estilo(actual)
            self._fila.invalidate()
            self._anchos = tuple(anchos)
        return self._anchos

    @property
    def compacta(self) -> bool:
        return bool(self._con_texto) and self._con_texto[0].toolButtonStyle() == Qt.ToolButtonIconOnly

    def minimumSizeHint(self) -> QSize:
        return QSize(self._medir()[1], super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        return QSize(self._medir()[0], super().sizeHint().height())

    def resizeEvent(self, evento) -> None:
        self._estilo(evento.size().width() < self._medir()[0])
        super().resizeEvent(evento)

    def fijar_visibilidad(self, tipo: str, visible: bool) -> None:
        self.botones_tipo[tipo].blockSignals(True)
        self.botones_tipo[tipo].setChecked(visible)
        self.botones_tipo[tipo].blockSignals(False)
