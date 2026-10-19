"""Panel único del modelo, en forma de árbol.

Reúne en un solo lugar todo lo que hay en el proyecto:

* **Elementos** (vigas, columnas, losas, muros), con las asignaciones que
  tiene cada uno colgando de él.
* **Apoyos**, **Conexiones**, **Cargas** (agrupadas por caso de carga).
* **Materiales**, **Casos de carga** y **Combinaciones**.
* **Resultados** (envolvente, combinaciones y casos) cuando ya se analizó.

Seleccionar un elemento (o una asignación) en el árbol lo selecciona en el
visor y viceversa; el clic derecho abre el menú de lo que se puede hacer con
cada cosa y *Supr* elimina las asignaciones seleccionadas. Hay un buscador
para filtrar por texto.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLineEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.cargas import ApoyoArea, CargaArea, CargaLineal, CargaPuntual, Conexion
from ..core.casos_carga import CasoCarga, Combinacion
from ..core.materiales import modulo_en_mpa, usos_de_material
from ..core.modelo import Proyecto
from .dialogo_casos import formula
from .iconos import ICONO_DE_TIPO, icono
from .panel_resultados import ENVOLVENTE

NOMBRES_GDL = {1: "Ux", 2: "Uy", 3: "Uz", 4: "Rx", 5: "Ry", 6: "Rz"}
ABIERTOS_POR_DEFECTO = {
    ("PROYECTO",), ("GRUPO", "Elementos"), ("GRUPO", "Apoyos"), ("GRUPO", "Cargas"), ("GRUPO", "Resultados"),
}
TIPOS_ELEMENTO = (("VIGA", "Vigas", "viga"), ("COLUMNA", "Columnas", "columna"), ("LOSA", "Losas", "losa"), ("MURO", "Muros", "muro"))


def tipo_de_asignacion(obj) -> str:
    if isinstance(obj, ApoyoArea):
        return "Apoyo de área"
    if isinstance(obj, Conexion):
        return "Conexión"
    if isinstance(obj, CargaArea):
        return "Carga de área"
    if isinstance(obj, CargaLineal):
        return "Carga lineal"
    if isinstance(obj, CargaPuntual):
        return "Carga puntual"
    return "Apoyo"


def _gdl(grados: Iterable[int]) -> str:
    grados = list(grados)
    if grados == [1, 2, 3, 4, 5, 6]:
        return "empotrado"
    if grados == [1, 2, 3]:
        return "articulado"
    return " ".join(NOMBRES_GDL[g] for g in grados)


def _partir_nombre(nombre: str) -> Tuple[str, str]:
    """«Viga B2: triangular 25 kN/m» -> («Viga B2», «triangular 25 kN/m»)."""

    if ": " in nombre:
        cabeza, cola = nombre.split(": ", 1)
        return cabeza, cola
    return nombre, ""


class PanelArbol(QWidget):
    elementos_seleccionados = Signal(list)  # [(tipo, id)]: lo elegido en el árbol, resuelto a elementos
    eliminar_solicitado = Signal(list)  # [(tipo de asignación, id)]
    menu_solicitado = Signal(list)  # claves de lo seleccionado, para el menú contextual
    resultado_solicitado = Signal(str)  # id de caso/combinación o ENVOLVENTE
    editar_materiales = Signal()
    editar_casos = Signal()
    propiedades_solicitadas = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._silencio = False
        self._abiertos: Set[tuple] = set()
        self._cerrados: Set[tuple] = set()
        self._destinos: Dict[tuple, List[Tuple[str, str]]] = {}

        self.buscador = QLineEdit()
        self.buscador.setPlaceholderText("Buscar: B1, apoyo, carga, acero…")
        self.buscador.setClearButtonEnabled(True)
        self.buscador.addAction(icono("buscar"), QLineEdit.LeadingPosition)
        self.buscador.textChanged.connect(self._filtrar)

        self.boton_expandir = QToolButton()
        self.boton_expandir.setIcon(icono("expandir"))
        self.boton_expandir.setToolTip("Expandir todo")
        self.boton_expandir.setAutoRaise(True)
        self.boton_expandir.clicked.connect(lambda: self.arbol.expandAll())
        self.boton_contraer = QToolButton()
        self.boton_contraer.setIcon(icono("contraer"))
        self.boton_contraer.setToolTip("Contraer todo (deja los grupos principales)")
        self.boton_contraer.setAutoRaise(True)
        self.boton_contraer.clicked.connect(self._contraer)

        cabecera = QHBoxLayout()
        cabecera.setContentsMargins(8, 8, 8, 4)
        cabecera.setSpacing(2)
        cabecera.addWidget(self.buscador, 1)
        cabecera.addWidget(self.boton_expandir)
        cabecera.addWidget(self.boton_contraer)

        self.arbol = QTreeWidget()
        self.arbol.setColumnCount(2)
        self.arbol.setHeaderLabels(["Modelo", "Detalle"])
        self.arbol.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.arbol.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.arbol.setIconSize(QSize(18, 18))
        self.arbol.setIndentation(16)
        self.arbol.setAnimated(True)
        self.arbol.setUniformRowHeights(True)
        self.arbol.setAlternatingRowColors(False)
        self.arbol.setTextElideMode(Qt.ElideRight)
        self.arbol.setContextMenuPolicy(Qt.CustomContextMenu)
        self.arbol.header().setStretchLastSection(True)
        self.arbol.header().resizeSection(0, 200)
        self.arbol.setHeaderHidden(True)
        self.arbol.setStyleSheet(
            "QTreeWidget { border: none; background: #ffffff; }"
            "QTreeWidget::item { padding: 3px 2px; }"
            "QTreeWidget::item:selected { background: #cfe2ff; color: #1d2b3a; }"
            "QTreeWidget::item:hover:!selected { background: #f1f5fb; }"
        )
        self.arbol.itemSelectionChanged.connect(self._al_cambiar_seleccion)
        self.arbol.itemExpanded.connect(lambda item: self._recordar(item, True))
        self.arbol.itemCollapsed.connect(lambda item: self._recordar(item, False))
        self.arbol.itemDoubleClicked.connect(self._al_doble_clic)
        self.arbol.customContextMenuRequested.connect(self._al_menu)

        capa = QVBoxLayout(self)
        capa.setContentsMargins(0, 0, 0, 0)
        capa.setSpacing(0)
        capa.addLayout(cabecera)
        capa.addWidget(self.arbol, 1)

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------
    def _nuevo(self, padre, texto: str, detalle: str = "", nombre_icono: Optional[str] = None,
               clave: tuple = (), ayuda: str = "", negrita: bool = False, destinos=None) -> QTreeWidgetItem:
        item = QTreeWidgetItem(padre, [texto, detalle])
        item.setData(0, Qt.UserRole, clave)
        if nombre_icono:
            item.setIcon(0, icono(nombre_icono))
        if ayuda:
            item.setToolTip(0, ayuda)
            item.setToolTip(1, ayuda)
        if negrita:
            fuente = QFont(item.font(0))
            fuente.setBold(True)
            item.setFont(0, fuente)
        item.setForeground(1, QBrush(QColor("#5b6b7c")))
        if destinos:
            self._destinos[clave] = list(destinos)
        return item

    def actualizar(
        self,
        proyecto: Proyecto,
        apoyos: Sequence,
        conexiones: Sequence[Conexion],
        casos: Dict[str, CasoCarga],
        combinaciones: Dict[str, Combinacion],
        cargas_area: Dict[str, List[CargaArea]],
        cargas_lineales: Dict[str, List[CargaLineal]],
        cargas_puntuales: Dict[str, List[CargaPuntual]],
        resultado=None,
        desactualizado: bool = False,
    ) -> None:
        """Reconstruye el árbol conservando lo expandido y lo seleccionado."""

        seleccionadas = {self._clave(i) for i in self.arbol.selectedItems()}
        self._silencio = True
        self.arbol.clear()
        self._destinos = {}

        # Todas las asignaciones, por elemento al que pertenecen
        por_elemento: Dict[Tuple[str, str], List[tuple]] = {}

        def adjuntar(tipo: str, id_: str, obj, caso: str = "") -> None:
            por_elemento.setdefault((tipo, id_), []).append((obj, caso))

        barras = proyecto.todas_las_barras()
        tipo_de_barra = {i: ("VIGA" if i in proyecto.vigas else "COLUMNA") for i in barras}
        for obj in apoyos:
            if isinstance(obj, ApoyoArea):
                adjuntar(obj.elemento_tipo, obj.elemento_id, obj)
        for c in conexiones:
            if c.barra_id in tipo_de_barra:
                adjuntar(tipo_de_barra[c.barra_id], c.barra_id, c)
        for caso_id in casos:
            for c in cargas_lineales.get(caso_id, []):
                if c.elemento_id in tipo_de_barra:
                    adjuntar(tipo_de_barra[c.elemento_id], c.elemento_id, c, caso_id)
            for c in cargas_puntuales.get(caso_id, []):
                if c.elemento_id in tipo_de_barra:
                    adjuntar(tipo_de_barra[c.elemento_id], c.elemento_id, c, caso_id)
            for c in cargas_area.get(caso_id, []):
                if c.losa_id:
                    adjuntar("LOSA", c.losa_id, c, caso_id)

        def destino_de(obj) -> List[Tuple[str, str]]:
            if isinstance(obj, ApoyoArea):
                return [(obj.elemento_tipo, obj.elemento_id)]
            if isinstance(obj, Conexion):
                return [(tipo_de_barra[obj.barra_id], obj.barra_id)] if obj.barra_id in tipo_de_barra else []
            if isinstance(obj, CargaArea):
                return [("LOSA", obj.losa_id)] if obj.losa_id else []
            if isinstance(obj, (CargaLineal, CargaPuntual)) and obj.elemento_id in tipo_de_barra:
                return [(tipo_de_barra[obj.elemento_id], obj.elemento_id)]
            nodo = getattr(obj, "nodo_id", None)  # apoyos/cargas en un nodo: las barras que llegan a él
            return [(tipo_de_barra[i], i) for i, b in barras.items() if nodo in (b.nodo_inicio, b.nodo_fin)]

        def fila_de(obj, caso: str = "", bajo_elemento: bool = False) -> Tuple[str, str]:
            if isinstance(obj, ApoyoArea):
                return (("Apoyo de área" if bajo_elemento else obj.nombre.split(": ")[0]),
                        f"{_gdl(obj.grados_restringidos())}")
            if isinstance(obj, Conexion):
                modo = "rígida" if obj.es_enlace_rigido else ("articulación" if not obj.transmite_giros else "continua")
                return ("Conexión " + modo if bajo_elemento else obj.nombre.split(" en ")[0].split(":")[0]), \
                    f"{modo} · {_gdl(obj.grados_transmitidos())}"
            if hasattr(obj, "grados_restringidos"):  # apoyo nodal
                return f"Nodo {obj.nodo_id}", _gdl(obj.grados_restringidos())
            cabeza, cola = _partir_nombre(obj.nombre)
            if bajo_elemento:  # el elemento ya es el padre: se muestra qué carga es y a qué caso pertenece
                return (cola or cabeza), caso
            return cabeza, (f"{caso} · {cola}" if caso else cola)

        def crear_asignacion(padre, obj, caso: str = "", bajo_elemento: bool = False) -> QTreeWidgetItem:
            tipo = tipo_de_asignacion(obj)
            texto, detalle = fila_de(obj, caso, bajo_elemento)
            return self._nuevo(
                padre, texto, detalle, ICONO_DE_TIPO.get(tipo, "apoyo"), ("ASIGNACION", tipo, obj.id),
                ayuda=obj.nombre, destinos=destino_de(obj),
            )

        # -- raíz --------------------------------------------------------------
        raiz = self._nuevo(
            self.arbol, proyecto.nombre or "Proyecto",
            f"{len(barras)} barras · {len(proyecto.losas)} losas · {len(proyecto.muros)} muros",
            "modelo", ("PROYECTO",), negrita=True,
        )

        # -- elementos ------------------------------------------------------------
        grupo = self._nuevo(raiz, "Elementos", "", "modelo", ("GRUPO", "Elementos"), negrita=True)
        for tipo, titulo, nombre_icono in TIPOS_ELEMENTO:
            coleccion = {"VIGA": proyecto.vigas, "COLUMNA": proyecto.columnas, "LOSA": proyecto.losas, "MURO": proyecto.muros}[tipo]
            sub = self._nuevo(grupo, f"{titulo} ({len(coleccion)})", "", nombre_icono, ("GRUPO", titulo))
            for id_, elemento in coleccion.items():
                if tipo in ("VIGA", "COLUMNA"):
                    perfil = proyecto.perfiles.get(elemento.perfil)
                    seccion = (
                        f"{perfil.ancho * 100:.0f}×{perfil.altura * 100:.0f} cm"
                        if perfil is not None and perfil.ancho and perfil.altura else elemento.perfil
                    )
                    detalle = f"{seccion} · {elemento.material}"
                else:
                    detalle = f"e = {elemento.espesor:.2f} m · {elemento.material}"
                hijos = por_elemento.get((tipo, id_), [])
                if hijos:
                    detalle += f"  ({len(hijos)})"
                item = self._nuevo(sub, elemento.id if elemento.nombre == elemento.id else f"{elemento.id} · {elemento.nombre}",
                                   detalle, nombre_icono, ("ELEMENTO", tipo, id_), destinos=[(tipo, id_)])
                for obj, caso in hijos:
                    crear_asignacion(item, obj, caso, bajo_elemento=True)

        # -- apoyos y conexiones ---------------------------------------------------------
        grupo = self._nuevo(raiz, f"Apoyos ({len(apoyos)})", "", "apoyo", ("GRUPO", "Apoyos"), negrita=True)
        for obj in apoyos:
            crear_asignacion(grupo, obj)
        grupo = self._nuevo(raiz, f"Conexiones ({len(conexiones)})", "", "conectar", ("GRUPO", "Conexiones"), negrita=True)
        for obj in conexiones:
            crear_asignacion(grupo, obj)

        # -- cargas, por caso -------------------------------------------------------------
        total = sum(len(v) for d in (cargas_area, cargas_lineales, cargas_puntuales) for v in d.values())
        grupo = self._nuevo(raiz, f"Cargas ({total})", "", "carga_dist", ("GRUPO", "Cargas"), negrita=True)
        for caso_id, caso in casos.items():
            lista = (
                list(cargas_area.get(caso_id, [])) + list(cargas_lineales.get(caso_id, [])) + list(cargas_puntuales.get(caso_id, []))
            )
            sub = self._nuevo(grupo, f"Caso {caso_id} — {caso.nombre} ({len(lista)})", "", "casos", ("GRUPO", "Caso", caso_id))
            for obj in lista:
                crear_asignacion(sub, obj)

        # -- materiales, casos, combinaciones ------------------------------------------------
        grupo = self._nuevo(raiz, f"Materiales ({len(proyecto.materiales)})", "", "materiales", ("GRUPO", "Materiales"), negrita=True)
        for nombre, material in proyecto.materiales.items():
            usos = len(usos_de_material(proyecto, nombre))
            self._nuevo(
                grupo, nombre,
                f"E {modulo_en_mpa(material):,.0f} MPa · ν {material.modulo_poisson:g} · γ {material.densidad:g}  ({usos})",
                "material", ("MATERIAL", nombre),
                ayuda=f"{usos} elemento(s) usan este material", destinos=[],
            )
        grupo = self._nuevo(raiz, f"Casos de carga ({len(casos)})", "", "casos", ("GRUPO", "Casos de carga"), negrita=True)
        for caso_id, caso in casos.items():
            self._nuevo(grupo, f"{caso_id} — {caso.nombre}", caso.tipo.value, "casos", ("CASO", caso_id))
        grupo = self._nuevo(raiz, f"Combinaciones ({len(combinaciones)})", "", "casos", ("GRUPO", "Combinaciones"), negrita=True)
        for combinacion in combinaciones.values():
            self._nuevo(grupo, combinacion.id, formula(combinacion, casos), "casos", ("COMBINACION", combinacion.id),
                        ayuda=combinacion.norma)

        # -- resultados -----------------------------------------------------------------------
        if resultado is not None:
            grupo = self._nuevo(
                raiz, "Resultados", "desactualizados" if desactualizado else "esfuerzos en MPa",
                "advertencia" if desactualizado else "esfuerzos", ("GRUPO", "Resultados"), negrita=True,
            )
            if resultado.tensiones_por_combinacion:
                self._nuevo(grupo, "Envolvente de combinaciones", "doble clic para ver", "esfuerzos", ("RESULTADO", ENVOLVENTE))
            for id_ in resultado.tensiones_por_combinacion:
                self._nuevo(grupo, f"Combinación {id_}", "", "esfuerzos", ("RESULTADO", id_))
            for id_ in resultado.tensiones_por_caso:
                self._nuevo(grupo, f"Caso {id_}", "", "esfuerzos", ("RESULTADO", id_))

        # -- estado: expandido y seleccionado --------------------------------------------------
        self._restaurar(self.arbol.invisibleRootItem(), seleccionadas)
        self.arbol.resizeColumnToContents(0) if self.arbol.columnWidth(0) < 120 else None
        self._silencio = False
        self._filtrar(self.buscador.text())

    @staticmethod
    def _clave(item: QTreeWidgetItem) -> tuple:
        return item.data(0, Qt.UserRole) or ()

    def _restaurar(self, padre: QTreeWidgetItem, seleccionadas: Set[tuple]) -> None:
        for i in range(padre.childCount()):
            item = padre.child(i)
            clave = self._clave(item)
            if clave in self._abiertos or (clave in ABIERTOS_POR_DEFECTO and clave not in self._cerrados):
                item.setExpanded(True)
            if clave in seleccionadas and clave[:1] in (("ELEMENTO",), ("ASIGNACION",), ("MATERIAL",), ("RESULTADO",)):
                item.setSelected(True)
            self._restaurar(item, seleccionadas)

    def _recordar(self, item: QTreeWidgetItem, abierto: bool) -> None:
        if self._silencio:
            return
        clave = self._clave(item)
        (self._abiertos if abierto else self._cerrados).add(clave)
        (self._cerrados if abierto else self._abiertos).discard(clave)

    def _contraer(self) -> None:
        self._abiertos.clear()
        self._cerrados.clear()
        self.arbol.collapseAll()
        raiz = self.arbol.topLevelItem(0)
        if raiz is not None:
            raiz.setExpanded(True)

    # ------------------------------------------------------------------
    # Selección
    # ------------------------------------------------------------------
    def claves_seleccionadas(self) -> List[tuple]:
        return [self._clave(i) for i in self.arbol.selectedItems()]

    def _elementos_de(self, claves: Iterable[tuple]) -> List[Tuple[str, str]]:
        resultado: List[Tuple[str, str]] = []
        for clave in claves:
            for destino in self._destinos.get(clave, []):
                if destino not in resultado:
                    resultado.append(destino)
        return resultado

    def _al_cambiar_seleccion(self) -> None:
        if self._silencio:
            return
        self.elementos_seleccionados.emit(self._elementos_de(self.claves_seleccionadas()))

    def resaltar(self, elementos: Sequence[Tuple[str, str]]) -> None:
        """Marca en el árbol los elementos seleccionados en el visor."""

        buscados = {("ELEMENTO", tipo, id_) for tipo, id_ in elementos}
        self._silencio = True
        self.arbol.clearSelection()
        ultimo = None

        def recorrer(padre: QTreeWidgetItem) -> None:
            nonlocal ultimo
            for i in range(padre.childCount()):
                item = padre.child(i)
                if self._clave(item) in buscados:
                    item.setSelected(True)
                    ultimo = item
                recorrer(item)

        recorrer(self.arbol.invisibleRootItem())
        if ultimo is not None:
            self.arbol.scrollToItem(ultimo)
        self._silencio = False

    def _al_doble_clic(self, item: QTreeWidgetItem, _columna: int) -> None:
        clave = self._clave(item)
        tipo = clave[0] if clave else ""
        if tipo == "RESULTADO":
            self.resultado_solicitado.emit(clave[1])
        elif tipo == "MATERIAL":
            self.editar_materiales.emit()
        elif tipo in ("CASO", "COMBINACION"):
            self.editar_casos.emit()
        elif tipo == "ELEMENTO":
            self.propiedades_solicitadas.emit()

    def _al_menu(self, posicion) -> None:
        item = self.arbol.itemAt(posicion)
        if item is not None and not item.isSelected():
            self.arbol.clearSelection()
            item.setSelected(True)
        self.menu_solicitado.emit(self.claves_seleccionadas())

    def keyPressEvent(self, evento) -> None:
        if evento.key() == Qt.Key_Delete:
            claves = [(c[1], c[2]) for c in self.claves_seleccionadas() if c[:1] == ("ASIGNACION",)]
            if claves:
                self.eliminar_solicitado.emit(claves)
                return
        super().keyPressEvent(evento)

    # ------------------------------------------------------------------
    # Filtro
    # ------------------------------------------------------------------
    def _filtrar(self, texto: str) -> None:
        texto = texto.strip().lower()

        def visible(item: QTreeWidgetItem, forzado: bool = False) -> bool:
            propio = not texto or texto in item.text(0).lower() or texto in item.text(1).lower()
            hijos = [visible(item.child(i), forzado or propio) for i in range(item.childCount())]
            mostrar = forzado or propio or any(hijos)
            item.setHidden(not mostrar)
            if texto and any(hijos):
                item.setExpanded(True)
            return mostrar

        for i in range(self.arbol.topLevelItemCount()):
            visible(self.arbol.topLevelItem(i))
        self.arbol.scheduleDelayedItemsLayout()
        self.arbol.viewport().update()
