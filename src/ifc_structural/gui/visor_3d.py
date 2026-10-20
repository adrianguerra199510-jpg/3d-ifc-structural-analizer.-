"""Visor 3D del modelo estructural (PyVista embebido en Qt).

Dos vistas:

- **Modelo**: vigas y columnas como prismas con su sección real, losas y
  muros con su espesor. Se seleccionan con el mouse, a nivel de elemento o
  de cara (modo «Elemento» / «Cara»); Ctrl o Mayús suman a la selección.
  También se dibujan los apoyos y las cargas del caso activo.
- **Esfuerzos**: mapa de contorno de una magnitud de esfuerzo sobre la
  malla expandida de las vigas y columnas, con escala configurable. Al
  hacer clic se muestra el valor en ese punto.

La selección se hace con un ``vtkCellPicker`` propio (un clic = presionar
y soltar sin arrastrar, para no interferir con girar la cámara), de modo
que se conoce con exactitud el elemento y la cara bajo el cursor: cada
elemento es una malla aparte cuya celda número ``i`` es su cara ``i``
(ver ``core.geometria``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
from PySide6.QtCore import Signal
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker

from ..core.asignacion import Objetivo, barras_de_apoyo_area
from ..core.cargas import Apoyo, ApoyoArea, CargaArea, CargaLineal, CargaPuntual, Conexion
from ..core.geometria import Solido, normal_losa, solido_elemento
from ..core.modelo import Proyecto
from ..solver.lector_resultados import MallaTensiones
from .simbolos_apoyo import COLORES_SIMBOLO, nombre_apoyo, simbolo_apoyo

logger = logging.getLogger(__name__)

COLORES_BASE = {
    "VIGA": (79, 134, 198),
    "COLUMNA": (212, 104, 90),
    "LOSA": (170, 182, 196),
    "MURO": (217, 198, 143),
}
COLOR_ELEMENTO_SELECCIONADO = (255, 200, 0)
COLOR_CARA_SELECCIONADA = (255, 120, 0)
COLOR_APOYO = "royalblue"
COLOR_CARGA = "crimson"
COLOR_MOMENTO = "darkorchid"
COLOR_APOYO_AREA = "seagreen"
COLOR_CONEXION = "deeppink"
OPACIDAD_PLANOS = 0.6
OPACIDAD_PLANOS_EN_ESFUERZOS = 0.12
#: Un clic es una pulsación que se mueve menos que esto (píxeles).
UMBRAL_ARRASTRE_PX = 4

MODO_ELEMENTO = "ELEMENTO"
MODO_CARA = "CARA"


@dataclass
class _Visual:
    tipo: str
    id: str
    solido: Solido
    malla: pv.PolyData
    actor: object


class Visor3D(QtInteractor):
    """Widget 3D con selección de elementos y caras por clic."""

    seleccion_cambiada = Signal(list)  # list[Objetivo]
    punto_sondeado = Signal(str)  # texto para mostrar al hacer clic en la vista de esfuerzos
    menu_contextual = Signal()  # clic derecho sobre el modelo (la selección ya está actualizada)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proyecto: Optional[Proyecto] = None
        self._visuales: Dict[Tuple[str, str], _Visual] = {}
        self._clave_por_actor: Dict[int, Tuple[str, str]] = {}
        self._seleccion: List[Objetivo] = []
        self._modo = MODO_ELEMENTO
        self._vista = "MODELO"
        self._visibles = {"VIGA": True, "COLUMNA": True, "LOSA": True, "MURO": True}
        self._escala = 1.0

        self._grid_esfuerzos: Optional[pv.UnstructuredGrid] = None
        self._valores_esfuerzos: Optional[np.ndarray] = None
        self._barra_de_celda: List[str] = []
        self._actor_esfuerzos = None
        self._titulo_esfuerzos = ""

        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self._pulsacion: Optional[Tuple[int, int]] = None
        self._pulsacion_derecha: Optional[Tuple[int, int]] = None

        self.set_background("#ffffff", top="#dde6f1")  # degradado suave de abajo (blanco) a arriba
        self.add_axes()
        self.iren.add_observer("LeftButtonPressEvent", self._al_presionar)
        self.iren.add_observer("LeftButtonReleaseEvent", self._al_soltar)
        self.iren.add_observer("RightButtonPressEvent", self._al_presionar_derecho)
        self.iren.add_observer("RightButtonReleaseEvent", self._al_soltar_derecho)

    # ------------------------------------------------------------------
    # Construcción de la vista del modelo
    # ------------------------------------------------------------------
    def mostrar_modelo(self, proyecto: Proyecto) -> None:
        self._proyecto = proyecto
        self.volver_al_modelo(reconstruir=True)
        self.reset_camera()
        self.view_isometric()
        self.render()

    def _limpiar_escena(self) -> None:
        self.clear_actors()
        self._visuales.clear()
        self._clave_por_actor.clear()
        self._actor_esfuerzos = None

    def _reconstruir_visuales(self) -> None:
        proyecto = self._proyecto
        if proyecto is None:
            return
        elementos = (
            [("VIGA", i) for i in proyecto.vigas]
            + [("COLUMNA", i) for i in proyecto.columnas]
            + [("LOSA", i) for i in proyecto.losas]
            + [("MURO", i) for i in proyecto.muros]
        )
        todos = [proyecto.nodos[n].coordenadas for n in proyecto.nodos]
        if todos:
            extension = np.ptp(np.array(todos, dtype=float), axis=0)
            self._escala = max(float(np.linalg.norm(extension)), 1.0)

        for tipo, id_ in elementos:
            solido = solido_elemento(proyecto, tipo, id_)
            faces = np.hstack([[len(c.indices), *c.indices] for c in solido.caras]).astype(np.int64)
            malla = pv.PolyData(np.asarray(solido.puntos, dtype=float), faces=faces)
            malla.cell_data["colores"] = np.tile(COLORES_BASE[tipo], (len(solido.caras), 1)).astype(np.uint8)
            es_plano = tipo in ("LOSA", "MURO")
            actor = self.add_mesh(
                malla,
                scalars="colores",
                rgb=True,
                show_edges=True,
                edge_color=(52, 68, 88),
                opacity=OPACIDAD_PLANOS if es_plano else 1.0,
                pickable=True,
                name=f"{tipo}_{id_}",
                show_scalar_bar=False,
            )
            self._visuales[(tipo, id_)] = _Visual(tipo, id_, solido, malla, actor)
            self._clave_por_actor[hash(actor)] = (tipo, id_)

    # ------------------------------------------------------------------
    # Selección
    # ------------------------------------------------------------------
    @property
    def seleccion(self) -> List[Objetivo]:
        return list(self._seleccion)

    @property
    def modo(self) -> str:
        return self._modo

    def set_modo(self, modo: str) -> None:
        if modo not in (MODO_ELEMENTO, MODO_CARA):
            raise ValueError(modo)
        if modo == self._modo:
            return
        self._modo = modo
        # Al cambiar de modo la selección anterior ya no significa lo mismo.
        self.seleccionar([])

    def seleccionar(self, objetivos: Sequence[Objetivo]) -> None:
        """Establece la selección (desde el código o desde una tabla)."""

        self._seleccion = list(objetivos)
        self._pintar_seleccion()
        self.seleccion_cambiada.emit(self.seleccion)

    def _pintar_seleccion(self) -> None:
        for clave, visual in self._visuales.items():
            colores = np.tile(COLORES_BASE[visual.tipo], (len(visual.solido.caras), 1)).astype(np.uint8)
            for objetivo in self._seleccion:
                if (objetivo.tipo, objetivo.id) != clave:
                    continue
                if objetivo.cara is None:
                    colores[:] = COLOR_ELEMENTO_SELECCIONADO
                else:
                    colores[objetivo.cara] = COLOR_CARA_SELECCIONADA
            visual.malla.cell_data["colores"][:] = colores
            visual.malla.Modified()
        self.render()

    def _elegir(self, x: int, y: int):
        """(clave, índice de celda) bajo el píxel, o (None, -1)."""

        if not self._picker.Pick(x, y, 0, self.renderer):
            return None, -1
        actor = self._picker.GetActor()
        if actor is None:
            return None, -1
        return self._clave_por_actor.get(hash(actor)), self._picker.GetCellId()

    def _al_presionar(self, *_args) -> None:
        # Los observadores de VTK reciben el "estilo" de interacción, no el
        # interactor: la posición y las teclas se leen del interactor.
        self._pulsacion = tuple(self.iren.interactor.GetEventPosition())

    def _al_soltar(self, *_args) -> None:
        interactor = self.iren.interactor
        inicio, self._pulsacion = self._pulsacion, None
        if inicio is None:
            return
        x, y = interactor.GetEventPosition()
        if abs(x - inicio[0]) > UMBRAL_ARRASTRE_PX or abs(y - inicio[1]) > UMBRAL_ARRASTRE_PX:
            return  # fue un arrastre (girar/mover la cámara), no un clic
        if self._vista == "ESFUERZOS":
            self._sondear(x, y)
            return
        sumar = bool(interactor.GetControlKey() or interactor.GetShiftKey())
        self._clic_en_modelo(x, y, sumar)

    def _al_presionar_derecho(self, *_args) -> None:
        self._pulsacion_derecha = tuple(self.iren.interactor.GetEventPosition())

    def _al_soltar_derecho(self, *_args) -> None:
        interactor = self.iren.interactor
        inicio, self._pulsacion_derecha = self._pulsacion_derecha, None
        if inicio is None or self._vista != "MODELO":
            return
        x, y = interactor.GetEventPosition()
        if abs(x - inicio[0]) > UMBRAL_ARRASTRE_PX or abs(y - inicio[1]) > UMBRAL_ARRASTRE_PX:
            return  # fue un arrastre (zoom con el botón derecho), no un clic
        self.clic_derecho_en(x, y)

    def clic_derecho_en(self, x: int, y: int) -> None:
        """Clic derecho en el píxel (x, y): si el elemento bajo el cursor no
        está en la selección pasa a ser la selección; luego pide el menú."""

        clave, celda = self._elegir(x, y)
        if clave is not None:
            tipo, id_ = clave
            ya_elegido = any((o.tipo, o.id) == (tipo, id_) for o in self._seleccion)
            if not ya_elegido:
                self.seleccionar([Objetivo(tipo, id_, celda if self._modo == MODO_CARA else None)])
        if self._seleccion:
            self.menu_contextual.emit()

    def _clic_en_modelo(self, x: int, y: int, sumar: bool) -> None:
        clave, celda = self._elegir(x, y)
        if clave is None:
            if not sumar:
                self.seleccionar([])
            return

        tipo, id_ = clave
        objetivo = Objetivo(tipo, id_, celda if self._modo == MODO_CARA else None)
        if sumar:
            seleccion = list(self._seleccion)
            if objetivo in seleccion:
                seleccion.remove(objetivo)  # Ctrl+clic sobre algo ya elegido lo quita
            else:
                seleccion.append(objetivo)
        else:
            seleccion = [objetivo]
        self.seleccionar(seleccion)

    # ------------------------------------------------------------------
    # Visibilidad y cámara
    # ------------------------------------------------------------------
    def set_visibilidad(self, tipo: str, visible: bool) -> None:
        self._visibles[tipo] = visible
        if self._vista == "MODELO":
            self._aplicar_visibilidad()
            self.render()

    def _aplicar_visibilidad(self) -> None:
        for visual in self._visuales.values():
            visual.actor.SetVisibility(self._visibles[visual.tipo])
            visual.actor.SetPickable(self._visibles[visual.tipo])

    def vista_isometrica(self) -> None:
        self.view_isometric()

    def vista_planta(self) -> None:
        self.view_xy()

    def vista_frontal(self) -> None:
        self.view_xz()

    def vista_lateral(self) -> None:
        self.view_yz()

    # ------------------------------------------------------------------
    # Apoyos y cargas
    # ------------------------------------------------------------------
    def mostrar_asignaciones(
        self,
        apoyos: Sequence,
        cargas_area: Sequence[CargaArea],
        cargas_lineales: Sequence[CargaLineal],
        cargas_puntuales: Sequence[CargaPuntual],
        conexiones: Sequence[Conexion] = (),
    ) -> None:
        """Dibuja los apoyos, las conexiones y las cargas del caso activo
        (las que se pasan). Las flechas de una carga variable tienen longitud
        proporcional a su valor, de modo que se ve su forma."""

        for nombre in tuple(f"ASIG_APOYOS_{c}" for c in COLORES_SIMBOLO) + ("ASIG_APOYOS_AREA", "ASIG_CONEXIONES", "ASIG_CARGAS", "ASIG_MOMENTOS", "ASIG_ETIQUETAS"):
            self.remove_actor(nombre, render=False)
        proyecto = self._proyecto
        if proyecto is None:
            return

        largo = float(np.clip(0.06 * self._escala, 0.3, 1.5))
        barras = proyecto.todas_las_barras()

        def punto_de(barra_id: str, s: float) -> np.ndarray:
            barra = proyecto.elemento_barra(barra_id)
            a = np.array(proyecto.nodos[barra.nodo_inicio].coordenadas)
            b = np.array(proyecto.nodos[barra.nodo_fin].coordenadas)
            return a + s * (b - a)

        # -- apoyos puntuales: símbolo según los grados de libertad restringidos ------------
        por_color: Dict[str, List[pv.PolyData]] = {}
        etiquetas_apoyo: List[np.ndarray] = []
        textos_apoyo: List[str] = []
        for a in apoyos:
            if isinstance(a, ApoyoArea) or a.nodo_id not in proyecto.nodos:
                continue
            grados = (a.ux, a.uy, a.uz, a.rx, a.ry, a.rz)
            centro = np.array(proyecto.nodos[a.nodo_id].coordenadas)
            for color, mallas in simbolo_apoyo(grados, centro, largo).items():
                por_color.setdefault(color, []).extend(mallas)
            etiquetas_apoyo.append(centro - np.array([0.0, 0.0, 1.35 * largo]))
            textos_apoyo.append(nombre_apoyo(grados))
        for color, mallas in por_color.items():
            self.add_mesh(pv.merge(mallas), color=color, name=f"ASIG_APOYOS_{color}", pickable=False, show_scalar_bar=False)

        # -- apoyos de área: conos pequeños a lo largo de las barras apoyadas ------------
        conos_area: List[pv.PolyData] = []
        for apoyo in apoyos:
            if not isinstance(apoyo, ApoyoArea):
                continue
            traslaciones = [i for i, activo in enumerate((apoyo.ux, apoyo.uy, apoyo.uz)) if activo]
            eje = 2 if 2 in traslaciones else (traslaciones[0] if traslaciones else 2)
            direccion = np.zeros(3)
            direccion[eje] = 1.0
            for barra_id in barras_de_apoyo_area(proyecto, apoyo):
                longitud = proyecto.longitud_barra(barra_id)
                n = max(3, int(round(longitud * (apoyo.s_fin - apoyo.s_inicio) / (largo * 1.2))) + 1)
                for s in np.linspace(apoyo.s_inicio, apoyo.s_fin, n):
                    conos_area.append(
                        pv.Cone(
                            center=punto_de(barra_id, s) - direccion * largo * 0.5, direction=direccion,
                            height=largo * 0.9, radius=largo * 0.4, resolution=12,
                        )
                    )
                etiquetas_apoyo.append(punto_de(barra_id, 0.5 * (apoyo.s_inicio + apoyo.s_fin)) - direccion * largo * 0.9)
                textos_apoyo.append("apoyo " + "".join(
                    n_ for n_, activo in zip(("Ux", "Uy", "Uz", "Rx", "Ry", "Rz"), (apoyo.ux, apoyo.uy, apoyo.uz, apoyo.rx, apoyo.ry, apoyo.rz)) if activo
                ))
        if conos_area:
            self.add_mesh(pv.merge(conos_area), color=COLOR_APOYO_AREA, name="ASIG_APOYOS_AREA", pickable=False, show_scalar_bar=False)

        # -- conexiones ----------------------------------------------------------------------
        marcas: List[pv.PolyData] = []
        for c in conexiones:
            if c.nodo_id not in proyecto.nodos:
                continue
            centro = np.array(proyecto.nodos[c.nodo_id].coordenadas)
            marcas.append(pv.Sphere(radius=largo * 0.3, center=centro))
            if c.nodo_destino is not None and c.nodo_destino in proyecto.nodos:
                destino = np.array(proyecto.nodos[c.nodo_destino].coordenadas)
                marcas.append(pv.Sphere(radius=largo * 0.3, center=destino))
                if np.linalg.norm(destino - centro) > 1e-9:
                    marcas.append(pv.Line(centro, destino).tube(radius=largo * 0.08))
        if marcas:
            self.add_mesh(pv.merge(marcas), color=COLOR_CONEXION, name="ASIG_CONEXIONES", pickable=False, show_scalar_bar=False)

        # -- cargas ------------------------------------------------------------------------------
        flechas: List[pv.PolyData] = []
        momentos: List[pv.PolyData] = []
        puntos_etiqueta: List[np.ndarray] = []
        textos: List[str] = []

        def flecha(punta, direccion, longitud):
            d = np.asarray(direccion, dtype=float)
            d = d / np.linalg.norm(d)
            return pv.Arrow(start=np.asarray(punta) - d * longitud, direction=d, scale=longitud,
                            tip_length=0.3, tip_radius=0.14, shaft_radius=0.05)

        for carga in cargas_area:
            if carga.losa_id not in proyecto.losas:
                continue
            solido = solido_elemento(proyecto, "LOSA", carga.losa_id)
            d = np.asarray(carga.direccion, dtype=float)
            d = d / np.linalg.norm(d)
            # La cara cargada es la que mira contra la carga.
            cara = min(solido.caras[:2], key=lambda c: float(np.dot(c.normal, d)))
            muestras = _puntos_en_cara(cara.puntos, 5)
            valores = [abs(carga.presion_en(m)) for m in muestras]
            maximo = max(valores, default=0.0) or 1.0
            for punto, valor in zip(muestras, valores):
                flechas.append(flecha(punto, d, largo * (0.3 + 1.1 * valor / maximo)))
            if muestras:
                puntos_etiqueta.append(muestras[0] - d * largo * 1.1)
                textos.append(
                    f"{carga.presion:g} kPa" if carga.perfil is None
                    else f"{carga.perfil.valor_a:g} a {carga.perfil.valor_b:g} kPa"
                )

        for carga in cargas_lineales:
            if carga.elemento_id not in barras:
                continue
            d = np.asarray(carga.direccion, dtype=float)
            d = d / np.linalg.norm(d)
            maximo = max(abs(carga.w_inicio), abs(carga.w_fin)) or 1.0
            n = 6 if not carga.es_uniforme else 5
            cola = []
            for s in np.linspace(carga.s_inicio, carga.s_fin, n):
                t = (s - carga.s_inicio) / max(carga.s_fin - carga.s_inicio, 1e-9)
                w = carga.w_inicio + t * (carga.w_fin - carga.w_inicio)
                alto = largo * (0.3 + 1.1 * abs(w) / maximo)
                base = punto_de(carga.elemento_id, s) - d * largo * 0.2
                flechas.append(flecha(base, d, alto))
                cola.append(base - d * alto)
            if len(cola) > 1 and not carga.es_uniforme:
                flechas.append(pv.MultipleLines(np.array(cola)).tube(radius=largo * 0.02))
            puntos_etiqueta.append(punto_de(carga.elemento_id, 0.5 * (carga.s_inicio + carga.s_fin)) - d * largo * 1.15)
            textos.append(
                f"{abs(carga.w_inicio):g} kN/m" if carga.es_uniforme
                else f"{carga.w_inicio:g} a {carga.w_fin:g} kN/m"
            )

        for carga in cargas_puntuales:
            if carga.nodo_id is not None and carga.nodo_id in proyecto.nodos:
                punto = np.array(proyecto.nodos[carga.nodo_id].coordenadas)
            elif carga.elemento_id in barras and carga.posicion is not None:
                punto = punto_de(carga.elemento_id, carga.posicion)
            else:
                continue
            f = np.array([carga.fx, carga.fy, carga.fz], dtype=float)
            m = np.array([carga.mx, carga.my, carga.mz], dtype=float)
            if np.linalg.norm(f) > 1e-12:
                d = f / np.linalg.norm(f)
                flechas.append(flecha(punto - d * largo * 0.15, d, largo * 1.4))
                puntos_etiqueta.append(punto - d * largo * 1.6)
                textos.append(f"{np.linalg.norm(f):g} kN")
            if np.linalg.norm(m) > 1e-12:
                d = m / np.linalg.norm(m)
                # Momento: flecha doble en la dirección del vector (regla de la mano derecha)
                momentos.append(flecha(punto + d * largo * 0.2, d, largo * 1.0))
                momentos.append(flecha(punto + d * largo * 0.55, d, largo * 1.0))
                puntos_etiqueta.append(punto + d * largo * 0.9)
                textos.append(f"M = {np.linalg.norm(m):g} kN·m")

        if flechas:
            self.add_mesh(pv.merge(flechas), color=COLOR_CARGA, name="ASIG_CARGAS", pickable=False, show_scalar_bar=False)
        if momentos:
            self.add_mesh(pv.merge(momentos), color=COLOR_MOMENTO, name="ASIG_MOMENTOS", pickable=False, show_scalar_bar=False)
        puntos_etiqueta += etiquetas_apoyo
        textos += textos_apoyo
        if puntos_etiqueta:
            self.add_point_labels(
                np.array(puntos_etiqueta), textos, name="ASIG_ETIQUETAS", font_size=14, text_color="darkred",
                shape=None, show_points=False, always_visible=True, pickable=False,
            )
        self.render()

    # ------------------------------------------------------------------
    # Vista de esfuerzos
    # ------------------------------------------------------------------
    def mostrar_esfuerzos(
        self,
        malla: MallaTensiones,
        valores: np.ndarray,
        barra_de_celda: Sequence[str],
        titulo: str,
        rango: Optional[Tuple[float, float]] = None,
        cmap: str = "turbo",
        texto: str = "",
        ocultar_planos: bool = False,
    ) -> None:
        """Mapa de contorno de ``valores`` (uno por nodo de ``malla``). Con ``ocultar_planos``
        (análisis sólido) las losas y los muros tampoco se dibujan: ya están en la malla."""

        self._vista = "ESFUERZOS"
        for visual in self._visuales.values():
            if visual.tipo in ("VIGA", "COLUMNA"):
                visual.actor.SetVisibility(False)
                visual.actor.SetPickable(False)
            else:
                visual.actor.SetVisibility(self._visibles[visual.tipo] and not ocultar_planos)
                visual.actor.SetPickable(False)
                visual.actor.GetProperty().SetOpacity(OPACIDAD_PLANOS_EN_ESFUERZOS)

        n_celdas = len(malla.ids_elementos)
        celdas = np.hstack([np.full((n_celdas, 1), 8, dtype=np.int64), malla.conectividad.astype(np.int64)]).ravel()
        tipos = np.full(n_celdas, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        grid = pv.UnstructuredGrid(celdas, tipos, np.asarray(malla.coordenadas, dtype=float))
        grid.point_data["esfuerzo"] = np.asarray(valores, dtype=float)

        self.remove_actor("ESFUERZOS", render=False)
        self.remove_scalar_bar_si_existe()
        self._actor_esfuerzos = self.add_mesh(
            grid,
            scalars="esfuerzo",
            cmap=cmap,
            clim=list(rango) if rango else None,
            show_edges=False,
            name="ESFUERZOS",
            pickable=True,
            scalar_bar_args={"title": "MPa", "vertical": True, "n_labels": 8, "fmt": "%.2f",
                             "title_font_size": 16, "label_font_size": 14},
        )
        self._grid_esfuerzos = grid
        self._valores_esfuerzos = np.asarray(valores, dtype=float)
        self._barra_de_celda = list(barra_de_celda)
        self._titulo_esfuerzos = titulo
        self.remove_actor("TEXTO_ESFUERZOS", render=False)
        if texto:
            self.add_text(texto, position="upper_left", font_size=11, color="black", name="TEXTO_ESFUERZOS")
        self.render()

    def remove_scalar_bar_si_existe(self) -> None:
        for titulo in list(self.scalar_bars.keys()):
            self.remove_scalar_bar(titulo, render=False)

    def volver_al_modelo(self, reconstruir: bool = False) -> None:
        """Vuelve a la vista del modelo (o la reconstruye desde cero)."""

        self._vista = "MODELO"
        if reconstruir:
            self._limpiar_escena()
            self.add_axes()
            self._reconstruir_visuales()
            self._seleccion = []
        else:
            self.remove_actor("ESFUERZOS", render=False)
            self.remove_actor("TEXTO_ESFUERZOS", render=False)
            self.remove_scalar_bar_si_existe()
            for visual in self._visuales.values():
                visual.actor.GetProperty().SetOpacity(OPACIDAD_PLANOS if visual.tipo in ("LOSA", "MURO") else 1.0)
        self._aplicar_visibilidad()
        self._grid_esfuerzos = None
        self._pintar_seleccion()

    @property
    def vista(self) -> str:
        return self._vista

    def _sondear(self, x: int, y: int) -> None:
        """Muestra el valor del esfuerzo en el punto bajo el cursor."""

        if self._grid_esfuerzos is None or self._actor_esfuerzos is None:
            return
        if not self._picker.Pick(x, y, 0, self.renderer) or self._picker.GetActor() is not self._actor_esfuerzos:
            return
        celda = self._picker.GetCellId()
        punto = np.array(self._picker.GetPickPosition())
        ids = np.array(self._grid_esfuerzos.get_cell(celda).point_ids)
        cercano = ids[np.argmin(np.linalg.norm(self._grid_esfuerzos.points[ids] - punto, axis=1))]
        barra = self._barra_de_celda[celda] if celda < len(self._barra_de_celda) else "?"
        self.punto_sondeado.emit(
            f"{self._titulo_esfuerzos}: {self._valores_esfuerzos[cercano]:.3f} — barra {barra} "
            f"(en x={punto[0]:.2f}, y={punto[1]:.2f}, z={punto[2]:.2f} m)"
        )


def _puntos_en_cara(puntos: np.ndarray, n: int) -> List[np.ndarray]:
    """Cuadrícula de n×n puntos dentro de una cara plana (para dibujar
    las flechas de una carga de área)."""

    from shapely.geometry import Point, Polygon

    centro = puntos.mean(axis=0)
    u = puntos[1] - puntos[0]
    u = u / np.linalg.norm(u)
    v = np.cross(normal_losa(puntos), u)
    plano = [((p - centro) @ u, (p - centro) @ v) for p in puntos]
    poligono = Polygon(plano)
    (a0, b0, a1, b1) = poligono.bounds
    resultado = []
    for i in range(n):
        for j in range(n):
            a = a0 + (i + 0.5) * (a1 - a0) / n
            b = b0 + (j + 0.5) * (b1 - b0) / n
            if poligono.contains(Point(a, b)):
                resultado.append(centro + a * u + b * v)
    return resultado
