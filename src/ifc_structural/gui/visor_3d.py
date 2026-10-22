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

import functools
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
from PySide6.QtCore import QTimer, Qt, Signal
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker

from ..core.asignacion import Objetivo, barras_de_apoyo_area
from ..core.cargas import Apoyo, ApoyoArea, ApoyoSuperficie, CargaArea, CargaLineal, CargaPuntual, CargaSuperficie, Conexion, UnionSuperficie
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
    "SOLIDO": (150, 163, 178),
}
COLOR_ELEMENTO_SELECCIONADO = (255, 200, 0)
COLOR_CARA_SELECCIONADA = (255, 120, 0)
COLOR_APOYO = "royalblue"
COLOR_CARGA = "crimson"
COLOR_MOMENTO = "darkorchid"
COLOR_APOYO_AREA = "seagreen"
COLOR_CONEXION = "deeppink"
OPACIDAD_PLANOS = 0.6

#: Estilos de vista del modelo: sólido, transparente (se ve a través de las piezas) y alambre (solo aristas).
ESTILO_SOLIDO = "SOLIDO"
ESTILO_TRANSPARENTE = "TRANSPARENTE"
ESTILO_ALAMBRE = "ALAMBRE"
OPACIDAD_TRANSPARENTE = 0.25
OPACIDAD_ALAMBRE = 0.04  # casi invisible pero sigue recibiendo los clics de selección
OPACIDAD_PLANOS_EN_ESFUERZOS = 0.12
#: Un clic es una pulsación que se mueve menos que esto (píxeles).
UMBRAL_ARRASTRE_PX = 4

MODO_ELEMENTO = "ELEMENTO"
MODO_CARA = "CARA"

#: Qué hace el arrastre con el botón izquierdo: seleccionar (por defecto, como en SAP2000, AutoCAD o Revit),
#: desplazar la vista o girarla. La rueda hace zoom; el botón central desplaza, y con Ctrl o Mayús gira.
#: Parámetros de tamaño de los símbolos y sus modos (porcentaje del tamaño automático o píxeles en pantalla).
PARAMETROS_ESCALA = ("apoyos", "cargas", "constraints", "etiquetas")
ESCALA_PORCENTAJE = "PORCENTAJE"
ESCALA_PIXELES = "PIXELES"
TAMANO_ETIQUETA_AUTOMATICO = 14

NAV_SELECCION = "SELECCION"
NAV_MOVER = "MOVER"
NAV_ROTAR = "ROTAR"


@dataclass
class _Visual:
    tipo: str
    id: str
    solido: Optional[Solido]  # barras, losas y muros (una celda por cara); None en los sólidos de forma libre
    malla: pv.PolyData
    actor: object
    #: Sólidos de forma libre: superficie (cara) a la que pertenece cada triángulo de la malla.
    cara_de_celda: Optional[np.ndarray] = None
    actor_bordes: object = None  # líneas de contorno de las superficies (solo sólidos de forma libre)


def conserva_camara(funcion):
    """Al dibujar o quitar actores, pyvista reajusta la cámara a lo que hay en escena y se pierde el zoom y el
    encuadre del usuario: este decorador devuelve la cámara a como estaba."""

    @functools.wraps(funcion)
    def envuelta(self, *args, **kwargs):
        cam = self.renderer.GetActiveCamera()
        estado = (cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp(), cam.GetViewAngle(), cam.GetParallelScale())
        try:
            return funcion(self, *args, **kwargs)
        finally:
            cam = self.renderer.GetActiveCamera()
            cam.SetPosition(*estado[0])
            cam.SetFocalPoint(*estado[1])
            cam.SetViewUp(*estado[2])
            cam.SetViewAngle(estado[3])
            cam.SetParallelScale(estado[4])
            self.renderer.ResetCameraClippingRange()

    return envuelta


class Visor3D(QtInteractor):
    """Widget 3D con selección de elementos y caras por clic."""

    seleccion_cambiada = Signal(list)  # list[Objetivo]
    punto_sondeado = Signal(str)  # texto para mostrar al hacer clic en la vista de esfuerzos
    menu_contextual = Signal()  # clic derecho sobre el modelo (la selección ya está actualizada)
    menu_fondo = Signal()  # clic derecho sobre el fondo (sin ningún elemento debajo)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proyecto: Optional[Proyecto] = None
        self._visuales: Dict[Tuple[str, str], _Visual] = {}
        self._clave_por_actor: Dict[int, Tuple[str, str]] = {}
        self._seleccion: List[Objetivo] = []
        self._modo = MODO_ELEMENTO
        self._vista = "MODELO"
        self._visibles = {"VIGA": True, "COLUMNA": True, "LOSA": True, "MURO": True, "SOLIDO": True}
        self._escala = 1.0
        self._flechas_visibles = True
        #: Tamaño de los símbolos: por parámetro, (modo, valor); ``PORCENTAJE`` (100 = automático) o ``PIXELES``.
        self._escalas = {p: (ESCALA_PORCENTAJE, 100.0) for p in PARAMETROS_ESCALA}
        self._temporizador_escala = QTimer(self)
        self._temporizador_escala.setSingleShot(True)
        self._temporizador_escala.setInterval(150)
        self._temporizador_escala.timeout.connect(self._redibujar_asignaciones)
        self._apoyos_visibles = True
        self._uniones_visibles = True
        self._estilo = ESTILO_SOLIDO
        self._argumentos_asignaciones = None

        self._grid_esfuerzos: Optional[pv.UnstructuredGrid] = None
        self._valores_esfuerzos: Optional[np.ndarray] = None
        self._barra_de_celda: List[str] = []
        self._actor_esfuerzos = None
        self._titulo_esfuerzos = ""
        self._puntos_sin_deformar: Optional[np.ndarray] = None
        self._desplazamientos: Optional[np.ndarray] = None
        self._escala_deformada = 1.0

        # navegación y selección por zona (arrastrar un rectángulo)
        self._navegacion = NAV_SELECCION
        self._arrastre: Optional[dict] = None  # gesto en curso: acción, botón, inicio y última posición (píxeles de VTK)
        self._caja_actor = None

        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0005)

        self.set_background("#ffffff", top="#dde6f1")  # degradado suave de abajo (blanco) a arriba
        self.add_axes()
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)  # para que Supr elimine lo seleccionado solo cuando el visor tiene el foco

    # ------------------------------------------------------------------
    # Construcción de la vista del modelo
    # ------------------------------------------------------------------
    def mostrar_modelo(self, proyecto: Proyecto) -> None:
        self._proyecto = proyecto
        self.volver_al_modelo(reconstruir=True)
        self.reset_camera()
        self.view_isometric()
        self.render()

    @property
    def estilo(self) -> str:
        return self._estilo

    def set_estilo(self, estilo: str) -> None:
        """Cambia el estilo de dibujo del modelo: ``ESTILO_SOLIDO``, ``ESTILO_TRANSPARENTE`` o ``ESTILO_ALAMBRE``."""

        if estilo not in (ESTILO_SOLIDO, ESTILO_TRANSPARENTE, ESTILO_ALAMBRE):
            raise ValueError(estilo)
        self._estilo = estilo
        if self._vista == "MODELO":
            self._aplicar_estilo()
            self.render()

    def _aplicar_estilo(self) -> None:
        """Opacidad, representación y aristas de cada elemento según el estilo. En todos se pueden seguir eligiendo."""

        for visual in self._visuales.values():
            propiedad = visual.actor.GetProperty()
            plano = visual.tipo in ("LOSA", "MURO")
            if self._estilo == ESTILO_TRANSPARENTE:
                propiedad.SetRepresentationToSurface()
                propiedad.SetOpacity(OPACIDAD_TRANSPARENTE * (0.7 if plano else 1.0))
            elif self._estilo == ESTILO_ALAMBRE:
                if visual.solido is not None:  # barras, losas y muros: sus aristas son las de las caras
                    propiedad.SetRepresentationToWireframe()
                    propiedad.SetOpacity(1.0)
                else:  # sólido libre: triángulos casi invisibles y solo las aristas vivas de sus superficies
                    propiedad.SetRepresentationToSurface()
                    propiedad.SetOpacity(OPACIDAD_ALAMBRE)
            else:
                propiedad.SetRepresentationToSurface()
                propiedad.SetOpacity(OPACIDAD_PLANOS if plano else 1.0)
            if visual.actor_bordes is not None:
                visual.actor_bordes.GetProperty().SetLineWidth(2.2 if self._estilo == ESTILO_ALAMBRE else 1.2)

    @conserva_camara
    def reemplazar_modelo(self, proyecto: Proyecto) -> None:
        """Dibuja otro estado del proyecto (deshacer, rehacer o eliminar) sin mover la cámara."""

        self._proyecto = proyecto
        self.volver_al_modelo(reconstruir=True)
        self.render()

    def _limpiar_escena(self) -> None:
        self.clear_actors()
        self._argumentos_asignaciones = None
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
        todos += [s.minimo for s in proyecto.solidos.values()] + [s.maximo for s in proyecto.solidos.values()]
        if todos:
            extension = np.ptp(np.array(todos, dtype=float), axis=0)
            self._escala = max(float(np.linalg.norm(extension)), 1e-3)

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

        for id_, solido_libre in proyecto.solidos.items():
            self._agregar_solido_libre(id_, solido_libre)
        self._aplicar_estilo()

    def _agregar_solido_libre(self, id_: str, s) -> None:
        """Geometría real de un elemento cualquiera: malla de triángulos con una celda por triángulo
        y el color de la superficie a la que pertenece."""

        faces = np.hstack([np.full((len(s.triangulos), 1), 3, dtype=np.int64), s.triangulos]).ravel()
        malla = pv.PolyData(np.asarray(s.vertices, dtype=float), faces=faces)
        malla.cell_data["colores"] = np.tile(COLORES_BASE["SOLIDO"], (len(s.triangulos), 1)).astype(np.uint8)
        actor = self.add_mesh(malla, scalars="colores", rgb=True, show_edges=False, pickable=True,
                              name=f"SOLIDO_{id_}", show_scalar_bar=False)
        bordes = malla.extract_feature_edges(feature_angle=35, boundary_edges=True, manifold_edges=False)
        actor_bordes = None
        if bordes.n_points:
            actor_bordes = self.add_mesh(bordes, color=(52, 68, 88), line_width=1.2, pickable=False, name=f"BORDES_{id_}",
                                         show_scalar_bar=False)
        self._visuales[("SOLIDO", id_)] = _Visual("SOLIDO", id_, None, malla, actor, cara_de_celda=s.cara_de_triangulo,
                                                  actor_bordes=actor_bordes)
        self._clave_por_actor[hash(actor)] = ("SOLIDO", id_)

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
            n_celdas = visual.malla.n_cells
            colores = np.tile(COLORES_BASE[visual.tipo], (n_celdas, 1)).astype(np.uint8)
            for objetivo in self._seleccion:
                if (objetivo.tipo, objetivo.id) != clave:
                    continue
                if objetivo.cara is None:
                    colores[:] = COLOR_ELEMENTO_SELECCIONADO
                elif visual.cara_de_celda is not None:  # sólido de forma libre: todos los triángulos de la superficie
                    colores[visual.cara_de_celda == objetivo.cara] = COLOR_CARA_SELECCIONADA
                else:
                    colores[objetivo.cara] = COLOR_CARA_SELECCIONADA
            visual.malla.cell_data["colores"][:] = colores
            visual.malla.Modified()
        self.render()

    # ------------------------------------------------------------------
    # Selección por zona: arrastrar un rectángulo
    # ------------------------------------------------------------------
    def set_navegacion(self, modo: str) -> None:
        """Qué hace arrastrar con el botón izquierdo: ``NAV_SELECCION`` (rectángulo de selección), ``NAV_MOVER``
        (desplaza la vista) o ``NAV_ROTAR`` (gira la vista)."""

        if modo not in (NAV_SELECCION, NAV_MOVER, NAV_ROTAR):
            raise ValueError(modo)
        self._navegacion = modo
        self.setCursor({NAV_SELECCION: Qt.ArrowCursor, NAV_MOVER: Qt.OpenHandCursor, NAV_ROTAR: Qt.SizeAllCursor}[modo])

    @property
    def navegacion(self) -> str:
        return self._navegacion

    def _pixel_vtk(self, ev) -> Tuple[float, float]:
        ratio = self.devicePixelRatioF()
        return ev.position().x() * ratio, (self.height() - 1 - ev.position().y()) * ratio

    # Todo el ratón se gestiona aquí (no se delega en el estilo de VTK, que gira con el botón izquierdo):
    #   izquierdo      seleccionar (clic) o rectángulo de selección (arrastre); Ctrl/Mayús suma a la selección
    #   central        desplazar la vista; con Ctrl o Mayús, girarla
    #   rueda          zoom
    #   derecho        menú contextual (clic)
    def mousePressEvent(self, ev) -> None:
        self.setFocus()
        boton, mod = ev.button(), ev.modifiers()
        con_tecla = bool(mod & (Qt.ControlModifier | Qt.ShiftModifier))
        x, y = self._pixel_vtk(ev)
        if boton == Qt.MiddleButton:
            accion = "ROTAR" if con_tecla else "MOVER"
        elif boton == Qt.LeftButton:
            accion = {NAV_MOVER: "MOVER", NAV_ROTAR: "ROTAR"}.get(self._navegacion, "CAJA" if self._vista == "MODELO" else "CLIC")
        elif boton == Qt.RightButton:
            accion = "DERECHO"
        else:
            return
        self._arrastre = {"accion": accion, "boton": boton, "inicio": (x, y), "ultimo": (x, y), "sumar": con_tecla}
        if accion == "MOVER":
            self.setCursor(Qt.ClosedHandCursor)
        elif accion == "ROTAR":
            self.setCursor(Qt.SizeAllCursor)

    def mouseMoveEvent(self, ev) -> None:
        a = self._arrastre
        if a is None:
            return
        x, y = self._pixel_vtk(ev)
        dx, dy = x - a["ultimo"][0], y - a["ultimo"][1]
        a["ultimo"] = (x, y)
        if a["accion"] == "ROTAR":
            self._orbitar(dx, dy)
        elif a["accion"] == "MOVER":
            self._desplazar(dx, dy)
        elif a["accion"] == "CAJA":
            self._dibujar_caja(a["inicio"], (x, y))

    def mouseReleaseEvent(self, ev) -> None:
        a = self._arrastre
        if a is None or ev.button() != a["boton"]:
            return
        self._arrastre = None
        self.set_navegacion(self._navegacion)  # restituye el cursor
        x, y = self._pixel_vtk(ev)
        inicio = a["inicio"]
        fue_clic = abs(x - inicio[0]) <= UMBRAL_ARRASTRE_PX and abs(y - inicio[1]) <= UMBRAL_ARRASTRE_PX
        accion = a["accion"]
        if accion == "CAJA":
            self._dibujar_caja(None, None)
            if fue_clic:
                self._clic_en_modelo(int(x), int(y), a["sumar"])
            else:
                self.seleccionar_zona(inicio, (x, y), sumar=a["sumar"])
        elif accion == "CLIC" and fue_clic:
            self._sondear(int(x), int(y))
        elif accion == "DERECHO" and fue_clic:
            if self._vista == "MODELO":
                self.clic_derecho_en(int(x), int(y))
            else:
                self.menu_fondo.emit()
        elif accion in ("MOVER", "ROTAR") and fue_clic and a["boton"] == Qt.LeftButton:
            # con la mano o el giro activos, un clic sin arrastrar también selecciona
            if self._vista == "ESFUERZOS":
                self._sondear(int(x), int(y))
            else:
                self._clic_en_modelo(int(x), int(y), a["sumar"])

    def _punto_bajo_cursor(self, x: float, y: float) -> np.ndarray:
        """Punto del mundo bajo el píxel: el de la superficie que se ve ahí o, si no hay nada, el del plano de enfoque."""

        if self._picker.Pick(x, y, 0, self.renderer):
            return np.array(self._picker.GetPickPosition())
        cam = self.renderer.GetActiveCamera()
        ancho, alto = self.renderer.GetSize()
        m = cam.GetCompositeProjectionTransformMatrix(ancho / max(alto, 1), -1, 1)
        inversa = np.linalg.inv(np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)]))

        def mundo(z: float) -> np.ndarray:
            p = inversa @ np.array([2.0 * x / ancho - 1.0, 2.0 * y / alto - 1.0, z, 1.0])
            return p[:3] / p[3]

        cerca, lejos = mundo(-1.0), mundo(1.0)
        foco, normal = np.array(cam.GetFocalPoint()), np.array(cam.GetDirectionOfProjection())
        direccion = lejos - cerca
        denominador = float(np.dot(normal, direccion))
        if abs(denominador) < 1e-12:
            return foco
        return cerca + direccion * (float(np.dot(normal, foco - cerca)) / denominador)

    def wheelEvent(self, ev) -> None:
        """Zoom hacia el cursor moviendo la cámara (sin cambiar el ángulo de visión, que deformaría la perspectiva):
        el punto que está bajo el cursor se queda donde está."""

        pasos = ev.angleDelta().y() / 120.0
        if not pasos:
            return
        x, y = self._pixel_vtk(ev)
        cam = self.renderer.GetActiveCamera()
        pos, foco = np.array(cam.GetPosition()), np.array(cam.GetFocalPoint())
        distancia = float(np.linalg.norm(foco - pos))
        factor = 1.15 ** pasos  # > 1 acerca
        # límites: ni dentro de una pieza ni tan lejos que el modelo sea un punto
        minima, maxima = 0.003 * self._escala, 30.0 * self._escala
        factor = float(np.clip(factor, distancia / maxima, distancia / minima))
        punto = self._punto_bajo_cursor(x, y)
        escala = 1.0 / factor
        cam.SetPosition(*(punto + (pos - punto) * escala))
        cam.SetFocalPoint(*(punto + (foco - punto) * escala))
        if cam.GetParallelProjection():
            cam.SetParallelScale(cam.GetParallelScale() * escala)
        self.renderer.ResetCameraClippingRange()
        self.render()
        if self._hay_tamanos_en_pixeles():  # los símbolos con tamaño en píxeles se recalculan al cambiar el zoom
            self._temporizador_escala.start()

    def keyPressEvent(self, ev) -> None:
        ev.ignore()  # las teclas de VTK (w, s, f...) cambian la representación: las atajos son los de la ventana

    def _desplazar(self, dx: float, dy: float) -> None:
        """Mueve la vista: el modelo sigue al cursor."""

        cam = self.renderer.GetActiveCamera()
        alto = max(self.renderer.GetSize()[1], 1)
        pos, foco = np.array(cam.GetPosition()), np.array(cam.GetFocalPoint())
        direccion = foco - pos
        distancia = float(np.linalg.norm(direccion))
        direccion /= max(distancia, 1e-12)
        arriba = np.array(cam.GetViewUp())
        derecha = np.cross(direccion, arriba)
        derecha /= max(np.linalg.norm(derecha), 1e-12)
        arriba = np.cross(derecha, direccion)
        if cam.GetParallelProjection():
            por_pixel = 2.0 * cam.GetParallelScale() / alto
        else:
            por_pixel = 2.0 * distancia * math.tan(math.radians(cam.GetViewAngle()) / 2.0) / alto
        delta = -(dx * derecha + dy * arriba) * por_pixel
        cam.SetPosition(*(pos + delta))
        cam.SetFocalPoint(*(foco + delta))
        self.renderer.ResetCameraClippingRange()
        self.render()

    def _orbitar(self, dx: float, dy: float) -> None:
        """Gira la vista alrededor del punto enfocado manteniendo el eje Z vertical (como SAP2000 o Revit)."""

        cam = self.renderer.GetActiveCamera()
        pos, foco = np.array(cam.GetPosition()), np.array(cam.GetFocalPoint())
        angulo = math.radians(-dx * 0.4)
        c, s_ = math.cos(angulo), math.sin(angulo)
        giro_z = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        pos = foco + giro_z @ (pos - foco)
        arriba = giro_z @ np.array(cam.GetViewUp())
        cam.SetPosition(*pos)
        cam.SetViewUp(*arriba)
        direccion = foco - pos
        elevacion = math.degrees(math.asin(np.clip(-direccion[2] / max(np.linalg.norm(direccion), 1e-12), -1.0, 1.0)))
        delta = float(np.clip(-dy * 0.4, -89.0 - elevacion, 89.0 - elevacion))
        cam.Elevation(delta)
        if abs(elevacion + delta) < 89.5:
            cam.SetViewUp(0.0, 0.0, 1.0)
        else:
            cam.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self.render()

    def _dibujar_caja(self, a, b) -> None:
        """Rectángulo de selección: continuo y azul si se arrastra de izquierda a derecha (solo lo que queda dentro);
        discontinuo y verde de derecha a izquierda (todo lo que toca). Lleva un relleno traslúcido para verse bien."""

        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D

        if self._caja_actor is None:
            self._caja_actor = (vtkActor2D(), vtkActor2D())  # (contorno, relleno)
            for actor in self._caja_actor:
                actor.SetMapper(vtkPolyDataMapper2D())
            self._caja_actor[0].GetProperty().SetLineWidth(2.0)
        contorno, relleno = self._caja_actor
        for actor in self._caja_actor:  # la escena se reconstruye (importar, deshacer...) y borra todos los actores
            if not self.renderer.HasViewProp(actor):
                self.add_actor(actor, reset_camera=False, render=False, pickable=False)
        if a is None:
            for actor in self._caja_actor:
                actor.SetVisibility(False)
            self.render()
            return
        puntos = vtkPoints()
        for x, y in ((a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])):
            puntos.InsertNextPoint(x, y, 0.0)
        lineas = vtkCellArray()
        lineas.InsertNextCell(5)
        for i in (0, 1, 2, 3, 0):
            lineas.InsertCellPoint(i)
        caras = vtkCellArray()
        caras.InsertNextCell(4)
        for i in (0, 1, 2, 3):
            caras.InsertCellPoint(i)
        datos_linea, datos_relleno = vtkPolyData(), vtkPolyData()
        datos_linea.SetPoints(puntos)
        datos_linea.SetLines(lineas)
        datos_relleno.SetPoints(puntos)
        datos_relleno.SetPolys(caras)
        contorno.GetMapper().SetInputData(datos_linea)
        relleno.GetMapper().SetInputData(datos_relleno)
        cruzando = b[0] < a[0]
        color = (0.1, 0.65, 0.25) if cruzando else (0.1, 0.35, 0.9)
        contorno.GetProperty().SetColor(*color)
        contorno.GetProperty().SetLineStipplePattern(0x00FF if cruzando else 0xFFFF)
        relleno.GetProperty().SetColor(*color)
        relleno.GetProperty().SetOpacity(0.12)
        for actor in self._caja_actor:
            actor.SetVisibility(True)
        self.render()
        self.update()

    def seleccionar_zona(self, a, b, sumar: bool = False) -> None:
        """Selecciona lo que hay en el rectángulo de esquinas ``a`` y ``b`` (píxeles de VTK). De izquierda a derecha
        se eligen solo los elementos (o caras) que quedan por completo dentro; de derecha a izquierda, los que tocan
        el rectángulo."""

        nuevos = self.objetivos_en_zona(a, b, cruzando=b[0] < a[0])
        if sumar:
            actuales = list(self._seleccion)
            nuevos = actuales + [o for o in nuevos if o not in actuales]
        self.seleccionar(nuevos)

    def _proyectar(self, puntos: np.ndarray):
        """Píxeles (x, y) de VTK de los puntos del mundo y máscara de los que están delante de la cámara."""

        camara = self.renderer.GetActiveCamera()
        ancho, alto = self.renderer.GetSize()
        m = camara.GetCompositeProjectionTransformMatrix(ancho / max(alto, 1), -1, 1)
        matriz = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        h = np.hstack([puntos, np.ones((len(puntos), 1))]) @ matriz.T
        delante = h[:, 3] > 1e-9
        w = np.where(delante, h[:, 3], 1.0)
        return np.column_stack([(h[:, 0] / w + 1.0) * 0.5 * ancho, (h[:, 1] / w + 1.0) * 0.5 * alto]), delante

    def objetivos_en_zona(self, a, b, cruzando: bool) -> List[Objetivo]:
        from vtkmodules.util.numpy_support import vtk_to_numpy

        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        posicion = np.array(self.renderer.GetActiveCamera().GetPosition())
        resultado: List[Objetivo] = []
        for (tipo, id_), visual in self._visuales.items():
            if not self._visibles[tipo]:
                continue
            malla = visual.malla
            puntos = np.asarray(malla.points, dtype=float)
            pantalla, delante = self._proyectar(puntos)
            dentro = delante & (pantalla[:, 0] >= x0) & (pantalla[:, 0] <= x1) & (pantalla[:, 1] >= y0) & (pantalla[:, 1] <= y1)
            polis = malla.GetPolys()
            desplazamientos = vtk_to_numpy(polis.GetOffsetsArray()).astype(np.int64)
            conectividad = vtk_to_numpy(polis.GetConnectivityArray()).astype(np.int64)
            n_celdas = len(desplazamientos) - 1
            tamanos = np.diff(desplazamientos)
            celda_de_vertice = np.repeat(np.arange(n_celdas), tamanos)
            todos = np.bincount(celda_de_vertice, weights=dentro[conectividad], minlength=n_celdas) == tamanos
            # recuadro en pantalla de cada celda y su solape con el rectángulo
            px, py = pantalla[conectividad, 0], pantalla[conectividad, 1]
            minx, maxx = np.minimum.reduceat(px, desplazamientos[:-1]), np.maximum.reduceat(px, desplazamientos[:-1])
            miny, maxy = np.minimum.reduceat(py, desplazamientos[:-1]), np.maximum.reduceat(py, desplazamientos[:-1])
            detras = np.bincount(celda_de_vertice, weights=~delante[conectividad], minlength=n_celdas) > 0
            solapa = (maxx >= x0) & (minx <= x1) & (maxy >= y0) & (miny <= y1) & ~detras
            if visual.solido is not None:  # barras, losas y muros: normal y centro de cada cara
                normales = np.array([c.normal for c in visual.solido.caras])
                centros = np.array([c.centro for c in visual.solido.caras])
            else:
                p = puntos[conectividad.reshape(-1, 3)]
                normales = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
                centros = p.mean(axis=1)
            de_frente = np.einsum("ij,ij->i", normales, posicion - centros) > 0
            if self._modo == MODO_ELEMENTO:
                elegido = bool(dentro[np.unique(conectividad)].all()) if not cruzando else bool(solapa.any())
                if elegido:
                    resultado.append(Objetivo(tipo, id_, None))
                continue
            if visual.cara_de_celda is None:
                cumple = (todos if not cruzando else solapa) & de_frente
                resultado += [Objetivo(tipo, id_, int(i)) for i in np.where(cumple)[0]]
            else:
                cara = visual.cara_de_celda
                n_cara = np.bincount(cara)
                if cruzando:
                    elegidas = np.bincount(cara, weights=(solapa & de_frente), minlength=len(n_cara)) > 0
                else:
                    dentro_cara = np.bincount(cara, weights=todos, minlength=len(n_cara)) == n_cara
                    con_frente = np.bincount(cara, weights=de_frente, minlength=len(n_cara)) > 0
                    elegidas = dentro_cara & con_frente
                resultado += [Objetivo(tipo, id_, int(i)) for i in np.where(elegidas)[0]]
        return resultado

    def _elegir(self, x: int, y: int):
        """(clave, índice de celda) bajo el píxel, o (None, -1)."""

        if not self._picker.Pick(x, y, 0, self.renderer):
            return None, -1
        actor = self._picker.GetActor()
        if actor is None:
            return None, -1
        return self._clave_por_actor.get(hash(actor)), self._picker.GetCellId()

    def _cara_en_celda(self, clave, celda: int) -> int:
        """Cara (superficie) a la que pertenece la celda picada: una celda por cara en barras, losas y
        muros; en los sólidos de forma libre, la superficie del triángulo."""

        visual = self._visuales.get(clave)
        if visual is not None and visual.cara_de_celda is not None:
            return int(visual.cara_de_celda[celda])
        return celda

    def clic_derecho_en(self, x: int, y: int) -> None:
        """Clic derecho en el píxel (x, y): si el elemento bajo el cursor no
        está en la selección pasa a ser la selección; luego pide el menú."""

        clave, celda = self._elegir(x, y)
        if clave is None:  # sobre el fondo: opciones de la vista (estilo, vistas, apoyos, cargas...)
            self.menu_fondo.emit()
            return
        tipo, id_ = clave
        ya_elegido = any((o.tipo, o.id) == (tipo, id_) for o in self._seleccion)
        if not ya_elegido:
            self.seleccionar([Objetivo(tipo, id_, self._cara_en_celda(clave, celda) if self._modo == MODO_CARA else None)])
        if self._seleccion:
            self.menu_contextual.emit()

    def _clic_en_modelo(self, x: int, y: int, sumar: bool) -> None:
        clave, celda = self._elegir(x, y)
        if clave is None:
            if not sumar:
                self.seleccionar([])
            return

        tipo, id_ = clave
        objetivo = Objetivo(tipo, id_, self._cara_en_celda(clave, celda) if self._modo == MODO_CARA else None)
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
            if visual.actor_bordes is not None:
                visual.actor_bordes.SetVisibility(self._visibles[visual.tipo])

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
    @conserva_camara
    def mostrar_asignaciones(
        self,
        apoyos: Sequence,
        cargas_area: Sequence[CargaArea],
        cargas_lineales: Sequence[CargaLineal],
        cargas_puntuales: Sequence[CargaPuntual],
        conexiones: Sequence[Conexion] = (),
        cargas_superficie: Sequence[CargaSuperficie] = (),
        uniones: Sequence[UnionSuperficie] = (),
    ) -> None:
        """Dibuja los apoyos, las conexiones y las cargas del caso activo
        (las que se pasan). Las flechas de una carga variable tienen longitud
        proporcional a su valor, de modo que se ve su forma."""

        for nombre in tuple(f"ASIG_APOYOS_{c}" for c in COLORES_SIMBOLO) + ("ASIG_APOYOS_AREA", "ASIG_CONEXIONES", "ASIG_CARGAS", "ASIG_MOMENTOS", "ASIG_ETIQUETAS",
                                                                                "ASIG_SUPERFICIE_APOYO", "ASIG_SUPERFICIE_CARGA", "ASIG_UNIONES", "ASIG_UNIONES_CARAS"):
            self.remove_actor(nombre, render=False)
        self._argumentos_asignaciones = (apoyos, cargas_area, cargas_lineales, cargas_puntuales, conexiones, cargas_superficie, uniones)
        proyecto = self._proyecto
        if proyecto is None:
            return

        # Tamaño de los símbolos: una fracción del modelo y, si hay barras, nunca mayor que parte de la barra típica
        barras = proyecto.todas_las_barras()
        largo = float(np.clip(0.018 * self._escala, 0.003, 0.8))
        longitudes = [proyecto.longitud_barra(i) for i in barras]
        if longitudes:
            largo = min(largo, 0.12 * float(np.median(longitudes)))
        largo_base = largo
        largo_apoyo = self._tamano("apoyos", largo_base)
        largo_union = self._tamano("constraints", largo_base)

        def punto_de(barra_id: str, s: float) -> np.ndarray:
            barra = proyecto.elemento_barra(barra_id)
            a = np.array(proyecto.nodos[barra.nodo_inicio].coordenadas)
            b = np.array(proyecto.nodos[barra.nodo_fin].coordenadas)
            return a + s * (b - a)

        # -- apoyos puntuales: símbolo según los grados de libertad restringidos ------------
        por_color: Dict[str, List[pv.PolyData]] = {}
        por_color_sup: Dict[str, List[pv.PolyData]] = {}
        etiquetas_apoyo: List[np.ndarray] = []
        textos_apoyo: List[str] = []
        for a in apoyos:
            if isinstance(a, (ApoyoArea, ApoyoSuperficie)) or a.nodo_id not in proyecto.nodos:
                continue
            grados = (a.ux, a.uy, a.uz, a.rx, a.ry, a.rz)
            centro = np.array(proyecto.nodos[a.nodo_id].coordenadas)
            for color, mallas in simbolo_apoyo(grados, centro, largo_apoyo).items():
                por_color.setdefault(color, []).extend(mallas)
            etiquetas_apoyo.append(centro - np.array([0.0, 0.0, 1.35 * largo_apoyo]))
            textos_apoyo.append(nombre_apoyo(grados))
        for color, mallas in por_color.items():
            if self._apoyos_visibles:
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
                n = max(3, int(round(longitud * (apoyo.s_fin - apoyo.s_inicio) / (largo_apoyo * 1.2))) + 1)
                for s in np.linspace(apoyo.s_inicio, apoyo.s_fin, n):
                    conos_area.append(
                        pv.Cone(
                            center=punto_de(barra_id, s) - direccion * largo_apoyo * 0.5, direction=direccion,
                            height=largo_apoyo * 0.9, radius=largo_apoyo * 0.4, resolution=12,
                        )
                    )
                etiquetas_apoyo.append(punto_de(barra_id, 0.5 * (apoyo.s_inicio + apoyo.s_fin)) - direccion * largo_apoyo * 0.9)
                textos_apoyo.append("apoyo " + "".join(
                    n_ for n_, activo in zip(("Ux", "Uy", "Uz", "Rx", "Ry", "Rz"), (apoyo.ux, apoyo.uy, apoyo.uz, apoyo.rx, apoyo.ry, apoyo.rz)) if activo
                ))
        if conos_area and self._apoyos_visibles:
            self.add_mesh(pv.merge(conos_area), color=COLOR_APOYO_AREA, name="ASIG_APOYOS_AREA", pickable=False, show_scalar_bar=False)

        # -- conexiones ----------------------------------------------------------------------
        marcas: List[pv.PolyData] = []
        for c in conexiones:
            if c.nodo_id not in proyecto.nodos:
                continue
            centro = np.array(proyecto.nodos[c.nodo_id].coordenadas)
            marcas.append(pv.Sphere(radius=largo_apoyo * 0.3, center=centro))
            if c.nodo_destino is not None and c.nodo_destino in proyecto.nodos:
                destino = np.array(proyecto.nodos[c.nodo_destino].coordenadas)
                marcas.append(pv.Sphere(radius=largo_apoyo * 0.3, center=destino))
                if np.linalg.norm(destino - centro) > 1e-9:
                    marcas.append(pv.Line(centro, destino).tube(radius=largo_apoyo * 0.08))
        if marcas:
            self.add_mesh(pv.merge(marcas), color=COLOR_CONEXION, name="ASIG_CONEXIONES", pickable=False, show_scalar_bar=False)

        # -- cargas ------------------------------------------------------------------------------
        largo = self._tamano("cargas", largo_base)  # tamaño de las flechas: Ver → Escala de símbolos
        factor_cargas = largo / max(largo_base, 1e-12)
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

        # -- apoyos y cargas sobre superficies (análisis 3D) -----------------------------------
        caras_apoyo: List[pv.PolyData] = []
        for apoyo in apoyos:
            if not isinstance(apoyo, ApoyoSuperficie):
                continue
            geo = self._geometria_de_cara(apoyo.elemento_tipo, apoyo.elemento_id, apoyo.cara)
            if geo is None:
                continue
            malla, centro, normal, muestras = geo
            caras_apoyo.append(malla)
            grados = (apoyo.ux, apoyo.uy, apoyo.uz, apoyo.rx, apoyo.ry, apoyo.rz)
            for punto in muestras[:4]:
                for color, mallas in simbolo_apoyo(grados, punto, largo_apoyo * 0.7).items():
                    por_color_sup.setdefault(color, []).extend(mallas)
            etiquetas_apoyo.append(centro + normal * largo_apoyo * 0.4)
            textos_apoyo.append("pasador" if getattr(apoyo, "pasador", False)
                                else "empotrada" if all(grados[:3]) else nombre_apoyo(grados))
        if caras_apoyo and self._apoyos_visibles:
            self.add_mesh(pv.merge(caras_apoyo), color=COLOR_APOYO_AREA, opacity=0.65, name="ASIG_SUPERFICIE_APOYO",
                          pickable=False, show_scalar_bar=False)
        for color, mallas in por_color_sup.items():
            if self._apoyos_visibles:
                self.add_mesh(pv.merge(mallas), color=color, name=f"ASIG_APOYOS_{color}", pickable=False, show_scalar_bar=False)

        caras_carga: List[pv.PolyData] = []
        maxima_presion = max((abs(c.valor) for c in cargas_superficie if c.tipo != "MOMENTO"), default=1.0) or 1.0
        for carga in cargas_superficie:
            geo = self._geometria_de_cara(carga.elemento_tipo, carga.elemento_id, carga.cara)
            if geo is None:
                continue
            malla, centro, normal, muestras = geo
            caras_carga.append(malla)
            # la flecha nunca es mayor que la superficie que carga: así no tapa el modelo en piezas pequeñas
            tamano_cara = float(np.sqrt(max(malla.area, 1e-12)))
            l_flecha = min(largo, 0.6 * tamano_cara * factor_cargas)
            if carga.tipo == "MOMENTO":
                d = np.asarray(carga.direccion, dtype=float) * (1.0 if carga.valor > 0 else -1.0)
                momentos.append(flecha(centro + d * l_flecha * 0.2, d, l_flecha))
                momentos.append(flecha(centro + d * l_flecha * 0.55, d, l_flecha))
                puntos_etiqueta.append(centro + d * l_flecha * 0.9)
                textos.append(f"M = {abs(carga.valor):g} kN·m")
                continue
            if carga.direccion is not None:
                d = np.asarray(carga.direccion, dtype=float) * (1.0 if carga.valor > 0 else -1.0)
            else:
                d = -normal * (1.0 if carga.valor > 0 else -1.0)
            d = d / np.linalg.norm(d)
            # en una presión, la flecha es proporcional al valor respecto de la mayor presión dibujada
            longitud = l_flecha * (0.5 + 0.7 * abs(carga.valor) / maxima_presion) if carga.tipo == "PRESION" else l_flecha * 1.1
            for punto in muestras:
                flechas.append(flecha(punto, d, longitud))
            puntos_etiqueta.append(centro - d * longitud * 1.15)
            textos.append(f"{abs(carga.valor):g} " + ("kPa" if carga.tipo == "PRESION" else "kN"))
        # -- constraints entre superficies --------------------------------------------------------
        caras_union: List[pv.PolyData] = []
        enlaces: List[pv.PolyData] = []
        for u in (uniones if self._uniones_visibles else ()):
            ga = self._geometria_de_cara(u.a_tipo, u.a_id, u.a_cara)
            gb = self._geometria_de_cara(u.b_tipo, u.b_id, u.b_cara)
            if ga is None or gb is None:
                continue
            caras_union += [ga[0], gb[0]]
            enlaces += [pv.Sphere(radius=largo_union * 0.25, center=ga[1]), pv.Sphere(radius=largo_union * 0.25, center=gb[1])]
            if np.linalg.norm(gb[1] - ga[1]) > 1e-9:
                enlaces.append(pv.Line(ga[1], gb[1]).tube(radius=largo_union * 0.07))
        if caras_union:
            self.add_mesh(pv.merge(caras_union), color=COLOR_CONEXION, opacity=0.5, name="ASIG_UNIONES_CARAS",
                          pickable=False, show_scalar_bar=False)
            self.add_mesh(pv.merge(enlaces), color=COLOR_CONEXION, name="ASIG_UNIONES", pickable=False, show_scalar_bar=False)

        if not self._flechas_visibles:  # las cargas se ocultan (los apoyos siguen a la vista)
            flechas, momentos, caras_carga, puntos_etiqueta, textos = [], [], [], [], []
        if caras_carga:
            self.add_mesh(pv.merge(caras_carga), color=COLOR_CARGA, opacity=0.45, name="ASIG_SUPERFICIE_CARGA",
                          pickable=False, show_scalar_bar=False)

        if flechas:
            self.add_mesh(pv.merge(flechas), color=COLOR_CARGA, name="ASIG_CARGAS", pickable=False, show_scalar_bar=False)
        if momentos:
            self.add_mesh(pv.merge(momentos), color=COLOR_MOMENTO, name="ASIG_MOMENTOS", pickable=False, show_scalar_bar=False)
        if not self._apoyos_visibles:  # los apoyos se ocultan con sus etiquetas
            etiquetas_apoyo, textos_apoyo = [], []
        puntos_etiqueta += etiquetas_apoyo
        textos += textos_apoyo
        if puntos_etiqueta:
            self.add_point_labels(
                np.array(puntos_etiqueta), textos, name="ASIG_ETIQUETAS", font_size=self._tamano_etiqueta(), text_color="darkred",
                shape=None, show_points=False, always_visible=True, pickable=False,
            )
        self.render()

    # ------------------------------------------------------------------
    # Vista de esfuerzos
    # ------------------------------------------------------------------
    @conserva_camara
    def mostrar_esfuerzos(
        self,
        malla: MallaTensiones,
        valores: np.ndarray,
        barra_de_celda: Sequence[str],
        titulo: str,
        rango: Optional[Tuple[float, float]] = None,
        cmap: str = "turbo",
        unidad: str = "MPa",
        texto: str = "",
        ocultar_planos: bool = False,
        desplazamientos: Optional[np.ndarray] = None,
        escala_deformada: float = 1.0,
        factor_deformada: float = 1.0,
        mostrar_original: bool = True,
    ) -> None:
        """Mapa de contorno de ``valores`` (uno por nodo de ``malla``). Con ``ocultar_planos``
        (análisis sólido) las losas y los muros tampoco se dibujan: ya están en la malla.

        Con ``desplazamientos`` (uno por nodo, en m) se dibuja la forma deformada: los
        puntos se mueven ``escala_deformada × factor_deformada × desplazamiento`` (el
        factor, de 0 a 1, es el de la animación) y, si se pide, se deja el contorno sin deformar."""

        self._vista = "ESFUERZOS"
        for nombre in ("MALLA", "MALLA_LAMINAS"):
            self.remove_actor(nombre, render=False)
        for visual in self._visuales.values():
            if visual.tipo in ("VIGA", "COLUMNA"):
                visual.actor.SetVisibility(False)
                visual.actor.SetPickable(False)
            else:
                visual.actor.SetVisibility(self._visibles[visual.tipo] and not ocultar_planos)
                visual.actor.SetPickable(False)
                if visual.actor_bordes is not None:
                    visual.actor_bordes.SetVisibility(self._visibles[visual.tipo] and not ocultar_planos)
                visual.actor.GetProperty().SetOpacity(OPACIDAD_PLANOS_EN_ESFUERZOS)

        celdas, tipos = malla.celdas_vtk()  # hexaedros, tetraedros y cuñas (láminas) con su forma real
        coordenadas = np.asarray(malla.coordenadas, dtype=float)
        grid = pv.UnstructuredGrid(celdas, tipos, coordenadas.copy())
        grid.point_data["esfuerzo"] = np.asarray(valores, dtype=float)
        self._puntos_sin_deformar = coordenadas
        self._desplazamientos = None if desplazamientos is None else np.asarray(desplazamientos, dtype=float)
        self._escala_deformada = escala_deformada
        self.remove_actor("SIN_DEFORMAR", render=False)
        if self._desplazamientos is not None:
            grid.points = coordenadas + escala_deformada * factor_deformada * self._desplazamientos
            if mostrar_original:
                with warnings.catch_warnings():  # pyvista cambiará el algoritmo por defecto de extract_surface
                    warnings.simplefilter("ignore")
                    original = pv.UnstructuredGrid(celdas, tipos, coordenadas.copy()).extract_surface()
                self.add_mesh(original.extract_feature_edges(feature_angle=35, boundary_edges=True, manifold_edges=False),
                              color="#7a8794", line_width=1.2, opacity=0.7, name="SIN_DEFORMAR", pickable=False,
                              show_scalar_bar=False)
        self._mostrar_asignaciones(self._desplazamientos is None)

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
            scalar_bar_args={"title": unidad, "vertical": True, "n_labels": 8, "fmt": "%.2f",
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

    def mostrar_malla(self, celdas: np.ndarray, tipos: np.ndarray, coordenadas: np.ndarray, es_lamina: np.ndarray,
                      texto: str = "") -> None:
        """Malla de elementos finitos (botón Mallar): sólidos en gris azulado y láminas en ocre, con las aristas de cada
        elemento. Los elementos del modelo se ocultan, como en la vista de esfuerzos."""

        self._quitar_resultados()
        self._vista = "MALLA"
        self._grid_esfuerzos = None
        for visual in self._visuales.values():
            visual.actor.SetVisibility(False)
            visual.actor.SetPickable(False)
            if visual.actor_bordes is not None:
                visual.actor_bordes.SetVisibility(False)
        grid = pv.UnstructuredGrid(celdas, tipos, np.asarray(coordenadas, dtype=float).copy())
        for nombre, mascara, color in (("MALLA", ~es_lamina, "#c9d6e3"), ("MALLA_LAMINAS", es_lamina, "#e3c48f")):
            if mascara.any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parte = grid.extract_cells(np.flatnonzero(mascara)).extract_surface()
                self.add_mesh(parte, color=color, show_edges=True, edge_color="#2f3b47", line_width=0.6, name=nombre,
                              pickable=False, show_scalar_bar=False)
        self._mostrar_asignaciones(False)  # las caras resaltadas de apoyos y constraints taparían la malla
        if texto:
            self.add_text(texto, position="upper_left", font_size=11, color="black", name="TEXTO_ESFUERZOS")
        self.render()

    def _quitar_resultados(self) -> None:
        for nombre in ("ESFUERZOS", "TEXTO_ESFUERZOS", "SIN_DEFORMAR", "MALLA", "MALLA_LAMINAS"):
            self.remove_actor(nombre, render=False)
        self.remove_scalar_bar_si_existe()

    # ------------------------------------------------------------------
    # Tamaño de los símbolos
    # ------------------------------------------------------------------
    @property
    def escalas(self) -> dict:
        return dict(self._escalas)

    def set_escalas(self, escalas: dict) -> None:
        """Fija el tamaño de los símbolos: ``{parámetro: (modo, valor)}`` con el modo ``PORCENTAJE`` (100 = automático)
        o ``PIXELES`` (tamaño en pantalla, que se mantiene al acercar o alejar)."""

        for parametro, (modo, valor) in escalas.items():
            if parametro in self._escalas and modo in (ESCALA_PORCENTAJE, ESCALA_PIXELES) and valor > 0:
                self._escalas[parametro] = (modo, float(valor))
        self._redibujar_asignaciones()

    def set_escala(self, parametro: str, modo: str, valor: float) -> None:
        self.set_escalas({parametro: (modo, valor)})

    def _metros_por_pixel(self) -> float:
        cam = self.renderer.GetActiveCamera()
        alto = max(self.renderer.GetSize()[1], 1)
        if cam.GetParallelProjection():
            return 2.0 * cam.GetParallelScale() / alto
        distancia = float(np.linalg.norm(np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())))
        return 2.0 * distancia * math.tan(math.radians(cam.GetViewAngle()) / 2.0) / alto

    def _tamano(self, parametro: str, automatico: float) -> float:
        modo, valor = self._escalas[parametro]
        return valor * self._metros_por_pixel() if modo == ESCALA_PIXELES else automatico * valor / 100.0

    def _tamano_etiqueta(self) -> int:
        modo, valor = self._escalas["etiquetas"]
        puntos = valor * 0.75 if modo == ESCALA_PIXELES else TAMANO_ETIQUETA_AUTOMATICO * valor / 100.0
        return int(np.clip(round(puntos), 5, 72))

    def _hay_tamanos_en_pixeles(self) -> bool:
        return any(modo == ESCALA_PIXELES for modo, _ in self._escalas.values())

    def _redibujar_asignaciones(self) -> None:
        if self._argumentos_asignaciones is not None:
            self.mostrar_asignaciones(*self._argumentos_asignaciones)
            if self._vista == "ESFUERZOS":
                self._mostrar_asignaciones(self._desplazamientos is None)
            elif self._vista == "MALLA":
                self._mostrar_asignaciones(False)

    def configurar_apoyos(self, visibles: bool) -> None:
        """Muestra u oculta los símbolos de apoyo."""

        self._apoyos_visibles = bool(visibles)
        self._redibujar_asignaciones()

    def configurar_uniones(self, visibles: bool) -> None:
        """Muestra u oculta los constraints (superficies resaltadas, enlaces y etiquetas)."""

        self._uniones_visibles = bool(visibles)
        if self._argumentos_asignaciones is not None:
            self.mostrar_asignaciones(*self._argumentos_asignaciones)
            if self._vista == "ESFUERZOS":
                self._mostrar_asignaciones(self._desplazamientos is None)
            elif self._vista == "MALLA":
                self._mostrar_asignaciones(False)

    def configurar_flechas(self, visibles: bool) -> None:
        """Muestra u oculta las flechas de las cargas."""

        self._flechas_visibles = bool(visibles)
        self._redibujar_asignaciones()

    def _geometria_de_cara(self, tipo: str, id_: str, cara: int):
        """(malla de la cara, centro, normal hacia fuera, puntos de muestra) o None si ya no existe."""

        proyecto = self._proyecto
        if cara is not None and cara < 0:  # elemento completo (constraint entre elementos 3D)
            try:
                if tipo == "SOLIDO":
                    s = proyecto.solidos[id_]
                    todo = pv.PolyData(np.asarray(s.vertices, dtype=float),
                                       np.hstack([np.full((len(s.triangulos), 1), 3), s.triangulos]).ravel().astype(np.int64))
                else:
                    sol = solido_elemento(proyecto, tipo, id_)
                    caras = np.hstack([[len(c.indices), *c.indices] for c in sol.caras]).astype(np.int64)
                    todo = pv.PolyData(np.asarray(sol.puntos, dtype=float), faces=caras)
            except (KeyError, IndexError):
                return None
            centro = np.asarray(todo.center, dtype=float)
            todo.points = centro + (np.asarray(todo.points) - centro) * 1.004  # un poco más grande: evita el parpadeo con el elemento
            return todo, centro, np.array([0.0, 0.0, 1.0]), np.array([centro])
        try:
            if tipo == "SOLIDO":
                s = proyecto.solidos[id_]
                c = s.caras[cara]
                tri = s.triangulos[c.triangulos]
                malla = pv.PolyData(np.asarray(s.vertices, dtype=float),
                                    np.hstack([np.full((len(tri), 1), 3), tri]).ravel().astype(np.int64))
                centro, normal = np.asarray(c.centro, dtype=float), np.asarray(c.normal, dtype=float)
            else:
                c = solido_elemento(proyecto, tipo, id_).caras[cara]
                pts = np.asarray(c.puntos, dtype=float)
                malla = pv.PolyData(pts, np.hstack([[len(pts)], np.arange(len(pts))]).astype(np.int64)).triangulate()
                centro, normal = np.asarray(c.centro, dtype=float), np.asarray(c.normal, dtype=float)
        except (KeyError, IndexError):
            return None
        malla = malla.clean()
        malla.points = malla.points + normal * 0.002 * self._escala  # evita el parpadeo con la cara original
        # puntos de muestra: muestreo del punto más lejano sobre los centros de los triángulos
        cel = malla.cell_centers().points
        elegidos = [int(np.argmin(np.linalg.norm(cel - centro, axis=1)))]
        distancia = np.linalg.norm(cel - cel[elegidos[0]], axis=1)
        while len(elegidos) < min(7, len(cel)):
            k = int(np.argmax(distancia))
            elegidos.append(k)
            distancia = np.minimum(distancia, np.linalg.norm(cel - cel[k], axis=1))
        return malla, centro, normal / max(np.linalg.norm(normal), 1e-12), cel[elegidos]

    def _mostrar_asignaciones(self, visibles: bool) -> None:
        """Apoyos y cargas no acompañan a la deformada: se ocultan mientras se ve."""

        for nombre, actor in list(self.actors.items()):
            if nombre.startswith("ASIG_"):
                actor.SetVisibility(visibles)

    def actualizar_deformada(self, factor: float) -> None:
        """Mueve la malla de esfuerzos a ``factor`` (0 = sin deformar, 1 = deformada completa)."""

        if self._grid_esfuerzos is None or self._desplazamientos is None or self._puntos_sin_deformar is None:
            return
        self._grid_esfuerzos.points = self._puntos_sin_deformar + self._escala_deformada * factor * self._desplazamientos
        self._grid_esfuerzos.Modified()
        self.render()

    @property
    def hay_deformada(self) -> bool:
        return self._grid_esfuerzos is not None and self._desplazamientos is not None

    def guardar_video(self, ruta: str, fotogramas: int = 60, fps: int = 24, al_avanzar=None) -> None:
        """Graba un ciclo de la deformada (0 → 1 → 0) con la vista actual: MP4 (necesita
        ``imageio-ffmpeg``) o GIF según la extensión de ``ruta``."""

        if not self.hay_deformada:
            raise RuntimeError("No hay una deformada que grabar.")
        import imageio.v2 as imageio

        es_gif = str(ruta).lower().endswith(".gif")
        escritor = (imageio.get_writer(ruta, mode="I", duration=1000.0 / fps, loop=0) if es_gif
                    else imageio.get_writer(ruta, fps=fps, codec="libx264", quality=8, macro_block_size=None,
                                            pixelformat="yuv420p"))
        try:
            for k in range(fotogramas):
                self.actualizar_deformada(0.5 * (1.0 - math.cos(2.0 * math.pi * k / fotogramas)))
                imagen = np.asarray(self.screenshot(return_img=True))[:, :, :3]
                alto, ancho = (imagen.shape[0] // 2) * 2, (imagen.shape[1] // 2) * 2  # H.264 pide dimensiones pares
                escritor.append_data(imagen[:alto, :ancho])
                if al_avanzar is not None:
                    al_avanzar(k + 1, fotogramas)
        finally:
            escritor.close()
            self.actualizar_deformada(1.0)

    def remove_scalar_bar_si_existe(self) -> None:
        for titulo in list(self.scalar_bars.keys()):
            self.remove_scalar_bar(titulo, render=False)

    @conserva_camara
    def volver_al_modelo(self, reconstruir: bool = False) -> None:
        """Vuelve a la vista del modelo (o la reconstruye desde cero)."""

        self._vista = "MODELO"
        if reconstruir:
            self._limpiar_escena()
            self.add_axes()
            self._reconstruir_visuales()
            self._seleccion = []
        else:
            self._quitar_resultados()
            self._mostrar_asignaciones(True)
            self._aplicar_estilo()
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
