"""Visor 3D del modelo estructural, basado en PyVista embebido en Qt
(pyvistaqt). Permite seleccionar vigas, columnas, losas y muros con un
clic; la selección se comunica al resto de la aplicación mediante la
señal ``elemento_seleccionado``.

Nota de implementación: a diferencia de un enfoque que combina todos los
elementos de un mismo tipo en una sola malla (más eficiente para modelos
muy grandes), aquí cada elemento se agrega como una malla y un actor de
VTK independientes. Es una opción deliberada para la Fase 1: hace que la
identificación del elemento clicado sea directa y confiable (un actor =
un elemento), a costa de rendimiento en modelos con miles de elementos —
aceptable para el alcance de esta fase.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError:  # pragma: no cover - entornos sin PyVista instalado
    pv = None
    QtInteractor = object

import logging

from PySide6.QtCore import Signal

from ..core.modelo import Losa, Muro, Proyecto

logger = logging.getLogger(__name__)

COLOR_VIGA = "steelblue"
COLOR_COLUMNA = "indianred"
COLOR_LOSA = "lightgray"
COLOR_MURO = "khaki"
COLOR_SELECCION = "gold"

_COLORES_POR_TIPO = {"VIGA": COLOR_VIGA, "COLUMNA": COLOR_COLUMNA, "LOSA": COLOR_LOSA, "MURO": COLOR_MURO}


class Visor3D(QtInteractor):
    """Widget de visualización 3D con selección de elementos por clic."""

    elemento_seleccionado = Signal(str, str)  # (tipo, id)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proyecto: Optional[Proyecto] = None
        self._tipo_id_por_mesh_id: Dict[int, Tuple[str, str]] = {}
        self._actor_por_tipo_id: Dict[Tuple[str, str], object] = {}
        self.set_background("white")
        try:
            self.enable_mesh_picking(callback=self._al_seleccionar_malla, show=False, show_message=False)
        except Exception:
            # En equipos sin aceleración 3D disponible (p. ej. algunos
            # entornos remotos sin GPU) la selección por clic en el visor
            # no está disponible, pero el resto de la aplicación (listas,
            # diálogos, resultados) sigue funcionando con normalidad.
            logger.warning("No se pudo habilitar la selección por clic en el visor 3D.", exc_info=True)

    def mostrar_proyecto(self, proyecto: Proyecto) -> None:
        self._proyecto = proyecto
        self.clear()
        self._tipo_id_por_mesh_id.clear()
        self._actor_por_tipo_id.clear()

        for barra in proyecto.vigas.values():
            self._agregar_barra(proyecto, barra, "VIGA")
        for barra in proyecto.columnas.values():
            self._agregar_barra(proyecto, barra, "COLUMNA")
        for losa in proyecto.losas.values():
            self._agregar_superficie(proyecto, losa, "LOSA")
        for muro in proyecto.muros.values():
            self._agregar_superficie(proyecto, muro, "MURO")

        self.reset_camera()
        self.render()

    def _agregar_barra(self, proyecto: Proyecto, barra, tipo: str) -> None:
        n1 = proyecto.nodos[barra.nodo_inicio]
        n2 = proyecto.nodos[barra.nodo_fin]
        linea = pv.Line(n1.coordenadas, n2.coordenadas)
        tubo = linea.tube(radius=0.04)
        actor = self.add_mesh(tubo, color=_COLORES_POR_TIPO[tipo], pickable=True, name=f"{tipo}_{barra.id}")
        self._registrar(tubo, actor, tipo, barra.id)

    def _agregar_superficie(self, proyecto: Proyecto, elemento, tipo: str) -> None:
        puntos = np.array([proyecto.nodos[nid].coordenadas for nid in elemento.nodos_contorno])
        n = len(puntos)
        caras = np.hstack([[n], np.arange(n)])
        malla = pv.PolyData(puntos, faces=caras)
        actor = self.add_mesh(
            malla, color=_COLORES_POR_TIPO[tipo], opacity=0.55, pickable=True, name=f"{tipo}_{elemento.id}",
            show_edges=True,
        )
        self._registrar(malla, actor, tipo, elemento.id)

    def _registrar(self, malla, actor, tipo: str, id_: str) -> None:
        self._tipo_id_por_mesh_id[id(malla)] = (tipo, id_)
        self._actor_por_tipo_id[(tipo, id_)] = actor

    def _al_seleccionar_malla(self, malla) -> None:
        clave = self._tipo_id_por_mesh_id.get(id(malla))
        if clave is None:
            return
        tipo, id_ = clave
        self.resaltar(tipo, id_)
        self.elemento_seleccionado.emit(tipo, id_)

    def resaltar(self, tipo: str, id_: str) -> None:
        """Resalta un elemento (p. ej. tras seleccionarlo en una tabla, no
        solo con el clic en el visor) y restaura el color del resto."""

        for (t, i), actor in self._actor_por_tipo_id.items():
            color = COLOR_SELECCION if (t, i) == (tipo, id_) else _COLORES_POR_TIPO[t]
            actor.prop.color = color
        self.render()
