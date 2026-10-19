"""Hilo de trabajo que ejecuta el análisis sin congelar la ventana."""

from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QThread, Signal

from ..solver.analisis import ErrorAnalisis, ResultadoAnalisis
from ..solver.ejecutor_ccx import ErrorEjecucionCcx

logger = logging.getLogger(__name__)


class TrabajadorAnalisis(QThread):
    """Ejecuta ``funcion(al_avanzar)`` (que devuelve un ``ResultadoAnalisis``)
    en un hilo aparte y avisa con señales."""

    progreso = Signal(str)
    terminado = Signal(object)  # ResultadoAnalisis
    fallido = Signal(str)

    def __init__(self, funcion: Callable[[Callable[[str], None]], ResultadoAnalisis], parent=None):
        super().__init__(parent)
        self._funcion = funcion

    def run(self) -> None:
        try:
            resultado = self._funcion(self.progreso.emit)
        except (ErrorAnalisis, ErrorEjecucionCcx) as exc:
            self.fallido.emit(str(exc))
        except Exception as exc:  # no dejar morir el hilo en silencio
            logger.exception("Error inesperado durante el análisis")
            self.fallido.emit(f"Error inesperado durante el análisis: {exc}")
        else:
            self.terminado.emit(resultado)
