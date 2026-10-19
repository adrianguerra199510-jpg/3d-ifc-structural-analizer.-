"""Ejecución de CalculiX (ccx) mediante subprocess."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class ErrorEjecucionCcx(RuntimeError):
    """Error al ejecutar ccx: binario no encontrado, timeout, o análisis fallido."""


@dataclass
class ResultadoEjecucion:
    ruta_inp: Path
    codigo_salida: int
    salida: str
    exitoso: bool

    @property
    def ruta_dat(self) -> Path:
        return self.ruta_inp.with_suffix(".dat")

    @property
    def ruta_frd(self) -> Path:
        return self.ruta_inp.with_suffix(".frd")


def ejecutar_ccx(
    ruta_inp: Path,
    ruta_ejecutable: str = "ccx",
    tiempo_limite: float = 300.0,
) -> ResultadoEjecucion:
    """Ejecuta ccx sobre un .inp y verifica que haya terminado con éxito.

    CalculiX se invoca como ``ccx <nombre_sin_extension>`` (sin la
    extensión .inp) y, en esta versión, siempre devuelve código de salida
    0 incluso si el análisis falló por un error en el .inp; por eso la
    verificación real de éxito se hace buscando la cadena "Job finished"
    en la salida estándar, no solo el código de retorno.
    """

    nombre_base = ruta_inp.with_suffix("")
    try:
        proceso = subprocess.run(
            [ruta_ejecutable, str(nombre_base)],
            cwd=str(ruta_inp.parent),
            capture_output=True,
            text=True,
            timeout=tiempo_limite,
        )
    except FileNotFoundError as exc:
        raise ErrorEjecucionCcx(
            f"No se encontró el ejecutable de CalculiX ('{ruta_ejecutable}'). "
            "Verifique la clave 'ruta_ccx' en config.json."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ErrorEjecucionCcx(
            f"ccx no terminó dentro de {tiempo_limite} s para '{ruta_inp.name}'."
        ) from exc

    salida = (proceso.stdout or "") + (proceso.stderr or "")
    exitoso = proceso.returncode == 0 and "Job finished" in salida
    return ResultadoEjecucion(
        ruta_inp=ruta_inp,
        codigo_salida=proceso.returncode,
        salida=salida,
        exitoso=exitoso,
    )


def ejecutar_ccx_todos_los_casos(
    rutas_inp: dict,
    ruta_ejecutable: str = "ccx",
    tiempo_limite: float = 300.0,
) -> dict:
    """Ejecuta ccx para cada (caso_id -> ruta .inp) y devuelve los resultados.

    No se detiene ante el primer error: ejecuta todos los casos y deja que
    el llamador decida qué hacer con los que fallaron (están marcados con
    ``exitoso=False`` en su ``ResultadoEjecucion``).
    """

    resultados = {}
    for caso_id, ruta_inp in rutas_inp.items():
        resultados[caso_id] = ejecutar_ccx(ruta_inp, ruta_ejecutable, tiempo_limite)
    return resultados
