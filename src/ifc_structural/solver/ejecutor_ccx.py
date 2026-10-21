"""Ejecución de CalculiX (ccx) mediante subprocess."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


class ErrorEjecucionCcx(RuntimeError):
    """Error al ejecutar ccx: binario no encontrado, timeout, o análisis fallido."""


#: Causas conocidas de fallo de ccx y qué hacer al respecto.
_PISTAS = (
    (
        "zero coefficient on the",
        "CalculiX no pudo expandir una viga inclinada que tiene giros restringidos en uno de "
        "sus nodos (limitación conocida de ccx con ciertas inclinaciones). Pruebe a asignar "
        "un apoyo articulado (sin restringir los giros) en ese nodo, o a variar levemente la "
        "inclinación de la barra.",
    ),
    (
        "beam normal in 2-direction",
        "CalculiX no pudo definir la orientación de la sección de una barra (eje degenerado "
        "o de longitud casi nula). Revise las barras muy cortas o con nodos duplicados.",
    ),
)


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

    @property
    def errores(self) -> List[str]:
        return [linea.strip() for linea in self.salida.splitlines() if "*ERROR" in linea]

    def diagnostico(self) -> str:
        """Explicación legible de por qué falló la ejecución."""

        for fragmento, pista in _PISTAS:
            if fragmento in self.salida:
                return pista
        errores = self.errores
        if errores:
            return "CalculiX informó: " + " | ".join(errores[:3])
        return f"CalculiX terminó sin éxito (código {self.codigo_salida})."


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

    # solo el nombre, relativo a la carpeta de trabajo: CalculiX rechaza rutas de más de 127 caracteres (habitual en
    # Windows con carpetas de usuario u OneDrive largas)
    nombre_base = ruta_inp.with_suffix("").name
    try:
        proceso = subprocess.run(
            [ruta_ejecutable, nombre_base],
            cwd=str(ruta_inp.parent),
            capture_output=True,
            text=True,
            timeout=tiempo_limite,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # sin ventana de consola en Windows
        )
    except FileNotFoundError as exc:
        raise ErrorEjecucionCcx(
            f"No se encontró el ejecutable de CalculiX ('{ruta_ejecutable}'). "
            "Instálelo o indique su ruta en la clave 'ruta_ccx' de config.json."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ErrorEjecucionCcx(
            f"ccx no terminó dentro de {tiempo_limite} s para '{ruta_inp.name}'."
        ) from exc

    salida = (proceso.stdout or "") + (proceso.stderr or "")
    exitoso = proceso.returncode == 0 and "Job finished" in salida and not any(
        "*ERROR" in linea for linea in salida.splitlines()
    )
    return ResultadoEjecucion(
        ruta_inp=ruta_inp,
        codigo_salida=proceso.returncode,
        salida=salida,
        exitoso=exitoso,
    )


def ejecutar_ccx_todos_los_casos(
    rutas_inp: Dict[str, Path],
    ruta_ejecutable: str = "ccx",
    tiempo_limite: float = 300.0,
    al_avanzar: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, ResultadoEjecucion]:
    """Ejecuta ccx para cada (caso_id -> ruta .inp) y devuelve los resultados.

    No se detiene ante el primer error: ejecuta todos los casos y deja que
    el llamador decida qué hacer con los que fallaron (están marcados con
    ``exitoso=False`` en su ``ResultadoEjecucion``). ``al_avanzar`` recibe
    (caso_id, número de caso en curso, total) antes de cada ejecución.
    """

    resultados: Dict[str, ResultadoEjecucion] = {}
    total = len(rutas_inp)
    for i, (caso_id, ruta_inp) in enumerate(rutas_inp.items(), start=1):
        if al_avanzar is not None:
            al_avanzar(caso_id, i, total)
        resultados[caso_id] = ejecutar_ccx(ruta_inp, ruta_ejecutable, tiempo_limite)
    return resultados
