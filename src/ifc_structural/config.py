"""Carga de la configuración de la aplicación (config.json)."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

def carpeta_aplicacion() -> Path:
    """Carpeta del ejecutable (versión portable) o raíz del proyecto (código fuente)."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


def _ruta_config(nombre: str) -> Path:
    return carpeta_aplicacion() / nombre


def buscar_ccx_incluido() -> Optional[Path]:
    """CalculiX que viene con la aplicación, si existe: ``ccx/ccx.exe`` junto
    al ejecutable portable o ``herramientas/ccx/`` en el código fuente."""

    base = carpeta_aplicacion()
    for carpeta in (base / "ccx", base / "herramientas" / "ccx"):
        for nombre in (("ccx.exe",) if sys.platform == "win32" else ("ccx",)):
            candidato = carpeta / nombre
            if candidato.is_file():
                return candidato
    return None


def resolver_ruta_ccx(valor: str) -> str:
    """Interpreta ``ruta_ccx``: con ``"ccx"`` o ``"auto"`` usa el CalculiX
    incluido si existe y, si no, el que esté en el PATH; cualquier otro valor
    se respeta tal cual."""

    if valor.strip().lower() in ("", "ccx", "auto"):
        incluido = buscar_ccx_incluido()
        if incluido is not None:
            return str(incluido)
        return shutil.which("ccx") or "ccx"
    return valor


@dataclass
class Configuracion:
    ruta_ccx: str = "auto"
    directorio_trabajo: str = "./trabajo_ccx"

    @property
    def ruta_ccx_efectiva(self) -> str:
        return resolver_ruta_ccx(self.ruta_ccx)

    @property
    def directorio_trabajo_absoluto(self) -> Path:
        ruta = Path(self.directorio_trabajo).expanduser()
        if getattr(sys, "frozen", False) and not ruta.is_absolute():
            ruta = carpeta_aplicacion() / ruta  # portable: junto al ejecutable, no en el directorio actual
        return ruta.resolve()


def cargar_configuracion(ruta: Optional[Path] = None) -> Configuracion:
    """Carga config.json si existe; si no, usa config.example.json o,
    en su defecto, los valores por defecto (ccx en el PATH)."""

    for candidata in (ruta, _ruta_config("config.json"), _ruta_config("config.example.json")):
        if candidata is not None and Path(candidata).is_file():
            datos = json.loads(Path(candidata).read_text(encoding="utf-8"))
            return Configuracion(
                ruta_ccx=datos.get("ruta_ccx", "auto"),
                directorio_trabajo=datos.get("directorio_trabajo", "./trabajo_ccx"),
            )
    return Configuracion()
