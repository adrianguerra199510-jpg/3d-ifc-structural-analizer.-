"""Carga de la configuración de la aplicación (config.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

RUTA_CONFIG_POR_DEFECTO = Path(__file__).resolve().parent.parent.parent / "config.json"
RUTA_CONFIG_EJEMPLO = Path(__file__).resolve().parent.parent.parent / "config.example.json"


@dataclass
class Configuracion:
    ruta_ccx: str = "ccx"
    directorio_trabajo: str = "./trabajo_ccx"

    @property
    def directorio_trabajo_absoluto(self) -> Path:
        return Path(self.directorio_trabajo).expanduser().resolve()


def cargar_configuracion(ruta: Optional[Path] = None) -> Configuracion:
    """Carga config.json si existe; si no, usa config.example.json o,
    en su defecto, los valores por defecto (ccx en el PATH)."""

    for candidata in (ruta, RUTA_CONFIG_POR_DEFECTO, RUTA_CONFIG_EJEMPLO):
        if candidata is not None and Path(candidata).is_file():
            datos = json.loads(Path(candidata).read_text(encoding="utf-8"))
            return Configuracion(
                ruta_ccx=datos.get("ruta_ccx", "ccx"),
                directorio_trabajo=datos.get("directorio_trabajo", "./trabajo_ccx"),
            )
    return Configuracion()
