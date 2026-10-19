"""Guardado y carga del proyecto completo (modelo + cargas + casos +
combinaciones) en un archivo JSON propio de la aplicación.

No es un formato de intercambio estándar (eso es el IFC, en la
importación); es simplemente la forma de no perder el trabajo de un
proyecto entre sesiones de la aplicación.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from .cargas import Apoyo, ApoyoArea, CargaArea, CargaLineal, CargaPuntual, Conexion, PerfilPresion, SistemaReferencia
from .casos_carga import Combinacion, CasoCarga, TipoCaso
from .modelo import (
    DireccionLosa,
    ElementoBarra,
    Losa,
    Material,
    ModoAnalisisLosa,
    Muro,
    Nodo,
    PerfilSeccion,
    Proyecto,
    TipoElementoBarra,
)


def _a_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _a_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_a_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _a_dict(v) for k, v in obj.items()}
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


def guardar_proyecto(
    ruta: Path,
    proyecto: Proyecto,
    apoyos: list,
    casos: Dict[str, CasoCarga],
    combinaciones: Dict[str, Combinacion],
    cargas_area_por_caso: Dict[str, list],
    cargas_lineales_por_caso: Dict[str, list],
    cargas_puntuales_por_caso: Dict[str, list],
    conexiones: list = (),
) -> None:
    datos = {
        "nombre": proyecto.nombre,
        "nodos": [_a_dict(n) for n in proyecto.nodos.values()],
        "materiales": [_a_dict(m) for m in proyecto.materiales.values()],
        "perfiles": [_a_dict(p) for p in proyecto.perfiles.values()],
        "vigas": [_a_dict(v) for v in proyecto.vigas.values()],
        "columnas": [_a_dict(c) for c in proyecto.columnas.values()],
        "losas": [_a_dict(l) for l in proyecto.losas.values()],
        "muros": [_a_dict(m) for m in proyecto.muros.values()],
        "apoyos": [{**_a_dict(a), "clase": "area" if isinstance(a, ApoyoArea) else "nodal"} for a in apoyos],
        "conexiones": [_a_dict(c) for c in conexiones],
        "casos": [_a_dict(c) for c in casos.values()],
        "combinaciones": [_a_dict(c) for c in combinaciones.values()],
        "cargas_area": {caso: [_a_dict(c) for c in lista] for caso, lista in cargas_area_por_caso.items()},
        "cargas_lineales": {caso: [_a_dict(c) for c in lista] for caso, lista in cargas_lineales_por_caso.items()},
        "cargas_puntuales": {caso: [_a_dict(c) for c in lista] for caso, lista in cargas_puntuales_por_caso.items()},
    }
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar_proyecto(ruta: Path):
    """Devuelve (proyecto, apoyos, casos, combinaciones, cargas_area_por_caso,
    cargas_lineales_por_caso, cargas_puntuales_por_caso, conexiones)."""

    datos = json.loads(ruta.read_text(encoding="utf-8"))

    proyecto = Proyecto(nombre=datos.get("nombre", "Proyecto"))
    for n in datos.get("nodos", []):
        proyecto.nodos[n["id"]] = Nodo(**n)
    for m in datos.get("materiales", []):
        proyecto.materiales[m["nombre"]] = Material(**m)
    for p in datos.get("perfiles", []):
        proyecto.perfiles[p["nombre"]] = PerfilSeccion(**p)
    for v in datos.get("vigas", []):
        v = dict(v)
        v["tipo"] = TipoElementoBarra(v["tipo"])
        v["vector_n1"] = tuple(v["vector_n1"])
        proyecto.vigas[v["id"]] = ElementoBarra(**v)
    for c in datos.get("columnas", []):
        c = dict(c)
        c["tipo"] = TipoElementoBarra(c["tipo"])
        c["vector_n1"] = tuple(c["vector_n1"])
        proyecto.columnas[c["id"]] = ElementoBarra(**c)
    for l in datos.get("losas", []):
        l = dict(l)
        l["modo_analisis"] = ModoAnalisisLosa(l["modo_analisis"])
        l["direccion"] = DireccionLosa(l["direccion"])
        proyecto.losas[l["id"]] = Losa(**l)
    for m in datos.get("muros", []):
        proyecto.muros[m["id"]] = Muro(**m)

    apoyos = []
    for a in datos.get("apoyos", []):
        a = dict(a)
        clase = a.pop("clase", "nodal")  # los archivos antiguos solo tienen apoyos nodales
        apoyos.append(ApoyoArea(**a) if clase == "area" else Apoyo(**a))
    conexiones = [Conexion(**c) for c in datos.get("conexiones", [])]
    casos = {c["id"]: CasoCarga(**{**c, "tipo": TipoCaso(c["tipo"])}) for c in datos.get("casos", [])}
    combinaciones = {c["id"]: Combinacion(**c) for c in datos.get("combinaciones", [])}
    def carga_area(c: dict) -> CargaArea:
        c = dict(c)
        if c.get("perfil"):
            c["perfil"] = PerfilPresion(**c["perfil"])
        c["direccion"] = tuple(c["direccion"])
        return CargaArea(**c)

    cargas_area_por_caso = {caso: [carga_area(c) for c in lista] for caso, lista in datos.get("cargas_area", {}).items()}
    cargas_lineales_por_caso = {
        caso: [
            CargaLineal(**{**c, "sistema": SistemaReferencia(c["sistema"])})
            for c in lista
        ]
        for caso, lista in datos.get("cargas_lineales", {}).items()
    }
    cargas_puntuales_por_caso = {
        caso: [CargaPuntual(**c) for c in lista] for caso, lista in datos.get("cargas_puntuales", {}).items()
    }

    return (
        proyecto,
        apoyos,
        casos,
        combinaciones,
        cargas_area_por_caso,
        cargas_lineales_por_caso,
        cargas_puntuales_por_caso,
        conexiones,
    )
