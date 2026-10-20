"""Edición del modelo: eliminar elementos y todo lo que depende de ellos.

Sin dependencias de la interfaz gráfica, para poder probarlo aisladamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .cargas import ApoyoArea, ApoyoSuperficie, Conexion
from .modelo import Proyecto

TIPOS_ELIMINABLES = ("VIGA", "COLUMNA", "LOSA", "MURO", "SOLIDO")


@dataclass
class ResultadoEliminacion:
    elementos: List[Tuple[str, str]] = field(default_factory=list)  # (tipo, id) eliminados
    apoyos: int = 0
    cargas: int = 0
    conexiones: int = 0
    constraints: int = 0
    nodos: int = 0

    def resumen(self) -> str:
        partes = [f"{len(self.elementos)} elemento(s)"]
        for cantidad, texto in ((self.apoyos, "apoyo(s)"), (self.cargas, "carga(s)"), (self.conexiones, "conexión(es)"),
                                (self.constraints, "constraint(s)"), (self.nodos, "nodo(s) sin uso")):
            if cantidad:
                partes.append(f"{cantidad} {texto}")
        return ", ".join(partes)


def _nodos_en_uso(proyecto: Proyecto) -> Set[str]:
    usados: Set[str] = set()
    for b in proyecto.todas_las_barras().values():
        usados.update((b.nodo_inicio, b.nodo_fin))
    for e in list(proyecto.losas.values()) + list(proyecto.muros.values()):
        usados.update(e.nodos_contorno)
    return usados


def eliminar_elementos(
    proyecto: Proyecto, elementos: Iterable[Tuple[str, str]], apoyos: Sequence, conexiones: Sequence[Conexion],
    cargas_area: Dict[str, list], cargas_lineales: Dict[str, list], cargas_puntuales: Dict[str, list],
    cargas_superficie: Dict[str, list],
) -> Tuple[list, list, ResultadoEliminacion]:
    """Quita ``elementos`` (pares ``(tipo, id)``) del proyecto y todo lo que quedaría colgando: los apoyos, cargas,
    conexiones y constraints que los usan y los nodos que ya no pertenecen a nada. Modifica ``proyecto`` y los
    diccionarios de cargas en el sitio; devuelve las listas nuevas de apoyos y de conexiones y un resumen."""

    resultado = ResultadoEliminacion()
    quitar = {(t, i) for t, i in elementos if t in TIPOS_ELIMINABLES}
    colecciones = {"VIGA": proyecto.vigas, "COLUMNA": proyecto.columnas, "LOSA": proyecto.losas, "MURO": proyecto.muros,
                   "SOLIDO": proyecto.solidos}
    for tipo, id_ in sorted(quitar):
        if id_ in colecciones[tipo]:
            del colecciones[tipo][id_]
            resultado.elementos.append((tipo, id_))
    barras_quitadas = {i for t, i in quitar if t in ("VIGA", "COLUMNA")}
    for losa in proyecto.losas.values():  # una viga eliminada ya no soporta ningún lado
        losa.vigas_soporte = [None if v in barras_quitadas else v for v in losa.vigas_soporte]

    def es_de_quitado(tipo, id_) -> bool:
        return (tipo, id_) in quitar

    # nodos que siguen perteneciendo a algún elemento
    en_uso = _nodos_en_uso(proyecto)

    apoyos_nuevos = []
    for a in apoyos:
        if isinstance(a, (ApoyoArea, ApoyoSuperficie)):
            fuera = es_de_quitado(a.elemento_tipo, a.elemento_id)
        else:
            fuera = a.nodo_id not in en_uso
        if fuera:
            resultado.apoyos += 1
        else:
            apoyos_nuevos.append(a)

    conexiones_nuevas = []
    for c in conexiones:
        if (c.barra_id in barras_quitadas or c.nodo_id not in en_uso
                or (c.nodo_destino is not None and c.nodo_destino not in en_uso)):
            resultado.conexiones += 1
        else:
            conexiones_nuevas.append(c)

    def filtrar(por_caso: Dict[str, list], quitar_si) -> None:
        for caso in list(por_caso):
            nuevas = [c for c in por_caso[caso] if not quitar_si(c)]
            resultado.cargas += len(por_caso[caso]) - len(nuevas)
            por_caso[caso] = nuevas

    filtrar(cargas_lineales, lambda c: c.elemento_id in barras_quitadas)
    filtrar(cargas_puntuales, lambda c: (c.elemento_id in barras_quitadas) if c.nodo_id is None else c.nodo_id not in en_uso)
    filtrar(cargas_area, lambda c: (c.losa_id and es_de_quitado("LOSA", c.losa_id)) or (c.muro_id and es_de_quitado("MURO", c.muro_id)))
    filtrar(cargas_superficie, lambda c: es_de_quitado(c.elemento_tipo, c.elemento_id))

    restantes = []
    for u in proyecto.uniones:
        if es_de_quitado(u.a_tipo, u.a_id) or es_de_quitado(u.b_tipo, u.b_id):
            resultado.constraints += 1
        else:
            restantes.append(u)
    proyecto.uniones = restantes

    for nodo in [n for n in proyecto.nodos if n not in en_uso]:
        del proyecto.nodos[nodo]
        resultado.nodos += 1
    return apoyos_nuevos, conexiones_nuevas, resultado
