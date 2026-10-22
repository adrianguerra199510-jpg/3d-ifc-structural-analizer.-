"""Apoyos y cargas sobre una superficie cualquiera (una cara de un sólido de forma libre o de una
barra, losa o muro). Solo se aplican en el análisis 3D.

Sin dependencias de la interfaz gráfica, para poder probarlo aisladamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .asignacion import ErrorAsignacion, Objetivo, cara_de, descripcion
from .cargas import ApoyoSuperficie, CargaSuperficie, TIPOS_UNION, UnionSuperficie
from .geometria import solido_elemento
from .modelo import Proyecto

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Superficie:
    """Datos de una superficie elegida, vengan de donde vengan."""

    tipo: str
    elemento_id: str
    cara: int
    nombre: str
    area: float
    normal: Vector3  # unitaria, hacia fuera del sólido
    centro: Vector3


def superficie_de(proyecto: Proyecto, objetivo: Objetivo) -> Superficie:
    cara = cara_de(proyecto, objetivo)
    if cara is None:
        raise ErrorAsignacion(
            f"{descripcion(proyecto, objetivo)}: elija una superficie (modo «Cara») para asignarle apoyos o cargas."
        )
    return Superficie(
        tipo=objetivo.tipo, elemento_id=objetivo.id, cara=objetivo.cara, nombre=descripcion(proyecto, objetivo),
        area=float(cara.area), normal=tuple(float(c) for c in cara.normal), centro=tuple(float(c) for c in cara.centro),
    )


def apoyo_superficie_para(proyecto: Proyecto, objetivo: Objetivo, id_: str, grados: Sequence[bool]) -> ApoyoSuperficie:
    """Apoyo sobre la superficie elegida con los grados de libertad (Ux, Uy, Uz, Rx, Ry, Rz) restringidos."""

    sup = superficie_de(proyecto, objetivo)
    if not any(grados):
        raise ErrorAsignacion(f"{sup.nombre}: marque al menos un grado de libertad restringido.")
    nombres = "".join(n for n, g in zip(("Ux", "Uy", "Uz", "Rx", "Ry", "Rz"), grados) if g)
    return ApoyoSuperficie(
        id=id_, elemento_tipo=sup.tipo, elemento_id=sup.elemento_id, cara=sup.cara,
        ux=bool(grados[0]), uy=bool(grados[1]), uz=bool(grados[2]),
        rx=bool(grados[3]), ry=bool(grados[4]), rz=bool(grados[5]),
        nombre=f"{sup.nombre}: apoyo {nombres}",
    )


def carga_superficie_para(proyecto: Proyecto, objetivo: Objetivo, caso_id: str, id_: str, tipo: str, valor: float,
                          direccion: Optional[Vector3]) -> CargaSuperficie:
    """Carga sobre la superficie elegida: presión (kPa), fuerza total (kN) o momento (kN·m)."""

    sup = superficie_de(proyecto, objetivo)
    if abs(valor) < 1e-12:
        raise ErrorAsignacion(f"{sup.nombre}: el valor es cero.")
    if direccion is not None:
        v = np.asarray(direccion, dtype=float)
        if np.linalg.norm(v) < 1e-9:
            raise ErrorAsignacion(f"{sup.nombre}: la dirección no puede ser el vector nulo.")
        direccion = tuple(float(c) for c in v / np.linalg.norm(v))
    if tipo in ("FUERZA", "MOMENTO") and direccion is None:
        raise ErrorAsignacion(f"{sup.nombre}: indique la dirección de la fuerza o del eje del momento.")
    unidad = {"PRESION": "kPa", "FUERZA": "kN", "MOMENTO": "kN·m"}[tipo]
    donde = {"PRESION": "presión", "FUERZA": "fuerza", "MOMENTO": "momento"}[tipo]
    return CargaSuperficie(
        id=id_, caso_id=caso_id, elemento_tipo=sup.tipo, elemento_id=sup.elemento_id, cara=sup.cara,
        tipo=tipo, valor=float(valor), direccion=direccion, nombre=f"{sup.nombre}: {donde} {valor:g} {unidad}",
    )


def direccion_efectiva(carga: CargaSuperficie, normal_hacia_fuera: Sequence[float]) -> np.ndarray:
    """Sentido en que actúa una presión: el de la dirección elegida o, si no hay, hacia dentro de la superficie."""

    if carga.direccion is not None:
        return np.asarray(carga.direccion, dtype=float)
    return -np.asarray(normal_hacia_fuera, dtype=float)


# ---------------------------------------------------------------------------
# Constraints entre dos superficies
# ---------------------------------------------------------------------------

NOMBRES_UNION = {
    "SOLDADURA": "soldadura",
    "CUERPO_RIGIDO": "cuerpo rígido",
    "IGUALAR": "igualar movimientos",
    "ARTICULACION": "articulación",
}


def _caras_con_puntos(proyecto: Proyecto, tipo: str, id_: str):
    """[(índice de cara, puntos (k, 3), normal hacia fuera)] de todas las caras de un elemento."""

    if tipo == "SOLIDO":
        s = proyecto.solidos[id_]
        return [(i, np.vstack([s.vertices[np.unique(s.triangulos[c.triangulos])], [c.centro]]), np.asarray(c.normal))
                for i, c in enumerate(s.caras)]
    sol = solido_elemento(proyecto, tipo, id_)
    return [(i, np.vstack([c.puntos, [c.centro]]), np.asarray(c.normal)) for i, c in enumerate(sol.caras)]


def caras_mas_cercanas(proyecto: Proyecto, a: Objetivo, b: Objetivo) -> Tuple[Objetivo, Objetivo]:
    """Cuando se eligen elementos enteros: la cara de cada uno por la que se tocan. Se mide la distancia de la
    **superficie** de cada cara a la de la otra (no entre sus vértices: una placa pequeña apoyada en el centro de una
    losa grande está lejos de los vértices de la cara superior de la losa); entre las parejas igual de cercanas se elige
    la de mayor área en contacto y, a igualdad, la que se mira (normales opuestas)."""

    from ..solido.deteccion import Superficie, distancia_con_signo, muestras, superficie_de

    def caras(objetivo: Objetivo):
        sup = superficie_de(proyecto, objetivo.tipo, objetivo.id)
        resultado = []
        for indice in np.unique(sup.cara):
            if objetivo.cara is not None and int(indice) != int(objetivo.cara):
                continue
            sub = Superficie(sup.vertices, sup.triangulos[sup.cara == indice], sup.cara[sup.cara == indice])
            p = sub.vertices[sub.triangulos]
            n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).sum(axis=0)
            usados = sub.vertices[np.unique(sub.triangulos)]
            resultado.append((int(indice), sub, n / max(float(np.linalg.norm(n)), 1e-300), usados.min(axis=0), usados.max(axis=0)))
        return resultado

    caras_a, caras_b = caras(a), caras(b)
    if not caras_a or not caras_b:
        raise ErrorAsignacion("No se encontraron las caras de los elementos elegidos.")

    def hueco(x, y) -> float:  # distancia entre las cajas de dos caras (0 si se cruzan)
        return float(np.linalg.norm(np.maximum(0.0, np.maximum(x[3] - y[4], y[3] - x[4]))))

    parejas = [(hueco(x, y), x, y) for x in caras_a for y in caras_b]
    corte = min(h for h, _, _ in parejas)
    candidatas = []
    datos = {}
    for h, x, y in sorted(parejas, key=lambda q: q[0]):
        if h > corte + 0.05 and len(candidatas) >= 4:
            break
        if id(x[1]) not in datos:
            datos[id(x[1])] = muestras(x[1], 600)
        puntos, peso, _ = datos[id(x[1])]
        d = np.abs(distancia_con_signo(puntos, y[1]))
        dmin = float(d.min())
        en_contacto = float(peso[d <= dmin + max(1e-3, 0.1 * dmin)].sum())
        candidatas.append((dmin, en_contacto, float(np.dot(x[2], y[2])), x[0], y[0]))
    mejor_d = min(c[0] for c in candidatas)
    cercanas = [c for c in candidatas if c[0] <= mejor_d + max(1e-3, 0.1 * mejor_d)]
    _, _, _, ia, ib = max(cercanas, key=lambda c: (round(c[1], 6), -c[2]))
    return Objetivo(a.tipo, a.id, ia), Objetivo(b.tipo, b.id, ib)


def union_para(proyecto: Proyecto, a: Objetivo, b: Objetivo, id_: str, tipo: str, grados: Sequence[bool],
               completo: bool = False, tolerancia: float = 0.0) -> UnionSuperficie:
    """Constraint entre dos objetivos. Si alguno no tiene cara elegida se toma la más cercana al otro, salvo con
    ``completo``: entonces es entre los dos elementos 3D enteros (``tolerancia`` = separación máxima de la soldadura)."""

    if tipo not in TIPOS_UNION:
        raise ErrorAsignacion(f"Tipo de constraint desconocido: {tipo}")
    if completo:
        if a.tipo == b.tipo and a.id == b.id:
            raise ErrorAsignacion("Elija dos elementos distintos.")
        if tipo == "IGUALAR":
            raise ErrorAsignacion("Igualar movimientos se aplica entre superficies, no entre elementos completos.")
        a, b = Objetivo(a.tipo, a.id, -1), Objetivo(b.tipo, b.id, -1)
    elif a.tipo == b.tipo and a.id == b.id and (a.cara is None or b.cara is None or a.cara == b.cara):
        raise ErrorAsignacion("Elija dos superficies distintas (de elementos diferentes o dos caras distintas del mismo).")
    elif a.cara is None or b.cara is None:
        a, b = caras_mas_cercanas(proyecto, a, b)
    if tipo == "IGUALAR" and not any(grados[:3]):
        raise ErrorAsignacion("Marque al menos una traslación (Ux, Uy, Uz) que se iguale.")
    if tipo == "IGUALAR":
        grados = tuple(grados[:3]) + (False, False, False)
    if tipo in ("SOLDADURA", "CUERPO_RIGIDO"):
        grados = (True,) * 6
    if tipo == "ARTICULACION":
        grados = (True, True, True) + tuple(bool(g) for g in grados[3:6])
    if completo:
        nombre_a, nombre_b = descripcion(proyecto, Objetivo(a.tipo, a.id)), descripcion(proyecto, Objetivo(b.tipo, b.id))
    else:
        nombre_a, nombre_b = descripcion(proyecto, a), descripcion(proyecto, b)
    return UnionSuperficie(
        id=id_, tipo=tipo, a_tipo=a.tipo, a_id=a.id, a_cara=int(a.cara), b_tipo=b.tipo, b_id=b.id, b_cara=int(b.cara),
        ux=bool(grados[0]), uy=bool(grados[1]), uz=bool(grados[2]), rx=bool(grados[3]), ry=bool(grados[4]), rz=bool(grados[5]),
        nombre=f"{NOMBRES_UNION[tipo].capitalize()}: {nombre_a} ↔ {nombre_b}" + (" (elementos completos)" if completo else ""),
        tolerancia=float(tolerancia),
    )
