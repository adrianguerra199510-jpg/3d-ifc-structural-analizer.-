"""Detección automática de los elementos que se tocan (o casi) para conectarlos con constraints.

Se malla el modelo grueso, se buscan por parejas de piezas las caras de borde que se miran (normales
opuestas), están casi en el mismo plano y se solapan, y se devuelve cada pareja de elementos con el área de
contacto y la separación. Es el mismo criterio de las uniones automáticas del análisis 3D, pero con una
tolerancia que elige el usuario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..core.modelo import Proyecto
from .malla import MallaSolida, construir_malla_solida, estimar_hexaedros
from .miembros import construir_miembros, recortar_uniones


@dataclass
class Contacto:
    a_tipo: str
    a_id: str
    b_tipo: str
    b_id: str
    area: float  # m² de superficie en contacto
    separacion: float  # m: 0 si se tocan

    def descripcion(self) -> str:
        sep = "se tocan" if self.separacion < 5e-4 else f"separados {self.separacion * 1000:.1f} mm"
        return f"{self.a_tipo.title()} {self.a_id} ↔ {self.b_tipo.title()} {self.b_id}  ({self.area * 1e4:.0f} cm², {sep})"


def _elemento_de(proyecto: Proyecto, parte) -> Tuple[str, str]:
    if parte.tipo == "BARRA":
        barra = parte.miembro.barras[0]
        return ("VIGA" if barra in proyecto.vigas else "COLUMNA"), barra
    return parte.tipo, parte.id


def _contacto_entre(malla: MallaSolida, i: int, j: int, tol: float) -> Optional[Tuple[float, float]]:
    """(área, separación) de las caras de borde de las partes ``i`` y ``j`` que se miran a menos de ``tol``."""

    a, b = malla.fronteras(malla.partes[i]), malla.fronteras(malla.partes[j])
    if len(a.area) == 0 or len(b.area) == 0:
        return None
    sa, sb = np.sqrt(a.area), np.sqrt(b.area)
    arbol = cKDTree(b.centro)
    radio = 0.5 * (sa.max() + sb.max()) * 1.05 + tol
    area, separacion, hay = 0.0, np.inf, False
    for k, vecinas in enumerate(arbol.query_ball_point(a.centro, radio)):
        for m in vecinas:
            if np.dot(a.normal[k], b.normal[m]) > -0.99:
                continue
            d = b.centro[m] - a.centro[k]
            hacia = abs(np.dot(a.normal[k], d))
            if hacia > tol:
                continue
            if np.linalg.norm(d - np.dot(a.normal[k], d) * a.normal[k]) > 0.5 * (sa[k] + sb[m]) * 0.99 * 1.1:
                continue
            hay = True
            area += float(min(a.area[k], b.area[m]))
            separacion = min(separacion, hacia)
    return (area, float(separacion)) if hay else None


def detectar_contactos(proyecto: Proyecto, tolerancia: float, hexaedros_maximos: int = 25_000) -> List[Contacto]:
    """Parejas de elementos que se tocan o están a menos de ``tolerancia`` (m)."""

    miembros = construir_miembros(proyecto)
    recortar_uniones(proyecto, miembros)
    tamano = 0.25
    for candidato in (0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25):
        if estimar_hexaedros(proyecto, miembros, candidato) <= hexaedros_maximos:
            tamano = candidato
            break
    malla = construir_malla_solida(proyecto, miembros, tamano)
    tol = max(tolerancia, 2e-3) + 0.1 * tamano
    cajas = []
    for parte in malla.partes:
        pts = malla.coordenadas[np.unique(malla.hexaedros[parte.primer_hexaedro: parte.ultimo_hexaedro])]
        cajas.append((pts.min(axis=0) - tol, pts.max(axis=0) + tol))
    resultado: List[Contacto] = []
    vistos = set()
    for i in range(len(malla.partes)):
        for j in range(i + 1, len(malla.partes)):
            if not (np.all(cajas[i][0] <= cajas[j][1]) and np.all(cajas[j][0] <= cajas[i][1])):
                continue
            contacto = _contacto_entre(malla, i, j, tol)
            if contacto is None:
                continue
            a, b = _elemento_de(proyecto, malla.partes[i]), _elemento_de(proyecto, malla.partes[j])
            if (a, b) in vistos or a == b:
                continue
            vistos.add((a, b))
            resultado.append(Contacto(a[0], a[1], b[0], b[1], contacto[0], contacto[1]))
    return sorted(resultado, key=lambda c: -c.area)
