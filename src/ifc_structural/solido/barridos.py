"""Mallas barridas de hexaedros cuadráticos ``C3D20R`` para las piezas prismáticas.

Una pieza es **prismática** cuando es la extrusión recta de una sección constante (columnas, pernos, tubos, barras,
placas gruesas): todas sus caras son las dos tapas, perpendiculares a un eje, o caras laterales paralelas a él. Su
sección (la tapa inferior) se malla con cuadriláteros (Gmsh) y se extruye en capas a lo largo del eje: cada
cuadrilátero de cada capa es un hexaedro de 20 nodos con integración reducida, que da mejores esfuerzos con menos nodos
que los tetraedros y sigue la forma de la pieza con capas regulares.

Orden de nodos de ``C3D20`` (CalculiX): 1-4 la cara inferior (antihorario visto desde la superior), 5-8 la superior,
9-12 los nodos medios de 1-2, 2-3, 3-4, 4-1, 13-16 los de 5-6, 6-7, 7-8, 8-5 y 17-20 los de 1-5, 2-6, 3-7, 4-8. Caras
``S1`` = 1-2-3-4, ``S2`` = 5-8-7-6, ``S3`` = 1-5-6-2, ``S4`` = 2-6-7-3, ``S5`` = 3-7-8-4, ``S6`` = 4-8-5-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.solidos import SolidoGeneral

#: Vértices de cada cara (índices 0-based), en el orden S1..S6 de CalculiX (los mismos que en ``C3D8``).
CARAS_HEX20 = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
#: Nodos medios de cada cara, en el orden de sus aristas (c1-c2, c2-c3, c3-c4, c4-c1).
MEDIOS_CARA_HEX20 = ((8, 9, 10, 11), (15, 14, 13, 12), (16, 12, 17, 8), (17, 13, 18, 9), (18, 14, 19, 10), (19, 15, 16, 11))
#: Carga uniforme sobre una cara cuadrática de 8 nodos: −1/12 en cada vértice y 1/3 en cada nodo medio (cargas nodales
#: consistentes de la función de forma serendípita).
PESOS_CARA_HEX20 = np.array([-1 / 12] * 4 + [1 / 3] * 4)
#: Coste de un hexaedro de 20 nodos frente a un tetraedro C3D10 (nodos por elemento: ≈ 4 frente a ≈ 1,5), para estimar
#: la memoria y el tiempo del cálculo y elegir el tamaño sugerido.
EQUIVALENTE_TETRAEDROS = 2.7
#: Capas mínimas a lo largo del eje (con una sola, la flexión en esa dirección queda pobre).
CAPAS_MINIMAS = 2

_TOL_ANGULO = 1e-4  # |cos| que separa caras de tapa (≈ 1) de caras laterales (≈ 0)


class ErrorBarrido(RuntimeError):
    """La sección no se pudo mallar con cuadriláteros sanos: la pieza va con tetraedros."""


@dataclass
class Prisma:
    eje: np.ndarray  # (3,) unitario, de la tapa inferior a la superior
    z0: float  # posición de la tapa inferior a lo largo del eje
    largo: float
    vertices_tapa: np.ndarray  # (k, 3) tapa inferior
    triangulos_tapa: np.ndarray  # (m, 3)
    area_tapa: float


@dataclass
class MallaHex20:
    coordenadas: np.ndarray  # (N, 3)
    hexaedros: np.ndarray  # (M, 20) orden de CalculiX
    superficies: Dict[int, np.ndarray] = field(default_factory=dict)  # superficie -> (k, 2) (hexaedro local, cara 1..6)
    capas: int = 0
    volumen: float = 0.0


def _normales(vertices: np.ndarray, triangulos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = vertices[triangulos]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    largo = np.linalg.norm(n, axis=1)
    return n / np.maximum(largo, 1e-300)[:, None], 0.5 * largo


def _aristas_de_borde(triangulos: np.ndarray) -> np.ndarray:
    aristas = np.sort(np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]]), axis=1)
    unicas, cuenta = np.unique(aristas, axis=0, return_counts=True)
    return unicas[cuenta == 1]


def prisma_de_solido(solido: SolidoGeneral) -> Optional[Prisma]:
    """El prisma recto que forma el sólido (``None`` si no es la extrusión recta de una sección constante). Si lo es en
    varias direcciones (un bloque), se toma la de menor sección (la más larga)."""

    v = np.asarray(solido.vertices, dtype=float)
    t = np.asarray(solido.triangulos, dtype=np.int64)
    if len(t) < 8 or not np.isfinite(v).all():
        return None
    normal, area = _normales(v, t)
    validos = area > 1e-12 * max(float(area.max()), 1e-300)
    escala = float(np.linalg.norm(v[t].reshape(-1, 3).max(axis=0) - v[t].reshape(-1, 3).min(axis=0)))
    if escala <= 0:
        return None
    usados = np.unique(t[validos])
    # el eje es la normal de una tapa o es perpendicular a dos caras laterales: normales de los triángulos mayores y
    # productos vectoriales de parejas de ellas (en un cilindro las mayores son todas laterales)
    normales: List[np.ndarray] = []
    for k in np.argsort(-area):
        if not validos[k]:
            continue
        if all(abs(float(normal[k] @ c)) < 1 - 1e-6 for c in normales):
            normales.append(normal[k])
        if len(normales) >= 6:
            break
    candidatos: List[np.ndarray] = []
    for d in normales + [np.cross(a, b) for i, a in enumerate(normales) for b in normales[i + 1:]]:
        largo_d = float(np.linalg.norm(d))
        if largo_d < 1e-3:
            continue
        d = d / largo_d
        if all(abs(float(d @ c)) < 1 - 1e-6 for c in candidatos):
            candidatos.append(d)
    mejor: Optional[Prisma] = None
    for d in candidatos:
        c = normal @ d
        tapa = validos & (np.abs(c) > 1 - _TOL_ANGULO)
        lado = validos & (np.abs(c) < _TOL_ANGULO)
        if np.any(validos & ~tapa & ~lado):
            continue
        z = v @ d
        abajo, arriba = tapa & (c < 0), tapa & (c > 0)
        if not abajo.any() or not arriba.any():
            continue
        z0, z1 = float(z[usados].min()), float(z[usados].max())
        largo = z1 - z0
        tol = 1e-6 * escala
        if largo <= 1e3 * tol or not np.all((np.abs(z[usados] - z0) <= tol) | (np.abs(z[usados] - z1) <= tol)):
            continue
        if np.any(np.abs(z[t[abajo]] - z0) > tol) or np.any(np.abs(z[t[arriba]] - z1) > tol):
            continue
        a0, a1 = float(area[abajo].sum()), float(area[arriba].sum())
        if abs(a0 - a1) > 1e-3 * a0:
            continue
        borde = _aristas_de_borde(t[abajo])
        perimetro = float(np.linalg.norm(v[borde[:, 0]] - v[borde[:, 1]], axis=1).sum())
        lateral = float(area[lado].sum())
        if abs(lateral - perimetro * largo) > 2e-3 * max(lateral, 1e-300):
            continue
        if mejor is None or a0 < mejor.area_tapa:
            tri = t[abajo]
            nodos, inverso = np.unique(tri, return_inverse=True)
            mejor = Prisma(eje=np.asarray(d, dtype=float), z0=z0, largo=largo, vertices_tapa=v[nodos],
                           triangulos_tapa=inverso.reshape(-1, 3).astype(np.int64), area_tapa=a0)
    return mejor


def tamano_seccion(prisma: Prisma, tamano: float) -> float:
    """Tamaño de elemento de la sección: el pedido, pero con al menos unos 3 elementos a lo ancho (en una tuerca con
    elementos mayores que ella los cuadriláteros no seguirían su forma)."""

    return min(tamano, math.sqrt(prisma.area_tapa) / 3.0)


#: Relación máxima entre el largo de una capa y el lado de los cuadriláteros de la sección.
ALARGAMIENTO_MAXIMO = 4.0


def capas(prisma: Prisma, tamano: float) -> int:
    """Capas a lo largo del eje: de largo ``tamano`` como mucho, y no más de 4 veces el lado de la sección."""

    return max(CAPAS_MINIMAS, math.ceil(prisma.largo / tamano - 1e-9),
               math.ceil(prisma.largo / (ALARGAMIENTO_MAXIMO * tamano_seccion(prisma, tamano)) - 1e-9))


def estimar_hex20(prisma: Prisma, tamano: float) -> float:
    """Hexaedros aproximados (cuadriláteros de la sección × capas)."""

    cuadros = max(1.0, prisma.area_tapa / tamano_seccion(prisma, tamano) ** 2)
    # el contorno no baja de unos pocos cuadriláteros por lado aunque el tamaño crezca
    borde = _aristas_de_borde(prisma.triangulos_tapa)
    perimetro = float(np.linalg.norm(prisma.vertices_tapa[borde[:, 0]] - prisma.vertices_tapa[borde[:, 1]], axis=1).sum())
    cuadros = max(cuadros, 0.5 * perimetro / tamano, 4.0)
    return cuadros * capas(prisma, tamano)


def _base(eje: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.eye(3)[int(np.argmin(np.abs(eje)))]
    e1 = np.cross(eje, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(eje, e1)


def _orientar_cuadros(coordenadas: np.ndarray, cuadros: np.ndarray, eje: np.ndarray) -> np.ndarray:
    """Cuadriláteros recorridos en sentido antihorario visto desde la punta del eje; ``ErrorBarrido`` si alguno no es
    convexo o tiene un ángulo de menos de unos 6° (CalculiX daría jacobianos malos)."""

    e1, e2 = _base(eje)
    p = coordenadas[cuadros]
    x = np.stack([p @ e1, p @ e2], axis=-1)  # (q, 4, 2)
    area = 0.5 * sum(x[:, i, 0] * x[:, (i + 1) % 4, 1] - x[:, (i + 1) % 4, 0] * x[:, i, 1] for i in range(4))
    cuadros = np.where((area < 0)[:, None], cuadros[:, [0, 3, 2, 1]], cuadros)
    x = np.where((area < 0)[:, None, None], x[:, [0, 3, 2, 1]], x)
    for i in range(4):
        a = x[:, (i + 1) % 4] - x[:, i]
        b = x[:, (i - 1) % 4] - x[:, i]
        seno = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
        if np.any(seno < 0.1):
            raise ErrorBarrido("la sección tiene cuadriláteros deformados")
    return cuadros


def extruir(prisma: Prisma, coordenadas: np.ndarray, cuadros: np.ndarray, tamano: float) -> Tuple[np.ndarray, np.ndarray]:
    """Hexaedros de 20 nodos de la sección (cuadriláteros de la tapa inferior) extruida en capas: (coordenadas, (M, 20))."""

    d = prisma.eje
    s = np.asarray(coordenadas, dtype=float)
    s = s - np.outer(s @ d - prisma.z0, d)  # exactamente sobre el plano de la tapa
    q = _orientar_cuadros(s, np.asarray(cuadros, dtype=np.int64), d)
    n = capas(prisma, tamano)
    alturas = np.linspace(0.0, prisma.largo, n + 1)
    k = len(s)
    aristas = np.sort(np.concatenate([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]]), axis=1)
    unicas, inverso = np.unique(aristas, axis=0, return_inverse=True)
    inverso = inverso.reshape(4, -1).T  # (q, 4): arista c1-c2, c2-c3, c3-c4, c4-c1 de cada cuadrilátero
    e = len(unicas)
    medios = 0.5 * (s[unicas[:, 0]] + s[unicas[:, 1]])
    esquinas = np.concatenate([s + z * d for z in alturas])  # (n + 1)·k
    medios_capa = np.concatenate([medios + z * d for z in alturas])  # (n + 1)·e
    verticales = np.concatenate([s + 0.5 * (alturas[j] + alturas[j + 1]) * d for j in range(n)])  # n·k
    base_medios, base_verticales = (n + 1) * k, (n + 1) * k + (n + 1) * e
    bloques = []
    for j in range(n):
        abajo, arriba = j * k + q, (j + 1) * k + q
        bloques.append(np.hstack([abajo, arriba, base_medios + j * e + inverso, base_medios + (j + 1) * e + inverso,
                                  base_verticales + j * k + q]))
    return np.vstack([esquinas, medios_capa, verticales]), np.vstack(bloques).astype(np.int64)


def caras_de_borde(coordenadas: np.ndarray, hexaedros: np.ndarray):
    """Caras sin vecino: (hexaedro, cara 1..6, nodos (n, 8) vértices y medios, centro, normal hacia fuera, área)."""

    esquinas = np.stack([hexaedros[:, list(c)] for c in CARAS_HEX20], axis=1)  # (n, 6, 4)
    medios = np.stack([hexaedros[:, list(c)] for c in MEDIOS_CARA_HEX20], axis=1)
    _, primero, cuenta = np.unique(np.sort(esquinas.reshape(-1, 4), axis=1), axis=0, return_index=True, return_counts=True)
    h, c = np.divmod(primero[cuenta == 1], 6)
    nodos = np.concatenate([esquinas[h, c], medios[h, c]], axis=1)
    p = coordenadas[nodos[:, :4]]
    centro = p.mean(axis=1)
    normal = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    centro_hex = coordenadas[hexaedros[h, :8]].mean(axis=1)
    normal[np.einsum("ij,ij->i", normal, centro - centro_hex) < 0] *= -1.0
    return h, c + 1, nodos, centro, normal, area


def volumen(coordenadas: np.ndarray, hexaedros: np.ndarray) -> float:
    """Volumen de los hexaedros (de caras planas), partido cada uno en 6 tetraedros por sus vértices."""

    p = coordenadas[hexaedros[:, :8]]
    total = 0.0
    for a, b, c, d in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
        total += float(np.abs(np.einsum("ij,ij->i", np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]), p[:, d] - p[:, a])).sum()) / 6.0
    return total


def mallar_solido_hex20(solido: SolidoGeneral, prisma: Prisma, seccion, tamano: float) -> MallaHex20:
    """Malla barrida del sólido a partir de ``seccion`` = (coordenadas, cuadriláteros) de su tapa inferior (Gmsh).
    ``ErrorBarrido`` si la sección no sirve (Gmsh falló, cuadriláteros malos o la malla pierde volumen)."""

    from .libres import _superficie_de_caras

    if isinstance(seccion, str):
        raise ErrorBarrido(seccion)
    coordenadas, hexaedros = extruir(prisma, seccion[0], seccion[1], tamano)
    vol = volumen(coordenadas, hexaedros)
    if not 0.9 <= vol / max(abs(float(solido.volumen)), 1e-300) <= 1.1:
        raise ErrorBarrido(f"la malla barrida tiene el {vol / abs(float(solido.volumen)):.0%} del volumen de la pieza")
    superficies: Dict[int, np.ndarray] = {}
    h, cara, _, centro, normal, _ = caras_de_borde(coordenadas, hexaedros)
    if len(h) and len(solido.caras):
        superficie = _superficie_de_caras(solido, centro, normal, tamano)
        for s in np.unique(superficie[superficie >= 0]):
            marca = superficie == s
            superficies[int(s)] = np.stack([h[marca], cara[marca]], axis=1)
    return MallaHex20(coordenadas=coordenadas, hexaedros=hexaedros, superficies=superficies,
                      capas=capas(prisma, tamano), volumen=vol)
