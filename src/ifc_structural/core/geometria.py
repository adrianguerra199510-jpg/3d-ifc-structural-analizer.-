"""Geometría de los elementos: ejes locales y dimensiones de sección de
vigas/columnas, y sólidos con caras identificables (vigas, columnas,
losas y muros).

Este módulo es la **única fuente de verdad** de la orientación de las
secciones: lo usa tanto el generador de .inp (lo que se analiza) como el
visor 3D (lo que se dibuja y se selecciona), de modo que ambos coinciden
siempre. La convención de ``*BEAM SECTION, SECTION=RECT`` de CalculiX se
verificó empíricamente contra la malla expandida que devuelve ccx 2.21:

- El primer número de la sección es la dimensión medida a lo largo de
  ``e1`` (el vector ``n1`` proyectado perpendicularmente al eje).
- El segundo, a lo largo de ``e2 = t × e1`` (``t`` = eje de la barra).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .modelo import ElementoBarra, Losa, Muro, PerfilSeccion, Proyecto

_TOL = 1e-9


def dimensiones_seccion(perfil: PerfilSeccion) -> Tuple[float, float]:
    """(altura, ancho) de la sección rectangular usada en el análisis.

    Si el perfil trae dimensiones reales (sección rectangular, lo típico en
    hormigón) se usan tal cual. En caso contrario se aproxima con el
    rectángulo equivalente que conserva el área y la inercia fuerte
    (``inercia_z``) del perfil real: h = sqrt(12·Iz/A), b = A/h.
    """

    if perfil.altura and perfil.ancho:
        return perfil.altura, perfil.ancho

    inercia_fuerte = perfil.inercia_z if perfil.inercia_z > 0 else perfil.inercia_y
    altura = math.sqrt(12.0 * inercia_fuerte / perfil.area)
    return altura, perfil.area / altura


@dataclass(frozen=True)
class EjesSeccion:
    """Terna local de una barra: eje ``t`` y ejes de la sección ``e1``/``e2``,
    con ``dim1`` medida a lo largo de ``e1`` y ``dim2`` a lo largo de ``e2``."""

    t: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    dim1: float
    dim2: float


def _unitario(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _es_diagonal_exacta(t: np.ndarray) -> bool:
    """True si el eje es una diagonal de 45° exacta dentro de un plano
    coordenado (dos componentes iguales en magnitud y la tercera nula)."""

    a, b, c = sorted(np.abs(t), reverse=True)
    return c < 1e-6 and abs(a - b) < 1e-6 and a > 1e-3


def ejes_seccion(proyecto: Proyecto, barra: ElementoBarra) -> EjesSeccion:
    """Ejes locales y dimensiones de la sección de una barra, en el
    sistema global, tal como se pasan a CalculiX y como se dibujan."""

    a = np.array(proyecto.nodos[barra.nodo_inicio].coordenadas, dtype=float)
    b = np.array(proyecto.nodos[barra.nodo_fin].coordenadas, dtype=float)
    t = _unitario(b - a)

    n1 = None
    for candidato in (barra.vector_n1, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        c = np.array(candidato, dtype=float)
        if np.linalg.norm(np.cross(t, c)) > 1e-6:
            n1 = c
            break
    if n1 is None:  # no debería ocurrir: t no puede ser paralelo a tres vectores a la vez
        n1 = np.array([0.0, 0.0, 1.0])

    e1 = _unitario(n1 - np.dot(n1, t) * t)
    e2 = np.cross(t, e1)
    altura, ancho = dimensiones_seccion(proyecto.perfiles[barra.perfil])

    # Limitación de ccx 2.21 (comprobada): en diagonales de 45° exactas
    # dentro de un plano coordenado, con e2 alineado a un eje global, falla
    # la expansión de la viga ("zero coefficient on the dependent side").
    # Intercambiar e1/e2 describe exactamente la misma sección física y
    # evita el problema.
    if _es_diagonal_exacta(t) and np.max(np.abs(e2)) > 1.0 - 1e-6:
        e1, e2 = e2, np.cross(t, e2)
        altura, ancho = ancho, altura

    return EjesSeccion(t=t, e1=e1, e2=e2, dim1=altura, dim2=ancho)


# ---------------------------------------------------------------------------
# Sólidos con caras identificables
# ---------------------------------------------------------------------------


@dataclass
class Cara:
    """Cara plana de un sólido (viga, columna, losa o muro).

    ``indices`` son posiciones dentro de los puntos del sólido, de modo que
    el visor puede construir una malla cuyo número de celda es el índice de
    la cara en la lista devuelta por ``caras_*``.
    """

    elemento_tipo: str  # "VIGA", "COLUMNA", "LOSA", "MURO"
    elemento_id: str
    indice: int
    tipo: str  # EXTREMO_INICIO, EXTREMO_FIN, LATERAL, SUPERIOR, INFERIOR, BORDE, CARA_A, CARA_B
    nombre: str
    indices: List[int]
    puntos: np.ndarray
    normal: np.ndarray  # unitaria, hacia fuera del sólido
    area: float
    ancho: float = 0.0  # ancho de la cara perpendicular al eje (solo caras laterales de barras)
    nodo_id: Optional[str] = None  # nodo del extremo (solo caras extremas de barras)
    indice_borde: Optional[int] = None  # lado del contorno (solo bordes de losa)

    @property
    def centro(self) -> np.ndarray:
        return self.puntos.mean(axis=0)


@dataclass
class Solido:
    puntos: np.ndarray
    caras: List[Cara] = field(default_factory=list)


def area_poligono(puntos: np.ndarray) -> float:
    """Área de un polígono plano en 3D (suma de productos cruzados)."""

    origen = puntos[0]
    total = np.zeros(3)
    for i in range(1, len(puntos) - 1):
        total += np.cross(puntos[i] - origen, puntos[i + 1] - origen)
    return float(np.linalg.norm(total) / 2.0)


def normal_poligono(puntos: np.ndarray) -> np.ndarray:
    """Normal del polígono por el método de Newell (sentido según el orden
    de los vértices, regla de la mano derecha)."""

    n = np.zeros(3)
    for i in range(len(puntos)):
        a, b = puntos[i], puntos[(i + 1) % len(puntos)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    norma = np.linalg.norm(n)
    return n / norma if norma > _TOL else np.array([0.0, 0.0, 1.0])


def _orientar_hacia_fuera(puntos_cara: np.ndarray, centro_solido: np.ndarray, indices: List[int]):
    """Ordena los vértices de la cara para que su normal apunte hacia
    fuera del sólido. Devuelve (indices, normal)."""

    normal = normal_poligono(puntos_cara)
    if np.dot(normal, puntos_cara.mean(axis=0) - centro_solido) < 0:
        indices = indices[::-1]
        normal = -normal
    return indices, normal


def _nombre_por_normal(normal: np.ndarray) -> str:
    """Descripción legible de una cara por la dirección de su normal."""

    if normal[2] > 0.7:
        return "superior"
    if normal[2] < -0.7:
        return "inferior"
    eje = int(np.argmax(np.abs(normal[:2])))
    signo = "+" if normal[eje] > 0 else "-"
    return f"lateral ({signo}{'XY'[eje]})"


def solido_barra(proyecto: Proyecto, barra: ElementoBarra) -> Solido:
    """Prisma rectangular de una viga/columna con sus 6 caras.

    Orden de las caras (índice = número de celda en el visor): 0 extremo
    inicial, 1 extremo final, 2..5 laterales.
    """

    ejes = ejes_seccion(proyecto, barra)
    a = np.array(proyecto.nodos[barra.nodo_inicio].coordenadas, dtype=float)
    b = np.array(proyecto.nodos[barra.nodo_fin].coordenadas, dtype=float)
    h1, h2 = ejes.dim1 / 2.0, ejes.dim2 / 2.0

    desplazamientos = [
        h1 * ejes.e1 + h2 * ejes.e2,
        -h1 * ejes.e1 + h2 * ejes.e2,
        -h1 * ejes.e1 - h2 * ejes.e2,
        h1 * ejes.e1 - h2 * ejes.e2,
    ]
    puntos = np.array([a + d for d in desplazamientos] + [b + d for d in desplazamientos])
    centro = (a + b) / 2.0

    tipo_elemento = barra.tipo.value
    definiciones = [
        ("EXTREMO_INICIO", [0, 1, 2, 3], 0.0),
        ("EXTREMO_FIN", [4, 5, 6, 7], 0.0),
        ("LATERAL", [0, 3, 7, 4], ejes.dim2),  # +e1: la cara se extiende según e2
        ("LATERAL", [1, 2, 6, 5], ejes.dim2),  # -e1
        ("LATERAL", [0, 1, 5, 4], ejes.dim1),  # +e2: la cara se extiende según e1
        ("LATERAL", [3, 2, 6, 7], ejes.dim1),  # -e2
    ]

    caras: List[Cara] = []
    for indice, (tipo, idx, ancho) in enumerate(definiciones):
        idx, normal = _orientar_hacia_fuera(puntos[idx], centro, list(idx))
        if tipo == "EXTREMO_INICIO":
            nombre, nodo_id = f"extremo inicial (nodo {barra.nodo_inicio})", barra.nodo_inicio
        elif tipo == "EXTREMO_FIN":
            nombre, nodo_id = f"extremo final (nodo {barra.nodo_fin})", barra.nodo_fin
        else:
            nombre, nodo_id = _nombre_por_normal(normal), None
        caras.append(
            Cara(
                elemento_tipo=tipo_elemento,
                elemento_id=barra.id,
                indice=indice,
                tipo=tipo,
                nombre=nombre,
                indices=idx,
                puntos=puntos[idx],
                normal=normal,
                area=area_poligono(puntos[idx]),
                ancho=ancho,
                nodo_id=nodo_id,
            )
        )
    return Solido(puntos=puntos, caras=caras)


def normal_losa(puntos_contorno: np.ndarray) -> np.ndarray:
    """Normal del plano medio de una losa/muro; para losas aproximadamente
    horizontales se orienta hacia +Z."""

    n = normal_poligono(puntos_contorno)
    if abs(n[2]) > 0.1 and n[2] < 0:
        n = -n
    return n


def solido_plano(proyecto: Proyecto, elemento) -> Solido:
    """Prisma de una losa o muro: contorno extruido ± espesor/2 a cada
    lado del plano medio.

    Orden de las caras: 0 inferior (o cara B), 1 superior (o cara A),
    2.. un borde por cada lado del contorno (el borde ``i`` va del nodo
    ``i`` al ``i+1``).
    """

    contorno = np.array([proyecto.nodos[nid].coordenadas for nid in elemento.nodos_contorno], dtype=float)
    n_pts = len(contorno)
    normal = normal_losa(contorno)
    mitad = elemento.espesor / 2.0
    inferior = contorno - normal * mitad
    superior = contorno + normal * mitad
    puntos = np.vstack([inferior, superior])
    centro = contorno.mean(axis=0)

    es_losa = isinstance(elemento, Losa)
    tipo_elemento = "LOSA" if es_losa else "MURO"
    es_horizontal = abs(normal[2]) > 0.7

    caras: List[Cara] = []

    def agregar(indice, tipo, nombre, idx, indice_borde=None):
        idx, nrm = _orientar_hacia_fuera(puntos[idx], centro, list(idx))
        caras.append(
            Cara(
                elemento_tipo=tipo_elemento,
                elemento_id=elemento.id,
                indice=indice,
                tipo=tipo,
                nombre=nombre,
                indices=idx,
                puntos=puntos[idx],
                normal=nrm,
                area=area_poligono(puntos[idx]),
                indice_borde=indice_borde,
            )
        )

    agregar(0, "INFERIOR" if es_horizontal else "CARA_B", "inferior" if es_horizontal else "cara B", list(range(n_pts)))
    agregar(1, "SUPERIOR" if es_horizontal else "CARA_A", "superior" if es_horizontal else "cara A",
            list(range(n_pts, 2 * n_pts)))
    for i in range(n_pts):
        j = (i + 1) % n_pts
        agregar(2 + i, "BORDE", f"borde {i + 1}", [i, j, n_pts + j, n_pts + i], indice_borde=i)

    return Solido(puntos=puntos, caras=caras)


def solido_elemento(proyecto: Proyecto, tipo: str, id_: str) -> Solido:
    """Sólido (puntos + caras) de cualquier elemento del proyecto."""

    if tipo in ("VIGA", "COLUMNA"):
        return solido_barra(proyecto, proyecto.elemento_barra(id_))
    if tipo == "LOSA":
        return solido_plano(proyecto, proyecto.losas[id_])
    if tipo == "MURO":
        return solido_plano(proyecto, proyecto.muros[id_])
    raise KeyError(f"Tipo de elemento desconocido: {tipo}")
