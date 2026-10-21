"""Conversión de una malla de hexaedros ``C3D8`` en tetraedros cuadráticos ``C3D10`` conformes.

Cada cara cuadrilátera se parte por la diagonal que pasa por su nodo de **menor índice global**: dos hexaedros
vecinos parten igual la cara que comparten sin tener que mirarse, y la malla de tetraedros sale conforme. Si esa
diagonal deja un triángulo muy malo (un vértice a 180° en el contorno de una losa poligonal da un triángulo de área
cero y tetraedros de volumen nulo), se usa la otra: la elección depende solo de los 4 nodos de la cara, así que los dos
hexaedros vecinos siguen coincidiendo. Después
cada hexaedro se divide en 5 o 6 tetraedros compatibles con las diagonales de sus 6 caras (tabla calculada una vez
por búsqueda exhaustiva sobre el cubo de referencia; si hay varias divisiones posibles se elige la de tetraedros de
mejor forma en el hexaedro real). Con esta regla siempre hay división sin punto interior (el nodo mínimo del
hexaedro tiene las diagonales de sus tres caras), pero si el patrón de diagonales no la admitiera, o si todas dan
tetraedros planos o invertidos (hexaedro muy deformado), se añade el centro del hexaedro y se une con los 12
triángulos de sus caras.

Los nodos originales conservan su índice y sus coordenadas; detrás van los centros añadidos y, al final, los nodos
medios de las aristas (uno por arista, compartido por todos los tetraedros que la usan, en su punto medio exacto).

Orden de nodos del hexaedro como en ``malla`` (``C3D8``: 1-4 la cara inferior, 5-8 la superior; caras ``S1`` =
1-2-3-4, ``S2`` = 5-8-7-6, ``S3`` = 1-5-6-2, ``S4`` = 2-6-7-3, ``S5`` = 3-7-8-4, ``S6`` = 4-8-5-1) y del tetraedro
como en ``tetraedros`` (``C3D10``: 1-4 los vértices, 5-10 los nodos medios de las aristas 1-2, 2-3, 3-1, 1-4, 2-4 y
3-4; caras ``S1`` = 1-2-3, ``S2`` = 1-4-2, ``S3`` = 2-4-3, ``S4`` = 3-4-1).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

#: Caras de un hexaedro (índices 0-based de sus 8 nodos), en el orden S1..S6 de CalculiX (las de ``malla``).
CARAS_HEXAEDRO = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
#: Vértices de cada cara de un tetraedro, en el orden S1..S4 de CalculiX (las de ``tetraedros``).
CARAS_TETRAEDRO = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
#: Nodos medios (índices 0-based dentro de los 10) de cada cara del tetraedro, en el mismo orden.
MEDIOS_CARA = ((4, 5, 6), (7, 8, 4), (8, 9, 5), (9, 7, 6))
#: Extremos de la arista de cada nodo medio 5..10 del tetraedro.
ARISTAS_TETRAEDRO = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
#: Cara del tetraedro (1..4) opuesta a cada vértice 1..4.
CARA_OPUESTA = (3, 4, 2, 1)
#: Por debajo de esta calidad (1 = tetraedro regular, 0 = plano) se prueba también la división con el centro.
CALIDAD_MINIMA = 1e-3
#: Si un triángulo de la diagonal del nodo menor tiene peor calidad que esta (1 = equilátero), se mira la otra diagonal.
CALIDAD_TRIANGULO = 0.15

#: Esquinas del cubo de referencia en el orden de ``C3D8``.
_CUBO = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int64)
#: Número de la cara del tetraedro tras intercambiar sus vértices 2 y 3 (para darle la vuelta): S2 <-> S4.
_CARA_VOLTEADA = np.array([0, 1, 4, 3, 2], dtype=np.int64)
#: Hexaedros (o tetraedros) que se evalúan de una vez, para acotar la memoria.
_BLOQUE = 1 << 15


@dataclass(frozen=True, eq=False)
class _Division:
    """Una división del hexaedro de referencia en tetraedros."""

    tetraedros: np.ndarray  # (n, 4) nodos locales con volumen positivo: 0-7 las esquinas, 8 el centro
    caras: np.ndarray  # (6, 2, 2) por cara del hexaedro y triángulo: (tetraedro local, cara 1..4)
    centro: bool


def _triangulos_cara(cara: Sequence[int], bit: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Los dos triángulos de un cuadrilátero ``a-b-c-d``: diagonal ``a-c`` (``bit`` 0) o ``b-d`` (1), con su sentido."""

    a, b, c, d = cara
    return ((a, b, c), (a, c, d)) if bit == 0 else ((a, b, d), (b, c, d))


def _bits(patron: int) -> List[int]:
    return [(patron >> f) & 1 for f in range(6)]


def _division(tetraedros: Sequence[Sequence[int]], patron: int, centro: bool) -> _Division:
    """Asocia cada triángulo de las caras del hexaedro con el tetraedro que lo tiene y su número de cara."""

    tetra = np.array(tetraedros, dtype=np.int64).reshape(-1, 4)
    caras = np.zeros((6, 2, 2), dtype=np.int64)
    for f, bit in enumerate(_bits(patron)):
        for k, triangulo in enumerate(_triangulos_cara(CARAS_HEXAEDRO[f], bit)):
            for t, vertices in enumerate(tetra.tolist()):
                if set(triangulo) <= set(vertices):
                    fuera = [v for v in vertices if v not in triangulo][0]
                    caras[f, k] = (t, CARA_OPUESTA[vertices.index(fuera)])
                    break
            else:  # no puede pasar con una división válida
                raise AssertionError(f"el triángulo {triangulo} no está en ningún tetraedro")
    return _Division(tetraedros=tetra, caras=caras, centro=centro)


def _division_con_centro(patron: int) -> _Division:
    """Los 12 triángulos de las caras unidos con el centro (nodo local 8)."""

    tetra = [triangulo + (8,) for f, bit in enumerate(_bits(patron)) for triangulo in _triangulos_cara(CARAS_HEXAEDRO[f], bit)]
    return _division(tetra, patron, centro=True)


def _triangulaciones_cubo() -> List[List[Tuple[int, int, int, int]]]:
    """Todas las divisiones del cubo en tetraedros con vértices en sus esquinas y caras coincidentes (son 74)."""

    candidatos, volumenes = [], []
    for c in itertools.combinations(range(8), 4):
        p = _CUBO[list(c)]
        v6 = int(round(np.linalg.det((p[1:] - p[0]).astype(float))))
        if v6:
            candidatos.append(c if v6 > 0 else (c[0], c[2], c[1], c[3]))
            volumenes.append(abs(v6))  # 6 × volumen: 1 o 2 en el cubo unidad
    t = np.array(candidatos, dtype=np.int64)
    n = len(t)
    # dos tetraedros no se solapan si un plano los separa: el de una cara o el de una arista de cada uno
    p = _CUBO[t]
    normales = np.stack([np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]) for a, b, c in CARAS_TETRAEDRO], axis=1)
    aristas = np.stack([p[:, j] - p[:, i] for i, j in ARISTAS_TETRAEDRO], axis=1)
    i, j = np.triu_indices(n, 1)
    cruz = np.cross(aristas[i][:, :, None], aristas[j][:, None, :]).reshape(len(i), 36, 3)
    ejes = np.concatenate([normales[i], normales[j], cruz], axis=1)
    pa = np.einsum("pvk,pek->pev", p[i], ejes)
    pb = np.einsum("pvk,pek->pev", p[j], ejes)
    separa = ((pa.max(axis=2) <= pb.min(axis=2)) | (pb.max(axis=2) <= pa.min(axis=2))) & ejes.any(axis=2)
    compatibles = [0] * n
    for a, b in zip(i[separa.any(axis=1)].tolist(), j[separa.any(axis=1)].tolist()):
        compatibles[a] |= 1 << b
        compatibles[b] |= 1 << a

    disecciones: List[List[int]] = []

    def buscar(elegidos: List[int], volumen: int, libres: int, desde: int) -> None:
        if volumen == 6:
            disecciones.append(list(elegidos))
            return
        libres >>= desde
        k = desde
        while libres:
            if libres & 1 and volumen + volumenes[k] <= 6:
                elegidos.append(k)
                buscar(elegidos, volumen + volumenes[k], (libres << k) & compatibles[k], k + 1)
                elegidos.pop()
            libres >>= 1
            k += 1

    buscar([], 0, (1 << n) - 1, 0)

    def en_el_borde(triangulo) -> bool:
        return bool((np.ptp(_CUBO[list(triangulo)], axis=0) == 0).any())

    resultado = []
    for diseccion in disecciones:  # se descartan las que tienen caras que no coinciden
        cuenta: dict = {}
        for k in diseccion:
            for triangulo in itertools.combinations(sorted(candidatos[k]), 3):
                cuenta[triangulo] = cuenta.get(triangulo, 0) + 1
        if all(v == (1 if en_el_borde(tr) else 2) for tr, v in cuenta.items()):
            resultado.append([candidatos[k] for k in diseccion])
    return resultado


def _patron_de(tetraedros: Sequence[Tuple[int, int, int, int]]) -> int:
    """Patrón de diagonales de las caras que deja una división del cubo."""

    patron = 0
    for f, (a, b, c, d) in enumerate(CARAS_HEXAEDRO):
        if not any({a, c} <= set(t) and len({a, b, c, d} & set(t)) == 3 for t in tetraedros):
            patron |= 1 << f
    return patron


@lru_cache(maxsize=None)
def _tabla() -> Tuple[Tuple[_Division, ...], Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
    """(divisiones, por patrón los índices de sus divisiones sin punto interior, por patrón la de con el centro)."""

    por_patron: List[List[_Division]] = [[] for _ in range(64)]
    for tetra in _triangulaciones_cubo():
        patron = _patron_de(tetra)
        por_patron[patron].append(_division(tetra, patron, centro=False))
    divisiones: List[_Division] = []
    sin_centro, con_centro = [], []
    for patron in range(64):
        ids = []
        for division in sorted(por_patron[patron], key=lambda d: (len(d.tetraedros), d.tetraedros.tolist())):
            ids.append(len(divisiones))
            divisiones.append(division)
        sin_centro.append(tuple(ids))
        con_centro.append(len(divisiones))
        divisiones.append(_division_con_centro(patron))
    return tuple(divisiones), tuple(sin_centro), tuple(con_centro)


def divisiones_de_patron(patron: int) -> List[np.ndarray]:
    """Divisiones sin punto interior (arrays (n, 4) de nodos locales) compatibles con un patrón de diagonales
    0..63; lista vacía si el patrón necesita el centro."""

    divisiones, sin_centro, _ = _tabla()
    return [divisiones[d].tetraedros.copy() for d in sin_centro[patron]]


def patrones_diagonales(hexaedros: np.ndarray) -> np.ndarray:
    """Patrón (M,) de 6 bits: el bit ``f`` vale 0 si la cara ``S(f+1)`` = ``a-b-c-d`` se parte por ``a-c`` y 1 si por
    ``b-d``; la diagonal es la que pasa por el nodo de menor índice global de la cara."""

    hexaedros = np.asarray(hexaedros, dtype=np.int64).reshape(-1, 8)
    minimo = np.argmin(hexaedros[:, np.array(CARAS_HEXAEDRO)], axis=2)  # (M, 6) posición 0..3 del nodo menor
    return ((minimo % 2) << np.arange(6)).sum(axis=1).astype(np.int64)


def _calidad_triangulos(puntos: np.ndarray) -> np.ndarray:
    """4√3·A / Σl² de triángulos (..., 3, 3): 1 equilátero, 0 degenerado."""

    area = 0.5 * np.linalg.norm(np.cross(puntos[..., 1, :] - puntos[..., 0, :], puntos[..., 2, :] - puntos[..., 0, :]), axis=-1)
    l2 = sum(((puntos[..., (k + 1) % 3, :] - puntos[..., k, :]) ** 2).sum(axis=-1) for k in range(3))
    return 4.0 * np.sqrt(3.0) * area / np.maximum(l2, 1e-300)


def patrones_geometricos(coordenadas: np.ndarray, hexaedros: np.ndarray) -> np.ndarray:
    """Como ``patrones_diagonales`` (diagonal del nodo menor) salvo en las caras donde esa diagonal da un triángulo de
    calidad menor que ``CALIDAD_TRIANGULO`` y la otra es mejor: ahí se toma la otra. Cada cara se decide con sus
    4 nodos y en un orden que no depende del hexaedro, así que los vecinos la parten igual."""

    hexaedros = np.asarray(hexaedros, dtype=np.int64).reshape(-1, 8)
    caras = hexaedros[:, np.array(CARAS_HEXAEDRO)]  # (M, 6, 4)
    patron_menor = patrones_diagonales(hexaedros)
    if len(hexaedros) == 0:
        return patron_menor
    # orden canónico de cada cara: empieza en el nodo menor y sigue hacia el menor de sus dos vecinos
    planas = caras.reshape(-1, 4)
    inicio = np.argmin(planas, axis=1)
    k = np.arange(len(planas))
    siguiente, previo = planas[k, (inicio + 1) % 4], planas[k, (inicio + 3) % 4]
    paso = np.where(siguiente < previo, 1, 3)
    canonica = np.stack([planas[k, (inicio + j * paso) % 4] for j in range(4)], axis=1)  # a = nodo menor
    p = coordenadas[canonica]
    # diagonal a-c (la del nodo menor) o b-d
    q_ac = np.minimum(_calidad_triangulos(p[:, [0, 1, 2]]), _calidad_triangulos(p[:, [0, 2, 3]]))
    q_bd = np.minimum(_calidad_triangulos(p[:, [0, 1, 3]]), _calidad_triangulos(p[:, [1, 2, 3]]))
    otra = ((q_ac < CALIDAD_TRIANGULO) & (q_bd > q_ac)).reshape(-1, 6)
    if not otra.any():
        return patron_menor
    return patron_menor ^ (otra.astype(np.int64) << np.arange(6)).sum(axis=1)


def _forma(puntos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(6 × volumen firmado, calidad firmada) de tetraedros de vértices ``puntos`` (..., 4, 3).

    La calidad es ``6 √2 V / l³``, con ``l`` la media cuadrática de las aristas: 1 en el tetraedro regular, 0 si es
    plano y negativa si está invertido."""

    d = puntos[..., 1:, :] - puntos[..., :1, :]
    v6 = np.einsum("...k,...k->...", np.cross(d[..., 0, :], d[..., 1, :]), d[..., 2, :])
    suma = sum(np.einsum("...k,...k->...", puntos[..., j, :] - puntos[..., i, :], puntos[..., j, :] - puntos[..., i, :])
               for i, j in ARISTAS_TETRAEDRO)
    return v6, np.sqrt(2.0) * v6 / np.maximum((suma / 6.0) ** 1.5, 1e-300)


@dataclass
class ConversionTetra:
    """Malla de tetraedros ``C3D10`` obtenida de una de hexaedros."""

    coordenadas: np.ndarray  # (N2, 3): los N nodos originales, los centros añadidos y los nodos medios
    tetraedros: np.ndarray  # (K, 10) orden de CalculiX, volumen positivo
    hexaedro_de_tetraedro: np.ndarray  # (K,) hexaedro de origen (los de cada hexaedro van seguidos)
    primer_tetraedro: np.ndarray  # (M + 1,) los tetraedros del hexaedro h son primer_tetraedro[h]:primer_tetraedro[h + 1]
    caras_de_hexaedro: np.ndarray  # (M, 6, 2, 2) por hexaedro, cara 1..6 y triángulo: (tetraedro, cara 1..4)
    aristas: np.ndarray  # (E, 2) extremos (i < j) de la arista de cada nodo medio primer_medio + e
    centro_de_hexaedro: np.ndarray  # (M,) nodo del centro añadido al hexaedro (-1 si no lo lleva)
    calidad: np.ndarray  # (K,) 6 √2 V / l³ de cada tetraedro (1 regular; ≤ 0 plano o invertido respecto a su hexaedro)
    n_originales: int
    primer_medio: int  # índice del primer nodo medio (= n_originales + número de centros añadidos)
    _claves: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._claves is None:
            self._claves = self.aristas[:, 0] * self.primer_medio + self.aristas[:, 1]

    @property
    def n_centros(self) -> int:
        return self.primer_medio - self.n_originales

    def tetraedros_de_hexaedros(self, primero: int, ultimo: int) -> range:
        """Tetraedros de los hexaedros ``primero`` a ``ultimo`` (exclusivo), p. ej. los de una ``ParteMalla``."""

        return range(int(self.primer_tetraedro[primero]), int(self.primer_tetraedro[ultimo]))

    def medio_de_arista(self, i, j):
        """Nodo medio de la arista entre los nodos ``i`` y ``j`` (enteros o arrays del mismo tamaño). ``KeyError`` si
        no son los extremos de una arista de la malla."""

        i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
        clave = np.minimum(i, j) * self.primer_medio + np.maximum(i, j)
        valido = (np.minimum(i, j) >= 0) & (np.maximum(i, j) < self.primer_medio)
        pos = np.minimum(np.searchsorted(self._claves, clave), max(len(self._claves) - 1, 0))
        if not len(self._claves) or not np.all(valido & (self._claves[pos] == clave)):
            raise KeyError(f"no hay arista entre los nodos {i.tolist()} y {j.tolist()}")
        resultado = self.primer_medio + pos
        return int(resultado) if resultado.ndim == 0 else resultado

    def ampliar_con_medios(self, nodos) -> np.ndarray:
        """Los ``nodos`` más los nodos medios de las aristas que tienen los dos extremos entre ellos, ordenados. Solo para
        conjuntos planos (el anillo de la sección de una barra): en una superficie escalonada entrarían también los medios
        de diagonales interiores; para superficies, ``caras_a_tetraedros`` con ``CARAS_TETRAEDRO`` y ``MEDIOS_CARA``."""

        nodos = np.unique(np.asarray(nodos, dtype=np.int64))
        dentro = np.zeros(len(self.coordenadas), dtype=bool)
        dentro[nodos] = True
        arista = dentro[self.aristas[:, 0]] & dentro[self.aristas[:, 1]]
        return np.union1d(nodos, self.primer_medio + np.flatnonzero(arista))

    def cara_hexaedro_a_tetraedros(self, h: int, cara: int) -> List[Tuple[int, int]]:
        """``(tetraedro, cara 1..4)`` de los dos triángulos que cubren la cara ``cara`` (1..6) del hexaedro ``h``."""

        if not 1 <= cara <= 6:
            raise ValueError(f"cara de hexaedro no válida: {cara} (1..6)")
        if not 0 <= h < len(self.caras_de_hexaedro):
            raise IndexError(f"hexaedro no válido: {h}")
        return [(int(t), int(c)) for t, c in self.caras_de_hexaedro[h, cara - 1]]

    def caras_a_tetraedros(self, hexaedros, caras) -> Tuple[np.ndarray, np.ndarray]:
        """Versión vectorizada: (tetraedros (2n,), caras 1..4 (2n,)) de las caras ``caras`` (1..6) de ``hexaedros``."""

        hexaedros, caras = np.asarray(hexaedros, dtype=np.int64), np.asarray(caras, dtype=np.int64)
        if caras.size and (caras.min() < 1 or caras.max() > 6):
            raise ValueError("cara de hexaedro no válida (1..6)")
        if hexaedros.size and (hexaedros.min() < 0 or hexaedros.max() >= len(self.caras_de_hexaedro)):
            raise IndexError("hexaedro no válido")
        pares = self.caras_de_hexaedro[hexaedros, caras - 1]
        return pares[..., 0].reshape(-1), pares[..., 1].reshape(-1)


def hexaedros_a_tetraedros(coordenadas: np.ndarray, hexaedros: np.ndarray) -> ConversionTetra:
    """Tetraedros ``C3D10`` conformes de una malla de hexaedros ``C3D8`` (0-based); conserva los nodos originales."""

    coordenadas = np.asarray(coordenadas, dtype=float)
    if coordenadas.ndim != 2 or coordenadas.shape[1] != 3:
        raise ValueError("las coordenadas deben ser una tabla (N, 3)")
    hexaedros = np.asarray(hexaedros, dtype=np.int64).reshape(-1, 8)
    if len(hexaedros) and (hexaedros.min() < 0 or hexaedros.max() >= len(coordenadas)):
        raise ValueError("los hexaedros hacen referencia a nodos que no existen")
    if len(hexaedros) and not np.isfinite(coordenadas[np.unique(hexaedros)]).all():
        raise ValueError("hay nodos con coordenadas no válidas (NaN o infinitas)")
    return _convertir(coordenadas, hexaedros, patrones_geometricos(coordenadas, hexaedros))


def _elegir(coordenadas: np.ndarray, hexaedros: np.ndarray, patrones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """División de cada hexaedro (índice en ``_tabla()[0]``) y signo de su volumen (-1 si su numeración está invertida)."""

    divisiones, sin_centro, con_centro = _tabla()
    eleccion = np.empty(len(hexaedros), dtype=np.int64)
    signo = np.ones(len(hexaedros))
    for patron in np.unique(patrones).tolist():
        todos = np.flatnonzero(patrones == patron)
        for inicio in range(0, len(todos), _BLOQUE):
            hs = todos[inicio: inicio + _BLOQUE]
            esquinas = coordenadas[hexaedros[hs]]
            puntos = np.concatenate([esquinas, esquinas.mean(axis=1, keepdims=True)], axis=1)  # (n, 9, 3)
            centro = divisiones[con_centro[patron]]
            opciones = list(sin_centro[patron])
            formas = [_forma(puntos[:, divisiones[d].tetraedros]) for d in (opciones or [con_centro[patron]])]
            # todas las divisiones tienen el mismo volumen total: el encerrado por los triángulos de las caras
            s = np.where(formas[0][0].sum(axis=1) < 0, -1.0, 1.0)
            if not opciones:
                eleccion[hs], signo[hs] = con_centro[patron], s
                continue
            puntuacion = np.stack([(s[:, None] * q).min(axis=1) for _, q in formas], axis=1)  # peor tetraedro
            mejor = np.argmax(puntuacion, axis=1)
            elegida = np.array(opciones, dtype=np.int64)[mejor]
            flojos = np.flatnonzero(puntuacion[np.arange(len(hs)), mejor] < CALIDAD_MINIMA)
            if len(flojos):  # hexaedro muy deformado: se prueba con el centro
                _, q = _forma(puntos[flojos][:, centro.tetraedros])
                mejora = (s[flojos, None] * q).min(axis=1) > puntuacion[flojos, mejor[flojos]]
                elegida[flojos[mejora]] = con_centro[patron]
            eleccion[hs], signo[hs] = elegida, s
    return eleccion, signo


def _convertir(coordenadas: np.ndarray, hexaedros: np.ndarray, patrones: np.ndarray) -> ConversionTetra:
    """Conversión con un patrón de diagonales dado por hexaedro (los vecinos deben partir igual su cara común)."""

    divisiones, _, _ = _tabla()
    n, m = len(coordenadas), len(hexaedros)
    eleccion, signo = _elegir(coordenadas, hexaedros, patrones)
    por_division = np.array([len(d.tetraedros) for d in divisiones], dtype=np.int64)
    usa_centro = np.array([d.centro for d in divisiones], dtype=bool)[eleccion]
    cuantos = por_division[eleccion]
    primer = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(cuantos, out=primer[1:])
    k = int(primer[-1])

    centro_de_hexaedro = np.full(m, -1, dtype=np.int64)
    centro_de_hexaedro[usa_centro] = n + np.arange(int(usa_centro.sum()))
    vertices = np.vstack([coordenadas, coordenadas[hexaedros[usa_centro]].mean(axis=1)]) if usa_centro.any() else coordenadas
    nodos = np.concatenate([hexaedros, centro_de_hexaedro[:, None]], axis=1)  # (M, 9)

    esquinas = np.empty((k, 4), dtype=np.int64)
    caras = np.empty((m, 6, 2, 2), dtype=np.int64)
    for d in np.unique(eleccion).tolist():
        hs = np.flatnonzero(eleccion == d)
        division = divisiones[d]
        filas = primer[hs][:, None] + np.arange(len(division.tetraedros))
        esquinas[filas] = nodos[hs][:, division.tetraedros]
        caras[hs, :, :, 0] = primer[hs][:, None, None] + division.caras[:, :, 0]
        caras[hs, :, :, 1] = division.caras[:, :, 1]
    hexaedro_de_tetraedro = np.repeat(np.arange(m, dtype=np.int64), cuantos)

    # volumen positivo: se intercambian los vértices 2 y 3 de los que salen al revés (hexaedros numerados al revés)
    calidad = np.empty(k)
    for inicio in range(0, k, _BLOQUE * 4):
        tramo = slice(inicio, inicio + _BLOQUE * 4)
        _, calidad[tramo] = _forma(vertices[esquinas[tramo]])
    negativo = calidad < 0
    esquinas[negativo] = esquinas[negativo][:, [0, 2, 1, 3]]
    volteada = negativo[caras[..., 0]]
    caras[..., 1] = np.where(volteada, _CARA_VOLTEADA[caras[..., 1]], caras[..., 1])
    calidad *= signo[hexaedro_de_tetraedro]

    # nodos medios: uno por arista
    nv = len(vertices)
    a = esquinas[:, [i for i, _ in ARISTAS_TETRAEDRO]]
    b = esquinas[:, [j for _, j in ARISTAS_TETRAEDRO]]
    claves, inversa = np.unique((np.minimum(a, b) * nv + np.maximum(a, b)).reshape(-1), return_inverse=True)
    aristas = np.stack([claves // nv, claves % nv], axis=1)
    medios = 0.5 * (vertices[aristas[:, 0]] + vertices[aristas[:, 1]])
    return ConversionTetra(
        coordenadas=np.vstack([vertices, medios]),
        tetraedros=np.concatenate([esquinas, nv + inversa.reshape(k, 6)], axis=1),
        hexaedro_de_tetraedro=hexaedro_de_tetraedro, primer_tetraedro=primer, caras_de_hexaedro=caras, aristas=aristas,
        centro_de_hexaedro=centro_de_hexaedro, calidad=calidad, n_originales=n, primer_medio=nv, _claves=claves)
