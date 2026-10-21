"""Malla de todo el modelo y sus utilidades (caras, anillos).

Para el análisis todo queda en tetraedros cuadráticos ``C3D10`` y láminas ``S6``: los sólidos libres macizos con
Gmsh, las piezas que son láminas en su superficie media, y las barras, losas y muros gruesos (y los sólidos que Gmsh
no pudo mallar) se mallan primero con hexaedros, que al final se parten en tetraedros cuadráticos conformes
(``hex_a_tet``). Con ``elemento_solido="HEXA"`` (la detección de contactos) se quedan como hexaedros.

Orden de nodos de cada hexaedro (CalculiX ``C3D8``): 1-4 la cara inferior
recorrida en sentido antihorario visto desde la superior, 5-8 la cara superior.
Cara ``S1`` = 1-2-3-4, ``S2`` = 5-8-7-6, ``S3`` = 1-5-6-2, ``S4`` = 2-6-7-3,
``S5`` = 3-7-8-4, ``S6`` = 4-8-5-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.geometria import normal_losa
from ..core.modelo import Proyecto
from .miembros import MiembroBarra
from .hex_a_tet import hexaedros_a_tetraedros
from .libres import estimar_hexaedros_solido, mallar_solido
from .laminas import ErrorLamina, MallaLamina, NoEsLamina, SuperficieMedia, es_lamina, lamina_de_superficie, superficie_media
from .tetraedros import (CARAS_TETRAEDRO, ErrorTetraedros, MEDIOS_CARA, MallaTetra, _es_variedad, estimar_tetraedros_solido,
                         mallar_lote_gmsh, mallar_solido_tetra)
from .tetraedros import motivo_sin_tetraedros as motivo_sin_tetraedros_de
from .seccion2d import malla_seccion, pesos_nodales

#: Caras de un hexaedro (índices 0-based de sus 8 nodos), en el orden S1..S6 de CalculiX.
CARAS_HEXAEDRO = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))


class ErrorMalla(RuntimeError):
    pass


@dataclass
class CarasFrontera:
    hexaedro: np.ndarray  # (n,) índice global (0-based, numeración de CalculiX) del elemento
    cara: np.ndarray  # (n,) número de cara 1..6 (hexaedro) o 1..4 (tetraedro)
    nodos: np.ndarray  # (n, 4) índices globales de nodo (hexaedro) o (n, 6) con los nodos medios (tetraedro)
    centro: np.ndarray  # (n, 3)
    normal: np.ndarray  # (n, 3) unitaria, hacia fuera
    area: np.ndarray  # (n,)
    pesos_nodo: Optional[np.ndarray] = None  # (k,) fracción del área de una cara que lleva cada nodo (None = 1/k)


@dataclass
class ParteMalla:
    """Un miembro (barra, losa o muro) ya mallado."""

    id: str
    tipo: str  # "BARRA", "LOSA", "MURO" o "SOLIDO" (sólido de forma libre)
    material: str
    primer_hexaedro: int
    ultimo_hexaedro: int  # exclusivo
    miembro: Optional[MiembroBarra] = None
    estaciones: np.ndarray = field(default_factory=lambda: np.zeros(0))
    anillos: List[np.ndarray] = field(default_factory=list)
    pesos: np.ndarray = field(default_factory=lambda: np.zeros(0))  # fracción de la sección que lleva cada nodo de un anillo
    cuadros: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int64))  # barras: cuadriláteros del anillo
    superficies: Dict[int, np.ndarray] = field(default_factory=dict)  # solo sólidos libres: superficie -> (k, 2) (elemento local, cara)
    tetra: bool = False  # True: la parte son tetraedros ``C3D10`` (``MallaSolida.tetraedros``), no hexaedros
    primer_tetraedro: int = 0
    ultimo_tetraedro: int = 0  # exclusivo
    lamina: bool = False  # True: la parte son láminas ``S6`` (``MallaSolida.laminas``): sólido de pared delgada
    primer_lamina: int = 0
    ultimo_lamina: int = 0  # exclusivo
    nodos_superficie: Dict[int, np.ndarray] = field(default_factory=dict)  # láminas: superficie -> nodos globales sobre ella
    _fronteras: Optional[CarasFrontera] = None

    @property
    def hexaedros(self) -> range:
        return range(self.primer_hexaedro, self.ultimo_hexaedro)


@dataclass
class MallaSolida:
    coordenadas: np.ndarray  # (N, 3)
    hexaedros: np.ndarray  # (M, 8) 0-based
    parte_de_hexaedro: np.ndarray  # (M,) índice en ``partes``
    elemento_de_hexaedro: List[str]  # id de la barra, losa o muro del modelo al que pertenece cada hexaedro
    partes: List[ParteMalla]
    avisos: List[str] = field(default_factory=list)
    tetraedros: np.ndarray = field(default_factory=lambda: np.zeros((0, 10), dtype=np.int64))  # (K, 10) orden de CalculiX
    parte_de_tetraedro: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    elemento_de_tetraedro: List[str] = field(default_factory=list)
    laminas: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), dtype=np.int64))  # (L, 6) ``S6`` de CalculiX
    espesor_de_lamina: np.ndarray = field(default_factory=lambda: np.zeros(0))
    normal_de_lamina: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    parte_de_lamina: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    elemento_de_lamina: List[str] = field(default_factory=list)
    # parejas de piezas (ids) que solapaban: la menor cedió el volumen común a la mayor y su borde allí es un corte, sin
    # caras planas enfrentadas (una tuerca metida en la placa); se unen por proximidad
    solapes: Set[frozenset] = field(default_factory=set)

    def fronteras(self, parte: ParteMalla) -> CarasFrontera:
        if parte._fronteras is None:
            if parte.lamina:
                parte._fronteras = _caras_frontera_lamina(self, parte)
            elif parte.tetra:
                parte._fronteras = _caras_frontera_tetra(self, parte)
            else:
                parte._fronteras = _caras_frontera(self, parte)
        return parte._fronteras

    @property
    def n_elementos(self) -> int:
        """Elementos de CalculiX: primero los hexaedros, después los tetraedros y al final las láminas."""

        return len(self.hexaedros) + len(self.tetraedros) + len(self.laminas)

    @property
    def elementos_en_orden(self) -> List[str]:
        """Id del elemento del modelo al que pertenece cada elemento finito, en el orden de numeración de CalculiX."""

        return list(self.elemento_de_hexaedro) + list(self.elemento_de_tetraedro) + list(self.elemento_de_lamina)

    def nodos_de_parte(self, parte: ParteMalla) -> np.ndarray:
        if parte.lamina:
            return np.unique(self.laminas[parte.primer_lamina: parte.ultimo_lamina])
        if parte.tetra:
            return np.unique(self.tetraedros[parte.primer_tetraedro: parte.ultimo_tetraedro])
        return np.unique(self.hexaedros[parte.primer_hexaedro: parte.ultimo_hexaedro])


# ---------------------------------------------------------------------------
# Barras
# ---------------------------------------------------------------------------


def _estaciones(inicio: float, fin: float, especiales: Sequence[float], tamano: float) -> np.ndarray:
    puntos = {round(inicio, 6), round(fin, 6)}
    puntos.update(round(s, 6) for s in especiales if inicio + 1e-6 < s < fin - 1e-6)
    ordenados = sorted(puntos)
    estaciones = [ordenados[0]]
    for a, b in zip(ordenados, ordenados[1:]):
        n = max(1, math.ceil((b - a) / tamano - 1e-9))
        estaciones.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(estaciones)


def _malla_barra(proyecto: Proyecto, miembro: MiembroBarra, tamano: float, especiales: Sequence[float],
                 nodos: List[np.ndarray], n_nodos: int, n_hex: int):
    perfil = proyecto.perfiles[miembro.perfil]
    puntos2d, quads = malla_seccion(perfil, miembro.ejes_ref, tamano)
    inicio, fin = miembro.s_util
    estaciones = _estaciones(inicio, fin, [miembro.coordenada(p) for p in []] + list(especiales)
                             + [tr[1] for tr in miembro.tramos[1:]], tamano)
    plantilla = puntos2d[:, 0:1] * miembro.e1 + puntos2d[:, 1:2] * miembro.e2  # (n2, 3)
    n2 = len(puntos2d)
    anillos, bloques = [], []
    for k, s in enumerate(estaciones):
        bloques.append(miembro.punto(s) + plantilla)
        anillos.append(np.arange(n_nodos + k * n2, n_nodos + (k + 1) * n2))
    nodos.append(np.vstack(bloques))
    abajo = np.arange(len(estaciones) - 1)[:, None, None] * n2 + n_nodos
    hexaedros = np.concatenate([quads[None] + abajo, quads[None] + abajo + n2], axis=2).reshape(-1, 8)
    elementos = []
    for k in range(len(estaciones) - 1):
        elementos.extend([miembro.barra_en(0.5 * (estaciones[k] + estaciones[k + 1]))] * len(quads))
    return estaciones, anillos, pesos_nodales(puntos2d, quads), hexaedros, elementos, len(estaciones) * n2, quads


# ---------------------------------------------------------------------------
# Losas y muros
# ---------------------------------------------------------------------------


def _quads_poligono(contorno2d: np.ndarray, tamano: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cuadriláteros de un polígono convexo en el plano (coordenadas 2D antihorarias)."""

    n = len(contorno2d)
    a = contorno2d[1] - contorno2d[0]
    b = contorno2d[-1] - contorno2d[0]
    es_rectangulo = n == 4 and abs(np.dot(a, b)) < 1e-6 * np.linalg.norm(a) * np.linalg.norm(b) and np.allclose(
        contorno2d[2], contorno2d[0] + a + b, atol=1e-6)
    puntos: Dict[Tuple[float, float], int] = {}
    lista: List[Tuple[float, float]] = []
    quads: List[Tuple[int, int, int, int]] = []

    def nodo(p) -> int:
        clave = (round(float(p[0]), 8) + 0.0, round(float(p[1]), 8) + 0.0)
        if clave not in puntos:
            puntos[clave] = len(lista)
            lista.append(clave)
        return puntos[clave]

    def cuadricula(origen, du, dv, nu, nv) -> None:
        for i in range(nu):
            for j in range(nv):
                p = [origen + du * i / nu + dv * j / nv, origen + du * (i + 1) / nu + dv * j / nv,
                     origen + du * (i + 1) / nu + dv * (j + 1) / nv, origen + du * i / nu + dv * (j + 1) / nv]
                quads.append(tuple(nodo(q) for q in p))

    if es_rectangulo:
        cuadricula(contorno2d[0], a, b, max(1, math.ceil(np.linalg.norm(a) / tamano - 1e-9)),
                   max(1, math.ceil(np.linalg.norm(b) / tamano - 1e-9)))
    else:
        centro = contorno2d.mean(axis=0)
        medios = [(contorno2d[i] + contorno2d[(i + 1) % n]) / 2.0 for i in range(n)]
        mitad_max = max(np.linalg.norm(contorno2d[(i + 1) % n] - contorno2d[i]) for i in range(n)) / 2.0
        k = max(1, math.ceil(mitad_max / tamano - 1e-9))
        for i in range(n):
            vertice, m_ant, m_sig = contorno2d[i], medios[i - 1], medios[i]
            # cuadrilátero [m_ant, vértice, m_sig, centro] como cuadrícula k×k (interpolación bilineal)
            for p in range(k):
                for q in range(k):
                    def f(s, t):
                        return ((1 - s) * (1 - t) * m_ant + s * (1 - t) * vertice + s * t * m_sig + (1 - s) * t * centro)

                    quads.append(tuple(nodo(f(*st)) for st in (
                        (p / k, q / k), ((p + 1) / k, q / k), ((p + 1) / k, (q + 1) / k), (p / k, (q + 1) / k))))
    puntos2d = np.array(lista)
    quads_a = np.array(quads, dtype=np.int64)
    c = puntos2d[quads_a]
    area = 0.5 * ((c[:, 0, 0] * c[:, 1, 1] - c[:, 1, 0] * c[:, 0, 1]) + (c[:, 1, 0] * c[:, 2, 1] - c[:, 2, 0] * c[:, 1, 1])
                  + (c[:, 2, 0] * c[:, 3, 1] - c[:, 3, 0] * c[:, 2, 1]) + (c[:, 3, 0] * c[:, 0, 1] - c[:, 0, 0] * c[:, 3, 1]))
    quads_a[area < 0] = quads_a[area < 0][:, ::-1]
    return puntos2d, quads_a


def _malla_plano(proyecto: Proyecto, elemento, tamano: float, nodos: List[np.ndarray], n_nodos: int):
    contorno = np.array([proyecto.nodos[n].coordenadas for n in elemento.nodos_contorno], dtype=float)
    normal = normal_losa(contorno)
    a = contorno[1] - contorno[0]
    a = a - np.dot(a, normal) * normal
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    centro = contorno.mean(axis=0)
    plano = np.column_stack([(contorno - centro) @ a, (contorno - centro) @ b])
    # sentido antihorario
    area = 0.5 * sum(plano[i, 0] * plano[(i + 1) % len(plano), 1] - plano[(i + 1) % len(plano), 0] * plano[i, 1]
                     for i in range(len(plano)))
    if area < 0:
        plano = plano[::-1].copy()
    puntos2d, quads = _quads_poligono(plano, tamano)
    capas = max(2, math.ceil(elemento.espesor / tamano - 1e-9)) if elemento.espesor > 0.08 else 1
    base = centro + puntos2d[:, 0:1] * a + puntos2d[:, 1:2] * b
    n2 = len(puntos2d)
    bloques = [base + normal * (-elemento.espesor / 2.0 + elemento.espesor * k / capas) for k in range(capas + 1)]
    nodos.append(np.vstack(bloques))
    abajo = np.arange(capas)[:, None, None] * n2 + n_nodos
    hexaedros = np.concatenate([quads[None] + abajo, quads[None] + abajo + n2], axis=2).reshape(-1, 8)
    return hexaedros, len(bloques) * n2


def plano_es_lamina(proyecto: Proyecto, elemento) -> bool:
    """Una losa o un muro es lámina si su espesor es como mucho 1/20 de su tamaño en planta (√área)."""

    from ..core.geometria import area_poligono
    from .laminas import RELACION_LAMINA

    contorno = np.array([proyecto.nodos[n].coordenadas for n in elemento.nodos_contorno], dtype=float)
    return elemento.espesor <= RELACION_LAMINA * float(np.sqrt(area_poligono(contorno)))


def _lamina_plano(proyecto: Proyecto, elemento, tamano: float, miembros: Sequence[MiembroBarra]) -> Optional[MallaLamina]:
    """Lámina S6 en el plano medio de una losa o un muro (su contorno), sin la parte que queda dentro de las vigas."""

    from .laminas import elementos_s6

    contorno = np.array([proyecto.nodos[n].coordenadas for n in elemento.nodos_contorno], dtype=float)
    normal = normal_losa(contorno)
    a = contorno[1] - contorno[0]
    a = a - np.dot(a, normal) * normal
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    centro = contorno.mean(axis=0)
    plano = np.column_stack([(contorno - centro) @ a, (contorno - centro) @ b])
    area = 0.5 * sum(plano[i, 0] * plano[(i + 1) % len(plano), 1] - plano[(i + 1) % len(plano), 0] * plano[i, 1]
                     for i in range(len(plano)))
    if area < 0:
        plano = plano[::-1].copy()
    puntos2d, quads = _quads_poligono(plano, tamano)
    vertices = centro + puntos2d[:, 0:1] * a + puntos2d[:, 1:2] * b
    # cada cuadrilátero en dos triángulos por su diagonal más corta
    d02 = np.linalg.norm(vertices[quads[:, 0]] - vertices[quads[:, 2]], axis=1)
    d13 = np.linalg.norm(vertices[quads[:, 1]] - vertices[quads[:, 3]], axis=1)
    corta = d02 <= d13
    triangulos = np.vstack([quads[corta][:, [0, 1, 2]], quads[corta][:, [0, 2, 3]],
                            quads[~corta][:, [0, 1, 3]], quads[~corta][:, [1, 2, 3]]])
    if miembros:
        dentro = _dentro_de_barras(proyecto, miembros, vertices[triangulos].mean(axis=1))
        if dentro.all():
            return None
        triangulos = triangulos[~dentro]
    return elementos_s6(vertices, triangulos, np.full(len(triangulos), float(elemento.espesor)), normal, tamano)


# ---------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------


def construir_malla_solida(proyecto: Proyecto, miembros: Sequence[MiembroBarra], tamano: float,
                           especiales: Optional[Dict[str, Sequence[float]]] = None,
                           elemento_solido: str = "HEXA", usar_laminas: bool = False,
                           al_avanzar: Optional[Callable[[str], None]] = None) -> MallaSolida:
    """Malla de todos los miembros y sólidos libres; ``especiales`` da, por miembro, coordenadas axiales donde debe
    haber una estación (cargas y apoyos).

    Con ``elemento_solido="TETRA"`` (el análisis) todo son tetraedros cuadráticos ``C3D10``: los sólidos libres con Gmsh
    y el resto (barras, losas, muros y las piezas que Gmsh no pudo mallar, por vóxeles) con hexaedros que al final se
    parten en tetraedros. Con ``"HEXA"`` quedan hexaedros ``C3D8I`` (los sólidos libres, por voxelización). Con
    ``usar_laminas`` las piezas que son láminas (``laminas.es_lamina``: espesor ≤ 1/20 de su tamaño en planta, y
    losas y muros igual) se modelan con cáscaras ``S6`` en su superficie media."""

    especiales = especiales or {}
    bloques_nodos: List[np.ndarray] = []
    lista_hex: List[np.ndarray] = []
    elementos: List[str] = []
    partes: List[ParteMalla] = []
    n_nodos = n_hex = 0
    con_gmsh = elemento_solido == "TETRA"

    def avisar(mensaje: str) -> None:
        if al_avanzar is not None:
            al_avanzar(mensaje)

    for miembro in miembros:
        estaciones, anillos, pesos, hexa, elems, n_nuevos, cuadros = _malla_barra(
            proyecto, miembro, tamano, especiales.get(miembro.id, ()), bloques_nodos, n_nodos, n_hex)
        partes.append(ParteMalla(
            id=miembro.id, tipo="BARRA", material=miembro.material, primer_hexaedro=n_hex,
            ultimo_hexaedro=n_hex + len(hexa), miembro=miembro, estaciones=estaciones, anillos=anillos, pesos=pesos,
            cuadros=np.asarray(cuadros, dtype=np.int64)))
        lista_hex.append(hexa)
        elementos.extend(elems)
        n_nodos += n_nuevos
        n_hex += len(hexa)

    laminas: List[Tuple[ParteMalla, MallaLamina]] = []
    for tipo, coleccion in (("LOSA", proyecto.losas), ("MURO", proyecto.muros)):
        for id_, elemento in coleccion.items():
            if usar_laminas and plano_es_lamina(proyecto, elemento):
                lamina = _lamina_plano(proyecto, elemento, tamano, miembros)
                if lamina is not None:
                    parte = ParteMalla(id=id_, tipo=tipo, material=elemento.material, primer_hexaedro=n_hex,
                                       ultimo_hexaedro=n_hex, lamina=True)
                    partes.append(parte)
                    laminas.append((parte, lamina))
                    continue
            hexa, n_nuevos = _malla_plano(proyecto, elemento, tamano, bloques_nodos, n_nodos)
            partes.append(ParteMalla(id=id_, tipo=tipo, material=elemento.material, primer_hexaedro=n_hex,
                                     ultimo_hexaedro=n_hex + len(hexa)))
            lista_hex.append(hexa)
            elementos.extend([id_] * len(hexa))
            n_nodos += n_nuevos
            n_hex += len(hexa)

    avisos: List[str] = []
    # Sólidos de forma libre: el de mayor volumen conserva el volumen común con los menores y las barras
    # (pernos, placas y soldaduras suelen solapar con los perfiles que unen).
    ordenados = sorted(proyecto.solidos.values(), key=lambda x: -x.volumen)
    recortados = 0
    solapes: Set[frozenset] = set()
    sin_volumen: List[str] = []  # piezas cuya malla de Gmsh recorta sus curvas (pequeñas frente al tamaño de elemento)
    delgados: List[str] = []
    tetras: List[Tuple[ParteMalla, MallaTetra]] = []
    sin_tetraedros: Dict[str, str] = {}  # sólido -> motivo por el que Gmsh no lo malló
    for solido in ordenados:
        if not solido.cerrado:
            raise ErrorMalla(f"El sólido {solido.id} ({solido.nombre}) no es una superficie cerrada y no se puede mallar en 3D.")
    # 1) Láminas: las piezas cuyo espesor es pequeño frente a su tamaño (criterio geométrico, ver laminas.es_lamina) se
    #    modelan con cáscaras S6 en su superficie media.
    medias: Dict[str, SuperficieMedia] = {}
    sin_laminas: Dict[str, str] = {}
    if usar_laminas:
        for solido in ordenados:
            if es_lamina(solido):
                try:
                    medias[solido.id] = superficie_media(solido, tamano)
                except NoEsLamina:  # barra o bloque macizo: va con tetraedros
                    pass
                except ErrorLamina as exc:
                    sin_laminas[solido.id] = str(exc)
    # 2) Gmsh, todas las piezas de una vez y en un solo proceso (lanzar uno por pieza hace saltar a los antivirus): las
    #    superficies medias de las láminas se vuelven a mallar con triángulos casi equiláteros y el resto se rellena con
    #    tetraedros C3D10.
    gmsh_por_solido: Dict[str, object] = {}

    def mallar_con_gmsh(lote) -> None:
        def avance(i: int, n: int) -> None:
            avisar(f"Mallando con Gmsh: pieza {i + 1} de {n}...")

        if lote:
            gmsh_por_solido.update(mallar_lote_gmsh(lote, tamano, al_avanzar=avance))

    def lote_tetraedros(solidos):
        lote = []
        for solido in solidos:
            motivo = motivo_sin_tetraedros_de(solido, tamano)
            if motivo:
                gmsh_por_solido[solido.id] = motivo
            else:
                lote.append((solido.id, solido.vertices, solido.triangulos, 3))
        return lote

    if con_gmsh:
        mallar_con_gmsh([(i, m.vertices, m.triangulos, 2) for i, m in medias.items()]
                        + lote_tetraedros([s for s in ordenados if s.id not in medias]))
    laminas_previas: Dict[str, MallaLamina] = {}
    for solido in ordenados:
        if solido.id in medias:
            remallada = gmsh_por_solido.pop(solido.id, None)
            try:
                laminas_previas[solido.id] = lamina_de_superficie(solido, medias[solido.id], tamano,
                                                                  None if isinstance(remallada, str) else remallada)
            except ErrorLamina as exc:
                sin_laminas[solido.id] = str(exc)
    if con_gmsh:  # las pocas láminas que fallaron al final van con tetraedros (otro proceso, solo si hace falta)
        mallar_con_gmsh(lote_tetraedros([s for s in ordenados if s.id in medias and s.id not in laminas_previas]))

    for k, solido in enumerate(ordenados):
        if solido.id in laminas_previas:
            lamina = laminas_previas[solido.id]
            parte = ParteMalla(id=solido.id, tipo="SOLIDO", material=solido.material, primer_hexaedro=n_hex,
                               ultimo_hexaedro=n_hex, superficies=lamina.superficies, lamina=True)
            partes.append(parte)
            laminas.append((parte, lamina))
            continue
        if con_gmsh:
            try:
                tetra = mallar_solido_tetra(solido, tamano, ordenados[:k], lambda p: _dentro_de_barras(proyecto, miembros, p),
                                            malla_gmsh=gmsh_por_solido.get(solido.id, "Gmsh no malló la pieza"))
            except ErrorTetraedros as exc:  # Gmsh no puede con esta pieza: se malla por vóxeles (y se pasa a C3D10)
                sin_tetraedros[solido.id] = str(exc)
            except ValueError as exc:  # coordenadas no válidas
                raise ErrorMalla(str(exc)) from exc
            else:
                if tetra is None:
                    avisos.append(f"El sólido {solido.id} ({solido.nombre}) queda dentro de otras piezas y no se analiza.")
                    continue
                recortados += bool(tetra.descartados)
                if tetra.volumen < 0.9 * abs(float(solido.volumen)):
                    sin_volumen.append(solido.id)
                solapes.update(frozenset((solido.id, otro)) for otro in tetra.cedido_a)
                parte = ParteMalla(id=solido.id, tipo="SOLIDO", material=solido.material, primer_hexaedro=n_hex,
                                   ultimo_hexaedro=n_hex, superficies=tetra.superficies, tetra=True)
                partes.append(parte)
                tetras.append((parte, tetra))
                continue
        try:
            libre = mallar_solido(solido, tamano, ordenados[:k], lambda p: _dentro_de_barras(proyecto, miembros, p))
        except ValueError as exc:  # coordenadas no válidas
            raise ErrorMalla(str(exc)) from exc
        if libre is None:
            avisos.append(f"El sólido {solido.id} ({solido.nombre}) queda dentro de otras piezas y no se analiza.")
            continue
        recortados += bool(libre.descartados)
        solapes.update(frozenset((solido.id, otro)) for otro in libre.cedido_a)
        area = sum(c.area for c in solido.caras)
        espesor = 2.0 * solido.volumen / area if area > 0 else np.inf  # espesor medio si es una pared delgada
        relacion = libre.volumen / solido.volumen if solido.volumen > 0 else 1.0
        if espesor < 0.7 * libre.celda_minima or not 0.75 <= relacion <= 1.33:
            delgados.append(solido.id)
        bloques_nodos.append(libre.coordenadas)
        lista_hex.append(libre.hexaedros + n_nodos)
        partes.append(ParteMalla(id=solido.id, tipo="SOLIDO", material=solido.material, primer_hexaedro=n_hex,
                                 ultimo_hexaedro=n_hex + len(libre.hexaedros), superficies=libre.superficies))
        elementos.extend([solido.id] * len(libre.hexaedros))
        n_nodos += len(libre.coordenadas)
        n_hex += len(libre.hexaedros)

    def lista_ids(ids: Sequence[str]) -> str:
        ids = list(ids)
        return ", ".join(ids[:8]) + (f" y {len(ids) - 8} más" if len(ids) > 8 else "")

    if laminas:
        avisos.append(f"{len(laminas)} pieza(s) son láminas (espesor ≤ 1/20 de su tamaño) y se modelaron con cáscaras S6 en su "
                      f"superficie media, con su espesor: {lista_ids(p.id for p, _ in laminas)}.")
    por_motivo: Dict[str, List[str]] = {}
    for id_, motivo in sin_laminas.items():
        por_motivo.setdefault(motivo, []).append(id_)
    for motivo, ids in por_motivo.items():
        avisos.append(f"No se pudo obtener la superficie media de {lista_ids(ids)} ({motivo}); se mallaron con tetraedros.")
    en_laminas = {p.id for p, _ in laminas}
    por_motivo = {}
    for id_, motivo in sin_tetraedros.items():
        if id_ not in en_laminas:
            por_motivo.setdefault(motivo, []).append(id_)
    for motivo, ids in por_motivo.items():
        avisos.append(f"{lista_ids(ids)}: Gmsh no pudo mallar {'esa pieza' if len(ids) == 1 else 'esas piezas'} ({motivo}); "
                      "se mallaron por vóxeles (superficies escalonadas)" + (", partidos en tetraedros C3D10." if con_gmsh else "."))
    if sin_volumen:
        avisos.append(f"{lista_ids(sin_volumen)}: son pequeñas frente al tamaño de elemento y su malla tiene menos del 90 % de su "
                      "volumen (las curvas quedan con pocos lados); reduzca el tamaño de elemento si sus esfuerzos importan.")
    if delgados:
        avisos.append(f"{len(delgados)} sólido(s) son de pared delgada para el tamaño de elemento usado ({lista_ids(delgados)}): la "
                      "malla no representa bien su espesor y los resultados son poco fiables. Reduzca el tamaño de elemento o "
                      "modele esas piezas con losas o muros.")
    if recortados:
        avisos.append(f"{recortados} sólido(s) se recortaron donde solapan con otras piezas (se conserva el volumen de la pieza "
                      "mayor o de la viga o columna, para no contarlo dos veces).")
    if not partes:
        raise ErrorMalla("No hay elementos que mallar.")
    parte_de_hex = np.concatenate([np.full(p.ultimo_hexaedro - p.primer_hexaedro, i) for i, p in enumerate(partes)])
    n_nodos_barras = sum(len(b) for b in bloques_nodos[: len(miembros)])
    malla = MallaSolida(
        coordenadas=np.vstack(bloques_nodos) if bloques_nodos else np.zeros((0, 3)),
        hexaedros=np.vstack(lista_hex).astype(np.int64) if lista_hex else np.zeros((0, 8), dtype=np.int64),
        parte_de_hexaedro=parte_de_hex, elemento_de_hexaedro=elementos, partes=partes, avisos=avisos, solapes=solapes,
    )
    _recortar_planos(proyecto, malla, miembros, n_nodos_barras)
    if con_gmsh and len(malla.hexaedros):
        _hexaedros_a_c3d10(malla)
    if tetras:
        _agregar_tetraedros(malla, tetras)
    if laminas:
        _agregar_laminas(malla, laminas)
    return malla


def _hexaedros_a_c3d10(malla: MallaSolida) -> None:
    """Parte todos los hexaedros de la malla en tetraedros cuadráticos ``C3D10`` conformes (``hex_a_tet``) y pasa las
    partes de hexaedros a tetraedros: sus superficies, sus anillos (con los nodos medios de las aristas de la sección y
    los pesos de una carga uniforme sobre triángulos cuadráticos: nada en los vértices, un tercio del área de cada
    triángulo en cada nodo medio)."""

    conversion = hexaedros_a_tetraedros(malla.coordenadas, malla.hexaedros)
    origen = conversion.hexaedro_de_tetraedro
    malos = np.unique(origen[conversion.calidad <= 0])
    if len(malos):
        piezas = sorted({malla.elemento_de_hexaedro[h] for h in malos.tolist()})
        malla.avisos.append(f"{len(malos)} hexaedro(s) muy deformados de {', '.join(piezas[:6])} dieron tetraedros planos o "
                            "invertidos: revise la geometría o cambie el tamaño de elemento.")
    elemento_de_hex = np.asarray(malla.elemento_de_hexaedro, dtype=object)
    malla.coordenadas = conversion.coordenadas
    malla.tetraedros = conversion.tetraedros
    malla.parte_de_tetraedro = malla.parte_de_hexaedro[origen].astype(np.int64)
    malla.elemento_de_tetraedro = elemento_de_hex[origen].tolist()
    for parte in malla.partes:
        if parte.tetra or parte.lamina:
            continue
        h0, h1 = parte.primer_hexaedro, parte.ultimo_hexaedro
        primero = int(conversion.primer_tetraedro[h0])
        parte.primer_tetraedro, parte.ultimo_tetraedro = primero, int(conversion.primer_tetraedro[h1])
        if parte.superficies:
            nuevas: Dict[int, np.ndarray] = {}
            for s, pares in parte.superficies.items():
                tetra, cara = conversion.caras_a_tetraedros(h0 + pares[:, 0], pares[:, 1])
                nuevas[s] = np.stack([tetra - primero, cara], axis=1).astype(np.int64)
            parte.superficies = nuevas
        if parte.anillos and len(parte.cuadros):
            # Los nodos del anillo k son primero + k·n + (nodo de la sección): la diagonal de cada cuadrilátero (la de su
            # nodo de menor índice) y por tanto los triángulos y los pesos son los mismos en todos los anillos.
            ampliados = [_anillo_cuadratico(conversion, malla.coordenadas, anillo, parte.cuadros) for anillo in parte.anillos]
            parte.anillos = [nodos for nodos, _ in ampliados]
            parte.pesos = ampliados[0][1]
        parte.tetra = True
        parte.primer_hexaedro = parte.ultimo_hexaedro = 0
        parte._fronteras = None
    malla.hexaedros = np.zeros((0, 8), dtype=np.int64)
    malla.parte_de_hexaedro = np.zeros(0, dtype=np.int64)
    malla.elemento_de_hexaedro = []


def _anillo_cuadratico(conversion, coordenadas: np.ndarray, anillo: np.ndarray, cuadros: np.ndarray):
    """(nodos, pesos) de un anillo de barra tras partir los hexaedros: sus nodos y, detrás, los nodos medios de las aristas
    de la sección; el peso de cada nodo medio es un tercio del área de los triángulos que lo tienen (los vértices, 0)."""

    anillo = np.asarray(anillo, dtype=np.int64)
    a, b, c, d = cuadros.T
    por_a_c = np.argmin(anillo[cuadros], axis=1) % 2 == 0  # la diagonal pasa por el nodo de menor índice
    triangulos = np.vstack([np.where(por_a_c[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1)),
                            np.where(por_a_c[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))])
    p = coordenadas[anillo[triangulos]]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    aristas = np.sort(np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]]), axis=1)
    unicas, inverso = np.unique(aristas, axis=0, return_inverse=True)
    pesos = np.bincount(inverso.ravel(), weights=np.tile(area / 3.0, 3), minlength=len(unicas))
    medios = np.atleast_1d(conversion.medio_de_arista(anillo[unicas[:, 0]], anillo[unicas[:, 1]]))
    return np.concatenate([anillo, medios]), np.concatenate([np.zeros(len(anillo)), pesos])


def _agregar_tetraedros(malla: MallaSolida, tetras: Sequence[Tuple[ParteMalla, MallaTetra]]) -> None:
    """Añade los tetraedros de los sólidos libres a la malla (después de recortar losas y muros, para no renumerar sus nodos)."""

    # van detrás de los que salen de partir hexaedros (si los hay)
    bloques, conectividad = [], [malla.tetraedros]
    partes, elementos = [np.asarray(malla.parte_de_tetraedro, dtype=np.int64)], list(malla.elemento_de_tetraedro)
    n_nodos, n_tetra = len(malla.coordenadas), len(malla.tetraedros)
    for parte, tetra in tetras:
        bloques.append(tetra.coordenadas)
        conectividad.append(tetra.tetraedros + n_nodos)
        parte.primer_tetraedro, parte.ultimo_tetraedro = n_tetra, n_tetra + len(tetra.tetraedros)
        partes.append(np.full(len(tetra.tetraedros), malla.partes.index(parte), dtype=np.int64))
        elementos.extend([parte.id] * len(tetra.tetraedros))
        n_nodos += len(tetra.coordenadas)
        n_tetra += len(tetra.tetraedros)
    malla.coordenadas = np.vstack([malla.coordenadas] + bloques)
    malla.tetraedros = np.vstack(conectividad).astype(np.int64)
    malla.parte_de_tetraedro = np.concatenate(partes)
    malla.elemento_de_tetraedro = elementos


def _agregar_laminas(malla: MallaSolida, laminas: Sequence[Tuple[ParteMalla, MallaLamina]]) -> None:
    """Añade las láminas ``S6`` de los sólidos de pared delgada (después de los tetraedros: son los últimos elementos)."""

    bloques, conectividad, espesores, normales, partes, elementos = [], [], [], [], [], []
    n_nodos, n_lamina = len(malla.coordenadas), 0
    for parte, lamina in laminas:
        bloques.append(lamina.coordenadas)
        conectividad.append(lamina.elementos + n_nodos)
        espesores.append(lamina.espesores)
        normales.append(lamina.normales)
        parte.primer_lamina, parte.ultimo_lamina = n_lamina, n_lamina + len(lamina.elementos)
        parte.nodos_superficie = {k: v + n_nodos for k, v in lamina.nodos_superficie.items()}
        partes.extend([malla.partes.index(parte)] * len(lamina.elementos))
        elementos.extend([parte.id] * len(lamina.elementos))
        n_nodos += len(lamina.coordenadas)
        n_lamina += len(lamina.elementos)
    malla.coordenadas = np.vstack([malla.coordenadas] + bloques)
    malla.laminas = np.vstack(conectividad).astype(np.int64)
    malla.espesor_de_lamina = np.concatenate(espesores)
    malla.normal_de_lamina = np.vstack(normales)
    malla.parte_de_lamina = np.array(partes, dtype=np.int64)
    malla.elemento_de_lamina = elementos


def _dentro_de_barras(proyecto: Proyecto, miembros: Sequence[MiembroBarra], puntos: np.ndarray) -> np.ndarray:
    """Máscara de los ``puntos`` que caen dentro del volumen de alguna viga o columna."""

    from shapely import contains_xy
    from shapely.geometry import Polygon

    from .seccion2d import contorno_local

    dentro = np.zeros(len(puntos), dtype=bool)
    for m in miembros:
        rel = puntos - m.p0
        s, u, v = rel @ m.t, rel @ m.e1, rel @ m.e2
        poligono = Polygon(contorno_local(proyecto.perfiles[m.perfil], m.ejes_ref))
        dentro |= (s > m.recorte0 + 1e-9) & (s < m.longitud - m.recorte1 - 1e-9) & contains_xy(poligono, u, v)
    return dentro


def _recortar_planos(proyecto: Proyecto, malla: MallaSolida, miembros: Sequence[MiembroBarra], n_nodos_barras: int) -> None:
    """Quita de las losas y los muros los hexaedros que quedan dentro de una viga o columna
    (el IFC suele colocar la losa solapada con las vigas): así no se cuenta dos veces el volumen
    y la losa queda apoyada contra los lados de las vigas."""

    indices_planos = [i for i, p in enumerate(malla.partes) if p.tipo in ("LOSA", "MURO") and not p.lamina]
    if not indices_planos:
        return
    centros = malla.coordenadas[malla.hexaedros].mean(axis=1)
    mantener = np.ones(len(malla.hexaedros), dtype=bool)
    primero = malla.partes[indices_planos[0]].primer_hexaedro
    ultimo = malla.partes[indices_planos[-1]].ultimo_hexaedro
    candidatos = np.arange(primero, ultimo)
    mantener[candidatos[_dentro_de_barras(proyecto, miembros, centros[candidatos])]] = False
    if mantener.all():
        return
    usados = np.zeros(len(malla.coordenadas), dtype=bool)
    usados[:n_nodos_barras] = True
    usados[malla.hexaedros[mantener].ravel()] = True
    nuevo = np.cumsum(usados) - 1
    malla.coordenadas = malla.coordenadas[usados]
    malla.hexaedros = nuevo[malla.hexaedros[mantener]]
    malla.parte_de_hexaedro = malla.parte_de_hexaedro[mantener]
    malla.elemento_de_hexaedro = [e for e, k in zip(malla.elemento_de_hexaedro, mantener) if k]
    inicio = 0
    for i, parte in enumerate(malla.partes):
        cuenta = int((malla.parte_de_hexaedro == i).sum())
        parte.primer_hexaedro, parte.ultimo_hexaedro, parte._fronteras = inicio, inicio + cuenta, None
        inicio += cuenta
    if any(p.ultimo_hexaedro == p.primer_hexaedro and not (p.tetra or p.lamina) for p in malla.partes):
        vacias = [p.id for p in malla.partes if p.ultimo_hexaedro == p.primer_hexaedro and not (p.tetra or p.lamina)]
        raise ErrorMalla(f"Las piezas {', '.join(vacias)} quedan totalmente dentro de otras: revise la geometría.")


#: Tetraedros por hexaedro al partirlos (5 o 6; con la numeración de las barras, casi siempre 6).
TETRAEDROS_POR_HEXAEDRO = 5.9


def estimar_hexaedros(proyecto: Proyecto, miembros: Sequence[MiembroBarra], tamano: float, tetraedros: bool = False) -> int:
    """Número aproximado de elementos con ese tamaño de elemento (sin mallar).

    Con ``tetraedros`` cuenta los elementos del análisis: tetraedros ``C3D10`` (los de Gmsh y los de partir los
    hexaedros de barras, losas y muros gruesos) más láminas ``S6`` de las piezas que son láminas. Sin él, hexaedros."""

    from ..core.geometria import area_poligono

    por_hexaedro = TETRAEDROS_POR_HEXAEDRO if tetraedros else 1.0
    total = 0.0
    for m in miembros:
        perfil = proyecto.perfiles[m.perfil]
        largo = max(m.longitud - m.recorte0 - m.recorte1, tamano)
        _, quads = malla_seccion(perfil, m.ejes_ref, tamano)
        total += por_hexaedro * len(quads) * (math.ceil(largo / tamano) + len(m.tramos))
    for coleccion in (proyecto.losas, proyecto.muros):
        for e in coleccion.values():
            area = area_poligono(np.array([proyecto.nodos[n].coordenadas for n in e.nodos_contorno], dtype=float))
            if tetraedros and plano_es_lamina(proyecto, e):
                total += 2.0 * area / tamano**2  # cuadriláteros partidos en dos triángulos S6
                continue
            capas = max(2, math.ceil(e.espesor / tamano)) if e.espesor > 0.08 else 1
            total += por_hexaedro * area / tamano**2 * capas
    for x in proyecto.solidos.values():
        if tetraedros and es_lamina(x):
            # superficie media (la mitad de la piel) en triángulos casi equiláteros de lado ``tamano``
            total += max(2.0, 0.5 * sum(c.area for c in x.caras) / (0.433 * tamano**2))
        elif tetraedros and _es_variedad(x.triangulos):
            total += estimar_tetraedros_solido(x, tamano)
        else:  # las piezas que Gmsh no puede mallar (superficies que se tocan por aristas) van por vóxeles
            total += por_hexaedro * estimar_hexaedros_solido(x, tamano)
    return int(total)


# ---------------------------------------------------------------------------
# Caras frontera y anillos
# ---------------------------------------------------------------------------


def _caras_frontera(malla: MallaSolida, parte: ParteMalla) -> CarasFrontera:
    hexa = malla.hexaedros[parte.primer_hexaedro: parte.ultimo_hexaedro]
    n = len(hexa)
    todas = np.stack([hexa[:, list(c)] for c in CARAS_HEXAEDRO], axis=1)  # (n, 6, 4)
    ordenadas = np.sort(todas.reshape(-1, 4), axis=1)
    _, primero, cuenta = np.unique(ordenadas, axis=0, return_index=True, return_counts=True)
    indices = primero[cuenta == 1]
    h, c = np.divmod(indices, 6)
    nodos = todas[h, c]
    p = malla.coordenadas[nodos]
    centro = p.mean(axis=1)
    normal = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    centro_hex = malla.coordenadas[hexa[h]].mean(axis=1)
    hacia_dentro = np.einsum("ij,ij->i", normal, centro - centro_hex) < 0
    normal[hacia_dentro] *= -1.0
    return CarasFrontera(hexaedro=h + parte.primer_hexaedro, cara=c + 1, nodos=nodos, centro=centro, normal=normal, area=area)


def _caras_frontera_tetra(malla: MallaSolida, parte: ParteMalla) -> CarasFrontera:
    tetra = malla.tetraedros[parte.primer_tetraedro: parte.ultimo_tetraedro]
    esquinas = np.stack([tetra[:, list(c)] for c in CARAS_TETRAEDRO], axis=1)  # (n, 4, 3)
    medios = np.stack([tetra[:, list(c)] for c in MEDIOS_CARA], axis=1)
    ordenadas = np.sort(esquinas.reshape(-1, 3), axis=1)
    _, primero, cuenta = np.unique(ordenadas, axis=0, return_index=True, return_counts=True)
    indices = primero[cuenta == 1]
    t, c = np.divmod(indices, 4)
    nodos = np.concatenate([esquinas[t, c], medios[t, c]], axis=1)  # (n, 6)
    p = malla.coordenadas[nodos[:, :3]]
    centro = p.mean(axis=1)
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    centro_tetra = malla.coordenadas[tetra[t, :4]].mean(axis=1)
    normal[np.einsum("ij,ij->i", normal, centro - centro_tetra) < 0] *= -1.0
    return CarasFrontera(hexaedro=len(malla.hexaedros) + parte.primer_tetraedro + t, cara=c + 1, nodos=nodos, centro=centro,
                         normal=normal, area=area, pesos_nodo=np.array([0, 0, 0, 1 / 3, 1 / 3, 1 / 3]))


def _caras_frontera_lamina(malla: MallaSolida, parte: ParteMalla) -> CarasFrontera:
    """Dos caras por lámina: la del lado de la normal (cara 1) y la opuesta (cara 2), en las pieles del sólido."""

    elementos = malla.laminas[parte.primer_lamina: parte.ultimo_lamina]
    normal = malla.normal_de_lamina[parte.primer_lamina: parte.ultimo_lamina]
    espesor = malla.espesor_de_lamina[parte.primer_lamina: parte.ultimo_lamina]
    p = malla.coordenadas[elementos[:, :3]]
    centro = p.mean(axis=1)
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    local = np.arange(len(elementos))
    base = len(malla.hexaedros) + len(malla.tetraedros) + parte.primer_lamina
    return CarasFrontera(
        hexaedro=np.concatenate([base + local, base + local]), cara=np.concatenate([np.full(len(local), 1), np.full(len(local), 2)]),
        nodos=np.vstack([elementos, elementos]),
        centro=np.vstack([centro + 0.5 * espesor[:, None] * normal, centro - 0.5 * espesor[:, None] * normal]),
        normal=np.vstack([normal, -normal]), area=np.concatenate([area, area]), pesos_nodo=np.array([0, 0, 0, 1 / 3, 1 / 3, 1 / 3]))
