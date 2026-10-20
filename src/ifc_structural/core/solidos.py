"""Sólidos de forma libre: la geometría real de cualquier elemento del IFC
(placas, pernos, zapatas, escaleras, piezas especiales...) como malla de
triángulos, agrupada en **superficies** (caras) que se pueden seleccionar para
asignarles apoyos y cargas.

Las superficies se obtienen sin depender de cómo vino el IFC: los triángulos
vecinos se unen en una misma cara mientras su normal no cambie bruscamente.
Una cara plana agrupa los triángulos coplanares; una superficie curva suave
(el fuste de un perno, un chaflán redondeado) forma una sola cara curva.
Es solo NumPy, sin interfaz gráfica, para poder probarlo aisladamente.
"""

from __future__ import annotations

import base64
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

#: Dos triángulos vecinos son de la misma cara plana si sus normales difieren menos de esto.
TOLERANCIA_PLANA_GRADOS = 4.0
#: Paso máximo entre triángulos vecinos de una misma superficie curva suave.
PASO_CURVA_GRADOS = 32.0
#: Una faceta (área / perímetro²) por debajo de esto es una tira estrecha: se une a las vecinas de una superficie curva.
COMPACIDAD_FACETA = 0.03
#: Una cara plana tiene al menos esta fracción del área total (si no, se junta a una curva).
FRACCION_AREA_PLANA_MINIMA = 0.004


@dataclass
class CaraSolido:
    indice: int
    triangulos: np.ndarray  # índices de los triángulos de la cara
    area: float
    centro: np.ndarray
    normal: np.ndarray  # normal media unitaria, hacia fuera
    tipo: str  # "PLANA" o "CURVA"
    nombre: str


@dataclass
class SolidoGeneral:
    """Geometría real de un elemento, con sus superficies."""

    id: str
    nombre: str
    tipo_ifc: str
    material: str
    vertices: np.ndarray  # (N, 3) m
    triangulos: np.ndarray  # (M, 3) índices, orientados hacia fuera
    ifc_guid: Optional[str] = None
    caras: List[CaraSolido] = field(default_factory=list)
    cara_de_triangulo: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    cerrado: bool = True  # malla estanca (cada arista en dos triángulos): hace falta para mallarlo en 3D
    volumen: float = 0.0

    def __post_init__(self) -> None:
        if not len(self.caras):
            self.recalcular()

    def recalcular(self) -> None:
        """Orienta los triángulos, comprueba que el sólido es cerrado y reagrupa las caras."""

        self.vertices = np.asarray(self.vertices, dtype=float)
        self.triangulos, self.cerrado, self.volumen = orientar_hacia_fuera(self.vertices, np.asarray(self.triangulos, dtype=np.int64))
        self.caras, self.cara_de_triangulo = agrupar_caras(self.vertices, self.triangulos)

    @property
    def minimo(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def maximo(self) -> np.ndarray:
        return self.vertices.max(axis=0)

    @property
    def extension(self) -> np.ndarray:
        return self.maximo - self.minimo


# ---------------------------------------------------------------------------
# Geometría de la malla
# ---------------------------------------------------------------------------


def soldar_vertices(vertices: np.ndarray, triangulos: np.ndarray, tolerancia: float = 1e-6):
    """Une los vértices repetidos y descarta los triángulos degenerados."""

    clave = np.round(np.asarray(vertices, dtype=float) / tolerancia).astype(np.int64)
    _, primero, inverso = np.unique(clave, axis=0, return_index=True, return_inverse=True)
    vertices_unicos = np.asarray(vertices, dtype=float)[primero]
    t = inverso.reshape(-1)[np.asarray(triangulos, dtype=np.int64)]
    valido = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return vertices_unicos, t[valido]


def normales_y_areas(vertices: np.ndarray, triangulos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = vertices[triangulos]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    longitud = np.linalg.norm(n, axis=1)
    return n / np.maximum(longitud[:, None], 1e-30), 0.5 * longitud


def volumen_firmado(vertices: np.ndarray, triangulos: np.ndarray) -> float:
    p = vertices[triangulos]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


def _aristas(triangulos: np.ndarray) -> np.ndarray:
    return np.sort(np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]]), axis=1)


def es_cerrado(triangulos: np.ndarray) -> bool:
    _, cuenta = np.unique(_aristas(triangulos), axis=0, return_counts=True)
    return bool(np.all(cuenta == 2))


def orientar_hacia_fuera(vertices: np.ndarray, triangulos: np.ndarray):
    """(triángulos, cerrado, volumen): si el volumen firmado es negativo los triángulos
    estaban orientados hacia dentro y se invierten."""

    if len(triangulos) == 0:
        return triangulos, False, 0.0
    volumen = volumen_firmado(vertices, triangulos)
    if volumen < 0:
        triangulos = triangulos[:, ::-1].copy()
        volumen = -volumen
    return triangulos, es_cerrado(triangulos), volumen


# ---------------------------------------------------------------------------
# Superficies
# ---------------------------------------------------------------------------


def _nombre_por_normal(normal: np.ndarray) -> str:
    if normal[2] > 0.7:
        return "superior"
    if normal[2] < -0.7:
        return "inferior"
    eje = int(np.argmax(np.abs(normal[:2])))
    return f"lateral ({'+' if normal[eje] > 0 else '-'}{'XY'[eje]})"


def agrupar_caras(vertices: np.ndarray, triangulos: np.ndarray):
    """Agrupa los triángulos en superficies. Devuelve (caras, cara_de_triangulo)."""

    m = len(triangulos)
    if m == 0:
        return [], np.zeros(0, dtype=np.int64)
    normales, areas = normales_y_areas(vertices, triangulos)

    # vecinos por arista: vecinos_arista[t][k] = triángulos que comparten la arista k de t
    por_arista: Dict[Tuple[int, int], List[int]] = {}
    lados = ((0, 1), (1, 2), (2, 0))
    for t in range(m):
        for a, b in lados:
            por_arista.setdefault(tuple(sorted((int(triangulos[t, a]), int(triangulos[t, b])))), []).append(t)
    vecinos_arista: List[List[List[int]]] = []
    longitud_arista: List[List[float]] = []
    for t in range(m):
        fila, largos = [], []
        for a, b in lados:
            clave = tuple(sorted((int(triangulos[t, a]), int(triangulos[t, b]))))
            fila.append([v for v in por_arista[clave] if v != t])
            largos.append(float(np.linalg.norm(vertices[triangulos[t, a]] - vertices[triangulos[t, b]])))
        vecinos_arista.append(fila)
        longitud_arista.append(largos)
    vecinos = [[v for lado in fila for v in lado] for fila in vecinos_arista]

    cos_plana = np.cos(np.radians(TOLERANCIA_PLANA_GRADOS))
    cos_curva = np.cos(np.radians(PASO_CURVA_GRADOS))
    etiqueta = np.full(m, -1, dtype=np.int64)
    area_total = float(areas.sum())
    siguiente = 0

    # 1) caras planas: se crece desde el triángulo más grande sin separarse de su normal
    for semilla in np.argsort(-areas):
        if etiqueta[semilla] >= 0:
            continue
        pila, grupo = [int(semilla)], [int(semilla)]
        etiqueta[semilla] = siguiente
        while pila:
            t = pila.pop()
            for v in vecinos[t]:
                if etiqueta[v] < 0 and np.dot(normales[v], normales[semilla]) >= cos_plana:
                    etiqueta[v] = siguiente
                    pila.append(v)
                    grupo.append(v)
        if areas[grupo].sum() >= FRACCION_AREA_PLANA_MINIMA * area_total:
            siguiente += 1
        else:  # demasiado pequeña: se devuelve para unirse a una superficie curva
            etiqueta[np.asarray(grupo)] = -1

    # 2) las facetas estrechas y consecutivas de una superficie curva (el fuste de un perno, un
    #    redondeo) se funden en una sola cara curva; una cara plana ancha nunca se les une
    n_planas = siguiente
    perimetro = np.zeros(n_planas)
    for t in range(m):
        if etiqueta[t] < 0:
            continue
        for k in range(3):
            if not any(etiqueta[v] == etiqueta[t] for v in vecinos_arista[t][k]):
                perimetro[etiqueta[t]] += longitud_arista[t][k]
    area_cara = np.array([areas[etiqueta == k].sum() for k in range(n_planas)])
    estrecha = (area_cara / np.maximum(perimetro, 1e-12) ** 2) < COMPACIDAD_FACETA
    normal_cara = np.array([(normales[etiqueta == k] * areas[etiqueta == k][:, None]).sum(axis=0) for k in range(n_planas)])
    if n_planas:
        normal_cara = normal_cara / np.maximum(np.linalg.norm(normal_cara, axis=1, keepdims=True), 1e-30)
    padre = list(range(n_planas))

    def raiz(x: int) -> int:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for t in range(m):
        if etiqueta[t] < 0:
            continue
        for v in vecinos[t]:
            a, b = etiqueta[t], etiqueta[v]
            if b >= 0 and a != b and estrecha[a] and estrecha[b] and np.dot(normal_cara[a], normal_cara[b]) >= cos_curva:
                padre[raiz(a)] = raiz(b)
    raices = sorted({raiz(k) for k in range(n_planas)})
    nuevo_id = {r: i for i, r in enumerate(raices)}
    miembros = {r: sum(1 for k in range(n_planas) if raiz(k) == r) for r in raices}
    for t in range(m):
        if etiqueta[t] >= 0:
            etiqueta[t] = nuevo_id[raiz(etiqueta[t])]
    tipos = ["CURVA" if miembros[r] > 1 else "PLANA" for r in raices]
    siguiente = len(raices)

    # 3) superficies curvas suaves con lo que queda
    for semilla in np.argsort(-areas):
        if etiqueta[semilla] >= 0:
            continue
        pila = [int(semilla)]
        etiqueta[semilla] = siguiente
        while pila:
            t = pila.pop()
            for v in vecinos[t]:
                if etiqueta[v] < 0 and np.dot(normales[v], normales[t]) >= cos_curva:
                    etiqueta[v] = siguiente
                    pila.append(v)
        tipos.append("CURVA")
        siguiente += 1

    # numeración estable: de arriba a abajo y de menor a mayor Y y X
    centros = np.array([(areas[etiqueta == k][:, None] * vertices[triangulos[etiqueta == k]].mean(axis=1)).sum(axis=0)
                        / areas[etiqueta == k].sum() for k in range(siguiente)])
    orden_caras = np.lexsort((centros[:, 0], centros[:, 1], -centros[:, 2].round(4)))
    renumerar = np.empty(siguiente, dtype=np.int64)
    renumerar[orden_caras] = np.arange(siguiente)
    etiqueta = renumerar[etiqueta]

    caras: List[CaraSolido] = []
    for nuevo, k in enumerate(orden_caras):
        indices = np.where(etiqueta == nuevo)[0]
        a = areas[indices]
        normal = (normales[indices] * a[:, None]).sum(axis=0)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 1e-12 else normales[indices[0]]
        centro = (vertices[triangulos[indices]].mean(axis=1) * a[:, None]).sum(axis=0) / a.sum()
        nombre = (f"cara {nuevo + 1} · {_nombre_por_normal(normal)}" if tipos[k] == "PLANA" else f"cara {nuevo + 1} · curva")
        caras.append(CaraSolido(nuevo, indices, float(a.sum()), centro, normal, tipos[k], nombre))
    return caras, etiqueta


def muestras_de_superficie(vertices: np.ndarray, triangulos: np.ndarray, espaciado: float):
    """Puntos repartidos sobre la superficie (para asociar la malla de elementos a las caras).
    Devuelve (puntos (K, 3), triángulo de cada punto (K,))."""

    p = vertices[triangulos]
    lado_max = np.maximum.reduce([np.linalg.norm(p[:, 1] - p[:, 0], axis=1), np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                                  np.linalg.norm(p[:, 0] - p[:, 2], axis=1)])
    puntos, origen = [], []
    for t in range(len(triangulos)):
        n = max(1, int(np.ceil(lado_max[t] / espaciado)))
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        mascara = (i + j) <= n
        a = i[mascara] / n
        b = j[mascara] / n
        puntos.append(p[t, 0] + a[:, None] * (p[t, 1] - p[t, 0]) + b[:, None] * (p[t, 2] - p[t, 0]))
        origen.append(np.full(len(a), t))
    return np.vstack(puntos), np.concatenate(origen)


# ---------------------------------------------------------------------------
# Persistencia compacta (base64 de arrays comprimidos)
# ---------------------------------------------------------------------------


def empaquetar(array: np.ndarray, dtype) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(array, dtype=dtype).tobytes(), 6)).decode("ascii")


def desempaquetar(texto: str, dtype, forma: Tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(texto)), dtype=dtype).reshape(forma).copy()


def a_diccionario(s: SolidoGeneral) -> Dict[str, object]:
    return {
        "id": s.id, "nombre": s.nombre, "tipo_ifc": s.tipo_ifc, "material": s.material, "ifc_guid": s.ifc_guid,
        "n_vertices": len(s.vertices), "n_triangulos": len(s.triangulos),
        "vertices": empaquetar(s.vertices, np.float64), "triangulos": empaquetar(s.triangulos, np.int32),
    }


def desde_diccionario(d: Dict[str, object]) -> SolidoGeneral:
    return SolidoGeneral(
        id=d["id"], nombre=d["nombre"], tipo_ifc=d["tipo_ifc"], material=d["material"], ifc_guid=d.get("ifc_guid"),
        vertices=desempaquetar(d["vertices"], np.float64, (d["n_vertices"], 3)),
        triangulos=desempaquetar(d["triangulos"], np.int32, (d["n_triangulos"], 3)).astype(np.int64),
    )
