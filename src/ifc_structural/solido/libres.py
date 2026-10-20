"""Mallado de hexaedros de un sólido de forma libre (geometría real leída del IFC).

Se rellena la caja que lo envuelve con una cuadrícula de hexaedros y se conservan los que
tienen el centro dentro de la superficie (voxelización). Las celdas se ajustan a la caja
del sólido, de modo que una placa o un bloque con caras paralelas a los ejes queda mallado
exactamente; las caras inclinadas o curvas quedan escalonadas (más finas cuanto menor el
tamaño de elemento). Cada cara de hexaedro que queda en el borde se asocia a la superficie
del sólido más cercana, y así se pueden aplicar apoyos y cargas a cualquier superficie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ..core.solidos import SolidoGeneral, muestras_de_superficie

#: Máximo de hexaedros por sólido: por encima se agranda el elemento.
MAXIMO_POR_SOLIDO = 120_000

#: Cara de hexaedro (S1..S6 de CalculiX) que mira en cada sentido (eje, signo).
_CARA_DE_SENTIDO = {(2, -1): 1, (2, 1): 2, (1, -1): 3, (0, 1): 4, (1, 1): 5, (0, -1): 6}


@dataclass
class MallaLibre:
    coordenadas: np.ndarray  # (N, 3)
    hexaedros: np.ndarray  # (M, 8)
    superficies: Dict[int, np.ndarray] = field(default_factory=dict)  # superficie -> (k, 2) (hexaedro local, cara 1..6)
    tamano: float = 0.0
    descartados: int = 0  # hexaedros quitados por solapar con otras piezas


def tamano_local(solido: SolidoGeneral, tamano: float) -> float:
    """Tamaño de celda del sólido: el pedido, pero al menos 2 celdas (3 si no es una caja) en su dimensión menor."""

    dimensiones = np.maximum(solido.extension, 1e-9)
    caja = float(np.prod(dimensiones))
    llenado = solido.volumen / caja if caja > 0 else 1.0
    h = min(tamano, float(dimensiones.min()) / (2.0 if llenado > 0.97 else 4.0))
    estimado = max(solido.volumen, 1e-12) / h**3
    if estimado > MAXIMO_POR_SOLIDO:
        h *= (estimado / MAXIMO_POR_SOLIDO) ** (1.0 / 3.0)
    return h


def estimar_hexaedros_solido(solido: SolidoGeneral, tamano: float) -> int:
    return int(max(solido.volumen, 1e-12) / tamano_local(solido, tamano) ** 3)


def malla_poligonal(vertices: np.ndarray, triangulos: np.ndarray) -> pv.PolyData:
    caras = np.hstack([np.full((len(triangulos), 1), 3), triangulos]).ravel().astype(np.int64)
    return pv.PolyData(np.asarray(vertices, dtype=float), caras)


def puntos_dentro(solido: SolidoGeneral, puntos: np.ndarray) -> np.ndarray:
    """Máscara de los ``puntos`` que están dentro del sólido (malla cerrada)."""

    if len(puntos) == 0:
        return np.zeros(0, dtype=bool)
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints

    filtro = vtkSelectEnclosedPoints()  # lanza un rayo desde cada punto y cuenta los cruces con la superficie
    filtro.SetInputData(pv.PolyData(np.asarray(puntos, dtype=float)))
    filtro.SetSurfaceData(malla_poligonal(solido.vertices, solido.triangulos))
    filtro.SetCheckSurface(False)
    filtro.SetTolerance(0.0)
    filtro.Update()
    return vtk_to_numpy(filtro.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)


def mallar_solido(solido: SolidoGeneral, tamano: float, mayores: Sequence[SolidoGeneral] = (),
                  excluir: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Optional[MallaLibre]:
    """Hexaedros del sólido. ``mayores`` son sólidos que lo contienen o lo solapan (se les cede
    el volumen común) y ``excluir`` devuelve, para una lista de puntos, cuáles están dentro de
    otras piezas (barras) que también le quitan volumen. ``None`` si no queda ningún hexaedro."""

    h = tamano_local(solido, tamano)
    minimo, maximo = solido.minimo, solido.maximo
    n = np.maximum(1, np.ceil((maximo - minimo) / h - 1e-3)).astype(int)
    paso = (maximo - minimo) / n
    ejes = [minimo[a] + (np.arange(n[a]) + 0.5) * paso[a] for a in range(3)]
    centros = np.stack(np.meshgrid(*ejes, indexing="ij"), axis=-1).reshape(-1, 3)
    dentro = puntos_dentro(solido, centros)
    descartados = 0
    for otro in mayores:
        if not dentro.any():
            break
        if np.any(otro.minimo - 1e-9 > maximo) or np.any(otro.maximo + 1e-9 < minimo):
            continue
        indices = np.where(dentro)[0]
        solapa = puntos_dentro(otro, centros[indices])
        descartados += int(solapa.sum())
        dentro[indices[solapa]] = False
    if excluir is not None and dentro.any():
        indices = np.where(dentro)[0]
        solapa = excluir(centros[indices])
        descartados += int(solapa.sum())
        dentro[indices[solapa]] = False
    if not dentro.any():
        return None

    ocupado = dentro.reshape(tuple(n))
    ii, jj, kk = np.where(ocupado)
    # nodos de las esquinas, numerados solo si se usan
    ny1, nz1 = n[1] + 1, n[2] + 1

    def clave(i, j, k):
        return (i * ny1 + j) * nz1 + k

    esquinas = np.stack([
        clave(ii, jj, kk), clave(ii + 1, jj, kk), clave(ii + 1, jj + 1, kk), clave(ii, jj + 1, kk),
        clave(ii, jj, kk + 1), clave(ii + 1, jj, kk + 1), clave(ii + 1, jj + 1, kk + 1), clave(ii, jj + 1, kk + 1),
    ], axis=1)
    usados, inverso = np.unique(esquinas, return_inverse=True)
    hexaedros = inverso.reshape(-1, 8).astype(np.int64)
    k_ = usados % nz1
    j_ = (usados // nz1) % ny1
    i_ = usados // (nz1 * ny1)
    coordenadas = minimo + np.stack([i_, j_, k_], axis=1) * paso

    # caras de borde y superficie a la que pertenecen
    relleno = np.pad(ocupado, 1)
    caras_borde = []  # (hexaedro, cara, centro, normal)
    indice_hex = np.full(relleno.shape, -1, dtype=np.int64)
    indice_hex[ii + 1, jj + 1, kk + 1] = np.arange(len(ii))
    for (eje, signo), numero in _CARA_DE_SENTIDO.items():
        d = [0, 0, 0]
        d[eje] = signo
        vecino = relleno[ii + 1 + d[0], jj + 1 + d[1], kk + 1 + d[2]]
        libres = np.where(~vecino)[0]
        if len(libres) == 0:
            continue
        centro = centros_de(ii[libres], jj[libres], kk[libres], minimo, paso)
        centro[:, eje] += signo * 0.5 * paso[eje]
        normal = np.zeros((len(libres), 3))
        normal[:, eje] = signo
        caras_borde.append((libres, np.full(len(libres), numero), centro, normal))
    superficies: Dict[int, np.ndarray] = {}
    if caras_borde and len(solido.caras):
        hexa = np.concatenate([c[0] for c in caras_borde])
        cara = np.concatenate([c[1] for c in caras_borde])
        centro = np.vstack([c[2] for c in caras_borde])
        normal = np.vstack([c[3] for c in caras_borde])
        superficie = _superficie_de_caras(solido, centro, normal, float(paso.max()))
        for s in np.unique(superficie[superficie >= 0]):
            marca = superficie == s
            superficies[int(s)] = np.stack([hexa[marca], cara[marca]], axis=1)
    return MallaLibre(coordenadas=coordenadas, hexaedros=hexaedros, superficies=superficies, tamano=h, descartados=descartados)


def centros_de(i, j, k, minimo, paso):
    return minimo + (np.stack([i, j, k], axis=1) + 0.5) * paso


def _superficie_de_caras(solido: SolidoGeneral, centro: np.ndarray, normal: np.ndarray, paso: float) -> np.ndarray:
    """Superficie del sólido (índice) a la que pertenece cada cara de hexaedro de borde, o -1 si
    no está sobre la superficie original (es el hueco que dejó otra pieza)."""

    puntos, triangulo = muestras_de_superficie(solido.vertices, solido.triangulos, max(paso * 0.5, 1e-4))
    p = solido.vertices[solido.triangulos]
    n_tri = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n_tri /= np.maximum(np.linalg.norm(n_tri, axis=1, keepdims=True), 1e-30)
    arbol = cKDTree(puntos)
    k = min(12, len(puntos))
    distancias, vecinos = arbol.query(centro, k=k)
    distancias, vecinos = np.atleast_2d(distancias.reshape(len(centro), -1)), np.atleast_2d(vecinos.reshape(len(centro), -1))
    alineadas = np.einsum("ijk,ik->ij", n_tri[triangulo[vecinos]], normal) > 0.25
    puntuacion = np.where(alineadas, distancias, np.inf)
    mejor = np.argmin(puntuacion, axis=1)
    fila = np.arange(len(centro))
    valido = np.isfinite(puntuacion[fila, mejor]) & (distancias[fila, mejor] <= 1.05 * paso)
    resultado = np.full(len(centro), -1, dtype=np.int64)
    resultado[valido] = solido.cara_de_triangulo[triangulo[vecinos[fila, mejor]]][valido]
    return resultado
