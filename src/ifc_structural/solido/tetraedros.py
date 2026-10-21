"""Mallado de un sólido de forma libre con tetraedros cuadráticos ``C3D10`` (Gmsh).

Es el mismo tipo de elemento del modelo sólido de Placa Base Pro: el contorno triangulado del sólido (la geometría
real leída del IFC) se vuelve a mallar con Gmsh y se rellena con tetraedros de segundo orden, que siguen las
superficies curvas e inclinadas sin escalonarlas. Cada cara de tetraedro que queda en el borde se asocia a la
superficie del sólido más cercana, de modo que se pueden aplicar apoyos y cargas a cualquier superficie.

Orden de nodos de ``C3D10`` (CalculiX): 1-4 los vértices, 5-10 los nodos medios de las aristas 1-2, 2-3, 3-1, 1-4,
2-4 y 3-4. Caras: ``S1`` = 1-2-3, ``S2`` = 1-4-2, ``S3`` = 2-4-3, ``S4`` = 3-4-1.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.solidos import SolidoGeneral
from .libres import _superficie_de_caras, puntos_dentro

#: Vértices de cada cara de un tetraedro (índices 0-based), en el orden S1..S4 de CalculiX.
CARAS_TETRAEDRO = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
#: Nodos medios (índices 0-based dentro de los 10) de cada cara, en el mismo orden.
MEDIOS_CARA = ((4, 5, 6), (7, 8, 4), (8, 9, 5), (9, 7, 6))
#: Máximo de tetraedros por sólido y en todo el modelo.
MAXIMO_POR_SOLIDO = 250_000


class ErrorTetraedros(RuntimeError):
    """Gmsh no está disponible o no pudo mallar el sólido (se usan hexaedros en su lugar)."""


@dataclass
class MallaTetra:
    coordenadas: np.ndarray  # (N, 3)
    tetraedros: np.ndarray  # (M, 10) orden de CalculiX
    superficies: Dict[int, np.ndarray] = field(default_factory=dict)  # superficie -> (k, 2) (tetraedro local, cara 1..4)
    tamano: float = 0.0
    descartados: int = 0  # tetraedros quitados por solapar con otras piezas
    volumen: float = 0.0  # volumen de la malla más el de los tetraedros cedidos a otras piezas


def gmsh_disponible() -> bool:
    try:
        import gmsh  # noqa: F401
    except Exception:  # falta la librería nativa o el paquete
        return False
    return True


def estimar_tetraedros_solido(solido: SolidoGeneral, tamano: float) -> int:
    """Número aproximado de tetraedros con elementos de ``tamano`` (Gmsh refina donde la pieza es delgada)."""

    area = sum(c.area for c in solido.caras)
    volumen = max(float(solido.volumen), 1e-12)
    espesor = 2.0 * volumen / area if area > 0 else tamano
    h = min(0.85 * tamano, 3.5 * espesor)
    return int(max(6.0, volumen / (0.118 * h**3)))


def volumenes_firmados(coordenadas: np.ndarray, tetraedros: np.ndarray) -> np.ndarray:
    p = coordenadas[tetraedros[:, :4]]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def _orientar(coordenadas: np.ndarray, tetraedros: np.ndarray) -> np.ndarray:
    """Tetraedros con volumen positivo (CalculiX rechaza jacobianos negativos): se intercambian los vértices 2 y 3."""

    t = tetraedros.copy()
    negativo = volumenes_firmados(coordenadas, t) < 0
    if negativo.any():
        t[negativo] = t[negativo][:, [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]]
    return t


#: Segundos que se le dan a Gmsh por pieza antes de pasar a hexaedros (hay superficies con las que se queda colgado).
TIEMPO_LIMITE_GMSH = 60.0


def _comando_hijo(entrada: str, salida: str) -> Tuple[list, dict]:
    """Comando y entorno para lanzar ``proceso_gmsh`` (en la versión empaquetada, el propio ejecutable)."""

    entorno = dict(os.environ)
    if getattr(sys, "frozen", False):
        return [sys.executable, "--malla-gmsh", entrada, salida], entorno
    raiz = str(Path(__file__).resolve().parents[2])  # carpeta que contiene el paquete ifc_structural
    entorno["PYTHONPATH"] = os.pathsep.join([raiz] + ([entorno["PYTHONPATH"]] if entorno.get("PYTHONPATH") else []))
    return [sys.executable, "-m", "ifc_structural.solido.proceso_gmsh", entrada, salida], entorno


def _es_variedad(triangulos: np.ndarray) -> bool:
    """Cada arista la comparten exactamente dos triángulos (Gmsh no puede rellenar superficies que se tocan por aristas)."""

    aristas = np.sort(np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]]), axis=1)
    _, cuenta = np.unique(aristas, axis=0, return_counts=True)
    return bool(np.all(cuenta == 2))


def _gmsh_tetraedros(vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float = 40.0,
                     tiempo_limite: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Tetraedros cuadráticos de la superficie triangulada, mallados por Gmsh en un proceso aparte.
    Devuelve (coordenadas, tetraedros (M, 10) de CalculiX)."""

    if not gmsh_disponible():
        raise ErrorTetraedros("Gmsh no está disponible")
    if not _es_variedad(np.asarray(triangulos)):
        raise ErrorTetraedros("la superficie tiene aristas compartidas por más de dos triángulos")
    tiempo_limite = TIEMPO_LIMITE_GMSH if tiempo_limite is None else tiempo_limite
    with tempfile.TemporaryDirectory(prefix="gmsh_", ignore_cleanup_errors=True) as carpeta:
        entrada, salida = str(Path(carpeta) / "entrada.npz"), str(Path(carpeta) / "salida.npz")
        np.savez(entrada, vertices=np.asarray(vertices, float), triangulos=np.asarray(triangulos, np.int64),
                 tamano=tamano, angulo=angulo)
        comando, entorno = _comando_hijo(entrada, salida)
        try:
            r = subprocess.run(comando, capture_output=True, text=True, timeout=tiempo_limite, env=entorno,
                               creationflags=0x08000000 if os.name == "nt" else 0)  # sin ventana de consola
        except subprocess.TimeoutExpired as exc:
            raise ErrorTetraedros(f"Gmsh no terminó en {tiempo_limite:.0f} s") from exc
        except OSError as exc:
            raise ErrorTetraedros(f"no se pudo lanzar Gmsh ({exc})") from exc
        if r.returncode != 0 or not Path(salida).exists():
            mensaje = next((l[6:].strip() for l in reversed((r.stdout or "").splitlines()) if l.startswith("ERROR:")),
                           f"Gmsh terminó con código {r.returncode}")
            raise ErrorTetraedros(mensaje)
        # El .npz se lee entero y se cierra antes de borrar la carpeta: en Windows un archivo abierto no se puede borrar.
        with np.load(salida) as datos:
            coordenadas, tetraedros = np.array(datos["coordenadas"]), np.array(datos["tetraedros"], dtype=np.int64)
    return coordenadas, _orientar(coordenadas, tetraedros)


def _caras_de_borde(coordenadas: np.ndarray, tetraedros: np.ndarray):
    """Caras de tetraedro sin vecino: (tetraedro, cara 1..4, centro, normal hacia fuera, área)."""

    n = len(tetraedros)
    esquinas = np.stack([tetraedros[:, list(c)] for c in CARAS_TETRAEDRO], axis=1)  # (n, 4, 3)
    claves = np.sort(esquinas.reshape(-1, 3), axis=1)
    _, primero, cuenta = np.unique(claves, axis=0, return_index=True, return_counts=True)
    indices = primero[cuenta == 1]
    t, c = np.divmod(indices, 4)
    p = coordenadas[esquinas[t, c]]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    centro = p.mean(axis=1)
    hacia_dentro = np.einsum("ij,ij->i", normal, centro - coordenadas[tetraedros[t, :4]].mean(axis=1)) < 0
    normal[hacia_dentro] *= -1.0
    return t, c + 1, centro, normal, area


def mallar_solido_tetra(solido: SolidoGeneral, tamano: float, mayores: Sequence[SolidoGeneral] = (),
                        excluir: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Optional[MallaTetra]:
    """Tetraedros del sólido. ``mayores`` son sólidos que lo contienen o lo solapan (se les cede el volumen común) y
    ``excluir`` devuelve, para una lista de puntos, cuáles están dentro de otras piezas (barras). ``None`` si no
    queda ningún tetraedro. Lanza ``ErrorTetraedros`` si Gmsh no puede mallar la pieza."""

    minimo, maximo = solido.minimo, solido.maximo
    if not (np.isfinite(minimo).all() and np.isfinite(maximo).all() and np.isfinite(solido.volumen)):
        raise ValueError(f"el sólido {solido.id} ({solido.nombre}) tiene coordenadas no válidas")
    if estimar_tetraedros_solido(solido, tamano) > MAXIMO_POR_SOLIDO:
        raise ErrorTetraedros(f"tendría más de {MAXIMO_POR_SOLIDO:,} tetraedros".replace(",", " "))
    coordenadas, tetraedros = _gmsh_tetraedros(solido.vertices, solido.triangulos, tamano)
    vol = np.abs(volumenes_firmados(coordenadas, tetraedros))
    volumen_malla = float(vol.sum())

    mantener = np.ones(len(tetraedros), dtype=bool)
    centros = coordenadas[tetraedros[:, :4]].mean(axis=1)
    descartados = 0
    for otro in mayores:
        if not mantener.any():
            break
        if np.any(otro.minimo - 1e-9 > maximo) or np.any(otro.maximo + 1e-9 < minimo):
            continue
        indices = np.where(mantener)[0]
        solapa = puntos_dentro(otro, centros[indices])
        descartados += int(solapa.sum())
        mantener[indices[solapa]] = False
    if excluir is not None and mantener.any():
        indices = np.where(mantener)[0]
        solapa = excluir(centros[indices])
        descartados += int(solapa.sum())
        mantener[indices[solapa]] = False
    if not mantener.any():
        return None
    tetraedros = tetraedros[mantener]
    usados, inverso = np.unique(tetraedros, return_inverse=True)
    tetraedros = inverso.reshape(-1, 10).astype(np.int64)
    coordenadas = coordenadas[usados]

    superficies: Dict[int, np.ndarray] = {}
    t, cara, centro, normal, _ = _caras_de_borde(coordenadas, tetraedros)
    if len(t) and len(solido.caras):
        superficie = _superficie_de_caras(solido, centro, normal, tamano)
        for s in np.unique(superficie[superficie >= 0]):
            marca = superficie == s
            superficies[int(s)] = np.stack([t[marca], cara[marca]], axis=1)
    return MallaTetra(coordenadas=coordenadas, tetraedros=tetraedros, superficies=superficies, tamano=tamano,
                      descartados=descartados, volumen=volumen_malla)
