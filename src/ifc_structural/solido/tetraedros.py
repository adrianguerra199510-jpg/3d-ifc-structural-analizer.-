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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    cedido_a: List[str] = field(default_factory=list)  # sólidos mayores con los que solapa (les cedió volumen)


def gmsh_disponible() -> bool:
    try:
        import gmsh  # noqa: F401
    except Exception:  # falta la librería nativa o el paquete
        return False
    return True


#: Constantes del estimador de tetraedros de Gmsh, ajustadas midiendo mallas reales: tetraedros ≈ A·V/h³ + B·(triángulos
#: de la superficie), con la superficie partida en parches como hace ``classifySurfaces`` (ángulo de 40°).
_A_VOLUMEN, _B_SUPERFICIE, _C_CURVO, _ANGULO_PARCHES = 2.8, 1.38, 1.2, 40.0


def _parches(solido: SolidoGeneral):
    """(área, nº de triángulos y si es curvo, por parche; largo y parches de cada curva entre parches)."""

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v, t = np.asarray(solido.vertices, float), np.asarray(solido.triangulos, np.int64)
    p = v[t]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    m = len(t)
    clave = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    dueno = np.tile(np.arange(m), 3)
    unicas, inverso, cuenta = np.unique(clave, axis=0, return_inverse=True, return_counts=True)
    orden = np.argsort(inverso.ravel(), kind="stable")
    inicio = np.concatenate([[0], np.cumsum(cuenta)[:-1]])
    dobles = np.where(cuenta == 2)[0]
    ta, tb = dueno[orden][inicio[dobles]], dueno[orden][inicio[dobles] + 1]
    suave = np.einsum("ij,ij->i", n[ta], n[tb]) > np.cos(np.radians(_ANGULO_PARCHES))
    _, parche = connected_components(coo_matrix((np.ones(int(suave.sum())), (ta[suave], tb[suave])), shape=(m, m)),
                                     directed=False)
    npar = int(parche.max()) + 1 if m else 0
    area_p = np.bincount(parche, weights=area, minlength=npar)
    ntri_p = np.bincount(parche, minlength=npar).astype(float)
    media = np.zeros((npar, 3))
    np.add.at(media, parche, n * area[:, None])
    curvo = np.linalg.norm(media, axis=1) / np.maximum(area_p, 1e-30) < 0.995
    # curvas: aristas vivas agrupadas por pareja de parches y por continuidad
    e = unicas[dobles[~suave]]
    if not len(e):
        return area_p, ntri_p, curvo, np.zeros(0), np.zeros((0, 2), np.int64)
    par = np.sort(np.stack([parche[ta[~suave]], parche[tb[~suave]]], axis=1), axis=1)
    largo = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    _, grupo = np.unique(par, axis=0, return_inverse=True)
    grupo = grupo.ravel()
    nodos, inv_n = np.unique(np.concatenate([grupo * len(v) + e[:, 0], grupo * len(v) + e[:, 1]]), return_inverse=True)
    k = len(e)
    _, comp = connected_components(coo_matrix((np.ones(k), (inv_n[:k], inv_n[k:])), shape=(len(nodos),) * 2), directed=False)
    curva = comp[inv_n[:k]]
    nc = int(curva.max()) + 1
    pares = np.zeros((nc, 2), np.int64)
    pares[curva] = par
    return area_p, ntri_p, curvo, np.bincount(curva, weights=largo, minlength=nc), pares


def estimar_tetraedros_solido(solido: SolidoGeneral, tamano: float) -> int:
    """Número aproximado de tetraedros que hará Gmsh con elementos de ``tamano``: un término de volumen y otro de los
    triángulos de la superficie (cada curva entre parches lleva al menos 3 segmentos y los parches curvos conservan
    su detalle, por eso las piezas pequeñas no bajan de un mínimo aunque el tamaño crezca)."""

    if len(solido.triangulos) == 0:
        return 0
    area_p, ntri_p, curvo, largo_c, pares_c = _parches(solido)
    h = float(tamano)
    segmentos = np.maximum(3.0, np.ceil(largo_c / h - 1e-9))
    borde, largo_borde = np.zeros(len(area_p)), np.zeros(len(area_p))
    for lado in (0, 1):
        np.add.at(borde, pares_c[:, lado], segmentos)
        np.add.at(largo_borde, pares_c[:, lado], largo_c)
    h_parche = np.minimum(h, np.where(borde > 0, largo_borde / np.maximum(borde, 1.0), h))
    triangulos = np.maximum(np.maximum(borde - 2.0, 1.0), area_p / (0.433 * h_parche**2))
    triangulos = np.where(curvo, np.maximum(triangulos, _C_CURVO * ntri_p), triangulos)
    return int(_A_VOLUMEN * abs(float(solido.volumen)) / h**3 + _B_SUPERFICIE * float(triangulos.sum()))


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
#: Veces que se relanza el proceso de Gmsh si una pieza lo cuelga o lo cierra; después, las piezas que faltan van a
#: hexaedros. Cada lanzamiento es un proceso nuevo, y muchos seguidos hacen saltar a los antivirus.
MAXIMO_RELANZAMIENTOS = 3


def _comando_hijo(argumentos: Sequence[str]) -> Tuple[list, dict]:
    """Comando y entorno para lanzar ``proceso_gmsh`` (en la versión empaquetada, el propio ejecutable)."""

    entorno = dict(os.environ)
    if getattr(sys, "frozen", False):
        return [sys.executable, "--malla-gmsh", *argumentos], entorno
    raiz = str(Path(__file__).resolve().parents[2])  # carpeta que contiene el paquete ifc_structural
    entorno["PYTHONPATH"] = os.pathsep.join([raiz] + ([entorno["PYTHONPATH"]] if entorno.get("PYTHONPATH") else []))
    return [sys.executable, "-m", "ifc_structural.solido.proceso_gmsh", *argumentos], entorno


def _es_variedad(triangulos: np.ndarray) -> bool:
    """Cada arista la comparten exactamente dos triángulos (Gmsh no puede rellenar superficies que se tocan por aristas)."""

    aristas = np.sort(np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]]), axis=1)
    _, cuenta = np.unique(aristas, axis=0, return_counts=True)
    return bool(np.all(cuenta == 2))


def _leer_entero(ruta: Path) -> Optional[int]:
    try:
        return int(ruta.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):  # todavía no existe, o el hijo lo está reemplazando
        return None


def mallar_lote_gmsh(piezas: Sequence[Tuple], tamano: float, angulo: float = 40.0,
                     tiempo_por_pieza: Optional[float] = None,
                     al_avanzar: Optional[Callable[[int, int], None]] = None) -> Dict[str, object]:
    """Malla con Gmsh varias piezas ``(id, vértices, triángulos[, dimensión])`` con **un solo** proceso.

    Dimensión 3 (por defecto): tetraedros cuadráticos dentro de la superficie cerrada; devuelve, por id,
    ``(coordenadas, tetraedros (M, 10) de CalculiX)``. Dimensión 2: la superficie vuelta a mallar con triángulos casi
    equiláteros de lado ``tamano`` (la superficie media de una lámina); devuelve ``(coordenadas, triángulos (M, 3))``.
    Si no se pudo, el motivo (texto). Si una pieza cuelga Gmsh más de ``tiempo_por_pieza`` segundos (o lo cierra), se
    termina el proceso y se relanza desde la pieza siguiente, como mucho ``MAXIMO_RELANZAMIENTOS`` veces."""

    resultado: Dict[str, object] = {}
    if not piezas:
        return resultado
    piezas = [(p[0], p[1], p[2], int(p[3]) if len(p) > 3 else 3) for p in piezas]
    if not gmsh_disponible():
        return {p[0]: "Gmsh no está disponible" for p in piezas}
    validas = []
    for id_, vertices, triangulos, dimension in piezas:
        if dimension == 2 or _es_variedad(np.asarray(triangulos)):
            validas.append((id_, vertices, triangulos, dimension))
        else:
            resultado[id_] = "la superficie tiene aristas compartidas por más de dos triángulos"
    if not validas:
        return resultado
    limite = TIEMPO_LIMITE_GMSH if tiempo_por_pieza is None else tiempo_por_pieza
    n = len(validas)
    with tempfile.TemporaryDirectory(prefix="gmsh_", ignore_cleanup_errors=True) as nombre:
        carpeta = Path(nombre)
        entrada = carpeta / "entrada.npz"
        datos = {"n": n, "tamano": tamano, "angulo": angulo}
        for i, (_, vertices, triangulos, dimension) in enumerate(validas):
            datos[f"v{i}"] = np.asarray(vertices, float)
            datos[f"t{i}"] = np.asarray(triangulos, np.int64)
            datos[f"d{i}"] = dimension
        np.savez(entrada, **datos)
        actual = carpeta / "actual.txt"
        desde, lanzamientos = 0, 0
        while desde < n:
            if lanzamientos > MAXIMO_RELANZAMIENTOS:
                for i in range(desde, n):
                    resultado[validas[i][0]] = "Gmsh se cerró o se colgó demasiadas veces"
                break
            lanzamientos += 1
            comando, entorno = _comando_hijo(["--lote", str(entrada), str(carpeta), "--desde", str(desde)])
            try:
                proceso = subprocess.Popen(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=entorno,
                                           creationflags=0x08000000 if os.name == "nt" else 0)  # sin ventana de consola
            except OSError as exc:
                for i in range(desde, n):
                    resultado[validas[i][0]] = f"no se pudo lanzar Gmsh ({exc})"
                break
            pieza, inicio, colgada = None, time.monotonic(), None
            try:
                while proceso.poll() is None:
                    time.sleep(0.2)
                    leida = _leer_entero(actual)
                    if leida is not None and leida != pieza and leida >= desde:
                        pieza, inicio = leida, time.monotonic()
                        if al_avanzar is not None and pieza < n:
                            al_avanzar(pieza, n)
                    if time.monotonic() - inicio > limite:
                        colgada = pieza if pieza is not None else desde
                        break
            finally:
                if proceso.poll() is None:  # colgado, o error/cierre en este proceso: no dejar un Gmsh huérfano
                    proceso.kill()
                    proceso.wait()
            if colgada is not None:
                resultado[validas[colgada][0]] = f"Gmsh no terminó en {limite:.0f} s"
                desde = colgada + 1
                continue
            ultima = _leer_entero(actual)
            if ultima is not None and ultima >= n:
                break  # terminó todas
            if ultima is None or ultima < desde:
                # el proceso no llegó ni a empezar (Gmsh ausente, ejecutable bloqueado...): no se relanza en bucle
                for i in range(desde, n):
                    resultado[validas[i][0]] = f"el proceso de Gmsh terminó sin mallar (código {proceso.returncode})"
                break
            if not (carpeta / f"res_{ultima}.npz").exists() and not (carpeta / f"err_{ultima}.txt").exists():
                resultado[validas[ultima][0]] = f"Gmsh se cerró inesperadamente (código {proceso.returncode})"
            desde = ultima + 1
        for i, (id_, _, _, dimension) in enumerate(validas):
            if id_ in resultado:
                continue
            res, err = carpeta / f"res_{i}.npz", carpeta / f"err_{i}.txt"
            if res.exists():
                # se lee entero y se cierra antes de borrar la carpeta: en Windows un archivo abierto no se puede borrar
                with np.load(res) as d:
                    coordenadas = np.array(d["coordenadas"])
                    elementos = np.array(d["triangulos" if dimension == 2 else "tetraedros"], dtype=np.int64)
                resultado[id_] = (coordenadas, elementos if dimension == 2 else _orientar(coordenadas, elementos))
            elif err.exists():
                resultado[id_] = err.read_text(encoding="utf-8").strip() or "error de Gmsh"
            else:
                resultado[id_] = "Gmsh no llegó a mallar la pieza"
    return resultado


def _gmsh_tetraedros(vertices: np.ndarray, triangulos: np.ndarray, tamano: float, angulo: float = 40.0,
                     tiempo_limite: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Tetraedros cuadráticos de una sola superficie triangulada. Devuelve (coordenadas, tetraedros (M, 10))."""

    malla = mallar_lote_gmsh([("pieza", vertices, triangulos)], tamano, angulo, tiempo_limite)["pieza"]
    if isinstance(malla, str):
        raise ErrorTetraedros(malla)
    return malla


def motivo_sin_tetraedros(solido: SolidoGeneral, tamano: float) -> Optional[str]:
    """Por qué no se intenta mallar el sólido con Gmsh (``None`` si se puede intentar)."""

    if estimar_tetraedros_solido(solido, tamano) > MAXIMO_POR_SOLIDO:
        return f"tendría más de {MAXIMO_POR_SOLIDO:,} tetraedros".replace(",", " ")
    if not _es_variedad(np.asarray(solido.triangulos)):
        return "la superficie tiene aristas compartidas por más de dos triángulos"
    return None


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


def _sin_trozos_sueltos(tetraedros: np.ndarray, mantener: np.ndarray, fraccion: float = 0.05) -> np.ndarray:
    """Máscara de los tetraedros de ``mantener`` que están en un trozo conexo (por vértices) con al menos ``fraccion``
    de ellos; el mayor se conserva siempre."""

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    indices = np.where(mantener)[0]
    esquinas = tetraedros[indices, :4]
    n = int(tetraedros.max()) + 1
    aristas = np.concatenate([esquinas[:, [0, 1]], esquinas[:, [0, 2]], esquinas[:, [0, 3]]])
    _, etiqueta = connected_components(coo_matrix((np.ones(len(aristas)), (aristas[:, 0], aristas[:, 1])), shape=(n, n)),
                                       directed=False)
    trozo = etiqueta[esquinas[:, 0]]
    cuenta = np.bincount(trozo)
    vale = (cuenta[trozo] >= fraccion * len(indices)) | (trozo == int(np.argmax(cuenta)))
    resultado = np.zeros(len(mantener), dtype=bool)
    resultado[indices[vale]] = True
    return resultado


def mallar_solido_tetra(solido: SolidoGeneral, tamano: float, mayores: Sequence[SolidoGeneral] = (),
                        excluir: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                        malla_gmsh: object = None) -> Optional[MallaTetra]:
    """Tetraedros del sólido. ``mayores`` son sólidos que lo contienen o lo solapan (se les cede el volumen común) y
    ``excluir`` devuelve, para una lista de puntos, cuáles están dentro de otras piezas (barras). ``malla_gmsh`` es el
    resultado de ``mallar_lote_gmsh`` para esta pieza (si no se da, se malla aquí). ``None`` si no queda ningún
    tetraedro. Lanza ``ErrorTetraedros`` si Gmsh no puede mallar la pieza."""

    minimo, maximo = solido.minimo, solido.maximo
    if not (np.isfinite(minimo).all() and np.isfinite(maximo).all() and np.isfinite(solido.volumen)):
        raise ValueError(f"el sólido {solido.id} ({solido.nombre}) tiene coordenadas no válidas")
    if malla_gmsh is None:
        motivo = motivo_sin_tetraedros(solido, tamano)
        if motivo:
            raise ErrorTetraedros(motivo)
        malla_gmsh = _gmsh_tetraedros(solido.vertices, solido.triangulos, tamano)
    if isinstance(malla_gmsh, str):
        raise ErrorTetraedros(malla_gmsh)
    coordenadas, tetraedros = malla_gmsh
    vol = np.abs(volumenes_firmados(coordenadas, tetraedros))
    volumen_malla = float(vol.sum())

    mantener = np.ones(len(tetraedros), dtype=bool)
    centros = coordenadas[tetraedros[:, :4]].mean(axis=1)
    descartados = 0
    cedido_a: List[str] = []
    for otro in mayores:
        if not mantener.any():
            break
        if np.any(otro.minimo - 1e-9 > maximo) or np.any(otro.maximo + 1e-9 < minimo):
            continue
        indices = np.where(mantener)[0]
        solapa = puntos_dentro(otro, centros[indices])
        descartados += int(solapa.sum())
        mantener[indices[solapa]] = False
        if solapa.any():
            cedido_a.append(otro.id)
    if excluir is not None and mantener.any():
        indices = np.where(mantener)[0]
        solapa = excluir(centros[indices])
        descartados += int(solapa.sum())
        mantener[indices[solapa]] = False
    if not mantener.any():
        return None
    if descartados:
        # el recorte puede dejar trocitos sueltos (la punta de una ménsula al otro lado de la pieza que la atraviesa): sin
        # nada que los sujete quedarían flotando; se descartan los que tienen menos del 5 % de los tetraedros
        mantener &= _sin_trozos_sueltos(tetraedros, mantener)
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
                      descartados=descartados, volumen=volumen_malla, cedido_a=cedido_a)
