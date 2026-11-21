"""Elementos «solo carga»: no se mallan; su peso y sus cargas pasan a las piezas malladas que tocan.

Teoría
------
Las piezas «solo carga» que se tocan entre sí forman un **grupo**, que se trata como un cuerpo rígido: no aporta
rigidez, solo carga. En cada caso de carga se suman su peso propio (γ·V en el centro de gravedad de cada pieza, si el
caso lleva peso propio) y las cargas asignadas a sus superficies (presión, fuerza o momento), y se obtiene la
resultante **F** y el momento **M** respecto de un punto de referencia **c**.

El grupo apoya en las piezas malladas a través de sus **puntos de contacto** p_j (puntos de su superficie a menos de
la tolerancia de una pieza mallada, o dentro de ella), cada uno con el área a_j que representa. Las fuerzas que las
piezas malladas ejercen sobre el grupo, g_j, deben equilibrarlo:

    Σ g_j + F = 0            Σ (p_j − c) × g_j + M = 0

Son 6 ecuaciones y 3·n incógnitas: entre todas las soluciones se toma la de **mínima energía ponderada por el área**,
min Σ |g_j|² / a_j, que reparte la carga en proporción al área de contacto y, si el grupo apoya en varios sitios,
según el brazo de palanca de cada uno (como una viga sobre varios apoyos tratada como rígida). Su solución cerrada es

    g_j = a_j · B_jᵀ · λ,   (Σ a_j B_j B_jᵀ) λ = −[F; M],   B_j = [I; [p_j − c]×]

La pieza mallada recibe −g_j en p_j, que se aplica al nodo de su superficie más cercano. Si los contactos no pueden
equilibrar el grupo (p. ej. apoya en una línea y la carga lo haría volcar alrededor de ella), el sistema no tiene
solución exacta: se aplica la de mínimos cuadrados y se avisa.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.casos_carga import casos_con_peso_propio
from ..core.modelo import Proyecto

Elemento = Tuple[str, str]
COLECCIONES = {"VIGA": "vigas", "COLUMNA": "columnas", "LOSA": "losas", "MURO": "muros", "SOLIDO": "solidos"}
#: Puntos de muestra por pieza para buscar sus contactos (espaciado entre 2 mm y 5 cm, como «Conectar»).
MUESTRAS_CONTACTO = 4000


def elementos_solo_carga(proyecto: Proyecto) -> Set[Elemento]:
    """Los elementos marcados «solo carga» que existen en el proyecto."""

    marcados = getattr(proyecto, "solo_carga", None) or set()
    return {(t, i) for t, i in marcados if i in getattr(proyecto, COLECCIONES.get(t, ""), {})}


def proyecto_mallado(proyecto: Proyecto) -> Proyecto:
    """El proyecto sin los elementos «solo carga» (lo que se malla). Si no hay ninguno, el mismo proyecto."""

    fuera = elementos_solo_carga(proyecto)
    if not fuera:
        return proyecto
    copia = copy.copy(proyecto)
    for tipo, nombre in COLECCIONES.items():
        setattr(copia, nombre, {i: e for i, e in getattr(proyecto, nombre).items() if (tipo, i) not in fuera})
    copia.uniones = [u for u in proyecto.uniones if (u.a_tipo, u.a_id) not in fuera and (u.b_tipo, u.b_id) not in fuera]
    copia.solo_carga = set()
    return copia


@dataclass
class Transferencia:
    """Fuerzas que un grupo «solo carga» transmite a una pieza mallada, en cada caso de carga."""

    destino: Elemento
    puntos: np.ndarray  # (n, 3) puntos de contacto, m
    fuerzas: Dict[str, np.ndarray] = field(default_factory=dict)  # caso -> (n, 3) kN sobre la pieza mallada


@dataclass
class GrupoSoloCarga:
    piezas: List[Elemento]
    destinos: List[Elemento]
    resultantes: Dict[str, np.ndarray] = field(default_factory=dict)  # caso -> (3,) kN, peso y cargas del grupo
    desequilibrio: Dict[str, float] = field(default_factory=dict)  # caso -> fracción de la carga que no se equilibra


@dataclass
class ResultadoSoloCarga:
    transferencias: List[Transferencia] = field(default_factory=list)
    grupos: List[GrupoSoloCarga] = field(default_factory=list)
    avisos: List[str] = field(default_factory=list)
    sueltos: List[List[Elemento]] = field(default_factory=list)  # grupos que no tocan ninguna pieza mallada


def _volumen_y_centro(sup) -> Tuple[float, np.ndarray]:
    p = sup.vertices[sup.triangulos]
    v6 = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2]))  # 6 × volumen de cada tetraedro con el origen
    volumen = float(v6.sum()) / 6.0
    if abs(volumen) < 1e-15:
        return 0.0, sup.vertices.mean(axis=0)
    centro = (v6[:, None] * p.sum(axis=1)).sum(axis=0) / (4.0 * float(v6.sum()))
    return abs(volumen), centro


def _carga_de_superficie(c, sup) -> Tuple[np.ndarray, np.ndarray]:
    """(fuerzas (k, 3), puntos (k, 3)) de una carga sobre una superficie, y el momento puro en ``momento``."""

    tri = sup.triangulos[np.asarray(sup.cara) == c.cara]
    if len(tri) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    p = sup.vertices[tri]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    normal = n / np.maximum(2 * area[:, None], 1e-30)
    centros = p.mean(axis=1)
    if c.tipo == "PRESION":
        d = np.broadcast_to(np.asarray(c.direccion, float), normal.shape) if c.direccion is not None else -normal
        return c.valor * area[:, None] * d, centros
    if c.tipo == "FUERZA" and c.direccion is not None:
        reparto = area / max(float(area.sum()), 1e-30)
        return c.valor * reparto[:, None] * np.asarray(c.direccion, float)[None, :], centros
    return np.zeros((0, 3)), np.zeros((0, 3))


def _producto_cruz(r: np.ndarray) -> np.ndarray:
    """Matrices [r]× (n, 3, 3) tales que [r]× g = r × g."""

    m = np.zeros((len(r), 3, 3))
    m[:, 0, 1], m[:, 0, 2] = -r[:, 2], r[:, 1]
    m[:, 1, 0], m[:, 1, 2] = r[:, 2], -r[:, 0]
    m[:, 2, 0], m[:, 2, 1] = -r[:, 1], r[:, 0]
    return m


def repartir(puntos: np.ndarray, areas: np.ndarray, fuerza: np.ndarray, momento: np.ndarray, referencia: np.ndarray
             ) -> Tuple[np.ndarray, float]:
    """Fuerzas g_j (n, 3) en los puntos de contacto que equilibran (F, M respecto de ``referencia``) con la mínima
    energía ponderada por el área, y la fracción de la carga que no se pudo equilibrar (0 si el equilibrio es exacto)."""

    escala = max(float(np.linalg.norm(np.ptp(puntos, axis=0))), 1e-3)
    r = (puntos - referencia) / escala
    b = -np.concatenate([fuerza, np.asarray(momento, float) / escala])
    bloques = np.concatenate([np.broadcast_to(np.eye(3), (len(r), 3, 3)), _producto_cruz(r)], axis=1)  # (n, 6, 3)
    w = areas / max(float(areas.sum()), 1e-30)
    matriz = np.einsum("n,nij,nkj->ik", w, bloques, bloques)  # Σ w B Bᵀ (6×6)
    lam = np.linalg.pinv(matriz, rcond=1e-9) @ b
    g = w[:, None] * np.einsum("nij,i->nj", bloques, lam)
    residuo = float(np.linalg.norm(np.einsum("nij,nj->i", bloques, g) - b))
    return g, residuo / max(float(np.linalg.norm(b)), 1e-12)


def calcular_transferencias(proyecto: Proyecto, casos: Dict, cargas_superficie_por_caso: Dict[str, list],
                            tolerancia: float, al_avanzar=None) -> ResultadoSoloCarga:
    """Fuerzas que los grupos «solo carga» transmiten a las piezas malladas en cada caso (``proyecto`` completo)."""

    from .deteccion import _polidata, distancia_con_signo, elementos_del_proyecto, muestras, superficie_de

    resultado = ResultadoSoloCarga()
    fuera = elementos_solo_carga(proyecto)
    if not fuera:
        return resultado
    if al_avanzar:
        al_avanzar("Elementos «solo carga»: buscando sus contactos...")
    tol = max(float(tolerancia), 5e-4)
    sups: Dict[Elemento, object] = {}
    for e in elementos_del_proyecto(proyecto):
        try:
            sups[e] = superficie_de(proyecto, *e)
        except (KeyError, ValueError, IndexError):
            continue
    cajas = {e: s.caja for e, s in sups.items()}
    mallas: Dict[Elemento, object] = {}

    def malla_de(e):
        if e not in mallas:
            mallas[e] = _polidata(sups[e])
        return mallas[e]

    def cerca(a, b) -> bool:
        (amin, amax), (bmin, bmax) = cajas[a], cajas[b]
        return not (np.any(amin - tol > bmax) or np.any(bmin - tol > amax))

    datos = {e: muestras(sups[e], MUESTRAS_CONTACTO) for e in fuera if e in sups}

    def contacto(e, otro) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(puntos, áreas) de la superficie de ``e`` que tocan a ``otro`` (o están dentro)."""

        puntos, peso, _ = datos[e]
        minimo, maximo = cajas[otro]
        dentro = np.all((puntos >= minimo - tol) & (puntos <= maximo + tol), axis=1)
        if not dentro.any():
            return None
        d = distancia_con_signo(puntos[dentro], sups[otro], malla_de(otro))
        toca = d <= tol
        if not toca.any():
            return None
        return puntos[dentro][toca], peso[dentro][toca]

    # grupos: piezas «solo carga» que se tocan entre sí
    lista = [e for e in sorted(fuera) if e in sups]
    padre = {e: e for e in lista}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for k, a in enumerate(lista):
        for b in lista[k + 1:]:
            if raiz(a) != raiz(b) and cerca(a, b) and contacto(a, b) is not None:
                padre[raiz(a)] = raiz(b)
    grupos: Dict[Elemento, List[Elemento]] = {}
    for e in lista:
        grupos.setdefault(raiz(e), []).append(e)

    mallados = [e for e in sups if e not in fuera]
    con_peso = set(casos_con_peso_propio(casos))
    for piezas in grupos.values():
        # contactos con las piezas malladas
        puntos, areas, destino_de = [], [], []
        for e in piezas:
            for m in mallados:
                if not cerca(e, m):
                    continue
                c = contacto(e, m)
                if c is not None:
                    puntos.append(c[0])
                    areas.append(c[1])
                    destino_de += [m] * len(c[0])
        nombres = ", ".join(i for _, i in piezas[:6]) + (f" y {len(piezas) - 6} más" if len(piezas) > 6 else "")
        if not puntos:
            resultado.sueltos.append(piezas)
            resultado.avisos.append(f"Los elementos «solo carga» {nombres} no tocan ninguna pieza mallada: su peso y sus "
                                    "cargas no se aplican. Mállelos (quite «solo carga») o únalos a la estructura.")
            continue
        puntos = np.vstack(puntos)
        areas = np.concatenate(areas)
        destino_de = np.array([f"{t}\x00{i}" for t, i in destino_de])
        destinos = sorted(set(destino_de))
        # cargas del grupo en cada caso: fuerzas puntuales y momentos puros
        volumenes = {e: _volumen_y_centro(sups[e]) for e in piezas}
        referencia = (sum(v * c for v, c in volumenes.values()) / max(sum(v for v, _ in volumenes.values()), 1e-30)
                      if sum(v for v, _ in volumenes.values()) > 0 else puntos.mean(axis=0))
        grupo = GrupoSoloCarga(piezas, [tuple(d.split("\x00")) for d in destinos])
        transferencias = {d: Transferencia(tuple(d.split("\x00")), puntos[destino_de == d]) for d in destinos}
        for caso_id in casos:
            fuerza, momento = np.zeros(3), np.zeros(3)
            if caso_id in con_peso:
                for e in piezas:
                    material = proyecto.materiales.get(getattr(getattr(proyecto, COLECCIONES[e[0]])[e[1]], "material", ""))
                    if material is None:
                        continue
                    v, centro = volumenes[e]
                    peso = np.array([0.0, 0.0, -material.densidad * v])
                    fuerza += peso
                    momento += np.cross(centro - referencia, peso)
            for c in cargas_superficie_por_caso.get(caso_id, []):
                if (c.elemento_tipo, c.elemento_id) not in piezas:
                    continue
                if c.tipo == "MOMENTO":
                    if c.direccion is not None:
                        momento += c.valor * np.asarray(c.direccion, float)
                    continue
                f, p = _carga_de_superficie(c, sups[(c.elemento_tipo, c.elemento_id)])
                fuerza += f.sum(axis=0)
                momento += np.cross(p - referencia, f).sum(axis=0)
            grupo.resultantes[caso_id] = fuerza
            if np.linalg.norm(fuerza) + np.linalg.norm(momento) < 1e-12:
                grupo.desequilibrio[caso_id] = 0.0
                g = np.zeros_like(puntos)
            else:
                g, falta = repartir(puntos, areas, fuerza, momento, referencia)
                grupo.desequilibrio[caso_id] = falta
            for d, t in transferencias.items():
                t.fuerzas[caso_id] = -g[destino_de == d]  # la pieza mallada recibe lo contrario de lo que da
        peor = max(grupo.desequilibrio.values(), default=0.0)
        if peor > 1e-3:
            resultado.avisos.append(f"Los elementos «solo carga» {nombres} no quedan en equilibrio con sus contactos (falta "
                                    f"el {peor * 100:.0f} % de la carga): apoyan en una línea o en un punto y la carga los "
                                    "haría girar. Mállelos o revise cómo apoyan.")
        resultado.grupos.append(grupo)
        resultado.transferencias += list(transferencias.values())
    return resultado


def aplicar_a_la_malla(malla, transferencias: Sequence[Transferencia], caso_id: str, cargas) -> int:
    """Suma a ``cargas`` (``CargasCaso``) las fuerzas del caso en los nodos de superficie de cada pieza mallada de
    destino (el más cercano a cada punto de contacto). Devuelve cuántas transferencias no encontraron su pieza."""

    from scipy.spatial import cKDTree

    perdidas = 0
    for t in transferencias:
        f = t.fuerzas.get(caso_id)
        if f is None or not len(f) or not np.abs(f).any():
            continue
        tipo, id_ = t.destino
        partes = [p for p in malla.partes
                  if p.id == id_ or (p.miembro is not None and id_ in getattr(p.miembro, "barras", ()))]
        nodos = []
        for p in partes:
            nodos.append(malla.nodos_de_parte(p) if p.lamina else np.unique(malla.fronteras(p).nodos))
        if not nodos:
            perdidas += 1
            continue
        nodos = np.unique(np.concatenate(nodos))
        _, cercano = cKDTree(malla.coordenadas[nodos]).query(t.puntos)
        cargas.sumar(nodos[cercano], f)
    return perdidas
