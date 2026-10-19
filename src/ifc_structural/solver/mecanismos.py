"""Detección de mecanismos (modos sin rigidez) con un modelo de pórtico propio.

CalculiX no avisa cuando una estructura es inestable: devuelve esfuerzos
absurdos (o falla) sin un mensaje claro. Las articulaciones y los enlaces
(``Conexion``) pueden dejar mecanismos internos que la comprobación de
cuerpo rígido de ``analisis.validar_modelo`` no ve (p. ej. un pórtico con
todas las uniones articuladas). Aquí se arma la matriz de rigidez de un
pórtico espacial de barras de Euler-Bernoulli (12 gdl por barra), se aplican
apoyos y conexiones y se cuentan los valores propios nulos de la matriz
reducida. Solo cuenta si existe rigidez, no su valor numérico, por lo que las
propiedades de la sección son aproximadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import null_space

from ..core.asignacion import barras_de_apoyo_area
from ..core.cargas import ApoyoArea, Conexion
from ..core.geometria import ejes_seccion
from ..core.modelo import Proyecto
from .generador_inp import eje_global_de_barra

#: Por encima de este número de nudos no se comprueba (matrices densas).
MAXIMO_DE_NUDOS = 600
#: Una barra con apoyo continuo se divide en tantos tramos para el chequeo.
TRAMOS_APOYO_CONTINUO = 4
#: Valor propio relativo por debajo del cual se considera "sin rigidez".
TOLERANCIA_RELATIVA = 1e-9


@dataclass
class ResultadoMecanismos:
    cantidad: int
    barras: List[str] = field(default_factory=list)


def _rigidez_local(E: float, G: float, A: float, Iy: float, Iz: float, J: float, L: float) -> np.ndarray:
    """Matriz de rigidez 12×12 de una barra de Euler-Bernoulli (ejes locales:
    x a lo largo de la barra; gdl: u,v,w,θx,θy,θz en cada extremo)."""

    k = np.zeros((12, 12))
    a, t = E * A / L, G * J / L

    def anadir(i, j, v):
        k[i, j] += v
        if i != j:
            k[j, i] += v

    anadir(0, 0, a); anadir(6, 6, a); anadir(0, 6, -a)
    anadir(3, 3, t); anadir(9, 9, t); anadir(3, 9, -t)
    # Flexión en el plano x-y (v, θz) con Iz
    c1, c2, c3, c4 = 12 * E * Iz / L**3, 6 * E * Iz / L**2, 4 * E * Iz / L, 2 * E * Iz / L
    anadir(1, 1, c1); anadir(7, 7, c1); anadir(1, 7, -c1)
    anadir(1, 5, c2); anadir(1, 11, c2); anadir(5, 7, -c2); anadir(7, 11, -c2)
    anadir(5, 5, c3); anadir(11, 11, c3); anadir(5, 11, c4)
    # Flexión en el plano x-z (w, θy) con Iy
    d1, d2, d3, d4 = 12 * E * Iy / L**3, 6 * E * Iy / L**2, 4 * E * Iy / L, 2 * E * Iy / L
    anadir(2, 2, d1); anadir(8, 8, d1); anadir(2, 8, -d1)
    anadir(2, 4, -d2); anadir(2, 10, -d2); anadir(4, 8, d2); anadir(8, 10, d2)
    anadir(4, 4, d3); anadir(10, 10, d3); anadir(4, 10, d4)
    return k


def detectar_mecanismos(
    proyecto: Proyecto, apoyos: Sequence, conexiones: Sequence[Conexion]
) -> Optional[ResultadoMecanismos]:
    """Cuenta los mecanismos (modos de rigidez nula) del modelo con sus
    apoyos y conexiones. Devuelve ``None`` si el modelo es demasiado grande
    para comprobarlo."""

    barras = proyecto.todas_las_barras()
    if len(proyecto.nodos) > MAXIMO_DE_NUDOS:
        return None

    # -- Nudos ------------------------------------------------------------
    indice: Dict[tuple, int] = {}
    coordenadas: List[np.ndarray] = []

    def nudo(clave: tuple, punto) -> int:
        if clave not in indice:
            indice[clave] = len(coordenadas)
            coordenadas.append(np.array(punto, dtype=float))
        return indice[clave]

    separados = {(c.barra_id, c.nodo_id) for c in conexiones if c.barra_id is not None}
    continuas = set()
    for apoyo in apoyos:
        if isinstance(apoyo, ApoyoArea):
            continuas.update(barras_de_apoyo_area(proyecto, apoyo))

    elementos: List[Tuple[str, int, int, float]] = []  # (barra, i, f, longitud)
    nudos_de_barra: Dict[str, List[int]] = {}
    tramos: Dict[str, List[Tuple[int, float]]] = {}  # barra -> [(nudo, s)]
    for barra_id, barra in barras.items():
        a, b = proyecto.nodos[barra.nodo_inicio], proyecto.nodos[barra.nodo_fin]
        pa, pb = np.array(a.coordenadas), np.array(b.coordenadas)
        n = TRAMOS_APOYO_CONTINUO if barra_id in continuas else 1
        ids: List[int] = []
        for k in range(n + 1):
            s = k / n
            if k == 0:
                clave = ("x", barra_id, barra.nodo_inicio) if (barra_id, barra.nodo_inicio) in separados else ("n", barra.nodo_inicio)
            elif k == n:
                clave = ("x", barra_id, barra.nodo_fin) if (barra_id, barra.nodo_fin) in separados else ("n", barra.nodo_fin)
            else:
                clave = ("m", barra_id, k)
            ids.append(nudo(clave, pa + s * (pb - pa)))
        nudos_de_barra[barra_id] = ids
        tramos[barra_id] = [(ids[k], k / n) for k in range(n + 1)]
        for i, j in zip(ids, ids[1:]):
            elementos.append((barra_id, i, j, float(np.linalg.norm(pb - pa)) / n))

    n_nudos = len(coordenadas)
    ndof = 6 * n_nudos
    K = np.zeros((ndof, ndof))
    for barra_id, i, j, L in elementos:
        barra = barras[barra_id]
        material = proyecto.materiales[barra.material]
        ejes = ejes_seccion(proyecto, barra)
        A = ejes.dim1 * ejes.dim2
        Iz = ejes.dim2 * ejes.dim1**3 / 12.0
        Iy = ejes.dim1 * ejes.dim2**3 / 12.0
        J = A * (ejes.dim1**2 + ejes.dim2**2) / 12.0
        kl = _rigidez_local(material.modulo_elasticidad, material.modulo_corte, A, Iy, Iz, J, L)
        R = np.array([ejes.t, ejes.e1, ejes.e2], dtype=float)  # filas = ejes locales en coordenadas globales
        T = np.kron(np.eye(4), R)
        kg = T.T @ kl @ T
        gdl = np.r_[6 * i : 6 * i + 6, 6 * j : 6 * j + 6]
        K[np.ix_(gdl, gdl)] += kg

    # -- Apoyos -------------------------------------------------------------
    fijos = np.zeros(ndof, dtype=bool)
    for apoyo in apoyos:
        if isinstance(apoyo, ApoyoArea):
            for barra_id in barras_de_apoyo_area(proyecto, apoyo):
                for n_i, s in tramos.get(barra_id, []):
                    if apoyo.s_inicio - 1e-6 <= s <= apoyo.s_fin + 1e-6:
                        for g in apoyo.grados_restringidos():
                            fijos[6 * n_i + g - 1] = True
        elif ("n", apoyo.nodo_id) in indice:
            for g in apoyo.grados_restringidos():
                fijos[6 * indice[("n", apoyo.nodo_id)] + g - 1] = True

    # -- Conexiones (ecuaciones) ---------------------------------------------
    ecuaciones: List[List[Tuple[int, float]]] = []
    for c in conexiones:
        if c.barra_id is not None:
            dep, prin = indice.get(("x", c.barra_id, c.nodo_id)), indice.get(("n", c.nodo_id))
            if dep is None or prin is None:
                continue
            for g in c.traslaciones_transmitidas():
                ecuaciones.append([(6 * dep + g - 1, 1.0), (6 * prin + g - 1, -1.0)])
            eje = eje_global_de_barra(proyecto, c.barra_id)
            if c.fijar_torsion and not c.transmite_giros and eje is not None:
                fijos[6 * dep + 3 + eje] = True
            continue
        o, d = indice.get(("n", c.nodo_id)), indice.get(("n", c.nodo_destino))
        if o is None or d is None:
            continue
        r = coordenadas[d] - coordenadas[o]
        if c.es_enlace_rigido:  # sólido rígido: u_d = u_o + θ_o × r ; θ_d = θ_o
            brazos = {1: [(4, -r[2]), (5, r[1])], 2: [(5, -r[0]), (3, r[2])], 3: [(3, -r[1]), (4, r[0])]}
            for g in range(1, 4):
                terminos = [(6 * d + g - 1, 1.0), (6 * o + g - 1, -1.0)]
                terminos.extend((6 * o + gdl, coef) for gdl, coef in brazos[g] if abs(coef) > 1e-12)
                ecuaciones.append(terminos)
            for g in range(4, 7):
                ecuaciones.append([(6 * d + g - 1, 1.0), (6 * o + g - 1, -1.0)])
        else:
            for g in c.traslaciones_transmitidas():
                ecuaciones.append([(6 * d + g - 1, 1.0), (6 * o + g - 1, -1.0)])

    libres = np.flatnonzero(~fijos)
    if libres.size == 0:
        return ResultadoMecanismos(0)
    posicion = {int(g): k for k, g in enumerate(libres)}
    if ecuaciones:
        C = np.zeros((len(ecuaciones), libres.size))
        for fila, terminos in enumerate(ecuaciones):
            for g, coef in terminos:
                if g in posicion:
                    C[fila, posicion[g]] += coef
        C = C[np.abs(C).sum(axis=1) > 0]
        Z = null_space(C) if C.size else np.eye(libres.size)
    else:
        Z = np.eye(libres.size)
    if Z.shape[1] == 0:
        return ResultadoMecanismos(0)

    Kr = Z.T @ K[np.ix_(libres, libres)] @ Z
    Kr = (Kr + Kr.T) / 2.0
    valores, vectores = np.linalg.eigh(Kr)
    umbral = TOLERANCIA_RELATIVA * max(float(valores.max()), 1.0)
    nulos = np.flatnonzero(valores < umbral)
    if nulos.size == 0:
        return ResultadoMecanismos(0)

    # Barras que más se mueven en los modos nulos
    afectadas: Dict[str, float] = {}
    for k in nulos:
        u = np.zeros(ndof)
        u[libres] = Z @ vectores[:, k]
        traslacion = np.linalg.norm(u.reshape(-1, 6)[:, :3], axis=1)
        giro = np.linalg.norm(u.reshape(-1, 6)[:, 3:], axis=1)
        amplitud = traslacion + giro
        pico = float(amplitud.max()) or 1.0
        for barra_id, ids in nudos_de_barra.items():
            valor = float(amplitud[ids].max()) / pico
            if valor > 0.3:
                afectadas[barra_id] = max(afectadas.get(barra_id, 0.0), valor)
    nombres = [b for b, _ in sorted(afectadas.items(), key=lambda kv: -kv[1])][:8]
    return ResultadoMecanismos(int(nulos.size), nombres)
