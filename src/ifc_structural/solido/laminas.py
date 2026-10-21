"""Láminas (cáscaras ``S6`` de CalculiX) para sólidos de pared delgada.

Un sólido cuyo espesor es mucho menor que su tamaño (el cuerpo de un depósito, una tolva, una chapa curva) no se puede
representar bien con tetraedros ni con hexaedros: haría falta un elemento por cada milímetro de pared. Se modela como
lámina: se obtiene su **superficie media** a partir de la superficie triangulada del sólido (cada triángulo de una piel se
lanza un rayo hacia dentro hasta la piel opuesta; la superficie media está a medio camino y el espesor es la distancia),
Gmsh la vuelve a mallar con triángulos casi equiláteros del tamaño de elemento (en el mismo proceso hijo que los
tetraedros; si no puede, se refina por bisección de aristas) y se convierte a triángulos cuadráticos de 6 nodos
(``S6``) con el espesor de cada elemento. No necesita que la superficie sea una variedad cerrada perfecta: sirve también
para las piezas con aristas compartidas por cuatro triángulos que Gmsh no puede rellenar con tetraedros.

Orden de nodos de ``S6``: vértices 1-3 y medios 4-6 de las aristas 1-2, 2-3 y 3-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..core.solidos import SolidoGeneral, normales_y_areas

#: Máximo de triángulos de una lámina tras el refinamiento.
MAXIMO_TRIANGULOS = 120_000
#: Fracción del tamaño de elemento por debajo de la cual el espesor medio hace que un sólido sea una pared delgada.
RELACION_PARED_DELGADA = 0.4
#: Una pieza es lámina si su espesor medio es como mucho esta fracción de su tamaño en planta (placa o cáscara delgada).
RELACION_LAMINA = 1.0 / 20.0
#: La bisección de aristas deja elementos de entre 0,5 y 1 vez el límite: con este factor el lado medio es ~ el tamaño de elemento.
FACTOR_REFINAMIENTO = 2.2


class ErrorLamina(RuntimeError):
    """No se pudo obtener la superficie media del sólido."""


class NoEsLamina(ErrorLamina):
    """La pieza es maciza (una barra, un bloque): se malla con tetraedros, sin avisar."""


@dataclass
class SuperficieMedia:
    """Superficie media de un sólido de pared delgada, antes de mallarla con el tamaño de elemento."""

    vertices: np.ndarray  # (n, 3)
    triangulos: np.ndarray  # (m, 3)
    origen: np.ndarray  # (m,) triángulo de la piel conservada del que sale cada triángulo
    espesores: np.ndarray  # (k,) espesor de cada triángulo de la piel conservada
    normales: np.ndarray  # (k, 3) normal hacia fuera de cada triángulo de la piel conservada


@dataclass
class MallaLamina:
    coordenadas: np.ndarray  # (N, 3) vértices y nodos medios
    elementos: np.ndarray  # (M, 6) orden de CalculiX
    espesores: np.ndarray  # (M,) m
    normales: np.ndarray  # (M, 3) unitarias, hacia fuera de la piel conservada
    superficies: Dict[int, np.ndarray] = field(default_factory=dict)  # superficie -> (k, 2) (elemento local, lado 1 | 2)
    nodos_superficie: Dict[int, np.ndarray] = field(default_factory=dict)  # superficie -> nodos que están sobre ella
    tamano: float = 0.0
    volumen: float = 0.0  # suma de área × espesor


def espesor_medio(solido: SolidoGeneral) -> float:
    """Espesor medio si el sólido es una pared delgada: 2 V / A."""

    area = sum(c.area for c in solido.caras)
    return 2.0 * float(solido.volumen) / area if area > 0 else float("inf")


def dimension_en_planta(solido: SolidoGeneral) -> float:
    """Lado de la placa cuadrada equivalente: √(A/2), con A el área de toda la superficie (las dos caras)."""

    return float(np.sqrt(max(sum(c.area for c in solido.caras), 0.0) / 2.0))


def es_lamina(solido: SolidoGeneral) -> bool:
    """¿Se modela como lámina (cáscara S6)? Sí si su espesor medio ``t = 2V/A`` es como mucho 1/20 de su tamaño en planta
    ``L = √(A/2)`` (el criterio habitual de placa o cáscara delgada). No depende del tamaño de elemento."""

    t, lado = espesor_medio(solido), dimension_en_planta(solido)
    return bool(np.isfinite(t) and lado > 0 and t <= RELACION_LAMINA * lado)


def es_pared_delgada(solido: SolidoGeneral, tamano: float) -> bool:
    return espesor_medio(solido) <= max(0.015, RELACION_PARED_DELGADA * tamano)


def _vecinos_por_arista(triangulos: np.ndarray, validos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pares de triángulos válidos que comparten una arista usada por exactamente dos triángulos (las aristas donde
    se tocan dos cascarones, con cuatro triángulos, no unen pieles distintas)."""

    m = len(triangulos)
    aristas = np.sort(np.concatenate([triangulos[:, [0, 1]], triangulos[:, [1, 2]], triangulos[:, [2, 0]]]), axis=1)
    propietario = np.tile(np.arange(m), 3)
    unicas, inverso, cuenta = np.unique(aristas, axis=0, return_inverse=True, return_counts=True)
    inverso = inverso.ravel()
    orden = np.argsort(inverso, kind="stable")
    ordenado = propietario[orden]
    inicio = np.concatenate([[0], np.cumsum(cuenta)[:-1]])
    dobles = np.where(cuenta == 2)[0]
    a, b = ordenado[inicio[dobles]], ordenado[inicio[dobles] + 1]
    bien = validos[a] & validos[b]
    return a[bien], b[bien]


def _piel_conservada(solido: SolidoGeneral) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triángulos de una de las dos pieles y su espesor.

    Devuelve (índices de los triángulos conservados, espesor de cada uno, triángulo opuesto de cada triángulo del sólido
    (-1 si no tiene), espesor de cada triángulo del sólido)."""

    import pyvista as pv

    v, t = solido.vertices, solido.triangulos
    n, area = normales_y_areas(v, t)
    centro = v[t].mean(axis=1)
    medio = espesor_medio(solido)
    if not np.isfinite(medio) or medio <= 0:
        raise ErrorLamina("el sólido no tiene espesor")
    limite = 3.0 * medio
    malla = pv.PolyData(v, np.hstack([np.full((len(t), 1), 3), t]).ravel().astype(np.int64))
    opuesto = np.full(len(t), -1, dtype=np.int64)
    espesor = np.zeros(len(t))
    eps = 1e-4 * medio
    for i in range(len(t)):
        puntos, celdas = malla.ray_trace(centro[i] - n[i] * eps, centro[i] - n[i] * limite)
        if len(celdas) == 0:
            continue
        d = np.linalg.norm(np.asarray(puntos) - centro[i], axis=1)
        k = int(np.argmin(d))
        j = int(celdas[k])
        if j != i and float(np.dot(n[i], n[j])) < -0.3:
            opuesto[i], espesor[i] = j, d[k]
    validos = opuesto >= 0
    if not validos.any():
        raise ErrorLamina("no se encontró la pared opuesta de ningún triángulo")

    # Pieles: componentes conexas de triángulos válidos. Dos vecinos con normales opuestas (el filo donde se juntan las dos
    # pieles de un canto afilado) no se unen: si no, las dos pieles forman una sola componente y se conservan las dos.
    a, b = _vecinos_por_arista(t, validos)
    une = np.einsum("ij,ij->i", n[a], n[b]) > -0.3
    a, b = a[une], b[une]
    grafo = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(t), len(t)))
    _, etiqueta = connected_components(grafo, directed=False)
    etiqueta = np.where(validos, etiqueta, -1)
    # De mayor a menor área: se conserva cada piel y se descartan TODAS las que tiene enfrente (en los dos sentidos del
    # rayo), no solo la más frecuente; así una piel partida en varias componentes no se queda con las dos caras.
    i_val = np.where(validos)[0]
    ca, cb = etiqueta[i_val], etiqueta[opuesto[i_val]]
    distinta = (cb >= 0) & (ca != cb)
    enfrente: Dict[int, set] = {}
    for x, y in zip(ca[distinta].tolist(), cb[distinta].tolist()):
        enfrente.setdefault(x, set()).add(y)
        enfrente.setdefault(y, set()).add(x)
    area_piel = np.bincount(etiqueta[i_val], weights=area[i_val])
    conservada: List[int] = []
    descartadas = set()
    for c in sorted(set(ca.tolist()), key=lambda c: -area_piel[c]):
        if c in descartadas:
            continue
        conservada.append(c)
        descartadas.update(d for d in enfrente.get(c, ()) if d not in conservada)
    mantener = np.isin(etiqueta, conservada) & validos
    # si aún queda algún triángulo enfrentado a otro conservado, se quita uno de los dos
    mantener &= ~(mantener & (opuesto >= 0) & mantener[np.maximum(opuesto, 0)] & (opuesto < np.arange(len(t))))
    return np.where(mantener)[0], espesor[mantener], opuesto, espesor


def _aristas_de_rasgo(v: np.ndarray, t: np.ndarray, angulo: float = 30.0) -> np.ndarray:
    """Número de aristas de rasgo (borde, arista con más de dos triángulos o pliegue de más de ``angulo``) en cada vértice."""

    n, _ = normales_y_areas(v, t)
    aristas = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    propietario = np.tile(np.arange(len(t)), 3)
    unicas, inverso, cuenta = np.unique(aristas, axis=0, return_inverse=True, return_counts=True)
    orden = np.argsort(inverso.ravel(), kind="stable")
    inicio = np.concatenate([[0], np.cumsum(cuenta)[:-1]])
    rasgo = cuenta != 2
    dobles = np.where(cuenta == 2)[0]
    a, b = propietario[orden][inicio[dobles]], propietario[orden][inicio[dobles] + 1]
    rasgo[dobles] = np.einsum("ij,ij->i", n[a], n[b]) < np.cos(np.radians(angulo))
    peso = np.zeros(len(v))
    np.add.at(peso, unicas[rasgo, 0], 1.0)
    np.add.at(peso, unicas[rasgo, 1], 1.0)
    return peso


def _soldar(v: np.ndarray, t: np.ndarray, atributo: np.ndarray, tolerancia: float):
    """Une los vértices a menos de ``tolerancia`` (las teselas diminutas de esquinas redondeadas dan elementos más
    estrechos que su espesor) y descarta los triángulos que quedan degenerados o repetidos.

    Sin encadenar: cada vértice «líder» (primero los de esquinas y bordes) absorbe a los vértices libres a menos de
    ``tolerancia`` y el grupo queda en la posición del líder, así que ningún vértice se mueve más de ``tolerancia``. (Con
    componentes conexas de los pares cercanos, una fila de vértices juntos, como el borde de un anillo o las generatrices
    de un cono, se fundía entera en un punto y la lámina quedaba vacía o deformada.)"""

    if tolerancia <= 0 or len(v) == 0:
        return v, t, atributo
    peso = _aristas_de_rasgo(v, t)
    vecinos = cKDTree(v).query_ball_point(v, tolerancia)
    grupo = np.full(len(v), -1, dtype=np.int64)
    for i in np.lexsort((np.arange(len(v)), -peso)):
        if grupo[i] < 0:
            for j in vecinos[i]:
                if grupo[j] < 0:
                    grupo[j] = i
    nt = grupo[t]
    bien = (nt[:, 0] != nt[:, 1]) & (nt[:, 1] != nt[:, 2]) & (nt[:, 0] != nt[:, 2])
    usados, inverso = np.unique(nt[bien], return_inverse=True)
    t2 = inverso.reshape(-1, 3)
    _, unico = np.unique(np.sort(t2, axis=1), axis=0, return_index=True)
    unico = np.sort(unico)
    return v[usados], t2[unico], atributo[bien][unico]


def _refinar(v: np.ndarray, t: np.ndarray, atributos: np.ndarray, h: float):
    """Bisección conforme de aristas más largas que ``h``. ``atributos`` (M,) se hereda en los triángulos hijos."""

    v = v.copy()
    for _ in range(30):
        arista = np.stack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=1)  # (M, 3, 2)
        largo = np.linalg.norm(v[arista[..., 0]] - v[arista[..., 1]], axis=2)
        cortar = largo > h
        if not cortar.any():
            break
        claves = np.sort(arista, axis=2)
        todas = claves[cortar]
        unicas, inverso = np.unique(todas, axis=0, return_inverse=True)
        medios = len(v) + np.arange(len(unicas))
        v = np.vstack([v, 0.5 * (v[unicas[:, 0]] + v[unicas[:, 1]])])
        indice = np.full(cortar.shape, -1, dtype=np.int64)
        indice[cortar] = medios[inverso.ravel()]
        mascara = cortar[:, 0] * 1 + cortar[:, 1] * 2 + cortar[:, 2] * 4
        nuevos, hereda = [], []
        for m in range(8):
            sel = np.where(mascara == m)[0]
            if len(sel) == 0:
                continue
            a, b, c = t[sel, 0], t[sel, 1], t[sel, 2]
            m01, m12, m20 = indice[sel, 0], indice[sel, 1], indice[sel, 2]
            if m == 0:
                partes = [(a, b, c)]
            elif m == 1:
                partes = [(a, m01, c), (m01, b, c)]
            elif m == 2:
                partes = [(b, m12, a), (m12, c, a)]
            elif m == 4:
                partes = [(c, m20, b), (m20, a, b)]
            elif m == 3:
                partes = [(m01, b, m12), (a, m01, m12), (a, m12, c)]
            elif m == 6:
                partes = [(m12, c, m20), (b, m12, m20), (b, m20, a)]
            elif m == 5:
                partes = [(m20, a, m01), (m01, b, c), (m01, c, m20)]
            else:
                partes = [(a, m01, m20), (m01, b, m12), (m20, m12, c), (m01, m12, m20)]
            for p in partes:
                nuevos.append(np.stack(p, axis=1))
                hereda.append(atributos[sel])
        t = np.vstack(nuevos)
        atributos = np.concatenate(hereda)
        if len(t) > MAXIMO_TRIANGULOS:
            raise ErrorLamina(f"la lámina tendría más de {MAXIMO_TRIANGULOS:,} elementos".replace(",", " "))
    return v, t, atributos


def _orientacion_coherente(tm: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Elementos que hay que voltear para que dos vecinos recorran su arista común en sentidos opuestos (normales
    coherentes: CalculiX expande cada S6 según la normal media del nodo). En cada parte conexa se voltea la menor área."""

    from scipy.sparse.csgraph import breadth_first_order

    m = len(tm)
    dirigidas = np.concatenate([tm[:, [0, 1]], tm[:, [1, 2]], tm[:, [2, 0]]])
    propietario = np.tile(np.arange(m), 3)
    _, inverso, cuenta = np.unique(np.sort(dirigidas, axis=1), axis=0, return_inverse=True, return_counts=True)
    orden = np.argsort(inverso.ravel(), kind="stable")
    inicio = np.concatenate([[0], np.cumsum(cuenta)[:-1]])
    dobles = np.where(cuenta == 2)[0]
    k1, k2 = orden[inicio[dobles]], orden[inicio[dobles] + 1]
    a, b = propietario[k1], propietario[k2]
    paridad = dirigidas[k1, 0] == dirigidas[k2, 0]  # recorren la arista en el mismo sentido: uno de los dos va volteado
    grafo = coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m)).tocsr()
    grafo = grafo + grafo.T
    par = {}
    for x, y, p in zip(a.tolist(), b.tolist(), paridad.tolist()):
        par[(x, y)] = par[(y, x)] = p
    voltear = np.zeros(m, dtype=bool)
    visto = np.zeros(m, dtype=bool)
    for semilla in np.argsort(-area):
        if visto[semilla]:
            continue
        nodos, previo = breadth_first_order(grafo, int(semilla), directed=False, return_predecessors=True)
        for k in nodos[1:]:
            voltear[k] = voltear[previo[k]] ^ par[(int(previo[k]), int(k))]
        visto[nodos] = True
        if area[nodos][voltear[nodos]].sum() > 0.5 * area[nodos].sum():
            voltear[nodos] = ~voltear[nodos]
    return voltear


def superficie_media(solido: SolidoGeneral, tamano: float) -> SuperficieMedia:
    """Superficie media del sólido (todavía con los triángulos de su piel). Lanza ``ErrorLamina`` si no se puede obtener
    y ``NoEsLamina`` si la pieza es maciza."""

    v, t = solido.vertices, solido.triangulos
    if not (np.isfinite(v).all() and len(t)):
        raise ErrorLamina("coordenadas no válidas")
    n, area_tri = normales_y_areas(v, t)
    conservados, espesores, opuesto, espesor_tri = _piel_conservada(solido)

    # cada vértice de la piel conservada se desplaza la mitad del espesor hacia dentro
    tc = t[conservados]
    usados, local = np.unique(tc, return_inverse=True)
    local = local.reshape(-1, 3)
    desplazamiento = np.zeros((len(usados), 3))
    cuenta = np.zeros(len(usados))
    for k in range(3):
        np.add.at(desplazamiento, local[:, k], -0.5 * espesores[:, None] * n[conservados])
        np.add.at(cuenta, local[:, k], 1.0)
    vm = v[usados] + desplazamiento / cuenta[:, None]
    # Una barra o un bloque macizo no es una pared: en una pared el espesor que miden los rayos es ~2V/A (una chapa de
    # lados a, b: 1 + t/a + t/b), en una barra cuadrada o redonda el doble y en un cubo el triple; además su «superficie
    # media» se encoge hasta el eje.
    validos = opuesto >= 0
    orden = np.argsort(espesor_tri[validos])
    acumulada = np.cumsum(area_tri[validos][orden])
    espesor_rayos = float(espesor_tri[validos][orden][np.searchsorted(acumulada, 0.5 * acumulada[-1])])
    _, area_media = normales_y_areas(vm, local)
    if espesor_rayos > 1.75 * espesor_medio(solido) or area_media.sum() < 0.5 * area_tri[conservados].sum():
        raise NoEsLamina(f"la pieza es maciza (espesor {1000 * espesor_rayos:.0f} mm), no una pared delgada")
    # se unen los vértices más cercanos que una fracción del espesor (y, con elementos grandes, algo más, pero nunca
    # más que 1,5 espesores: con 0,12 veces un tamaño de 25 cm un anillo de 10 mm se fundía entero)
    e_med = float(np.median(espesores))
    tolerancia = max(0.4 * e_med, min(0.12 * FACTOR_REFINAMIENTO * tamano, 1.5 * e_med))
    vm, local, origen = _soldar(vm, local, np.arange(len(conservados)), tolerancia)
    if len(local) == 0:
        raise ErrorLamina("la superficie media quedó vacía")
    return SuperficieMedia(vertices=vm, triangulos=local, origen=origen, espesores=espesores, normales=n[conservados])


def _partes_conexas(triangulos: np.ndarray) -> int:
    """Número de trozos de una superficie triangulada (unidos por vértices compartidos)."""

    t = np.asarray(triangulos, dtype=np.int64)
    if len(t) == 0:
        return 0
    n = int(t.max()) + 1
    aristas = np.concatenate([t[:, [0, 1]], t[:, [1, 2]]])
    _, etiqueta = connected_components(coo_matrix((np.ones(len(aristas)), (aristas[:, 0], aristas[:, 1])), shape=(n, n)),
                                       directed=False)
    return len(np.unique(etiqueta[np.unique(t)]))


def remallado_valido(media: SuperficieMedia, vertices: np.ndarray, triangulos: np.ndarray) -> np.ndarray:
    """Triángulo de la superficie media más cercano a cada triángulo de la superficie vuelta a mallar, o ``None`` si
    el remallado no sirve (triángulos degenerados, área distinta, triángulos girados respecto de la de partida, o la
    superficie partida en más trozos: Gmsh a veces separa dos parches que solo se tocan por una arista, y el trozo que
    no se acopla a nada quedaría flotando)."""

    import pyvista as pv

    v, t = np.asarray(vertices, dtype=float), np.asarray(triangulos, dtype=np.int64)
    if len(t) == 0 or not np.isfinite(v).all():
        return None
    n1, a1 = normales_y_areas(v, t)
    p = v[t]
    l2 = sum(((p[:, (k + 1) % 3] - p[:, k]) ** 2).sum(axis=1) for k in range(3))
    n0, a0 = normales_y_areas(media.vertices, media.triangulos)
    if float((4 * np.sqrt(3) * a1 / np.maximum(l2, 1e-30)).min()) < 0.05 or abs(a1.sum() / a0.sum() - 1.0) > 0.03:
        return None
    malla = pv.PolyData(media.vertices, np.hstack([np.full((len(media.triangulos), 1), 3), media.triangulos]).ravel()
                        .astype(np.int64))
    cercano = np.asarray(malla.find_closest_cell(p.mean(axis=1)), dtype=np.int64)
    if (cercano < 0).any() or (np.abs(np.einsum("ij,ij->i", n1, n0[cercano])) < 0.5).any():
        return None
    if _partes_conexas(t) > _partes_conexas(media.triangulos):
        return None
    return cercano


def lamina_de_superficie(solido: SolidoGeneral, media: SuperficieMedia, tamano: float,
                         remallada: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> MallaLamina:
    """Lámina ``S6`` a partir de la superficie media: con la superficie vuelta a mallar por Gmsh (``remallada`` =
    (vértices, triángulos)) si es válida y, si no, refinando la de partida por bisección de aristas."""

    v, t = solido.vertices, solido.triangulos
    n, _ = normales_y_areas(v, t)
    vm = tm = None
    if remallada is not None:
        cercano = remallado_valido(media, *remallada)
        if cercano is not None:
            vm, tm, origen = np.asarray(remallada[0], dtype=float), np.asarray(remallada[1], dtype=np.int64), media.origen[cercano]
    if tm is None:
        vm, tm, origen = _refinar(media.vertices, media.triangulos, media.origen, FACTOR_REFINAMIENTO * tamano)
    espesores = media.espesores

    e = espesores[origen]
    p = vm[tm]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    keep = area > 1e-12
    tm, origen, e, normal, area = tm[keep], origen[keep], e[keep], normal[keep], area[keep]
    if len(tm) == 0:
        raise ErrorLamina("la superficie media quedó vacía")
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    # la normal de cada elemento sigue la de la piel conservada (hacia fuera)...
    voltea = np.einsum("ij,ij->i", normal, media.normales[origen]) < 0
    tm[voltea] = tm[voltea][:, [0, 2, 1]]
    normal[voltea] *= -1.0
    # ...y es coherente entre vecinos (si se conservaron pieles de lados distintos, p. ej. la exterior del cilindro y la
    # interior del cono, las normales quedarían opuestas en la unión)
    voltea = _orientacion_coherente(tm, area)
    tm[voltea] = tm[voltea][:, [0, 2, 1]]
    normal[voltea] *= -1.0

    # nodos medios de las aristas
    arista = np.concatenate([tm[:, [0, 1]], tm[:, [1, 2]], tm[:, [2, 0]]])
    claves = np.sort(arista, axis=1)
    unicas, inverso = np.unique(claves, axis=0, return_inverse=True)
    inverso = inverso.reshape(3, -1).T
    coordenadas = np.vstack([vm, 0.5 * (vm[unicas[:, 0]] + vm[unicas[:, 1]])])
    elementos = np.hstack([tm, len(vm) + inverso]).astype(np.int64)

    # superficies del sólido: cada triángulo original se asocia al elemento más cercano (y a su lado)
    centro_e = vm[tm].mean(axis=1)
    arbol = cKDTree(centro_e)
    centro_o = v[t].mean(axis=1)
    distancia, cercano = arbol.query(centro_o)
    alcance = max(2.0 * tamano, 4.0 * float(espesores.max()))
    lado = np.where(np.einsum("ij,ij->i", n, normal[cercano]) >= 0, 1, 2)
    superficies: Dict[int, np.ndarray] = {}
    cara_de = solido.cara_de_triangulo
    valido = distancia <= alcance
    for s in np.unique(cara_de[valido]):
        marca = valido & (cara_de == s)
        pares = np.unique(np.stack([cercano[marca], lado[marca]], axis=1), axis=0)
        superficies[int(s)] = pares
    # nodos sobre cada superficie del sólido (en un canto solo los del borde; en una piel, todos los de la lámina bajo ella)
    from ..core.solidos import muestras_de_superficie

    espaciado = max(0.5 * float(np.median(espesores)), 0.25 * tamano)
    arbol_nodos = cKDTree(coordenadas)
    nodos_superficie: Dict[int, np.ndarray] = {}
    for s_id in superficies:
        triangulos_cara = t[cara_de == s_id]
        puntos, _ = muestras_de_superficie(v, triangulos_cara, espaciado)
        cerca = arbol_nodos.query_ball_point(puntos, 0.5 * float(espesores.max()) + 0.55 * espaciado)
        nodos_superficie[s_id] = np.unique(np.fromiter((n for lista in cerca for n in lista), dtype=np.int64))
    return MallaLamina(coordenadas=coordenadas, elementos=elementos, espesores=e, normales=normal, superficies=superficies,
                       nodos_superficie=nodos_superficie, tamano=tamano, volumen=float((area * e).sum()))


def mallar_lamina(solido: SolidoGeneral, tamano: float,
                  remallar: Optional[Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]] = None
                  ) -> MallaLamina:
    """Lámina ``S6`` de la superficie media del sólido. Lanza ``ErrorLamina`` si no se puede obtener.

    ``remallar(vértices, triángulos, tamaño) -> (vértices, triángulos)`` vuelve a mallar la superficie media; si falta
    o falla, se refina por bisección (el análisis la remalla con Gmsh en el lote de ``tetraedros.mallar_lote_gmsh``)."""

    media = superficie_media(solido, tamano)
    remallada = None
    if remallar is not None:
        try:
            remallada = remallar(media.vertices, media.triangulos, tamano)
        except Exception:  # noqa: BLE001 - Gmsh no pudo: se refina por bisección
            remallada = None
    return lamina_de_superficie(solido, media, tamano, remallada)


def elementos_s6(vertices: np.ndarray, triangulos: np.ndarray, espesores: np.ndarray, normal_ref: np.ndarray,
                 tamano: float) -> MallaLamina:
    """Lámina S6 de una superficie triangulada (vértices + triángulos lineales): orienta los triángulos según
    ``normal_ref`` (una normal (3,) o una por triángulo) y añade los nodos medios de las aristas."""

    usados, local = np.unique(np.asarray(triangulos, dtype=np.int64), return_inverse=True)
    vm, tm = np.asarray(vertices, dtype=float)[usados], local.reshape(-1, 3)
    p = vm[tm]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    referencia = np.broadcast_to(np.asarray(normal_ref, dtype=float), normal.shape)
    voltea = np.einsum("ij,ij->i", normal, referencia) < 0
    tm[voltea] = tm[voltea][:, [0, 2, 1]]
    normal[voltea] *= -1.0
    claves = np.sort(np.concatenate([tm[:, [0, 1]], tm[:, [1, 2]], tm[:, [2, 0]]]), axis=1)
    unicas, inverso = np.unique(claves, axis=0, return_inverse=True)
    inverso = inverso.reshape(3, -1).T
    coordenadas = np.vstack([vm, 0.5 * (vm[unicas[:, 0]] + vm[unicas[:, 1]])])
    elementos = np.hstack([tm, len(vm) + inverso]).astype(np.int64)
    espesores = np.asarray(espesores, dtype=float)
    return MallaLamina(coordenadas=coordenadas, elementos=elementos, espesores=espesores, normales=normal, tamano=tamano,
                       volumen=float((area * espesores).sum()))
