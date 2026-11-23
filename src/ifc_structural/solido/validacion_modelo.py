"""Validación del modelo antes de analizar: ¿va a correr el análisis y va a converger?

Comprobaciones rápidas (segundos, sin resolver nada):

* **Datos**: casos, combinaciones, y que apoyos, cargas y constraints apunten a elementos y superficies que existen.
* **Materiales**: que cada pieza tenga un material con E > 0, 0 ≤ ν < 0,5 y, en el análisis elastoplástico, fy.
* **Geometría**: coordenadas finitas y sólidos cerrados (una malla abierta no se puede mallar).
* **Conectividad**: qué piezas se tocan (geometría real) o están unidas por constraints; cada grupo necesita apoyos.
* **Estabilidad**: los apoyos de cada grupo deben impedir sus 6 movimientos de cuerpo rígido (3 traslaciones y 3
  giros); si no, la matriz de rigidez es singular y CalculiX falla o da desplazamientos absurdos.
* **Cargas**: resultante de cada caso (un caso sin cargas ni peso propio da resultados nulos).
* **Malla y tiempo**: tamaño de la malla y, en el análisis no lineal, cuántos análisis hay que resolver.

Prueba con CalculiX (opcional, minutos): se hace el análisis elástico completo. Muestra si CalculiX resuelve el
modelo, si quedan piezas en mecanismo y, con materiales con fluencia, qué parte del modelo pasaría de fy en cada
combinación: si fluye una zona amplia, el análisis elastoplástico probablemente no converja (mecanismo plástico).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.asignacion import Objetivo
from ..core.cargas import Apoyo, ApoyoArea, ApoyoSuperficie, CargaSuperficie, UnionSuperficie
from ..core.casos_carga import CasoCarga, Combinacion, casos_con_peso_propio
from ..core.modelo import Proyecto
from .solo_carga import calcular_transferencias, elementos_solo_carga, proyecto_mallado

ERROR, AVISO, CORRECTO = "ERROR", "AVISO", "OK"
#: Fracción de los nodos de materiales con fluencia que, pasando de fy en el análisis elástico, hace probable que el
#: análisis elastoplástico no converja (la plastificación deja de ser local).
FRACCION_FLUENCIA_AMPLIA = 0.05
#: Puntos de muestra por pieza para saber si dos piezas se tocan (las mismas reglas de «Conectar», con menos puntos).
MUESTRAS_CONECTIVIDAD = 600
#: Área de contacto mínima (m²) para contar como unión (como «Conectar automáticamente»).
AREA_MINIMA_CONTACTO = 5e-5

NOMBRES_MODOS = ("traslación en X", "traslación en Y", "traslación en Z", "giro alrededor de X", "giro alrededor de Y",
                 "giro alrededor de Z")


@dataclass
class Hallazgo:
    nivel: str  # ERROR, AVISO u OK
    seccion: str  # "Datos", "Materiales", "Geometría", "Conectividad", "Estabilidad", "Cargas", "Malla", "CalculiX"
    texto: str
    objetivos: List[Objetivo] = field(default_factory=list)  # elementos a los que se refiere (para seleccionarlos)


@dataclass
class InformeValidacion:
    hallazgos: List[Hallazgo] = field(default_factory=list)
    prueba_ccx: bool = False  # se hizo la prueba con CalculiX
    malla_previa: object = None  # la malla generada al validar (ANALIZAR la vuelve a usar)

    def agregar(self, nivel: str, seccion: str, texto: str, objetivos: Sequence[Objetivo] = ()) -> None:
        self.hallazgos.append(Hallazgo(nivel, seccion, texto, list(objetivos)))

    @property
    def errores(self) -> List[Hallazgo]:
        return [h for h in self.hallazgos if h.nivel == ERROR]

    @property
    def avisos(self) -> List[Hallazgo]:
        return [h for h in self.hallazgos if h.nivel == AVISO]

    @property
    def correcto(self) -> bool:
        return not self.errores

    def resumen(self) -> str:
        if self.errores:
            return (f"El análisis no va a correr correctamente: {len(self.errores)} error(es)"
                    + (f" y {len(self.avisos)} aviso(s)." if self.avisos else "."))
        texto = "El modelo está listo para analizar"
        if self.avisos:
            texto += f", con {len(self.avisos)} aviso(s) para revisar"
        if not self.prueba_ccx:
            texto += " (sin la prueba con CalculiX)"
        return texto + "."


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _coleccion(proyecto: Proyecto, tipo: str) -> dict:
    return {"VIGA": proyecto.vigas, "COLUMNA": proyecto.columnas, "LOSA": proyecto.losas, "MURO": proyecto.muros,
            "SOLIDO": proyecto.solidos}.get(tipo, {})


def _existe(proyecto: Proyecto, tipo: str, id_: str) -> bool:
    return id_ in _coleccion(proyecto, tipo)


def _nombre(tipo: str, id_: str) -> str:
    return {"VIGA": "viga", "COLUMNA": "columna", "LOSA": "losa", "MURO": "muro", "SOLIDO": "sólido"}.get(tipo, tipo.lower()) + f" {id_}"


def _lista(ids: Sequence[str], maximo: int = 8) -> str:
    ids = list(ids)
    return ", ".join(ids[:maximo]) + (f" y {len(ids) - maximo} más" if len(ids) > maximo else "")


class _Superficies:
    """Superficies reales de los elementos (las mismas de «Conectar»), calculadas una sola vez."""

    def __init__(self, proyecto: Proyecto):
        self.proyecto = proyecto
        self._cache: Dict[Tuple[str, str], object] = {}

    def de(self, tipo: str, id_: str):
        from .deteccion import superficie_de

        clave = (tipo, id_)
        if clave not in self._cache:
            try:
                self._cache[clave] = superficie_de(self.proyecto, tipo, id_)
            except (KeyError, ValueError, IndexError):
                self._cache[clave] = None
        return self._cache[clave]

    def cara(self, tipo: str, id_: str, cara: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(puntos de muestra, centro del área) de una cara del elemento, o None si no existe."""

        sup = self.de(tipo, id_)
        if sup is None:
            return None
        tri = sup.triangulos[np.asarray(sup.cara) == cara]
        if len(tri) == 0:
            return None
        p = sup.vertices[tri]
        area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        centros = p.mean(axis=1)
        centro = (centros * area[:, None]).sum(axis=0) / max(area.sum(), 1e-30)
        puntos = np.unique(sup.vertices[np.unique(tri)], axis=0)
        if len(puntos) > 60:
            puntos = puntos[np.linspace(0, len(puntos) - 1, 60).astype(int)]
        return puntos, centro

    def area_y_normal(self, tipo: str, id_: str, cara: int) -> Optional[Tuple[float, np.ndarray]]:
        sup = self.de(tipo, id_)
        if sup is None:
            return None
        tri = sup.triangulos[np.asarray(sup.cara) == cara]
        if len(tri) == 0:
            return None
        p = sup.vertices[tri]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area = 0.5 * float(np.linalg.norm(n, axis=1).sum())
        suma = n.sum(axis=0)
        return area, suma / max(float(np.linalg.norm(suma)), 1e-30)


# ---------------------------------------------------------------------------
# Comprobaciones rápidas
# ---------------------------------------------------------------------------
def _datos(informe: InformeValidacion, proyecto: Proyecto, apoyos, casos, combinaciones, cargas_superficie, uniones,
           sups: _Superficies) -> None:
    s = "Datos"
    if not casos:
        informe.agregar(ERROR, s, "No hay casos de carga elegidos para analizar.")
    if not apoyos:
        informe.agregar(ERROR, s, "No hay apoyos: el modelo no está sujeto a nada.")
    for comb in combinaciones.values():
        faltan = [c for c, f in comb.factores.items() if f and c not in casos]
        if faltan:
            informe.agregar(AVISO, s, f"La combinación {comb.id} usa casos que no se analizan ({', '.join(faltan)}): "
                                      "no se calculará.")
    for a in apoyos:
        tipo, id_ = getattr(a, "elemento_tipo", None), getattr(a, "elemento_id", None)
        if isinstance(a, ApoyoSuperficie):
            if not _existe(proyecto, tipo, id_):
                informe.agregar(ERROR, s, f"El apoyo {a.nombre} está sobre un elemento que ya no existe ({_nombre(tipo, id_)}).")
            elif sups.cara(tipo, id_, a.cara) is None:
                informe.agregar(ERROR, s, f"El apoyo {a.nombre}: la superficie {a.cara} de {_nombre(tipo, id_)} no existe.",
                                [Objetivo(tipo, id_)])
            elif not a.grados_restringidos():
                informe.agregar(AVISO, s, f"El apoyo {a.nombre} no restringe ningún movimiento.", [Objetivo(tipo, id_)])
    for caso_id, lista in cargas_superficie.items():
        if caso_id not in casos:
            continue
        for c in lista:
            if not _existe(proyecto, c.elemento_tipo, c.elemento_id):
                informe.agregar(ERROR, s, f"La carga {c.nombre or c.id} (caso {caso_id}) está sobre un elemento que ya no "
                                          f"existe ({_nombre(c.elemento_tipo, c.elemento_id)}).")
            elif sups.cara(c.elemento_tipo, c.elemento_id, c.cara) is None:
                informe.agregar(ERROR, s, f"La carga {c.nombre or c.id} (caso {caso_id}): la superficie {c.cara} de "
                                          f"{_nombre(c.elemento_tipo, c.elemento_id)} no existe.",
                                [Objetivo(c.elemento_tipo, c.elemento_id)])
    for u in uniones:
        for tipo, id_, cara in ((u.a_tipo, u.a_id, u.a_cara), (u.b_tipo, u.b_id, u.b_cara)):
            if not _existe(proyecto, tipo, id_):
                informe.agregar(ERROR, s, f"El constraint {u.nombre or u.id} une un elemento que ya no existe "
                                          f"({_nombre(tipo, id_)}).")
            elif cara is not None and cara >= 0 and sups.cara(tipo, id_, cara) is None:
                informe.agregar(ERROR, s, f"El constraint {u.nombre or u.id}: la superficie {cara} de {_nombre(tipo, id_)} "
                                          "no existe.", [Objetivo(tipo, id_)])


def _materiales(informe: InformeValidacion, proyecto: Proyecto, elementos, no_lineal: bool) -> None:
    s = "Materiales"
    por_material: Dict[str, List[Tuple[str, str]]] = {}
    for tipo, id_ in elementos:
        por_material.setdefault(getattr(_coleccion(proyecto, tipo)[id_], "material", None) or "", []).append((tipo, id_))
    plasticos = []
    for nombre, elems in por_material.items():
        objetivos = [Objetivo(t, i) for t, i in elems]
        m = proyecto.materiales.get(nombre)
        if m is None:
            informe.agregar(ERROR, s, f"Los elementos {_lista([i for _, i in elems])} usan el material «{nombre or '(ninguno)'}», "
                                      "que no está definido: asígneles un material.", objetivos)
            continue
        if not (m.modulo_elasticidad > 0 and math.isfinite(m.modulo_elasticidad)):
            informe.agregar(ERROR, s, f"El material {nombre} tiene un módulo de elasticidad no válido.", objetivos)
        if not (0.0 <= m.modulo_poisson < 0.5):
            informe.agregar(ERROR, s, f"El material {nombre} tiene un coeficiente de Poisson fuera de 0 ≤ ν < 0,5 "
                                      f"(ν = {m.modulo_poisson:g}): el sólido sería incompresible o inestable.", objetivos)
        if m.densidad < 0:
            informe.agregar(ERROR, s, f"El material {nombre} tiene peso específico negativo.", objetivos)
        if nombre.endswith("_DEFECTO") or "DEFECTO" in nombre.upper():
            informe.agregar(AVISO, s, f"El material {nombre} tiene valores por defecto (el IFC no traía sus propiedades): "
                                      "revíselo en Materiales.", objetivos)
        if getattr(m, "es_plastico", False):
            plasticos.append(nombre)
            if m.fluencia >= m.modulo_elasticidad * 0.05:
                informe.agregar(ERROR, s, f"El material {nombre}: la fluencia fy no es coherente con E (revise las unidades).",
                                objetivos)
    if no_lineal and not plasticos:
        informe.agregar(AVISO, s, "Se pidió el análisis elastoplástico, pero ningún material tiene fluencia (fy): se hará "
                                  "el análisis elástico.")


def _geometria(informe: InformeValidacion, proyecto: Proyecto) -> None:
    from ..core.validacion import problemas_de_geometria

    s = "Geometría"
    for problema in problemas_de_geometria(proyecto):
        informe.agregar(ERROR, s, problema)
    abiertos = [i for i, sol in proyecto.solidos.items() if not getattr(sol, "cerrado", True)]
    if abiertos:
        informe.agregar(ERROR, s, f"Los sólidos {_lista(abiertos)} tienen la superficie abierta o con errores: no se pueden "
                                  "mallar. Corríjalos en el modelo de origen o elimínelos.",
                        [Objetivo("SOLIDO", i) for i in abiertos])
    diminutos = [i for i, sol in proyecto.solidos.items() if getattr(sol, "cerrado", True) and abs(getattr(sol, "volumen", 1.0)) < 1e-9]
    if diminutos:
        informe.agregar(AVISO, s, f"Los sólidos {_lista(diminutos)} tienen un volumen casi nulo.",
                        [Objetivo("SOLIDO", i) for i in diminutos])


def _grupos(proyecto: Proyecto, elementos, uniones, tolerancia: float, al_avanzar=None) -> List[List[Tuple[str, str]]]:
    """Grupos de elementos unidos entre sí (se tocan, se solapan o los une un constraint)."""

    from .deteccion import _polidata, contacto_entre, muestras, superficie_de

    indice = {e: k for k, e in enumerate(elementos)}
    padre = list(range(len(elementos)))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for u in uniones:
        a, b = indice.get((u.a_tipo, u.a_id)), indice.get((u.b_tipo, u.b_id))
        if a is not None and b is not None:
            padre[raiz(a)] = raiz(b)
    if len(elementos) > 1:
        if al_avanzar:
            al_avanzar("Validación: buscando qué piezas se tocan...")
        # Solo importa si se tocan, no cuánto: pocos puntos por pieza, y no se mide una pareja que ya está en el mismo
        # grupo por otro camino (en una estructura grande casi todas las parejas se resuelven así).
        sups, cajas, datos, mallas = {}, {}, {}, {}
        for e in elementos:
            try:
                sups[e] = superficie_de(proyecto, *e)
            except (KeyError, ValueError, IndexError):
                continue
            cajas[e] = sups[e].caja
        claves = list(sups)
        tol = max(float(tolerancia), 5e-4)
        for i, ea in enumerate(claves):
            for eb in claves[i + 1:]:
                ka, kb = indice[ea], indice[eb]
                if raiz(ka) == raiz(kb):
                    continue
                (amin, amax), (bmin, bmax) = cajas[ea], cajas[eb]
                if np.any(amin - tol > bmax) or np.any(bmin - tol > amax):
                    continue
                for e in (ea, eb):
                    if e not in datos:
                        datos[e] = muestras(sups[e], MUESTRAS_CONECTIVIDAD)
                        mallas[e] = _polidata(sups[e])
                contacto = contacto_entre(sups[ea], sups[eb], tol, datos[ea], datos[eb], mallas[ea], mallas[eb])
                if contacto is not None and contacto[0] >= AREA_MINIMA_CONTACTO:
                    padre[raiz(ka)] = raiz(kb)
    grupos: Dict[int, List[Tuple[str, str]]] = {}
    for e, k in indice.items():
        grupos.setdefault(raiz(k), []).append(e)
    return list(grupos.values())


def _filas_de_apoyo(apoyo, proyecto: Proyecto, sups: _Superficies, referencia: np.ndarray, escala: float) -> List[np.ndarray]:
    """Filas de la matriz de restricciones de cuerpo rígido: cada fila es el desplazamiento restringido en un punto
    para cada uno de los 6 modos (traslaciones X, Y, Z y giros, estos multiplicados por ``escala``)."""

    def filas_punto(p: np.ndarray, grados: Sequence[int]) -> List[np.ndarray]:
        r = (np.asarray(p, dtype=float) - referencia) / escala
        filas = {1: [1, 0, 0, 0, r[2], -r[1]], 2: [0, 1, 0, -r[2], 0, r[0]], 3: [0, 0, 1, r[1], -r[0], 0]}
        return [np.array(filas[g], dtype=float) for g in grados if g in filas]

    grados = list(apoyo.grados_restringidos()) if hasattr(apoyo, "grados_restringidos") else []
    traslaciones = [g for g in grados if g <= 3]
    giros = [g for g in grados if g > 3]
    resultado: List[np.ndarray] = []
    if isinstance(apoyo, ApoyoSuperficie):
        cara = sups.cara(apoyo.elemento_tipo, apoyo.elemento_id, apoyo.cara)
        if cara is None:
            return []
        puntos, centro = cara
        if getattr(apoyo, "pasador", False) or giros:
            # pasador o cara rígida: se fija el centro; los giros marcados de la cara rígida, aparte
            resultado += filas_punto(centro, traslaciones)
            if not getattr(apoyo, "pasador", False):
                resultado += [np.eye(6)[g - 1] for g in giros]
        else:
            for p in puntos:
                resultado += filas_punto(p, traslaciones)
        return resultado
    if isinstance(apoyo, ApoyoArea):
        sup = sups.de(apoyo.elemento_tipo, apoyo.elemento_id)
        if sup is None:
            return []
        v = sup.vertices
        abajo = v[v[:, 2] < v[:, 2].min() + 1e-3]
        for p in abajo[:: max(1, len(abajo) // 30)]:
            resultado += filas_punto(p, traslaciones)
        return resultado
    if isinstance(apoyo, Apoyo) and apoyo.nodo_id in proyecto.nodos:
        n = proyecto.nodos[apoyo.nodo_id]
        resultado += filas_punto(np.array([n.x, n.y, n.z]), traslaciones)
        resultado += [np.eye(6)[g - 1] for g in giros]
    return resultado


def _modos_libres(filas: List[np.ndarray]) -> List[str]:
    """Movimientos de cuerpo rígido que las filas de restricción no impiden, descritos con palabras."""

    if not filas:
        return ["todos (no hay apoyos)"]
    a = np.vstack(filas)
    _, s, vt = np.linalg.svd(a, full_matrices=True)
    rango = int((s > 1e-6 * max(float(s[0]), 1e-30)).sum())
    if rango >= 6:
        return []
    nulo = vt[rango:]  # base de los movimientos libres
    # primero los movimientos «puros» (traslación en X, giro alrededor de Z...) que caben enteros en ese espacio
    puros = [k for k in range(6) if np.linalg.norm(nulo @ np.eye(6)[k]) > 0.999]
    libres = [NOMBRES_MODOS[k] for k in puros]
    if len(puros) == len(nulo):
        return libres
    # lo que queda, fuera de los puros: combinaciones (p. ej. un giro alrededor de un eje inclinado)
    if puros:
        resto = nulo - (nulo @ np.eye(6)[puros].T) @ np.eye(6)[puros]
        u, s_r, _ = np.linalg.svd(resto.T, full_matrices=False)
        nulo = u[:, s_r > 1e-6].T
    for v in nulo:
        v = v / max(float(np.abs(v).max()), 1e-30)
        traslacion, giro = v[:3], v[3:]
        if np.linalg.norm(giro) < 1e-6:
            k = int(np.argmax(np.abs(traslacion)))
            libres.append(NOMBRES_MODOS[k] if np.sum(np.abs(traslacion) > 0.2) == 1 else
                          f"traslación en la dirección ({traslacion[0]:.2f}; {traslacion[1]:.2f}; {traslacion[2]:.2f})")
        else:
            eje = giro / np.linalg.norm(giro)
            k = int(np.argmax(np.abs(eje)))
            libres.append(NOMBRES_MODOS[3 + k] if np.sum(np.abs(eje) > 0.2) == 1 else
                          f"giro alrededor del eje ({eje[0]:.2f}; {eje[1]:.2f}; {eje[2]:.2f})")
    return libres


def _conectividad_y_estabilidad(informe: InformeValidacion, proyecto: Proyecto, elementos, apoyos, uniones, cargados,
                                tolerancia: float, sups: _Superficies, al_avanzar=None) -> None:
    grupos = _grupos(proyecto, elementos, uniones, tolerancia, al_avanzar)
    apoyos_de: Dict[Tuple[str, str], list] = {}
    for a in apoyos:
        clave = (getattr(a, "elemento_tipo", None), getattr(a, "elemento_id", None))
        apoyos_de.setdefault(clave, []).append(a)
    nodales = [a for a in apoyos if isinstance(a, Apoyo) and not isinstance(a, (ApoyoArea, ApoyoSuperficie))]
    con_constraint = {(u.a_tipo, u.a_id) for u in uniones} | {(u.b_tipo, u.b_id) for u in uniones}
    if al_avanzar:
        al_avanzar("Validación: comprobando la estabilidad de cada grupo de piezas...")
    if len(grupos) > 1:
        informe.agregar(CORRECTO, "Conectividad", f"El modelo forma {len(grupos)} grupos de piezas independientes (no se "
                                                  "tocan entre sí ni los une un constraint); cada uno necesita sus apoyos.")
    else:
        informe.agregar(CORRECTO, "Conectividad", "Todas las piezas están unidas entre sí (se tocan o las une un constraint).")
    for grupo in grupos:
        ids = sorted(i for _, i in grupo)
        objetivos = [Objetivo(t, i) for t, i in grupo]
        propios = [a for e in grupo for a in apoyos_de.get(e, [])]
        if not propios and not nodales:
            asignados = [e for e in grupo if e in cargados or e in con_constraint]
            if asignados:
                informe.agregar(ERROR, "Conectividad", f"Las piezas {_lista(ids)} no tocan a ninguna pieza con apoyos y "
                                                       "tienen cargas o constraints: quedarían flotando. Únalas (Conectar) o "
                                                       "asígneles apoyos.", objetivos)
            else:
                informe.agregar(AVISO, "Conectividad", f"Las piezas {_lista(ids)} no tocan a ninguna otra ni tienen apoyos, "
                                                       "cargas o constraints: el análisis las dejará fijas en su sitio (no "
                                                       "participan). Elimínelas si sobran.", objetivos)
            continue
        puntos = np.vstack([sups.de(*e).vertices for e in grupo if sups.de(*e) is not None] or [np.zeros((1, 3))])
        referencia = puntos.mean(axis=0)
        escala = max(float(np.linalg.norm(np.ptp(puntos, axis=0))), 1e-6)
        filas = [f for a in propios + nodales for f in _filas_de_apoyo(a, proyecto, sups, referencia, escala)]
        libres = _modos_libres(filas)
        nombre = "El modelo" if len(grupos) == 1 else f"El grupo de piezas {_lista(ids, 5)}"
        if libres:
            informe.agregar(ERROR, "Estabilidad", f"{nombre} es inestable: sus apoyos no impiden {', '.join(libres)}. La "
                                                  "matriz de rigidez sería singular (CalculiX falla o da desplazamientos "
                                                  "enormes). Restrinja más grados de libertad o añada apoyos: p. ej. "
                                                  "pasadores en una misma recta dejan libre el giro alrededor de ella.",
                            objetivos)
        else:
            informe.agregar(CORRECTO, "Estabilidad", f"{nombre}: los apoyos impiden los 6 movimientos de cuerpo rígido.")


def _cargas(informe: InformeValidacion, proyecto: Proyecto, casos, cargas_superficie, otras_cargas: Dict[str, int],
            elementos, sups: _Superficies) -> None:
    s = "Cargas"
    con_peso = set(casos_con_peso_propio(casos))
    peso_total = 0.0
    for tipo, id_ in elementos:
        el = _coleccion(proyecto, tipo)[id_]
        m = proyecto.materiales.get(getattr(el, "material", ""))
        if m is None:
            continue
        volumen = getattr(el, "volumen", None)
        if volumen is None:
            sup = sups.de(tipo, id_)
            if sup is None:
                continue
            p = sup.vertices[sup.triangulos]
            volumen = abs(float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) / 6.0)
        peso_total += m.densidad * abs(float(volumen))
    for caso_id in casos:
        resultante = np.zeros(3)
        for c in cargas_superficie.get(caso_id, []):
            if c.tipo == "FUERZA" and c.direccion is not None:
                resultante += c.valor * np.asarray(c.direccion, dtype=float)
            elif c.tipo == "PRESION":
                datos = sups.area_y_normal(c.elemento_tipo, c.elemento_id, c.cara)
                if datos is not None:
                    area, normal = datos
                    d = np.asarray(c.direccion, dtype=float) if c.direccion is not None else -normal
                    resultante += c.valor * area * d
        if caso_id in con_peso:
            resultante += np.array([0.0, 0.0, -peso_total])
        n_otras = otras_cargas.get(caso_id, 0)
        hay = bool(cargas_superficie.get(caso_id)) or caso_id in con_peso or n_otras
        if not hay:
            informe.agregar(AVISO, s, f"El caso {caso_id} no tiene cargas ni peso propio: sus resultados serán nulos.")
            continue
        texto = (f"Caso {caso_id}: resultante de las cargas ({resultante[0]:.2f}; {resultante[1]:.2f}; {resultante[2]:.2f}) kN"
                 + (f", con {peso_total:.2f} kN de peso propio" if caso_id in con_peso else "")
                 + (f" (más {n_otras} carga(s) de barras o de área no sumadas aquí)" if n_otras else "") + ".")
        informe.agregar(CORRECTO, s, texto)


def memoria_disponible_mb() -> Optional[float]:
    """RAM física del equipo (o el límite del contenedor, si es menor), en MB; None si no se sabe."""

    import os
    import sys

    total = None
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class _Estado(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            estado = _Estado()
            estado.dwLength = ctypes.sizeof(_Estado)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado)):
                total = estado.ullTotalPhys / 2**20
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20
            for ruta in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
                try:
                    texto = Path(ruta).read_text().strip()
                except OSError:
                    continue
                if texto.isdigit() and int(texto) < 2**60:
                    total = min(total, int(texto) / 2**20)
    except (OSError, ValueError, AttributeError):
        return None
    return total


def memoria_calculix_mb(nodos_solidos: int, nodos_laminas: int) -> float:
    """Memoria aproximada (MB) que necesita CalculiX (SPOOLES) para resolver la malla. Medida con mallas de prueba:
    un sólido macizo de 34 651 nodos C3D10 pide unos 830 MB y crece como N^1,8; las láminas S6 se expanden a tres
    capas de nodos (C3D15) y crecen como N^1,3 (153 759 nodos expandidos, unos 1 030 MB). Es un orden de magnitud: el
    reparto real depende de la forma del modelo."""

    return (150.0 + 707.0 * (max(nodos_solidos, 0) / 34651.0) ** 1.82
            + 910.0 * (3.0 * max(nodos_laminas, 0) / 153759.0) ** 1.30)


def _memoria(informe: InformeValidacion, malla) -> None:
    s = "Malla"
    nodos_laminas = len(np.unique(malla.laminas)) if len(malla.laminas) else 0
    nodos_solidos = len(malla.coordenadas) - nodos_laminas
    necesaria = memoria_calculix_mb(nodos_solidos, nodos_laminas)
    disponible = memoria_disponible_mb()
    texto = f"CalculiX necesitará unos {necesaria / 1024:.1f} GB de memoria por análisis (estimación)"
    if disponible is None:
        informe.agregar(CORRECTO, s, texto + ".")
        return
    texto += f"; el equipo tiene {disponible / 1024:.1f} GB."
    if necesaria > 0.9 * disponible:
        informe.agregar(ERROR, s, texto + " CalculiX se quedaría sin memoria y el sistema lo detendría: agrande el tamaño "
                                          "de elemento en Mallar (la memoria baja mucho: con el doble de tamaño, unas 3 a "
                                          "6 veces menos) o use un equipo con más memoria.")
    elif necesaria > 0.6 * disponible:
        informe.agregar(AVISO, s, texto + " Va justo: cierre otros programas o agrande algo el tamaño de elemento.")
    else:
        informe.agregar(CORRECTO, s, texto)


def _malla(informe: InformeValidacion, proyecto: Proyecto, malla_previa, tamano: float, no_lineal: bool,
           n_casos: int, n_combinaciones: int) -> None:
    from .analisis_solido import OBJETIVO_ELEMENTOS, estimar

    s = "Malla"
    if malla_previa is not None:
        informe.agregar(CORRECTO, s, malla_previa.resumen())
        _memoria(informe, malla_previa.malla)
    else:
        try:
            n = estimar(proyecto, tamano)  # sin mallar: aproximado (las láminas curvas pueden dar bastante más)
        except Exception as exc:  # una geometría que el estimador no sabe tratar: lo dirá el mallado
            informe.agregar(AVISO, s, f"No se pudo estimar el tamaño de la malla: {exc}")
            n = 0
        if n:
            texto = (f"Con elementos de {tamano * 1000:.0f} mm saldrán unos {n:,} elementos (estimación sin mallar; la "
                     "validación con malla calcula también la memoria).").replace(",", " ")
            nivel = AVISO if n > 3 * OBJETIVO_ELEMENTOS else CORRECTO
            if nivel == AVISO:
                texto += " Es una malla muy grande: puede tardar mucho o quedarse sin memoria; agrande el tamaño en Mallar."
            informe.agregar(nivel, s, texto)
    if no_lineal:
        informe.agregar(AVISO, s, f"Análisis elastoplástico: se resuelven {n_casos + n_combinaciones} análisis no lineales "
                                  f"({n_casos} caso(s) y {n_combinaciones} combinación(es)), cada uno varias veces más lento "
                                  "que el elástico.")


def _solo_carga(informe: InformeValidacion, proyecto: Proyecto, casos, superficiales, apoyos, tolerancia: float,
                cargados: set, al_avanzar=None) -> None:
    """Grupos «solo carga»: dónde apoyan, qué transmiten y si se pueden equilibrar con sus contactos."""

    s = "Solo carga"
    fuera = elementos_solo_carga(proyecto)
    sobre = [a for a in apoyos if (getattr(a, "elemento_tipo", None), getattr(a, "elemento_id", None)) in fuera]
    r = calcular_transferencias(proyecto, casos, superficiales, tolerancia, al_avanzar, apoyos)
    propios = {e for piezas in r.apoyados for e in piezas}  # grupos sostenidos solo por sus apoyos: está bien
    for a in sobre:
        if (a.elemento_tipo, a.elemento_id) in propios:
            continue
        informe.agregar(AVISO, s, f"El apoyo {a.nombre} está sobre un elemento «solo carga» que carga sobre la parte mallada: "
                                  "no se aplica, y toda la carga de ese elemento baja a las piezas malladas que toca (del "
                                  "lado seguro para ellas). Si ese apoyo debe sostener la parte mallada, póngalo en una "
                                  "pieza mallada.", [Objetivo(a.elemento_tipo, a.elemento_id)])
    for piezas in r.apoyados:
        informe.agregar(CORRECTO, s, f"Los elementos «solo carga» {_lista([i for _, i in piezas])} no tocan ninguna pieza "
                                     "mallada y tienen apoyos propios: su peso y sus cargas van directamente a esos apoyos, "
                                     "no a la estructura.", [Objetivo(t, i) for t, i in piezas])
    for piezas in r.sueltos:
        informe.agregar(AVISO, s, f"Los elementos «solo carga» {_lista([i for _, i in piezas])} no tocan ninguna pieza mallada "
                                  "ni tienen apoyos: su peso y sus cargas no se aplican a la estructura. Si deben cargarla, "
                                  "únalos a ella o mállelos (quite «solo carga»).",
                        [Objetivo(t, i) for t, i in piezas])
    for g in r.grupos:
        cargados.update(g.destinos)
        resultantes = "; ".join(f"{c}: {abs(f[2]):.2f} kN" + (f" (+ {np.linalg.norm(f[:2]):.2f} kN horizontal)"
                                                              if np.linalg.norm(f[:2]) > 1e-6 else "")
                                for c, f in g.resultantes.items() if np.linalg.norm(f) > 1e-9)
        texto = (f"Los elementos {_lista([i for _, i in g.piezas], 5)} no se mallan y apoyan en "
                 f"{_lista([i for _, i in g.destinos], 6)}: les transmiten {resultantes or 'nada (sin peso ni cargas)'}.")
        peor = max(g.desequilibrio.values(), default=0.0)
        if peor > 1e-3:
            informe.agregar(AVISO, s, texto + f" No quedan en equilibrio con sus contactos (falta el {peor * 100:.0f} % de la "
                                              "carga): apoyan en una línea o un punto y la carga los haría girar.",
                            [Objetivo(t, i) for t, i in g.piezas])
        else:
            informe.agregar(CORRECTO, s, texto, [Objetivo(t, i) for t, i in g.piezas])


# ---------------------------------------------------------------------------
# Prueba con CalculiX
# ---------------------------------------------------------------------------
def _prueba_ccx(informe: InformeValidacion, proyecto: Proyecto, casos, combinaciones, apoyos, area, lineales, puntuales,
                superficiales, uniones, directorio: Path, ruta_ccx: str, tamano: float, malla_previa, no_lineal: bool,
                al_avanzar=None) -> None:
    from ..combinaciones import esfuerzos
    from ..solver.analisis import ErrorAnalisis
    from .analisis_solido import ejecutar_analisis_solido

    s = "CalculiX"
    informe.prueba_ccx = True
    try:
        r = ejecutar_analisis_solido(proyecto, casos, combinaciones, apoyos, area, lineales, puntuales, directorio, ruta_ccx,
                                     al_avanzar, tamano=tamano, cargas_superficie_por_caso=superficiales,
                                     uniones_manuales=uniones, malla_previa=malla_previa)
    except (ErrorAnalisis, ValueError, RuntimeError) as exc:
        informe.agregar(ERROR, s, f"El análisis elástico de prueba falló: {exc}")
        return
    informe.agregar(CORRECTO, s, "CalculiX resolvió el análisis elástico de prueba de todos los casos.")
    for a in r.advertencias:
        if "mecanismo" in a or "se desplazan más" in a or "no estaban unidas" in a or "no son físicamente razonables" in a:
            informe.agregar(ERROR if "no son válidos" in a or "no son físicamente" in a else AVISO, s, a)
    coords = np.asarray(r.malla_tensiones.coordenadas, dtype=float)
    tamano_modelo = max(float(np.linalg.norm(np.ptp(coords, axis=0))), 1e-9)
    celdas = list(r.malla_tensiones.conectividad)
    piezas = r.barra_de_cada_celda()
    tipo_de = {id_: t for t in ("VIGA", "COLUMNA", "LOSA", "MURO", "SOLIDO") for id_ in _coleccion(proyecto, t)}
    for nombre, u in {**r.desplazamientos_por_caso, **r.desplazamientos_por_combinacion}.items():
        norma = np.linalg.norm(u, axis=1) if len(u) else np.zeros(0)
        u_max = float(norma.max()) if len(norma) else 0.0
        if u_max <= tamano_modelo / 50:
            continue
        # las piezas que se mueven tanto: casi siempre cuelgan de un punto o de una arista (unión que no transmite)
        por_pieza: Dict[str, float] = {}
        for celda, pieza in zip(celdas, piezas):
            por_pieza[pieza] = max(por_pieza.get(pieza, 0.0), float(norma[np.asarray(celda)].max()))
        grandes = sorted((p for p, v in por_pieza.items() if v > tamano_modelo / 50), key=lambda p: -por_pieza[p])
        informe.agregar(AVISO, s, f"{nombre}: desplazamiento máximo {u_max * 1000:.1f} mm, más de 1/50 del tamaño del "
                                  f"modelo, en las piezas {_lista(grandes)}. Suelen ser piezas que cuelgan de un punto o "
                                  "de una arista (la unión no transmite): únalas con un constraint (Conectar), revise "
                                  "apoyos y materiales; el análisis no lineal puede no converger.",
                        [Objetivo(tipo_de[p], p) for p in grandes if p in tipo_de])
    # ¿Cuánto del modelo pasaría de fy? (solo con materiales con fluencia)
    fy_por_elemento = {}
    for tipo in ("VIGA", "COLUMNA", "LOSA", "MURO", "SOLIDO"):
        for id_, el in _coleccion(proyecto, tipo).items():
            m = proyecto.materiales.get(getattr(el, "material", ""))
            if m is not None and getattr(m, "es_plastico", False):
                fy_por_elemento[id_] = m.fluencia
    if not fy_por_elemento:
        return
    conectividad = np.asarray(r.malla_tensiones.conectividad)
    fy_nodo = np.full(len(coords), np.nan)
    for celda, elemento in zip(conectividad, r.barra_de_cada_celda()):
        if elemento in fy_por_elemento:
            fy_nodo[np.asarray(celda)] = fy_por_elemento[elemento]
    con_fy = np.isfinite(fy_nodo)
    if not con_fy.any():
        return
    resultados = r.tensiones_por_combinacion or r.tensiones_por_caso
    peor_fraccion = 0.0
    for nombre, t in resultados.items():
        vm = esfuerzos.von_mises(t)
        razon = vm[con_fy] / fy_nodo[con_fy]
        fraccion = float((razon > 1.0).mean())
        peor_fraccion = max(peor_fraccion, fraccion)
        maximo = float(razon.max())
        if maximo <= 1.0:
            informe.agregar(CORRECTO, s, f"{nombre}: el esfuerzo de Von Mises no llega a fy (máximo {maximo * 100:.0f} % de fy): "
                                         "el análisis elastoplástico dará lo mismo que el elástico.")
        elif fraccion < FRACCION_FLUENCIA_AMPLIA:
            informe.agregar(CORRECTO, s, f"{nombre}: en el análisis elástico pasa de fy el {fraccion * 100:.2f} % de los nodos "
                                         f"(hasta {maximo:.1f}·fy): fluencia local, el análisis elastoplástico debería "
                                         "converger.")
        else:
            informe.agregar(AVISO if not no_lineal else ERROR, s,
                            f"{nombre}: en el análisis elástico pasa de fy el {fraccion * 100:.1f} % de los nodos (hasta "
                            f"{maximo:.1f}·fy). Es una plastificación amplia: el análisis elastoplástico probablemente no "
                            "converja (se formaría un mecanismo plástico), es decir, la carga supera la capacidad. Revise "
                            "cargas, secciones o material.")


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
def validar_modelo_solido(
    proyecto: Proyecto,
    casos: Dict[str, CasoCarga],
    combinaciones: Dict[str, Combinacion],
    apoyos: Sequence,
    cargas_superficie_por_caso: Optional[Dict[str, List[CargaSuperficie]]] = None,
    uniones: Sequence[UnionSuperficie] = (),
    tamano: float = 0.05,
    malla_previa=None,
    no_lineal: bool = False,
    otras_cargas_por_caso: Optional[Dict[str, int]] = None,
    prueba_ccx: bool = False,
    mallar: bool = False,
    directorio_trabajo: Optional[Path] = None,
    ruta_ccx: str = "ccx",
    cargas_area_por_caso=None,
    cargas_lineales_por_caso=None,
    cargas_puntuales_por_caso=None,
    al_avanzar: Optional[Callable[[str], None]] = None,
) -> InformeValidacion:
    """Comprueba el modelo para el análisis sólido 3D y dice qué impediría que el análisis corra o converja."""

    from .deteccion import elementos_del_proyecto

    informe = InformeValidacion()
    superficiales = {k: v for k, v in (cargas_superficie_por_caso or {}).items() if k in casos}
    sups = _Superficies(proyecto)
    elementos = elementos_del_proyecto(proyecto)
    if al_avanzar:
        al_avanzar("Validación: datos, materiales y geometría...")
    _datos(informe, proyecto, apoyos, casos, combinaciones, superficiales, uniones, sups)
    _materiales(informe, proyecto, elementos, no_lineal)
    _geometria(informe, proyecto)
    if any(h.seccion == "Geometría" and h.nivel == ERROR for h in informe.hallazgos):
        return informe  # sin geometría válida no se puede seguir
    cargados = {(c.elemento_tipo, c.elemento_id) for lista in superficiales.values() for c in lista}
    # Elementos «solo carga»: no se mallan; la conectividad y la estabilidad son las de lo que se malla, y cada grupo
    # «solo carga» tiene que apoyar en piezas malladas que reciban su peso y sus cargas
    fuera = elementos_solo_carga(proyecto)
    mallados = [e for e in elementos if e not in fuera]
    apoyos_mallados, uniones_malladas = list(apoyos), list(uniones)
    if fuera:
        _solo_carga(informe, proyecto, casos, superficiales, apoyos, max(2e-3, 0.25 * tamano), cargados, al_avanzar)
        apoyos_mallados = [a for a in apoyos if (getattr(a, "elemento_tipo", None), getattr(a, "elemento_id", None)) not in fuera]
        uniones_malladas = [u for u in uniones if (u.a_tipo, u.a_id) not in fuera and (u.b_tipo, u.b_id) not in fuera]
    if mallados:
        _conectividad_y_estabilidad(informe, proyecto_mallado(proyecto), mallados, apoyos_mallados, uniones_malladas, cargados,
                                    max(2e-3, 0.25 * tamano), sups, al_avanzar)
    else:
        informe.agregar(ERROR, "Geometría", "El modelo no tiene elementos que mallar (todos son «solo carga»)."
                        if elementos else "El modelo no tiene elementos.")
    _cargas(informe, proyecto, casos, superficiales, otras_cargas_por_caso or {}, elementos, sups)
    if mallar and malla_previa is None and not informe.errores:
        # la malla real (la que usará el análisis, que la vuelve a usar): da los nodos y la memoria de verdad
        from .analisis_solido import mallar_modelo

        try:
            malla_previa = mallar_modelo(proyecto, tamano, apoyos, cargas_puntuales_por_caso or {}, cargas_lineales_por_caso or {},
                                         al_avanzar=al_avanzar)
            informe.malla_previa = malla_previa
        except Exception as exc:  # el mismo error que daría el análisis al mallar
            informe.agregar(ERROR, "Malla", f"No se pudo mallar el modelo: {exc}")
    _malla(informe, proyecto, malla_previa, tamano, no_lineal, len(casos), len(combinaciones))
    if prueba_ccx:
        if informe.errores:
            informe.agregar(AVISO, "CalculiX", "No se hizo la prueba con CalculiX: corrija antes los errores.")
        else:
            _prueba_ccx(informe, proyecto, casos, combinaciones, apoyos, cargas_area_por_caso or {},
                        cargas_lineales_por_caso or {}, cargas_puntuales_por_caso or {}, superficiales, list(uniones),
                        Path(directorio_trabajo or Path.cwd() / "validacion"), ruta_ccx, tamano, malla_previa, no_lineal,
                        al_avanzar)
    return informe


def validar_modelo_barras(errores: Sequence[str], advertencias: Sequence[str]) -> InformeValidacion:
    """Informe con las comprobaciones del modelo de barras (``solver.analisis.validar_modelo``)."""

    informe = InformeValidacion()
    for e in errores:
        informe.agregar(ERROR, "Modelo de barras", e)
    for a in advertencias:
        informe.agregar(AVISO, "Modelo de barras", a)
    if not errores:
        informe.agregar(CORRECTO, "Modelo de barras", "Barras conectadas a apoyos, estructura estable y sin mecanismos.")
    return informe
