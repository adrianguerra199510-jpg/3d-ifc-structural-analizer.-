"""Análisis del modelo como sólido 3D: orquesta mallado, .inp, ccx y resultados.

Devuelve el mismo ``ResultadoAnalisis`` que el análisis de barras, de modo que la
interfaz muestra los esfuerzos igual (mapa de colores, casos, combinaciones y
envolvente). Cada hexaedro conserva el id de la viga, columna, losa o muro de
donde viene.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..combinaciones import esfuerzos
from ..combinaciones.motor_superposicion import superponer_todas_las_combinaciones
from ..core.cargas import Apoyo, ApoyoArea, ApoyoSuperficie, CargaArea, CargaLineal, CargaPuntual, CargaSuperficie, UnionSuperficie
from ..core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ..core.modelo import Proyecto
from ..solver.analisis import (
    LIMITE_ESFUERZO_RAZONABLE_KPA, ErrorAnalisis, ResultadoAnalisis,
)
from ..solver.ejecutor_ccx import ejecutar_ccx_todos_los_casos
from ..solver.generador_inp import ElementoMalla, MallaAnalisis
from ..solver.lector_resultados import leer_desplazamientos_frd, leer_malla_frd, leer_tensiones_frd
from .contactos import Union, _caras_en_contacto, detectar_uniones, partes_sin_conexion
from .generador import Apoyo3D, CargasCaso, Enlace3D, ModeloSolido, escribir, escribir_caso, escribir_comun, etiqueta
from .malla import ErrorMalla, MallaSolida, ParteMalla, construir_malla_solida, estimar_hexaedros
from .miembros import MiembroBarra, barras_de_longitud_cero, construir_miembros, recortar_uniones
from .superficies import _parte_de, fuerzas_nodales, superficie_en_malla
from .tetraedros import gmsh_disponible

#: Tamaño de elemento por defecto (m) y máximo de hexaedros que se aceptan sin avisar.
TAMANO_POR_DEFECTO = 0.05
MAXIMO_HEXAEDROS = 400_000
TIEMPO_LIMITE_CCX = 3600.0


def elemento_solido_por_defecto() -> str:
    """Elemento de los sólidos libres: tetraedros cuadráticos ``C3D10`` (Gmsh, como Placa Base Pro) si Gmsh está
    disponible; si no, hexaedros ``C3D8I`` por voxelización."""

    return "TETRA" if gmsh_disponible() else "HEXA"


def tamano_sugerido(proyecto: Proyecto, objetivo_hexaedros: int = 40_000, elemento_solido: Optional[str] = None) -> float:
    """Tamaño de elemento (m) más fino, entre 1 cm y 25 cm, que no pasa de ``objetivo_hexaedros``."""

    elemento_solido = elemento_solido or elemento_solido_por_defecto()
    miembros = construir_miembros(proyecto)
    recortar_uniones(proyecto, miembros)
    for tamano in (0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25):
        if estimar_hexaedros(proyecto, miembros, tamano, elemento_solido == "TETRA") <= objetivo_hexaedros:
            return tamano  # el más fino que no pasa del objetivo
    return 0.25


def estimar(proyecto: Proyecto, tamano: float, elemento_solido: Optional[str] = None) -> int:
    elemento_solido = elemento_solido or elemento_solido_por_defecto()
    miembros = construir_miembros(proyecto)
    recortar_uniones(proyecto, miembros)
    return estimar_hexaedros(proyecto, miembros, tamano, elemento_solido == "TETRA")


# ---------------------------------------------------------------------------
# Ubicación de nodos del modelo en la malla
# ---------------------------------------------------------------------------


def _parte_de_miembro(malla: MallaSolida, miembro_id: str) -> ParteMalla:
    return next(p for p in malla.partes if p.id == miembro_id)


def _miembro_en_punto(proyecto: Proyecto, miembros: Sequence[MiembroBarra], punto: np.ndarray) -> Optional[MiembroBarra]:
    """Miembro de mayor sección cuyo eje pasa por ``punto`` (extremos incluidos)."""

    candidatos = [m for m in miembros
                  if m.distancia_al_eje(punto) < 1e-4 and -1e-4 <= m.coordenada(punto) <= m.longitud + 1e-4]
    return max(candidatos, key=lambda m: proyecto.perfiles[m.perfil].area, default=None)


def _anillo(parte: ParteMalla, s: float) -> int:
    return int(np.argmin(np.abs(parte.estaciones - s)))


def _punto_de_nodo(proyecto: Proyecto, nodo_id: str) -> np.ndarray:
    return np.array(proyecto.nodos[nodo_id].coordenadas, dtype=float)


def _repartir_en_anillo(parte: ParteMalla, k: int, fuerza: np.ndarray, momento: np.ndarray, malla: MallaSolida):
    """Fuerzas nodales sobre el anillo ``k`` equivalentes a una fuerza y un momento en su centro."""

    nodos = parte.anillos[k]
    w = parte.pesos / parte.pesos.sum()
    fuerzas = w[:, None] * fuerza[None, :]
    if np.linalg.norm(momento) > 0:
        r = malla.coordenadas[nodos] - (w[:, None] * malla.coordenadas[nodos]).sum(axis=0)
        K = np.zeros((3, 3))
        for wi, ri in zip(w, r):
            K += wi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
        m_eq = np.linalg.lstsq(K, momento, rcond=None)[0]
        fuerzas = fuerzas + w[:, None] * np.cross(m_eq[None, :], r)
    return nodos, fuerzas


def puntos_especiales(proyecto: Proyecto, miembros: Sequence[MiembroBarra], apoyos: Sequence,
                      puntuales: Dict[str, List[CargaPuntual]], lineales: Dict[str, List[CargaLineal]]) -> Dict[str, List[float]]:
    """Coordenadas axiales donde conviene una estación de la malla (cargas y apoyos)."""

    por_barra = {b: m for m in miembros for b in m.barras}
    resultado: Dict[str, List[float]] = {}

    def agregar(miembro: MiembroBarra, punto) -> None:
        resultado.setdefault(miembro.id, []).append(miembro.coordenada(np.asarray(punto)))

    for a in apoyos:
        if isinstance(a, Apoyo) and a.nodo_id in proyecto.nodos:
            p = _punto_de_nodo(proyecto, a.nodo_id)
            m = _miembro_en_punto(proyecto, miembros, p)
            if m is not None:
                agregar(m, p)
    for cargas in puntuales.values():
        for c in cargas:
            if c.nodo_id is not None and c.nodo_id in proyecto.nodos:
                p = _punto_de_nodo(proyecto, c.nodo_id)
                m = _miembro_en_punto(proyecto, miembros, p)
                if m is not None:
                    agregar(m, p)
            elif c.elemento_id in por_barra:
                b = proyecto.elemento_barra(c.elemento_id)
                a_, b_ = _punto_de_nodo(proyecto, b.nodo_inicio), _punto_de_nodo(proyecto, b.nodo_fin)
                agregar(por_barra[c.elemento_id], a_ + c.posicion * (b_ - a_))
    for cargas in lineales.values():
        for c in cargas:
            if c.elemento_id in por_barra:
                b = proyecto.elemento_barra(c.elemento_id)
                a_, b_ = _punto_de_nodo(proyecto, b.nodo_inicio), _punto_de_nodo(proyecto, b.nodo_fin)
                for s in (c.s_inicio, c.s_fin):
                    agregar(por_barra[c.elemento_id], a_ + s * (b_ - a_))
    return resultado


# ---------------------------------------------------------------------------
# Apoyos y cargas sobre la malla
# ---------------------------------------------------------------------------


def _apoyos_3d(proyecto: Proyecto, miembros: Sequence[MiembroBarra], malla: MallaSolida, apoyos: Sequence,
               avisos: List[str], tamano: float = TAMANO_POR_DEFECTO) -> Tuple[List[Apoyo3D], Set[int]]:
    resultado: List[Apoyo3D] = []
    partes_apoyadas: Set[int] = set()
    for a in apoyos:
        if isinstance(a, ApoyoSuperficie):
            sup = superficie_en_malla(proyecto, malla, a.elemento_tipo, a.elemento_id, a.cara, tamano)
            if sup is None:
                avisos.append(f"El apoyo {a.nombre} no encontró su superficie en la malla y se ignora.")
                continue
            grados = set(a.grados_restringidos())
            resultado.append(Apoyo3D(sup.nodos_unicos, grados, rigido=any(g > 3 for g in grados), nombre=a.nombre))
            partes_apoyadas.add(sup.parte)
            continue
        if isinstance(a, ApoyoArea):
            elementos = {a.elemento_id}
            if a.elemento_tipo == "LOSA" and a.elemento_id in proyecto.losas:
                elementos = {a.elemento_id}
            nodos: Set[int] = set()
            for i, parte in enumerate(malla.partes):
                ids = set(parte.miembro.barras) if parte.miembro is not None else {parte.id}
                if not ids & elementos:
                    continue
                f = malla.fronteras(parte)
                abajo = f.normal[:, 2] < -0.7
                if parte.miembro is not None:
                    longitud = parte.miembro.longitud
                    s = np.dot(f.centro - parte.miembro.p0, parte.miembro.t)
                    abajo &= (s >= a.s_inicio * longitud - 1e-6) & (s <= a.s_fin * longitud + 1e-6)
                if abajo.any():
                    nodos.update(int(n) for n in f.nodos[abajo].ravel())
                    partes_apoyadas.add(i)
            if nodos:
                resultado.append(Apoyo3D(np.array(sorted(nodos)), set(g for g in a.grados_restringidos() if g <= 3),
                                         rigido=False, nombre=f"AREA {a.nombre}"))
            else:
                avisos.append(f"El apoyo de área {a.nombre} no encontró una cara inferior donde apoyarse y se ignora.")
            continue
        if a.nodo_id not in proyecto.nodos:
            continue
        punto = _punto_de_nodo(proyecto, a.nodo_id)
        miembro = _miembro_en_punto(proyecto, miembros, punto)
        if miembro is None:
            avisos.append(f"El apoyo {a.nombre} no está sobre ninguna viga o columna y se ignora.")
            continue
        parte = _parte_de_miembro(malla, miembro.id)
        k = _anillo(parte, miembro.coordenada(punto))
        grados = set(a.grados_restringidos())
        resultado.append(Apoyo3D(parte.anillos[k], grados, rigido=grados != {1, 2, 3, 4, 5, 6}, nombre=a.nombre))
        partes_apoyadas.add(malla.partes.index(parte))
    return resultado, partes_apoyadas


def _cargas_del_caso(proyecto: Proyecto, miembros: Sequence[MiembroBarra], malla: MallaSolida, caso_id: str,
                     puntuales: Sequence[CargaPuntual], lineales: Sequence[CargaLineal], area: Sequence[CargaArea],
                     peso_propio: bool, avisos: List[str], superficiales: Sequence[CargaSuperficie] = (),
                     tamano: float = TAMANO_POR_DEFECTO) -> CargasCaso:
    cargas = CargasCaso(peso_propio=peso_propio)
    for c in superficiales:
        sup = superficie_en_malla(proyecto, malla, c.elemento_tipo, c.elemento_id, c.cara, tamano)
        if sup is None:
            avisos.append(f"La carga {c.nombre} no encontró su superficie en la malla y se ignora.")
            continue
        nodos, fuerzas = fuerzas_nodales(malla, sup, c)
        cargas.sumar(nodos, fuerzas)
    por_barra = {b: m for m in miembros for b in m.barras}

    for c in puntuales:
        fuerza, momento = np.array([c.fx, c.fy, c.fz], float), np.array([c.mx, c.my, c.mz], float)
        if c.nodo_id is not None:
            if c.nodo_id not in proyecto.nodos:
                continue
            punto = _punto_de_nodo(proyecto, c.nodo_id)
            miembro = _miembro_en_punto(proyecto, miembros, punto)
        elif c.elemento_id in por_barra:
            b = proyecto.elemento_barra(c.elemento_id)
            a_, b_ = _punto_de_nodo(proyecto, b.nodo_inicio), _punto_de_nodo(proyecto, b.nodo_fin)
            punto = a_ + c.posicion * (b_ - a_)
            miembro = por_barra[c.elemento_id]
        else:
            continue
        if miembro is None:
            avisos.append(f"La carga {c.nombre} no está sobre ninguna viga o columna y se ignora.")
            continue
        parte = _parte_de_miembro(malla, miembro.id)
        nodos, fuerzas = _repartir_en_anillo(parte, _anillo(parte, miembro.coordenada(punto)), fuerza, momento, malla)
        cargas.sumar(nodos, fuerzas)

    for c in lineales:
        miembro = por_barra.get(c.elemento_id)
        if miembro is None:
            continue
        parte = _parte_de_miembro(malla, miembro.id)
        b = proyecto.elemento_barra(c.elemento_id)
        a_, b_ = _punto_de_nodo(proyecto, b.nodo_inicio), _punto_de_nodo(proyecto, b.nodo_fin)
        longitud_barra = float(np.linalg.norm(b_ - a_))
        d = np.array(c.direccion, float)
        for k in range(len(parte.estaciones) - 1):
            s0, s1 = parte.estaciones[k], parte.estaciones[k + 1]
            medio = miembro.punto(0.5 * (s0 + s1))
            f_barra = float(np.dot(medio - a_, b_ - a_) / longitud_barra**2)
            if not (c.s_inicio - 1e-6 <= f_barra <= c.s_fin + 1e-6):
                continue
            t = 0.0 if c.s_fin - c.s_inicio < 1e-9 else (f_barra - c.s_inicio) / (c.s_fin - c.s_inicio)
            w = c.w_inicio + t * (c.w_fin - c.w_inicio)
            total = w * (s1 - s0) * d
            for anillo in (k, k + 1):
                nodos, fuerzas = _repartir_en_anillo(parte, anillo, total / 2.0, np.zeros(3), malla)
                cargas.sumar(nodos, fuerzas)

    for c in area:
        id_ = c.losa_id or c.muro_id
        parte = next((p for p in malla.partes if p.id == id_ and p.tipo in ("LOSA", "MURO")), None)
        if parte is None:
            continue
        d = np.array(c.direccion, float)
        f = malla.fronteras(parte)
        enfrentadas = np.einsum("ij,j->i", f.normal, -d) > 0.7
        if not enfrentadas.any():
            avisos.append(f"La carga de área {c.nombre} no encontró la cara de {id_} que la recibe.")
            continue
        # La losa pierde los hexaedros que solapan con las vigas; la carga sigue actuando sobre
        # toda su planta, así que se reparte sobre las caras que quedan (equilibrio conservado).
        elemento = proyecto.losas.get(id_) or proyecto.muros.get(id_)
        contorno = np.array([proyecto.nodos[n].coordenadas for n in elemento.nodos_contorno], dtype=float)
        area_total = 0.5 * float(np.linalg.norm(sum(np.cross(contorno[i], contorno[(i + 1) % len(contorno)])
                                                    for i in range(len(contorno)))))
        factor = area_total / float(f.area[enfrentadas].sum()) if f.area[enfrentadas].sum() > 0 else 1.0
        for i in np.where(enfrentadas)[0]:
            presion = c.presion_en(tuple(f.centro[i])) * factor
            for n in f.nodos[i]:
                cargas.sumar(np.array([n]), (presion * f.area[i] / 4.0 * d)[None, :])
    return cargas


def _enlaces_manuales(proyecto: Proyecto, malla: MallaSolida, manuales: Sequence[UnionSuperficie], tamano: float,
                      avisos: List[str]) -> Tuple[List[Union], List[Enlace3D], List[Tuple[int, int]]]:
    """Constraints manuales: soldaduras (``*TIE``), cuerpos rígidos, igualaciones y articulaciones."""

    from scipy.spatial import cKDTree

    ties: List[Union] = []
    enlaces: List[Enlace3D] = []
    partes: List[Tuple[int, int]] = []
    for u in manuales:
        if u.es_completa:  # constraint entre elementos 3D completos
            pa, pb = _parte_de(malla, u.a_tipo, u.a_id), _parte_de(malla, u.b_tipo, u.b_id)
            if pa is None or pb is None:
                avisos.append(f"El constraint «{u.nombre}» no encontró alguno de sus elementos en la malla y se ignora.")
                continue
            ia, ib = malla.partes.index(pa), malla.partes.index(pb)
            if ia == ib:
                avisos.append(f"El constraint «{u.nombre}» une un elemento consigo mismo y se ignora.")
                continue
            if u.tipo == "SOLDADURA":
                tol = max(u.tolerancia, 2e-3 + 0.25 * tamano)
                contacto = _caras_en_contacto(malla, ia, ib, tol)
                if contacto is None:
                    avisos.append(f"Soldadura «{u.nombre}»: los elementos no se tocan ni están a menos de {tol * 1000:.0f} mm "
                                  "(aumente la tolerancia); no se unieron.")
                    continue
                _, _, f_a, f_b, area_a, area_b = contacto
                esc, mae = (f_a, f_b) if area_a <= area_b else (f_b, f_a)
                p_esc, p_mae = (ia, ib) if area_a <= area_b else (ib, ia)
                ties.append(Union(f"UM{len(ties) + 1}", p_esc, p_mae, esc, mae, 1.5 * tol))
                partes.append((ia, ib))  # solo cuenta como unión si de verdad se soldó
                continue
            partes.append((ia, ib))

            def nodos(parte):
                return malla.nodos_de_parte(parte)

            enlaces.append(Enlace3D(u.tipo, nodos(pa), nodos(pb), set(u.grados()), u.nombre))
            continue
        sa = superficie_en_malla(proyecto, malla, u.a_tipo, u.a_id, u.a_cara, tamano)
        sb = superficie_en_malla(proyecto, malla, u.b_tipo, u.b_id, u.b_cara, tamano)
        if sa is None or sb is None:
            avisos.append(f"El constraint «{u.nombre}» no encontró alguna de sus superficies en la malla y se ignora.")
            continue
        partes.append((sa.parte, sb.parte))
        if u.tipo == "SOLDADURA":
            if sa.parte == sb.parte:
                avisos.append(f"El constraint «{u.nombre}» une dos superficies de la misma pieza y se ignora.")
                continue
            na, nb = sa.nodos_unicos, sb.nodos_unicos
            holgura = float(cKDTree(malla.coordenadas[nb]).query(malla.coordenadas[na])[0].min())  # distancia entre las superficies
            esclava_es_a = sa.area_total <= sb.area_total
            esc, mae = (sa, sb) if esclava_es_a else (sb, sa)
            ties.append(Union(
                f"UM{len(ties) + 1}", esc.parte, mae.parte,
                [(int(h), int(c)) for h, c in zip(esc.hexaedro, esc.cara)],
                [(int(h), int(c)) for h, c in zip(mae.hexaedro, mae.cara)],
                1.5 * holgura + 2e-3 + 0.25 * tamano))
            continue
        enlaces.append(Enlace3D(u.tipo, sa.nodos_unicos, sb.nodos_unicos, set(u.grados()), u.nombre))
    return ties, enlaces, partes


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def ejecutar_analisis_solido(
    proyecto: Proyecto,
    casos: Dict[str, CasoCarga],
    combinaciones: Dict[str, Combinacion],
    apoyos: Sequence,
    cargas_area_por_caso: Dict[str, List[CargaArea]],
    cargas_lineales_por_caso: Dict[str, List[CargaLineal]],
    cargas_puntuales_por_caso: Dict[str, List[CargaPuntual]],
    directorio_trabajo: Path,
    ruta_ccx: str = "ccx",
    al_avanzar: Optional[Callable[[str], None]] = None,
    tamano: float = TAMANO_POR_DEFECTO,
    conexiones: Sequence = (),
    cargas_superficie_por_caso: Optional[Dict[str, List[CargaSuperficie]]] = None,
    uniones_manuales: Sequence[UnionSuperficie] = (),
    elemento_solido: Optional[str] = None,
) -> ResultadoAnalisis:
    def avanzar(mensaje: str) -> None:
        if al_avanzar is not None:
            al_avanzar(mensaje)

    advertencias: List[str] = []
    cargas_superficie_por_caso = cargas_superficie_por_caso or {}
    if not casos:
        raise ErrorAnalisis("No hay casos de carga definidos.")
    if not apoyos:
        raise ErrorAnalisis("No hay apoyos asignados: seleccione la cara extrema de una columna o viga y asigne un apoyo.")
    if conexiones:
        advertencias.append("Las conexiones manuales (articulaciones y enlaces) no se aplican en el análisis sólido: "
                            "las piezas se unen por las superficies en contacto.")

    avanzar("Preparando los miembros sólidos...")
    degeneradas = barras_de_longitud_cero(proyecto)
    if degeneradas:
        advertencias.append(f"Las barras {', '.join(degeneradas)} tienen longitud cero (sus dos extremos coinciden) y se omiten.")
    miembros = construir_miembros(proyecto)
    advertencias.extend(recortar_uniones(proyecto, miembros))
    elemento_solido = elemento_solido or elemento_solido_por_defecto()
    estimados = estimar_hexaedros(proyecto, miembros, tamano, elemento_solido == "TETRA")
    if estimados > MAXIMO_HEXAEDROS:
        raise ErrorAnalisis(
            f"Con elementos de {tamano * 1000:.0f} mm el modelo tendría unos {estimados:,} elementos (máximo "
            f"{MAXIMO_HEXAEDROS:,}). Aumente el tamaño de elemento.".replace(",", " "))

    avanzar(f"Mallando (unos {estimados:,} elementos)...".replace(",", " "))
    especiales = puntos_especiales(proyecto, miembros, apoyos, cargas_puntuales_por_caso, cargas_lineales_por_caso)
    try:
        malla = construir_malla_solida(proyecto, miembros, tamano, especiales, elemento_solido)
    except ErrorMalla as exc:
        raise ErrorAnalisis(str(exc)) from exc

    avanzar("Detectando las uniones entre piezas...")
    espesores = {**{i: l.espesor for i, l in proyecto.losas.items()}, **{i: m.espesor for i, m in proyecto.muros.items()}}
    uniones, avisos_uniones = detectar_uniones(malla, tamano, espesores)
    advertencias.extend(avisos_uniones)
    advertencias.extend(malla.avisos)
    apoyos_3d, partes_apoyadas = _apoyos_3d(proyecto, miembros, malla, apoyos, advertencias, tamano)
    if not apoyos_3d:
        raise ErrorAnalisis("Ningún apoyo quedó sobre un elemento del modelo: asigne apoyos a una cara de un sólido, una columna, una viga o una losa.")
    ties_manuales, enlaces, partes_enlazadas = _enlaces_manuales(proyecto, malla, uniones_manuales, tamano, advertencias)
    pares_manuales = {frozenset(par) for par in partes_enlazadas}
    # un constraint manual sustituye a la unión automática entre las mismas dos piezas
    uniones = [u for u in uniones if frozenset((u.parte_esclava, u.parte_maestra)) not in pares_manuales] + ties_manuales
    flotantes = partes_sin_conexion(malla, uniones, partes_apoyadas, partes_enlazadas)
    if flotantes:
        # Un sólido suelto al que no se le asignó nada (apoyo, carga ni constraint) y que no toca a nada no participa
        # en la estructura (una placa de prueba, una pieza de otro conjunto...): se deja fija y se avisa.
        con_asignacion = {(a.elemento_tipo, a.elemento_id) for a in apoyos if isinstance(a, (ApoyoSuperficie, ApoyoArea))}
        for lista in cargas_area_por_caso.values():
            for c in lista:
                con_asignacion.add(("LOSA", c.losa_id) if c.losa_id else ("MURO", c.muro_id))
        con_asignacion |= {(c.elemento_tipo, c.elemento_id) for lista in cargas_superficie_por_caso.values() for c in lista}
        for u in uniones_manuales:
            con_asignacion |= {(u.a_tipo, u.a_id), (u.b_tipo, u.b_id)}
        sueltos = [i for i in flotantes if malla.partes[i].tipo in ("SOLIDO", "LOSA", "MURO")
                   and (malla.partes[i].tipo, malla.partes[i].id) not in con_asignacion]
        for i in sueltos:
            parte = malla.partes[i]
            nodos = malla.nodos_de_parte(parte)
            apoyos_3d.append(Apoyo3D(nodos, {1, 2, 3}, rigido=False, nombre=f"pieza suelta {parte.id}"))
        if sueltos:
            advertencias.append(
                f"Las piezas {', '.join(malla.partes[i].id for i in sueltos)} no tocan a ninguna otra ni tienen apoyos o "
                "cargas: no participan en la estructura y se dejaron fijos (sus esfuerzos no son significativos).")
        flotantes = [i for i in flotantes if i not in sueltos]
    if flotantes:
        nombres = ", ".join(malla.partes[i].id for i in flotantes)
        raise ErrorAnalisis(
            f"Hay piezas que no tocan a ninguna otra con apoyo (quedarían flotando): {nombres}. En el modelo sólido las "
            "piezas se unen por las caras que se tocan: revise que se apoyen unas en otras (o que haya apoyos en cada grupo); "
            "también puede unirlas con un constraint (Conectar).")

    materiales = {m: etiqueta(m) for m in {p.material for p in malla.partes}}
    modelo = ModeloSolido(malla=malla, uniones=uniones, apoyos=apoyos_3d, materiales=materiales, enlaces=enlaces)

    avanzar("Generando los archivos de CalculiX...")
    comun = escribir_comun(modelo, proyecto)
    caso_muerta = next((c.id for c in casos.values() if c.tipo == TipoCaso.D), None)
    avisos_cargas: List[str] = []
    rutas: Dict[str, Path] = {}
    directorio_trabajo.mkdir(parents=True, exist_ok=True)
    for caso_id in casos:
        cargas = _cargas_del_caso(
            proyecto, miembros, malla, caso_id, cargas_puntuales_por_caso.get(caso_id, []),
            cargas_lineales_por_caso.get(caso_id, []), cargas_area_por_caso.get(caso_id, []),
            caso_id == caso_muerta, avisos_cargas, cargas_superficie_por_caso.get(caso_id, []), tamano)
        ruta = escribir(directorio_trabajo / f"solido_{caso_id}.inp", escribir_caso(comun, cargas, caso_id))
        for sufijo in (".frd", ".dat"):
            ruta.with_suffix(sufijo).unlink(missing_ok=True)
        rutas[caso_id] = ruta
    advertencias.extend(dict.fromkeys(avisos_cargas))

    ejecuciones = ejecutar_ccx_todos_los_casos(
        rutas, ruta_ccx, tiempo_limite=TIEMPO_LIMITE_CCX,
        al_avanzar=lambda caso, i, total: avanzar(f"Resolviendo el caso {caso} ({i} de {total}) con CalculiX..."))
    fallidos = {c: r for c, r in ejecuciones.items() if not r.exitoso}
    if fallidos:
        detalle = "\n".join(f"Caso {c}: {r.diagnostico()}" for c, r in fallidos.items())
        raise ErrorAnalisis("CalculiX no pudo resolver el modelo sólido.\n" + detalle)

    avanzar("Leyendo los esfuerzos...")
    primer = next(iter(ejecuciones))
    malla_tensiones = leer_malla_frd(ejecuciones[primer].ruta_frd)
    tensiones_por_caso: Dict[str, np.ndarray] = {}
    desplazamientos_por_caso: Dict[str, np.ndarray] = {}
    for caso_id, ejecucion in ejecuciones.items():
        m_caso = leer_malla_frd(ejecucion.ruta_frd)
        if not np.array_equal(m_caso.ids_nodos, malla_tensiones.ids_nodos):
            raise ErrorAnalisis(f"La malla de resultados del caso {caso_id} no coincide con la de los demás casos.")
        tensiones_por_caso[caso_id] = leer_tensiones_frd(ejecucion.ruta_frd, malla_tensiones)
        desplazamientos_por_caso[caso_id] = leer_desplazamientos_frd(ejecucion.ruta_frd, malla_tensiones)

    # Solo los hexaedros del modelo (CalculiX no añade otros, pero los nodos de apoyo rígido no se dibujan)
    avanzar("Combinando casos de carga...")
    tensiones_por_combinacion = superponer_todas_las_combinaciones(tensiones_por_caso, list(combinaciones.values()))
    desplazamientos_por_combinacion = superponer_todas_las_combinaciones(desplazamientos_por_caso, list(combinaciones.values()))
    maximo = max((float(np.nanmax(esfuerzos.von_mises(t))) for t in tensiones_por_caso.values()), default=0.0)
    if not np.isfinite(maximo) or maximo > LIMITE_ESFUERZO_RAZONABLE_KPA:
        advertencias.append("Los esfuerzos obtenidos no son físicamente razonables: revise los apoyos y las uniones entre piezas.")
    advertencias.append(
        "Modelo sólido: las piezas se unen por las superficies que se tocan (como una soldadura continua); no se modelan "
        "pernos, contacto con separación ni fricción. Los esfuerzos junto a apoyos y cargas puntuales son locales y "
        "dependen del tamaño de elemento.")

    if elemento_solido == "TETRA" and len(malla.tetraedros):
        advertencias.insert(0, f"Los sólidos libres se mallaron con {len(malla.tetraedros):,} tetraedros cuadráticos (C3D10, Gmsh)."
                            .replace(",", " "))
    equivalente = MallaAnalisis()
    for i, id_ in enumerate(malla.elementos_en_orden, start=1):
        equivalente.elementos[i] = ElementoMalla(id=i, barra_id=id_, nodo_i=0, nodo_f=0, s_inicio=0.0, s_fin=1.0)
    return ResultadoAnalisis(
        malla=equivalente, malla_tensiones=malla_tensiones, tensiones_por_caso=tensiones_por_caso,
        tensiones_por_combinacion=tensiones_por_combinacion, advertencias=advertencias, es_solido=True,
        desplazamientos_por_caso=desplazamientos_por_caso, desplazamientos_por_combinacion=desplazamientos_por_combinacion,
    )
