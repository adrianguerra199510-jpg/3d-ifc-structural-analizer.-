"""Escritura del .inp de CalculiX del modelo sólido.

Elementos ``C3D8I`` (hexaedros con modos incompatibles: flexionan bien con un
solo elemento en el espesor de un ala o un alma) en barras, losas y muros, y
``C3D10`` (tetraedros cuadráticos de Gmsh, como en Placa Base Pro) en los sólidos libres, uniones ``*TIE`` entre
piezas, apoyos sobre las caras (cuerpo rígido con nodo de referencia cuando
se dejan giros libres) y cargas repartidas sobre los nodos de la sección o de
la cara cargada. El peso propio es ``*DLOAD,GRAV`` con ``*DENSITY`` igual al
peso específico (kN/m³) y aceleración 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from ..core.modelo import Proyecto
from .contactos import Union
from .malla import MallaSolida


@dataclass
class Apoyo3D:
    nodos: np.ndarray  # índices 0-based
    grados: Set[int]  # 1..6
    rigido: bool = False  # True: la cara se mueve como cuerpo rígido (con giros libres o restringidos aparte)
    nombre: str = ""
    directo: bool = False  # True: nodos de láminas, que tienen giros propios: se restringen los 6 grados directamente


@dataclass
class Enlace3D:
    """Constraint manual entre dos superficies (las soldaduras van como ``Union`` en ``*TIE``)."""

    tipo: str  # "CUERPO_RIGIDO", "IGUALAR" o "ARTICULACION"
    nodos_a: np.ndarray  # índices 0-based
    nodos_b: np.ndarray
    grados: Set[int]  # 1..6
    nombre: str = ""


@dataclass
class Acople:
    """Láminas acopladas a otra pieza: cada nodo ``a`` (de la lámina) sigue al nodo ``b`` (``*EQUATION``)."""

    nombre: str
    parte_a: int
    parte_b: int
    nodos_a: np.ndarray  # índices 0-based, uno por pareja
    nodos_b: np.ndarray
    giros: bool = False  # True si las dos piezas son láminas: también se igualan los giros


@dataclass
class ModeloSolido:
    malla: MallaSolida
    uniones: List[Union]
    apoyos: List[Apoyo3D]
    materiales: Dict[str, str]  # material del proyecto -> etiqueta ccx
    enlaces: List[Enlace3D] = field(default_factory=list)
    acoples: List[Acople] = field(default_factory=list)

    @property
    def n_nodos(self) -> int:
        return len(self.malla.coordenadas)


def etiqueta(nombre: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in nombre).upper()[:60]


def _numeros(valores: Sequence[int], por_linea: int = 8) -> List[str]:
    return [",".join(str(int(v)) for v in valores[i:i + por_linea]) for i in range(0, len(valores), por_linea)]


def escribir_comun(modelo: ModeloSolido, proyecto: Proyecto) -> List[str]:
    """Nodos, elementos, materiales, uniones y apoyos (iguales en todos los casos)."""

    malla = modelo.malla
    lineas = ["** Modelo sólido (generado por ifc_structural)", "*NODE"]
    c = malla.coordenadas
    lineas += [f"{i + 1},{x:.7g},{y:.7g},{z:.7g}" for i, (x, y, z) in enumerate(c)]

    por_material: Dict[str, List[int]] = {}
    por_material_tetra: Dict[str, List[int]] = {}
    for parte_i, parte in enumerate(malla.partes):
        por_material.setdefault(parte.material, []).extend(range(parte.primer_hexaedro, parte.ultimo_hexaedro))
        if parte.tetra:
            por_material_tetra.setdefault(parte.material, []).extend(range(parte.primer_tetraedro, parte.ultimo_tetraedro))
    por_lamina: Dict[Tuple[str, int], List[int]] = {}  # (material, espesor en 0,1 mm) -> láminas
    for parte in malla.partes:
        if parte.lamina:
            for k in range(parte.primer_lamina, parte.ultimo_lamina):
                por_lamina.setdefault((parte.material, int(round(float(malla.espesor_de_lamina[k]) * 1e4))), []).append(k)
    n_hex = len(malla.hexaedros)
    for material, hexas in por_material.items():
        if not hexas:
            continue
        elset = f"E_{modelo.materiales[material]}"
        lineas.append(f"*ELEMENT,TYPE=C3D8I,ELSET={elset}")
        lineas += [f"{h + 1}," + ",".join(str(int(n) + 1) for n in malla.hexaedros[h]) for h in hexas]
    for material, tetras in por_material_tetra.items():
        elset = f"T_{modelo.materiales[material]}"
        lineas.append(f"*ELEMENT,TYPE=C3D10,ELSET={elset}")
        lineas += [f"{n_hex + t + 1}," + ",".join(str(int(n) + 1) for n in malla.tetraedros[t]) for t in tetras]
    n_tetra = len(malla.tetraedros)
    for (material, decimas), laminas in por_lamina.items():
        elset = f"L_{modelo.materiales[material]}_{decimas}"
        lineas.append(f"*ELEMENT,TYPE=S6,ELSET={elset}")
        lineas += [f"{n_hex + n_tetra + k + 1}," + ",".join(str(int(n) + 1) for n in malla.laminas[k]) for k in laminas]
    for material in por_material:
        m = proyecto.materiales[material]
        etq = modelo.materiales[material]
        lineas += [f"*MATERIAL,NAME={etq}", "*ELASTIC", f"{m.modulo_elasticidad:.8g},{m.modulo_poisson:.5f}",
                   "*DENSITY", f"{m.densidad:.8g}"]
        if por_material.get(material):
            lineas.append(f"*SOLID SECTION,ELSET=E_{etq},MATERIAL={etq}")
        if material in por_material_tetra:
            lineas.append(f"*SOLID SECTION,ELSET=T_{etq},MATERIAL={etq}")
        for (m_lam, decimas) in por_lamina:
            if m_lam == material:
                lineas += [f"*SHELL SECTION,ELSET=L_{etq}_{decimas},MATERIAL={etq}", f"{decimas / 1e4:.6g}"]
    lineas += ["*ELSET,ELSET=EALL,GENERATE", f"1,{malla.n_elementos},1"]

    for u in modelo.uniones:
        for nombre, caras in ((f"{u.nombre}_E", u.esclava), (f"{u.nombre}_M", u.maestra)):
            lineas.append(f"*SURFACE,NAME={nombre},TYPE=ELEMENT")
            lineas += [f"{h + 1},S{cara}" for h, cara in caras]
        lineas += [f"*TIE,NAME={u.nombre},POSITION TOLERANCE={u.tolerancia:.6g},ADJUST=NO", f"{u.nombre}_E,{u.nombre}_M"]

    siguiente = modelo.n_nodos + 1
    condiciones: List[str] = []
    for k, apoyo in enumerate(modelo.apoyos):
        nodos = [int(n) + 1 for n in apoyo.nodos]
        traslaciones = sorted(g for g in apoyo.grados if g <= 3)
        giros = sorted(g - 3 for g in apoyo.grados if g > 3)
        if apoyo.directo:  # láminas: los 6 grados de libertad existen en cada nodo
            for g in sorted(apoyo.grados):
                condiciones += [f"{n},{g},{g}" for n in nodos]
            continue
        if not apoyo.rigido:
            for g in traslaciones:
                condiciones += [f"{n},{g},{g}" for n in nodos]
            continue
        # cuerpo rígido: la cara se mueve como una pieza con nodo de traslación y de giro libres
        ref, rot = siguiente, siguiente + 1
        siguiente += 2
        centro = c[apoyo.nodos].mean(axis=0)
        lineas += ["*NODE", f"{ref},{centro[0]:.7g},{centro[1]:.7g},{centro[2]:.7g}",
                   f"{rot},{centro[0]:.7g},{centro[1]:.7g},{centro[2]:.7g}", f"*NSET,NSET=APOYO{k}"]
        lineas += _numeros(nodos)
        lineas.append(f"*RIGID BODY,NSET=APOYO{k},REF NODE={ref},ROT NODE={rot}")
        condiciones += [f"{ref},{g},{g}" for g in traslaciones] + [f"{rot},{g},{g}" for g in giros]
    ecuaciones: List[str] = []
    apoyados = np.unique(np.concatenate([a.nodos for a in modelo.apoyos])) if modelo.apoyos else np.zeros(0, dtype=np.int64)
    for k, enlace in enumerate(modelo.enlaces):
        # un nodo con apoyo propio no puede además depender de un cuerpo rígido o una ecuación: se deja fuera
        enlace = Enlace3D(enlace.tipo, np.setdiff1d(enlace.nodos_a, apoyados), np.setdiff1d(enlace.nodos_b, apoyados),
                          enlace.grados, enlace.nombre)
        if len(enlace.nodos_a) == 0 or len(enlace.nodos_b) == 0:
            continue
        siguiente = _escribir_enlace(enlace, k, c, siguiente, lineas, ecuaciones)
    for acople in modelo.acoples:
        # un nodo con apoyo propio no puede además depender de otro: se deja fuera
        libres = ~np.isin(acople.nodos_a, apoyados)
        for g in (1, 2, 3, 4, 5, 6) if acople.giros else (1, 2, 3):
            for n, m in zip(acople.nodos_a[libres], acople.nodos_b[libres]):
                ecuaciones += ["2", f"{int(n) + 1},{g},1.,{int(m) + 1},{g},-1."]
    if ecuaciones:
        lineas.append("*EQUATION")
        lineas += ecuaciones
    if condiciones:
        lineas += ["*BOUNDARY"] + list(dict.fromkeys(condiciones))
    return lineas


def _cuerpo_rigido(nombre: str, nodos: np.ndarray, coordenadas: np.ndarray, siguiente: int, lineas: List[str]):
    """Escribe un ``*RIGID BODY`` con nodo de referencia y de giro en el centro. Devuelve (ref, rot, siguiente)."""

    ref, rot = siguiente, siguiente + 1
    centro = coordenadas[nodos].mean(axis=0)
    lineas += ["*NODE", f"{ref},{centro[0]:.7g},{centro[1]:.7g},{centro[2]:.7g}",
               f"{rot},{centro[0]:.7g},{centro[1]:.7g},{centro[2]:.7g}", f"*NSET,NSET={nombre}"]
    lineas += _numeros([int(n) + 1 for n in nodos])
    lineas.append(f"*RIGID BODY,NSET={nombre},REF NODE={ref},ROT NODE={rot}")
    return ref, rot, siguiente + 2


def _escribir_enlace(enlace: Enlace3D, k: int, c: np.ndarray, siguiente: int, lineas: List[str], ecuaciones: List[str]) -> int:
    from scipy.spatial import cKDTree

    a, b = np.unique(enlace.nodos_a), np.unique(enlace.nodos_b)
    if enlace.tipo == "CUERPO_RIGIDO":
        _, _, siguiente = _cuerpo_rigido(f"CR{k}", np.union1d(a, b), c, siguiente, lineas)
    elif enlace.tipo == "ARTICULACION":
        ref_a, rot_a, siguiente = _cuerpo_rigido(f"ARA{k}", a, c, siguiente, lineas)
        ref_b, rot_b, siguiente = _cuerpo_rigido(f"ARB{k}", b, c, siguiente, lineas)
        for g in sorted(enlace.grados):
            na, nb = (ref_a, ref_b) if g <= 3 else (rot_a, rot_b)
            gl = g if g <= 3 else g - 3
            ecuaciones += ["2", f"{na},{gl},1.,{nb},{gl},-1."]
    elif enlace.tipo == "IGUALAR":
        # cada nodo de la superficie con menos nodos sigue al nodo más cercano de la otra
        corto, largo = (a, b) if len(a) <= len(b) else (b, a)
        _, vecino = cKDTree(c[largo]).query(c[corto])
        for g in sorted(g for g in enlace.grados if g <= 3):
            for n, m in zip(corto, largo[vecino]):
                ecuaciones += ["2", f"{int(n) + 1},{g},1.,{int(m) + 1},{g},-1."]
    return siguiente


@dataclass
class CargasCaso:
    nodales: Dict[int, np.ndarray] = field(default_factory=dict)  # nodo 0-based -> (fx, fy, fz)
    peso_propio: bool = False

    def sumar(self, nodos: np.ndarray, fuerzas: np.ndarray) -> None:
        for n, f in zip(nodos, fuerzas):
            acumulada = self.nodales.setdefault(int(n), np.zeros(3))
            acumulada += f


def escribir_caso(comun: List[str], cargas: CargasCaso, caso_id: str) -> List[str]:
    lineas = list(comun)
    lineas += ["*STEP", "*STATIC"]
    if cargas.nodales:
        lineas.append("*CLOAD")
        for n in sorted(cargas.nodales):
            for g, valor in enumerate(cargas.nodales[n], start=1):
                if abs(valor) > 1e-12:
                    lineas.append(f"{n + 1},{g},{valor:.8g}")
    if cargas.peso_propio:
        lineas += ["*DLOAD", "EALL,GRAV,1.0,0.0,0.0,-1.0"]
    lineas += ["*NODE FILE", "U", "*EL FILE", "S", "*END STEP"]
    return lineas


def escribir(ruta: Path, lineas: List[str]) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta
