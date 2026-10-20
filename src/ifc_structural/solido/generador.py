"""Escritura del .inp de CalculiX del modelo sólido.

Elementos ``C3D8I`` (hexaedros con modos incompatibles: flexionan bien con un
solo elemento en el espesor de un ala o un alma), uniones ``*TIE`` entre
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


@dataclass
class ModeloSolido:
    malla: MallaSolida
    uniones: List[Union]
    apoyos: List[Apoyo3D]
    materiales: Dict[str, str]  # material del proyecto -> etiqueta ccx

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
    for parte_i, parte in enumerate(malla.partes):
        por_material.setdefault(parte.material, []).extend(range(parte.primer_hexaedro, parte.ultimo_hexaedro))
    for material, hexas in por_material.items():
        elset = f"E_{modelo.materiales[material]}"
        lineas.append(f"*ELEMENT,TYPE=C3D8I,ELSET={elset}")
        lineas += [f"{h + 1}," + ",".join(str(int(n) + 1) for n in malla.hexaedros[h]) for h in hexas]
    for material in por_material:
        m = proyecto.materiales[material]
        etq = modelo.materiales[material]
        lineas += [f"*MATERIAL,NAME={etq}", "*ELASTIC", f"{m.modulo_elasticidad:.8g},{m.modulo_poisson:.5f}",
                   "*DENSITY", f"{m.densidad:.8g}", f"*SOLID SECTION,ELSET=E_{etq},MATERIAL={etq}"]
    lineas += ["*ELSET,ELSET=EALL,GENERATE", f"1,{len(malla.hexaedros)},1"]

    for u in modelo.uniones:
        for nombre, caras in ((f"{u.nombre}_E", u.esclava), (f"{u.nombre}_M", u.maestra)):
            lineas.append(f"*SURFACE,NAME={nombre},TYPE=ELEMENT")
            lineas += [f"{h + 1},S{cara}" for h, cara in caras]
        lineas += [f"*TIE,NAME={u.nombre},POSITION TOLERANCE={u.tolerancia:.6g}", f"{u.nombre}_E,{u.nombre}_M"]

    siguiente = modelo.n_nodos + 1
    condiciones: List[str] = []
    for k, apoyo in enumerate(modelo.apoyos):
        nodos = [int(n) + 1 for n in apoyo.nodos]
        traslaciones = sorted(g for g in apoyo.grados if g <= 3)
        giros = sorted(g - 3 for g in apoyo.grados if g > 3)
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
    if condiciones:
        lineas += ["*BOUNDARY"] + condiciones
    return lineas


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
    lineas += ["*EL FILE", "S", "*END STEP"]
    return lineas


def escribir(ruta: Path, lineas: List[str]) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta
