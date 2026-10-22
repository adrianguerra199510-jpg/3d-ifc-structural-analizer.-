"""Escritura del .inp de CalculiX del modelo sólido.

Elementos ``C3D10`` (tetraedros cuadráticos, como en Placa Base Pro), ``C3D20R`` (hexaedros cuadráticos barridos de las
piezas prismáticas) y ``S6`` (láminas) en el análisis; ``C3D8I``
(hexaedros con modos incompatibles) solo si la malla aún tiene hexaedros. Uniones ``*TIE`` entre piezas, acoples
``*EQUATION`` de las láminas, apoyos sobre las caras (cuerpo rígido con nodo de referencia cuando se dejan giros
libres) y cargas repartidas sobre los nodos de la sección o de la cara cargada. El peso propio es ``*DLOAD,GRAV`` con ``*DENSITY`` igual al
peso específico (kN/m³) y aceleración 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
    # caras de cada superficie (nodos de cada cara: 3/6 triángulos, 4/8 cuadriláteros): IGUALAR interpola en ellas
    caras_a: Optional[np.ndarray] = None
    caras_b: Optional[np.ndarray] = None


@dataclass
class Acople:
    """Láminas acopladas a otra pieza: cada nodo ``a`` (de la lámina) sigue al nodo ``b`` (``*EQUATION``)."""

    nombre: str
    parte_a: int
    parte_b: int
    nodos_a: np.ndarray  # índices 0-based, uno por pareja
    nodos_b: np.ndarray
    giros: bool = False  # True si las dos piezas son láminas: también se igualan los giros
    # Lámina unida a un sólido: el giro de cada nodo de la lámina sigue al giro del parche de nodos del sólido que lo
    # rodea (ajuste de cuerpo rígido por mínimos cuadrados), para que la unión transmita también momentos.
    # Una entrada por ecuación: [(nodo 0-based, grado 1..6, coeficiente), ...]; el primer término es el dependiente.
    ecuaciones_giro: List[List[Tuple[int, int, float]]] = field(default_factory=list)


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
    por_material_hex20: Dict[str, List[int]] = {}
    for parte_i, parte in enumerate(malla.partes):
        por_material.setdefault(parte.material, []).extend(range(parte.primer_hexaedro, parte.ultimo_hexaedro))
        if parte.hex20:
            por_material_hex20.setdefault(parte.material, []).extend(range(parte.primer_hex20, parte.ultimo_hex20))
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
    for material, hexas in por_material_hex20.items():
        lineas.append(f"*ELEMENT,TYPE=C3D20R,ELSET=H_{modelo.materiales[material]}")
        for h in hexas:  # 20 nodos: en dos líneas (CalculiX admite 16 datos por línea)
            nodos = [str(int(n) + 1) for n in malla.hexaedros20[h]]
            lineas += [f"{malla.inicio_hex20 + h + 1}," + ",".join(nodos[:15]) + ",", ",".join(nodos[15:])]
    for material in por_material:
        m = proyecto.materiales[material]
        etq = modelo.materiales[material]
        lineas += [f"*MATERIAL,NAME={etq}", "*ELASTIC", f"{m.modulo_elasticidad:.8g},{m.modulo_poisson:.5f}",
                   "*DENSITY", f"{m.densidad:.8g}"]
        if por_material.get(material):
            lineas.append(f"*SOLID SECTION,ELSET=E_{etq},MATERIAL={etq}")
        if material in por_material_tetra:
            lineas.append(f"*SOLID SECTION,ELSET=T_{etq},MATERIAL={etq}")
        if material in por_material_hex20:
            lineas.append(f"*SOLID SECTION,ELSET=H_{etq},MATERIAL={etq}")
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
    ecuaciones: List[List[Tuple[int, int, float]]] = []  # términos (nodo 1-based, grado, coeficiente); el primero depende
    apoyados = np.unique(np.concatenate([a.nodos for a in modelo.apoyos])) if modelo.apoyos else np.zeros(0, dtype=np.int64)
    for k, enlace in enumerate(modelo.enlaces):
        # un nodo con apoyo propio no puede además depender de un cuerpo rígido o una ecuación: se deja fuera
        enlace = Enlace3D(enlace.tipo, np.setdiff1d(enlace.nodos_a, apoyados), np.setdiff1d(enlace.nodos_b, apoyados),
                          enlace.grados, enlace.nombre, enlace.caras_a, enlace.caras_b)
        if len(enlace.nodos_a) == 0 or len(enlace.nodos_b) == 0:
            continue
        siguiente = _escribir_enlace(enlace, k, c, siguiente, lineas, ecuaciones)
    for acople in modelo.acoples:
        # un nodo con apoyo propio no puede además depender de otro: se deja fuera
        libres = ~np.isin(acople.nodos_a, apoyados)
        for g in (1, 2, 3, 4, 5, 6) if acople.giros else (1, 2, 3):
            for n, m in zip(acople.nodos_a[libres], acople.nodos_b[libres]):
                ecuaciones.append([(int(n) + 1, g, 1.0), (int(m) + 1, g, -1.0)])
        for terminos in acople.ecuaciones_giro:
            ecuaciones.append([(n + 1, g, c) for n, g, c in terminos])
    lineas += _ecuaciones_validas(ecuaciones, lineas, condiciones)
    if condiciones:
        lineas += ["*BOUNDARY"] + list(dict.fromkeys(condiciones))
    return lineas


def _ecuaciones_validas(ecuaciones: List[List[Tuple[int, int, float]]], lineas: List[str], condiciones: List[str]) -> List[str]:
    """``*EQUATION`` sin conflictos: CalculiX aborta si un mismo grado de libertad es el dependiente de dos ecuaciones
    (o de una ecuación y un ``*RIGID BODY``) o si además tiene un apoyo. Pasa cuando un nodo de lámina queda a la vez en
    un acople, un constraint (igualar, cuerpo rígido, articulación) o junto a un apoyo; la ecuación repetida se omite."""

    ocupados = set()
    for condicion in condiciones:  # apoyos: "nodo,grado,grado"
        n, g, _ = condicion.split(",")
        ocupados.add((int(n), int(g)))
    nset, en_nset = {}, None  # nodos de cada *RIGID BODY: dependen en sus 6 grados
    for linea in lineas:
        if linea.startswith("*"):
            en_nset = linea.split("NSET=")[1].strip() if linea.upper().startswith("*NSET") else None
            if linea.upper().startswith("*RIGID BODY"):
                nombre = linea.split("NSET=")[1].split(",")[0].strip()
                ocupados.update((n, g) for n in nset.get(nombre, ()) for g in range(1, 7))
        elif en_nset is not None:
            nset.setdefault(en_nset, []).extend(int(x) for x in linea.split(",") if x.strip())
    texto: List[str] = []
    for terminos in ecuaciones:
        dependiente = terminos[0][:2]
        if dependiente in ocupados or any(t[:2] == dependiente for t in terminos[1:]):
            continue
        ocupados.add(dependiente)
        texto.append(str(len(terminos)))
        partes = [f"{n},{g},{c:.10g}" for n, g, c in terminos]
        texto += [",".join(partes[k:k + 4]) for k in range(0, len(partes), 4)]
    return ["*EQUATION"] + texto if texto else []


def _cuerpo_rigido(nombre: str, nodos: np.ndarray, coordenadas: np.ndarray, siguiente: int, lineas: List[str]):
    """Escribe un ``*RIGID BODY`` con nodo de referencia y de giro en el centro. Devuelve (ref, rot, siguiente)."""

    ref, rot = siguiente, siguiente + 1
    centro = coordenadas[nodos].mean(axis=0)
    lineas += ["*NODE", f"{ref},{centro[0]:.7g},{centro[1]:.7g},{centro[2]:.7g}",
               f"{rot},{centro[0]:.7g},{centro[1]:.7g},{centro[2]:.7g}", f"*NSET,NSET={nombre}"]
    lineas += _numeros([int(n) + 1 for n in nodos])
    lineas.append(f"*RIGID BODY,NSET={nombre},REF NODE={ref},ROT NODE={rot}")
    return ref, rot, siguiente + 2


def _escribir_enlace(enlace: Enlace3D, k: int, c: np.ndarray, siguiente: int, lineas: List[str], ecuaciones: list) -> int:
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
            ecuaciones.append([(na, gl, 1.0), (nb, gl, -1.0)])
    elif enlace.tipo == "IGUALAR":
        grados = sorted(g for g in enlace.grados if g <= 3)
        # cada nodo de la superficie con más nodos (la más fina) sigue al punto más cercano de la otra, interpolado con las
        # funciones de forma de su cara (como *TIE): con mallas distintas a cada lado, seguir al nodo más cercano deja
        # nodos sin unir y ablanda la unión
        esclava, maestra, caras_maestra = (a, b, enlace.caras_b) if len(a) >= len(b) else (b, a, enlace.caras_a)
        if caras_maestra is not None and np.asarray(caras_maestra).ndim == 2 and np.asarray(caras_maestra).shape[1] in (3, 4, 6, 8):
            for n, (nodos, pesos) in zip(esclava, interpolar_en_caras(c, c[esclava], np.asarray(caras_maestra))):
                if int(n) in set(nodos.tolist()):
                    continue
                for g in grados:
                    ecuaciones.append([(int(n) + 1, g, 1.0)] + [(int(m) + 1, g, -float(w)) for m, w in zip(nodos, pesos)])
        else:  # sin caras (canto de una lámina): cada nodo sigue al nodo más cercano de la otra
            corto, largo = (a, b) if len(a) <= len(b) else (b, a)
            _, vecino = cKDTree(c[largo]).query(c[corto])
            for g in grados:
                for n, m in zip(corto, largo[vecino]):
                    ecuaciones.append([(int(n) + 1, g, 1.0), (int(m) + 1, g, -1.0)])
    return siguiente


def _formas(k: int, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Funciones de forma (n, k) de una cara de ``k`` nodos en coordenadas naturales: triángulos (``xi``, ``eta``
    baricéntricas de los vértices 2 y 3) de 3 o 6 nodos, cuadriláteros (−1..1) de 4 u 8 nodos (serendípita)."""

    if k in (3, 6):
        l1, l2, l3 = 1.0 - xi - eta, xi, eta
        if k == 3:
            return np.stack([l1, l2, l3], axis=1)
        return np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], axis=1)
    sx, sy = np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0])
    lineales = 0.25 * (1 + xi[:, None] * sx) * (1 + eta[:, None] * sy)
    if k == 4:
        return lineales
    esquinas = lineales * (xi[:, None] * sx + eta[:, None] * sy - 1)
    medios = np.stack([0.5 * (1 - xi**2) * (1 - eta), 0.5 * (1 + xi) * (1 - eta**2), 0.5 * (1 - xi**2) * (1 + eta),
                       0.5 * (1 - xi) * (1 - eta**2)], axis=1)
    return np.hstack([esquinas, medios])


def _proyectar_en_caras(c: np.ndarray, puntos: np.ndarray, caras: np.ndarray):
    """Coordenadas naturales (recortadas a la cara) del punto de cada cara más cercano a cada punto, y su distancia."""

    k = caras.shape[1]
    p = c[caras[:, :3 if k in (3, 6) else 4]]
    if k in (3, 6):  # baricéntricas en el plano del triángulo
        a, ab, ac = p[:, 0], p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        ap = puntos - a
        d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
        d20, d21 = (ap * ab).sum(1), (ap * ac).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
        xi, eta = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        xi, eta = np.clip(xi, 0.0, 1.0), np.clip(eta, 0.0, 1.0)
        exceso = np.maximum(xi + eta - 1.0, 0.0)
        xi, eta = xi - 0.5 * exceso, eta - 0.5 * exceso
    else:  # cuadrilátero de lados rectos: Newton sobre la aplicación bilineal de sus vértices
        xi, eta = np.zeros(len(puntos)), np.zeros(len(puntos))
        sx, sy = np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0])
        for _ in range(8):
            n = 0.25 * (1 + xi[:, None] * sx) * (1 + eta[:, None] * sy)
            dxi = 0.25 * sx * (1 + eta[:, None] * sy)
            deta = 0.25 * sy * (1 + xi[:, None] * sx)
            x = np.einsum("nk,nkd->nd", n, p)
            tx, ty = np.einsum("nk,nkd->nd", dxi, p), np.einsum("nk,nkd->nd", deta, p)
            r = puntos - x
            a11, a12, a22 = (tx * tx).sum(1), (tx * ty).sum(1), (ty * ty).sum(1)
            b1, b2 = (r * tx).sum(1), (r * ty).sum(1)
            det = np.maximum(a11 * a22 - a12 * a12, 1e-300)
            xi = np.clip(xi + (a22 * b1 - a12 * b2) / det, -1.0, 1.0)
            eta = np.clip(eta + (a11 * b2 - a12 * b1) / det, -1.0, 1.0)
    formas = _formas(4 if k in (4, 8) else 3, xi, eta)
    x = np.einsum("nk,nkd->nd", formas, p)
    return xi, eta, np.linalg.norm(puntos - x, axis=1)


def interpolar_en_caras(c: np.ndarray, puntos: np.ndarray, caras: np.ndarray, candidatas: int = 8):
    """Para cada punto, (nodos, pesos) del punto más cercano de las caras: los nodos de esa cara y sus funciones de forma
    allí (los pesos de valor casi nulo se omiten)."""

    from scipy.spatial import cKDTree

    k = caras.shape[1]
    vertices = caras[:, :3 if k in (3, 6) else 4]
    centros = c[vertices].mean(axis=1)
    m = min(candidatas, len(caras))
    _, cercanas = cKDTree(centros).query(puntos, k=m)
    cercanas = np.asarray(cercanas).reshape(len(puntos), m)
    mejor = np.full(len(puntos), np.inf)
    eleccion = np.zeros(len(puntos), dtype=np.int64)
    coords = np.zeros((len(puntos), 2))
    for columna in range(m):
        f = cercanas[:, columna]
        xi, eta, d = _proyectar_en_caras(c, puntos, caras[f])
        mas_cerca = d < mejor - 1e-12
        mejor[mas_cerca], eleccion[mas_cerca] = d[mas_cerca], f[mas_cerca]
        coords[mas_cerca] = np.stack([xi, eta], axis=1)[mas_cerca]
    formas = _formas(k, coords[:, 0], coords[:, 1])
    resultado = []
    for i in range(len(puntos)):
        usar = np.abs(formas[i]) > 1e-9
        resultado.append((caras[eleccion[i]][usar], formas[i][usar]))
    return resultado


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
