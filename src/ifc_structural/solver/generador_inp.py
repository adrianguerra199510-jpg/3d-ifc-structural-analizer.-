"""Generación del archivo .inp de CalculiX para un caso de carga.

Este módulo encapsula varias decisiones de implementación que se
comprobaron de forma empírica contra el CalculiX realmente instalado
(versión 2.21, paquete `calculix-ccx` de Ubuntu), porque el comportamiento
real difiere en varios puntos de lo que sugiere la redacción "clásica" de
la documentación:

1. **Secciones**: ``*BEAM SECTION, SECTION=GENERAL`` (y su variante
   obsoleta ``*BEAM GENERAL SECTION``) NO funciona con elementos B31 en
   esta versión de CalculiX: el programa exige que el elemento sea de tipo
   "U1", que no existe como elemento utilizable. Por eso aquí siempre se
   usa ``SECTION=RECT``. Si el perfil importado del IFC no es rectangular
   (o no se conocen su altura y ancho reales), se aproxima mediante un
   **rectángulo equivalente** que conserva el área y la inercia fuerte
   (eje z) del perfil real — ver ``core.geometria.dimensiones_seccion``.

2. **Cargas lineales**: se comprobó que, para elementos de viga (B31), las
   claves de carga distribuida global ``PX``/``PY``/``PZ`` NO son válidas
   en esta versión de CalculiX (error de lectura de ``*DLOAD``), y que las
   claves locales ``P1``/``P2`` solo representan cargas transversales (no
   sirven para una carga axial, como el peso propio de una columna
   vertical). Por eso **todas** las cargas lineales (peso propio,
   manuales y transferidas desde losas) se aplican mediante
   ``*DLOAD, GRAV`` con una densidad ficticia (``*DENSITY`` = 1.0, que NO
   tiene relación con la densidad real del material: se usa únicamente
   como factor de escala) y una magnitud = (carga lineal real) / (densidad
   ficticia × área de la sección), en la dirección global de la carga.
   Esto permite representar cualquier carga lineal, en cualquier
   dirección global, de forma exacta y uniforme para todos los casos.

3. **Resultados (esfuerzos)**: CalculiX expande internamente cada viga
   B31 en hexaedros (``C3D8I``). Con ``*EL FILE`` + ``S`` entrega, en el
   .frd (ASCII), el tensor de esfuerzos en cada nodo de esa malla
   expandida, junto con sus coordenadas y conectividad. Es lo que se usa
   para dibujar mapas de contorno de esfuerzos sobre las vigas y
   columnas. Los valores en los extremos de cada tramo son
   extrapolaciones desde los puntos de integración, por lo que son menos
   precisos que los interiores; un mallado fino (ver
   ``LONGITUD_ELEMENTO_OBJETIVO``) lo mitiga.

4. **Limitación conocida de ccx**: la expansión de la viga falla ("zero
   coefficient on the dependent side of an equation") con algunas
   orientaciones de barras inclinadas cuando uno de sus nodos tiene los
   giros restringidos. La orientación de 45° exactos en un plano
   coordenado, la más común, se evita en ``core.geometria``; para el
   resto se informa al usuario con un mensaje claro (ver
   ``solver.ejecutor_ccx``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.cargas import Apoyo, CargaLineal, CargaPuntual
from ..core.geometria import ejes_seccion
from ..core.modelo import Proyecto

Vector3 = Tuple[float, float, float]

#: Longitud máxima objetivo (m) de cada elemento B31 de la malla de análisis.
#: Cada tramo continuo de una viga/columna (entre dos "quiebres" de carga)
#: se divide en tantos elementos como hagan falta para no superarla; da
#: buena resolución a los mapas de esfuerzos y reduce el error de
#: extrapolación en los extremos.
LONGITUD_ELEMENTO_OBJETIVO = 0.25

#: Densidad ficticia usada únicamente para escalar *DLOAD,GRAV (ver punto 2
#: del docstring del módulo). No representa la densidad real del material.
DENSIDAD_FICTICIA = 1.0

_TOL = 1e-9


def _norma(v: Vector3) -> float:
    return math.sqrt(sum(c * c for c in v))


@dataclass
class NodoMalla:
    id: int
    x: float
    y: float
    z: float
    nodo_original: Optional[str] = None


@dataclass
class ElementoMalla:
    id: int
    barra_id: str
    nodo_i: int
    nodo_f: int
    s_inicio: float
    s_fin: float


@dataclass
class MallaAnalisis:
    """Malla de elementos finitos (B31) derivada del modelo de barras.

    Cada viga/columna del ``Proyecto`` se traduce en una cadena de varios
    ``ElementoMalla``, para poder aplicar cargas no uniformes y obtener
    mapas de esfuerzos con buena resolución. ``nodos`` y ``elementos``
    usan etiquetas enteras (requeridas por CalculiX).
    """

    nodos: Dict[int, NodoMalla] = field(default_factory=dict)
    elementos: Dict[int, ElementoMalla] = field(default_factory=dict)
    nodo_ccx_de_nodo_original: Dict[str, int] = field(default_factory=dict)
    nodo_ccx_de_posicion: Dict[Tuple[str, float], int] = field(default_factory=dict)
    elementos_por_barra: Dict[str, List[int]] = field(default_factory=dict)
    orden_nodos_por_barra: Dict[str, List[int]] = field(default_factory=dict)
    posiciones_s_por_barra: Dict[str, List[float]] = field(default_factory=dict)


def construir_malla(
    proyecto: Proyecto,
    cargas_lineales_por_caso: Dict[str, List[CargaLineal]],
    cargas_puntuales_por_caso: Dict[str, List[CargaPuntual]],
    longitud_elemento: float = LONGITUD_ELEMENTO_OBJETIVO,
) -> MallaAnalisis:
    """Construye la malla de análisis, común a todos los casos de carga.

    Usar la MISMA malla para todos los casos es imprescindible para que la
    superposición de resultados (ver combinaciones/motor_superposicion.py)
    pueda comparar nodo a nodo entre casos.
    """

    malla = MallaAnalisis()
    contador = {"nodo": 0, "elemento": 0}

    def nuevo_id(clave: str) -> int:
        contador[clave] += 1
        return contador[clave]

    for barra_id, barra in proyecto.todas_las_barras().items():
        longitud_barra = proyecto.longitud_barra(barra_id)
        breakpoints = {0.0, 1.0}
        for cargas in cargas_lineales_por_caso.values():
            for cl in cargas:
                if cl.elemento_id == barra_id:
                    breakpoints.add(round(cl.s_inicio, 6))
                    breakpoints.add(round(cl.s_fin, 6))
        for cargas_p in cargas_puntuales_por_caso.values():
            for cp in cargas_p:
                if cp.elemento_id == barra_id and cp.posicion is not None:
                    breakpoints.add(round(cp.posicion, 6))

        puntos_quiebre = sorted(breakpoints)
        posiciones: List[float] = [puntos_quiebre[0]]
        for i in range(len(puntos_quiebre) - 1):
            s0, s1 = puntos_quiebre[i], puntos_quiebre[i + 1]
            if s1 - s0 < 1e-7:
                continue
            n_sub = max(2, math.ceil((s1 - s0) * longitud_barra / longitud_elemento))
            for k in range(1, n_sub + 1):
                posiciones.append(s0 + (s1 - s0) * k / n_sub)

        nodo_ini = proyecto.nodos[barra.nodo_inicio]
        nodo_fin = proyecto.nodos[barra.nodo_fin]

        ids_nodo: List[int] = []
        for s in posiciones:
            if s <= _TOL:
                id_ccx = malla.nodo_ccx_de_nodo_original.get(barra.nodo_inicio)
                if id_ccx is None:
                    id_ccx = nuevo_id("nodo")
                    malla.nodos[id_ccx] = NodoMalla(id_ccx, nodo_ini.x, nodo_ini.y, nodo_ini.z, barra.nodo_inicio)
                    malla.nodo_ccx_de_nodo_original[barra.nodo_inicio] = id_ccx
            elif s >= 1.0 - _TOL:
                id_ccx = malla.nodo_ccx_de_nodo_original.get(barra.nodo_fin)
                if id_ccx is None:
                    id_ccx = nuevo_id("nodo")
                    malla.nodos[id_ccx] = NodoMalla(id_ccx, nodo_fin.x, nodo_fin.y, nodo_fin.z, barra.nodo_fin)
                    malla.nodo_ccx_de_nodo_original[barra.nodo_fin] = id_ccx
            else:
                clave = (barra_id, round(s, 6))
                id_ccx = malla.nodo_ccx_de_posicion.get(clave)
                if id_ccx is None:
                    id_ccx = nuevo_id("nodo")
                    x = nodo_ini.x + s * (nodo_fin.x - nodo_ini.x)
                    y = nodo_ini.y + s * (nodo_fin.y - nodo_ini.y)
                    z = nodo_ini.z + s * (nodo_fin.z - nodo_ini.z)
                    malla.nodos[id_ccx] = NodoMalla(id_ccx, x, y, z, None)
                    malla.nodo_ccx_de_posicion[clave] = id_ccx
            ids_nodo.append(id_ccx)

        malla.orden_nodos_por_barra[barra_id] = ids_nodo
        malla.posiciones_s_por_barra[barra_id] = posiciones
        elementos_barra: List[int] = []
        for i in range(len(ids_nodo) - 1):
            id_el = nuevo_id("elemento")
            malla.elementos[id_el] = ElementoMalla(
                id=id_el,
                barra_id=barra_id,
                nodo_i=ids_nodo[i],
                nodo_f=ids_nodo[i + 1],
                s_inicio=posiciones[i],
                s_fin=posiciones[i + 1],
            )
            elementos_barra.append(id_el)
        malla.elementos_por_barra[barra_id] = elementos_barra

    return malla


def _rangos_contiguos(numeros: List[int]) -> List[Tuple[int, int]]:
    """Agrupa una lista de enteros ordenables en rangos contiguos, para
    escribir menos líneas de *BOUNDARY (p. ej. [1,2,3,4,5,6] -> [(1,6)])."""

    numeros = sorted(numeros)
    rangos: List[Tuple[int, int]] = []
    for n in numeros:
        if rangos and n == rangos[-1][1] + 1:
            rangos[-1] = (rangos[-1][0], n)
        else:
            rangos.append((n, n))
    return rangos


def _etiqueta_ccx(nombre: str) -> str:
    """Sanea un nombre para usarlo como etiqueta de material/ELSET en el
    .inp (CalculiX no admite espacios ni comas en las etiquetas)."""

    return "".join(c if c.isalnum() else "_" for c in nombre).upper()[:80]


def _carga_global_en_elemento(elemento: ElementoMalla, cargas: List[CargaLineal]) -> Vector3:
    """Fuerza total por unidad de longitud (vector global, kN/m) sobre un
    sub-elemento, sumando todas las CargaLineal del caso que lo cubren."""

    s_medio = (elemento.s_inicio + elemento.s_fin) / 2.0
    total = [0.0, 0.0, 0.0]
    for cl in cargas:
        if cl.elemento_id != elemento.barra_id:
            continue
        if not (cl.s_inicio - 1e-6 <= s_medio <= cl.s_fin + 1e-6):
            continue
        if cl.s_fin - cl.s_inicio < 1e-9:
            w = cl.w_inicio
        else:
            t = (s_medio - cl.s_inicio) / (cl.s_fin - cl.s_inicio)
            w = cl.w_inicio + t * (cl.w_fin - cl.w_inicio)
        for i in range(3):
            total[i] += w * cl.direccion[i]
    return (total[0], total[1], total[2])


def generar_inp_caso(
    proyecto: Proyecto,
    malla: MallaAnalisis,
    caso_id: str,
    cargas_lineales: List[CargaLineal],
    cargas_puntuales: List[CargaPuntual],
    apoyos: List[Apoyo],
    ruta_salida: Path,
) -> Path:
    """Escribe el .inp de CalculiX correspondiente a un único caso de carga."""

    lineas: List[str] = [f"** Caso de carga: {caso_id} (generado por ifc_structural)"]

    lineas.append("*NODE")
    for nid in sorted(malla.nodos):
        n = malla.nodos[nid]
        lineas.append(f"{nid},{n.x:.6f},{n.y:.6f},{n.z:.6f}")

    for barra_id, elementos_ids in malla.elementos_por_barra.items():
        lineas.append(f"*ELEMENT,TYPE=B31,ELSET=E_{barra_id}")
        for eid in elementos_ids:
            el = malla.elementos[eid]
            lineas.append(f"{eid},{el.nodo_i},{el.nodo_f}")

    for nombre, material in proyecto.materiales.items():
        lineas.append(f"*MATERIAL,NAME={_etiqueta_ccx(nombre)}")
        lineas.append("*ELASTIC")
        lineas.append(f"{material.modulo_elasticidad:.6g},{material.modulo_poisson:.4f}")
        lineas.append("*DENSITY")
        lineas.append(f"{DENSIDAD_FICTICIA}")

    areas: Dict[str, float] = {}
    for barra_id, barra in proyecto.todas_las_barras().items():
        ejes = ejes_seccion(proyecto, barra)
        areas[barra_id] = ejes.dim1 * ejes.dim2
        lineas.append(
            f"*BEAM SECTION,ELSET=E_{barra_id},MATERIAL={_etiqueta_ccx(barra.material)},SECTION=RECT"
        )
        lineas.append(f"{ejes.dim1:.6g},{ejes.dim2:.6g}")
        lineas.append(f"{ejes.e1[0]:.9f},{ejes.e1[1]:.9f},{ejes.e1[2]:.9f}")

    apoyos_validos = [a for a in apoyos if a.nodo_id in malla.nodo_ccx_de_nodo_original]
    if apoyos_validos:
        lineas.append("*BOUNDARY")
        for apoyo in apoyos_validos:
            nid = malla.nodo_ccx_de_nodo_original[apoyo.nodo_id]
            for ini, fin in _rangos_contiguos(apoyo.grados_restringidos()):
                lineas.append(f"{nid},{ini},{fin}")

    lineas.append("*STEP")
    lineas.append("*STATIC")

    lineas_cload: List[str] = []
    for c in cargas_puntuales:
        if c.nodo_id is not None:
            nid = malla.nodo_ccx_de_nodo_original.get(c.nodo_id)
        else:
            nid = malla.nodo_ccx_de_posicion.get((c.elemento_id, round(c.posicion, 6)))
        if nid is None:
            continue
        for dof, valor in zip(range(1, 7), (c.fx, c.fy, c.fz, c.mx, c.my, c.mz)):
            if abs(valor) > 1e-12:
                lineas_cload.append(f"{nid},{dof},{valor:.6g}")
    if lineas_cload:
        lineas.append("*CLOAD")
        lineas.extend(lineas_cload)

    lineas_dload: List[str] = []
    for barra_id, elementos_ids in malla.elementos_por_barra.items():
        area = areas[barra_id]
        for eid in elementos_ids:
            elemento = malla.elementos[eid]
            vector = _carga_global_en_elemento(elemento, cargas_lineales)
            magnitud = _norma(vector)
            if magnitud < 1e-9:
                continue
            direccion = (vector[0] / magnitud, vector[1] / magnitud, vector[2] / magnitud)
            magnitud_grav = magnitud / (DENSIDAD_FICTICIA * area)
            lineas_dload.append(
                f"{eid},GRAV,{magnitud_grav:.6g},"
                f"{direccion[0]:.6f},{direccion[1]:.6f},{direccion[2]:.6f}"
            )
    if lineas_dload:
        lineas.append("*DLOAD")
        lineas.extend(lineas_dload)

    lineas.append("*EL FILE")
    lineas.append("S")
    lineas.append("*END STEP")

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta_salida


def generar_inp_todos_los_casos(
    proyecto: Proyecto,
    malla: MallaAnalisis,
    cargas_lineales_por_caso: Dict[str, List[CargaLineal]],
    cargas_puntuales_por_caso: Dict[str, List[CargaPuntual]],
    apoyos: List[Apoyo],
    directorio_salida: Path,
) -> Dict[str, Path]:
    """Genera un .inp independiente por cada caso de carga, sobre la misma malla."""

    rutas: Dict[str, Path] = {}
    for caso_id in cargas_lineales_por_caso:
        ruta = directorio_salida / f"{caso_id}.inp"
        generar_inp_caso(
            proyecto,
            malla,
            caso_id,
            cargas_lineales_por_caso.get(caso_id, []),
            cargas_puntuales_por_caso.get(caso_id, []),
            apoyos,
            ruta,
        )
        rutas[caso_id] = ruta
    return rutas
