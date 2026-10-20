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
   Los perfiles en I (acero) se analizan con su forma real: ccx 2.21 solo
   expande a hexaedros (necesarios para obtener esfuerzos) las vigas B31
   rectangulares (``SECTION=I`` no existe y ``CIRC`` exige elementos
   cuadráticos), así que cada perfil se descompone en tres barras
   rectangulares —dos alas y el alma— desplazadas respecto del eje y unidas
   en cada nudo por tramos B31 muy rígidos (ver ``core.geometria.partes_seccion``
   y ``_agregar_partes``). La acción compuesta, la flexión fuerte y débil y
   la torsión de sección abierta salen del propio modelo. Comprobado con
   ccx 2.21: con coeficiente de Poisson no nulo la expansión de esas barras
   da esfuerzos laterales espurios (σyy, σzz ≈ 0,43·σxx) y un σxx 2 % alto,
   por lo que las barras compuestas usan un material con ν = 0 y el módulo
   de corte real (``*ELASTIC, TYPE=ENGINEERING CONSTANTS``), y el cortante
   se recalcula aparte (``solver.recuperacion_perfiles``), porque ccx lo
   reparte por igual entre alas y alma.

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
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..core.asignacion import barras_de_apoyo_area
from ..core.cargas import Apoyo, ApoyoArea, CargaLineal, CargaPuntual, Conexion
from ..core.geometria import ejes_seccion, es_perfil_compuesto, partes_seccion
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
    #: Índice de la parte de la sección a la que pertenece (ver ``partes_seccion``).
    parte: int = 0


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
    #: Nodo propio del extremo de una barra separado del nudo por una
    #: conexión de tipo EXTREMO: (barra_id, nodo_original) -> nodo ccx.
    nodo_ccx_de_extremo: Dict[Tuple[str, str], int] = field(default_factory=dict)
    elementos_por_barra: Dict[str, List[int]] = field(default_factory=dict)
    orden_nodos_por_barra: Dict[str, List[int]] = field(default_factory=dict)
    posiciones_s_por_barra: Dict[str, List[float]] = field(default_factory=dict)
    #: Elementos de las partes de una sección compuesta que no van sobre el
    #: eje: (barra_id, índice de parte) -> ids de elemento.
    elementos_por_parte: Dict[Tuple[str, int], List[int]] = field(default_factory=dict)
    #: Tramos rígidos que unen el eje con cada parte: etiqueta -> ids de elemento.
    elementos_auxiliares: Dict[str, List[int]] = field(default_factory=dict)
    #: Parte de la sección que lleva la cadena de elementos del eje de cada
    #: barra (-1: ninguna, el eje lleva solo un elemento casi sin rigidez).
    parte_en_el_eje: Dict[str, int] = field(default_factory=dict)


#: Prefijos de las etiquetas de elementos que no son barras del modelo (tramos
#: rígidos de enlaces y de secciones compuestas): no se dibujan ni cuentan en
#: los esfuerzos.
PREFIJOS_AUXILIARES = ("ENL_", "RIG_")

#: Lado (m) de la sección de la barra casi sin rigidez que lleva las cargas
#: en el eje de una sección compuesta sin parte central.
LADO_NUCLEO = 0.002


def es_elemento_auxiliar(etiqueta: str) -> bool:
    return etiqueta.startswith(PREFIJOS_AUXILIARES)


def _agregar_partes(proyecto: Proyecto, malla: "MallaAnalisis", barra_id: str, barra, ids_nodo: List[int],
                    posiciones: List[float], nuevo_id) -> None:
    """Crea las barras de las partes desplazadas de una sección compuesta y
    los tramos rígidos que las unen al eje, en cada nudo de la barra."""

    partes = partes_seccion(proyecto, barra)
    malla.parte_en_el_eje[barra_id] = next((k for k, p in enumerate(partes) if p.en_el_eje), -1)
    for id_el in malla.elementos_por_barra[barra_id]:
        malla.elementos[id_el].parte = malla.parte_en_el_eje[barra_id]
    ejes = ejes_seccion(proyecto, barra)
    for k, parte in enumerate(partes):
        if parte.en_el_eje:
            continue
        d = parte.desplazamiento1 * ejes.e1 + parte.desplazamiento2 * ejes.e2
        ids_parte: List[int] = []
        for nid in ids_nodo:
            n = malla.nodos[nid]
            nuevo = nuevo_id("nodo")
            malla.nodos[nuevo] = NodoMalla(nuevo, n.x + float(d[0]), n.y + float(d[1]), n.z + float(d[2]), None)
            ids_parte.append(nuevo)
        elementos_parte: List[int] = []
        for i in range(len(ids_parte) - 1):
            id_el = nuevo_id("elemento")
            malla.elementos[id_el] = ElementoMalla(id_el, barra_id, ids_parte[i], ids_parte[i + 1],
                                                   posiciones[i], posiciones[i + 1], parte=k)
            elementos_parte.append(id_el)
        malla.elementos_por_parte[(barra_id, k)] = elementos_parte
        etiqueta = f"RIG_{barra_id}_{k}"
        tramos: List[int] = []
        for nid, nid_parte in zip(ids_nodo, ids_parte):
            id_el = nuevo_id("elemento")
            malla.elementos[id_el] = ElementoMalla(id_el, etiqueta, nid, nid_parte, 0.0, 1.0, parte=k)
            tramos.append(id_el)
        malla.elementos_auxiliares[etiqueta] = tramos


def construir_malla(
    proyecto: Proyecto,
    cargas_lineales_por_caso: Dict[str, List[CargaLineal]],
    cargas_puntuales_por_caso: Dict[str, List[CargaPuntual]],
    longitud_elemento: float = LONGITUD_ELEMENTO_OBJETIVO,
    conexiones: Sequence[Conexion] = (),
    apoyos: Sequence = (),
) -> MallaAnalisis:
    """Construye la malla de análisis, común a todos los casos de carga.

    Usar la MISMA malla para todos los casos es imprescindible para que la
    superposición de resultados (ver combinaciones/motor_superposicion.py)
    pueda comparar nodo a nodo entre casos.
    """

    malla = MallaAnalisis()
    contador = {"nodo": 0, "elemento": 0}
    extremos_separados = {(c.barra_id, c.nodo_id) for c in conexiones if c.barra_id is not None}
    tramos_de_apoyo: Dict[str, List[Tuple[float, float]]] = {}
    for apoyo in apoyos:
        if isinstance(apoyo, ApoyoArea) and apoyo.elemento_tipo in ("VIGA", "COLUMNA"):
            tramos_de_apoyo.setdefault(apoyo.elemento_id, []).append((apoyo.s_inicio, apoyo.s_fin))

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
        for s_ini, s_fin in tramos_de_apoyo.get(barra_id, []):
            breakpoints.update((round(s_ini, 6), round(s_fin, 6)))

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
            if s <= _TOL and (barra_id, barra.nodo_inicio) in extremos_separados:
                id_ccx = nuevo_id("nodo")
                malla.nodos[id_ccx] = NodoMalla(id_ccx, nodo_ini.x, nodo_ini.y, nodo_ini.z, None)
                malla.nodo_ccx_de_extremo[(barra_id, barra.nodo_inicio)] = id_ccx
            elif s >= 1.0 - _TOL and (barra_id, barra.nodo_fin) in extremos_separados:
                id_ccx = nuevo_id("nodo")
                malla.nodos[id_ccx] = NodoMalla(id_ccx, nodo_fin.x, nodo_fin.y, nodo_fin.z, None)
                malla.nodo_ccx_de_extremo[(barra_id, barra.nodo_fin)] = id_ccx
            elif s <= _TOL:
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
        if len(partes_seccion(proyecto, barra)) > 1:
            _agregar_partes(proyecto, malla, barra_id, barra, ids_nodo, posiciones, nuevo_id)
        else:
            malla.parte_en_el_eje[barra_id] = 0

    # Enlaces rígidos: un tramo B31 muy rígido entre los dos nudos (ver tramos_rigidos)
    for c in conexiones:
        if not c.es_enlace_rigido:
            continue
        origen = malla.nodo_ccx_de_nodo_original.get(c.nodo_id)
        destino = malla.nodo_ccx_de_nodo_original.get(c.nodo_destino)
        if origen is None or destino is None or origen == destino:
            continue
        etiqueta = f"ENL_{c.id}"
        id_el = nuevo_id("elemento")
        malla.elementos[id_el] = ElementoMalla(id=id_el, barra_id=etiqueta, nodo_i=origen, nodo_f=destino, s_inicio=0.0, s_fin=1.0)
        malla.elementos_por_barra[etiqueta] = [id_el]

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


def _material_sin_poisson(nombre: str) -> str:
    """Etiqueta del material con ν = 0 y G real que usan las barras compuestas."""

    return _etiqueta_ccx(nombre) + "_SP"


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


def grados_restringidos_por_nodo(proyecto: Proyecto, malla: MallaAnalisis, apoyos: Sequence) -> Dict[int, Set[int]]:
    """Grados de libertad (1-6) restringidos en cada nodo de la malla, a
    partir de los apoyos nodales y de los apoyos de área (que restringen
    todos los nodos de la barra dentro de su tramo)."""

    restringidos: Dict[int, Set[int]] = {}
    for apoyo in apoyos:
        if isinstance(apoyo, ApoyoArea):
            for barra_id in barras_de_apoyo_area(proyecto, apoyo):
                if barra_id not in malla.orden_nodos_por_barra:
                    continue
                for nid, s in zip(malla.orden_nodos_por_barra[barra_id], malla.posiciones_s_por_barra[barra_id]):
                    if apoyo.s_inicio - 1e-6 <= s <= apoyo.s_fin + 1e-6:
                        restringidos.setdefault(nid, set()).update(apoyo.grados_restringidos())
        else:
            nid = malla.nodo_ccx_de_nodo_original.get(apoyo.nodo_id)
            if nid is not None:
                restringidos.setdefault(nid, set()).update(apoyo.grados_restringidos())
    return restringidos


Terminos = List[Tuple[int, int, float]]

#: Módulo de elasticidad del tramo rígido de un enlace = este factor × el mayor del modelo.
FACTOR_RIGIDEZ_ENLACE = 1000.0
#: Lado (m) de la sección cuadrada del tramo rígido.
LADO_ENLACE = 0.5


def ecuaciones_de_conexiones(proyecto: Proyecto, malla: MallaAnalisis, conexiones: Sequence[Conexion]) -> List[Terminos]:
    """Ecuaciones lineales ``Σ coef·u(nodo, gdl) = 0`` de las conexiones. El
    primer término de cada ecuación es el grado de libertad dependiente.

    Solo se escriben ecuaciones de TRASLACIÓN: en la versión 2.21 de
    CalculiX las ecuaciones sobre los giros de vigas B31 no se aplican (se
    comprobó que no tienen efecto), así que los giros se transmiten
    mediante la continuidad del nudo (no hay conexión), se liberan (EXTREMO
    articulado) o se transmiten con un tramo rígido (enlace rígido, ver
    ``tramos_rigidos``).
    """

    ecuaciones: List[Terminos] = []
    for c in conexiones:
        if c.es_enlace_rigido:
            continue
        if c.barra_id is not None:
            dependiente = malla.nodo_ccx_de_extremo.get((c.barra_id, c.nodo_id))
            principal = malla.nodo_ccx_de_nodo_original.get(c.nodo_id)
        else:
            dependiente = malla.nodo_ccx_de_nodo_original.get(c.nodo_destino)
            principal = malla.nodo_ccx_de_nodo_original.get(c.nodo_id)
        if dependiente is None or principal is None:
            continue
        for g in c.traslaciones_transmitidas():
            ecuaciones.append([(dependiente, g, 1.0), (principal, g, -1.0)])
    return ecuaciones


def giros_axiales_fijados(proyecto: Proyecto, malla: MallaAnalisis, conexiones: Sequence[Conexion]) -> Dict[int, Set[int]]:
    """Giro alrededor del eje de la barra impedido en los extremos
    articulados (``fijar_torsion``), cuando la barra es paralela a un eje global."""

    fijados: Dict[int, Set[int]] = {}
    for c in conexiones:
        if c.barra_id is None or not c.fijar_torsion or c.transmite_giros:
            continue
        nid = malla.nodo_ccx_de_extremo.get((c.barra_id, c.nodo_id))
        eje = eje_global_de_barra(proyecto, c.barra_id)
        if nid is not None and eje is not None:
            fijados.setdefault(nid, set()).add(4 + eje)
    return fijados


def eje_global_de_barra(proyecto: Proyecto, barra_id: str) -> Optional[int]:
    """0, 1 o 2 si la barra es paralela al eje X, Y o Z global; si no, None."""

    barra = proyecto.elemento_barra(barra_id)
    a, b = proyecto.nodos[barra.nodo_inicio], proyecto.nodos[barra.nodo_fin]
    d = (b.x - a.x, b.y - a.y, b.z - a.z)
    longitud = _norma(d)
    for i in range(3):
        if longitud > 0 and abs(abs(d[i]) - longitud) < 1e-6 * max(1.0, longitud):
            return i
    return None


def tramos_rigidos(proyecto: Proyecto, malla: MallaAnalisis, conexiones: Sequence[Conexion]) -> List[Tuple[str, int, int, Vector3]]:
    """Tramos B31 muy rígidos que materializan los enlaces rígidos:
    (etiqueta, nodo ccx origen, nodo ccx destino, vector n1 de la sección)."""

    tramos = []
    for c in conexiones:
        if not c.es_enlace_rigido:
            continue
        origen = malla.nodo_ccx_de_nodo_original.get(c.nodo_id)
        destino = malla.nodo_ccx_de_nodo_original.get(c.nodo_destino)
        if origen is None or destino is None or origen == destino:
            continue
        a, b = proyecto.nodos[c.nodo_id], proyecto.nodos[c.nodo_destino]
        t = (b.x - a.x, b.y - a.y, b.z - a.z)
        base = (0.0, 0.0, 1.0) if abs(t[2]) < 0.9 * _norma(t) else (1.0, 0.0, 0.0)
        n1 = (t[1] * base[2] - t[2] * base[1], t[2] * base[0] - t[0] * base[2], t[0] * base[1] - t[1] * base[0])
        norma = _norma(n1)
        tramos.append((f"ENL_{c.id}", origen, destino, (n1[0] / norma, n1[1] / norma, n1[2] / norma)))
    return tramos


def generar_inp_caso(
    proyecto: Proyecto,
    malla: MallaAnalisis,
    caso_id: str,
    cargas_lineales: List[CargaLineal],
    cargas_puntuales: List[CargaPuntual],
    apoyos: Sequence,
    ruta_salida: Path,
    conexiones: Sequence[Conexion] = (),
) -> Path:
    """Escribe el .inp de CalculiX correspondiente a un único caso de carga.

    ``apoyos`` mezcla ``Apoyo`` (nodales) y ``ApoyoArea`` (continuos).
    ``conexiones`` se traducen en ecuaciones lineales (``*EQUATION``).
    """

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
    for (barra_id, k), elementos_ids in malla.elementos_por_parte.items():
        lineas.append(f"*ELEMENT,TYPE=B31,ELSET=E_{barra_id}_P{k}")
        for eid in elementos_ids:
            el = malla.elementos[eid]
            lineas.append(f"{eid},{el.nodo_i},{el.nodo_f}")
    for etiqueta, elementos_ids in malla.elementos_auxiliares.items():
        lineas.append(f"*ELEMENT,TYPE=B31,ELSET=E_{etiqueta}")
        for eid in elementos_ids:
            el = malla.elementos[eid]
            lineas.append(f"{eid},{el.nodo_i},{el.nodo_f}")

    for nombre, material in proyecto.materiales.items():
        lineas.append(f"*MATERIAL,NAME={_etiqueta_ccx(nombre)}")
        lineas.append("*ELASTIC")
        lineas.append(f"{material.modulo_elasticidad:.6g},{material.modulo_poisson:.4f}")
        lineas.append("*DENSITY")
        lineas.append(f"{DENSIDAD_FICTICIA}")
        if any(es_perfil_compuesto(proyecto.perfiles[b.perfil]) and b.material == nombre
               for b in proyecto.todas_las_barras().values()):
            g = material.modulo_corte
            lineas.append(f"*MATERIAL,NAME={_material_sin_poisson(nombre)}")
            lineas.append("*ELASTIC,TYPE=ENGINEERING CONSTANTS")
            e = material.modulo_elasticidad
            lineas.append(f"{e:.6g},{e:.6g},{e:.6g},0,0,0,{g:.6g},{g:.6g}")
            lineas.append(f"{g:.6g}")
            lineas.append("*DENSITY")
            lineas.append(f"{DENSIDAD_FICTICIA}")

    enlaces = tramos_rigidos(proyecto, malla, conexiones)
    if enlaces or malla.elementos_auxiliares:
        e_max = max((m.modulo_elasticidad for m in proyecto.materiales.values()), default=1e7)
        lineas.append("*MATERIAL,NAME=RIGIDO_ENLACE")
        lineas.append("*ELASTIC")
        lineas.append(f"{e_max * FACTOR_RIGIDEZ_ENLACE:.6g},0.2000")
        lineas.append("*DENSITY")
        lineas.append(f"{DENSIDAD_FICTICIA}")
        for etiqueta in malla.elementos_auxiliares:
            barra_id = etiqueta[len("RIG_"):].rsplit("_", 1)[0]
            t = ejes_seccion(proyecto, proyecto.elemento_barra(barra_id)).t
            lineas.append(f"*BEAM SECTION,ELSET=E_{etiqueta},MATERIAL=RIGIDO_ENLACE,SECTION=RECT")
            lineas.append(f"{LADO_ENLACE},{LADO_ENLACE}")
            lineas.append(f"{t[0]:.9f},{t[1]:.9f},{t[2]:.9f}")
        for etiqueta, origen, destino, n1 in enlaces:
            lineas.append(f"*BEAM SECTION,ELSET=E_{etiqueta},MATERIAL=RIGIDO_ENLACE,SECTION=RECT")
            lineas.append(f"{LADO_ENLACE},{LADO_ENLACE}")
            lineas.append(f"{n1[0]:.9f},{n1[1]:.9f},{n1[2]:.9f}")

    areas: Dict[str, float] = {etiqueta: LADO_ENLACE**2 for etiqueta, *_ in enlaces}
    for barra_id, barra in proyecto.todas_las_barras().items():
        ejes = ejes_seccion(proyecto, barra)
        partes = partes_seccion(proyecto, barra)
        en_el_eje = malla.parte_en_el_eje.get(barra_id, 0)
        if en_el_eje >= 0:
            dim1, dim2 = partes[en_el_eje].dim1, partes[en_el_eje].dim2
        else:  # sin parte central: el eje lleva una barra casi sin rigidez que recibe las cargas
            dim1 = dim2 = LADO_NUCLEO
        areas[barra_id] = dim1 * dim2
        secciones = [(f"E_{barra_id}", dim1, dim2)]
        secciones += [(f"E_{barra_id}_P{k}", p.dim1, p.dim2) for k, p in enumerate(partes) if not p.en_el_eje]
        material_ccx = (_material_sin_poisson(barra.material) if es_perfil_compuesto(proyecto.perfiles[barra.perfil])
                        else _etiqueta_ccx(barra.material))
        for elset, d1, d2 in secciones:
            lineas.append(f"*BEAM SECTION,ELSET={elset},MATERIAL={material_ccx},SECTION=RECT")
            lineas.append(f"{d1:.6g},{d2:.6g}")
            lineas.append(f"{ejes.e1[0]:.9f},{ejes.e1[1]:.9f},{ejes.e1[2]:.9f}")

    ecuaciones = ecuaciones_de_conexiones(proyecto, malla, conexiones)
    dependientes: Set[Tuple[int, int]] = {(eq[0][0], eq[0][1]) for eq in ecuaciones}

    restringidos = grados_restringidos_por_nodo(proyecto, malla, apoyos)
    for nid, grados in giros_axiales_fijados(proyecto, malla, conexiones).items():
        restringidos.setdefault(nid, set()).update(grados)
    lineas_boundary: List[str] = []
    for nid in sorted(restringidos):
        grados = [g for g in restringidos[nid] if (nid, g) not in dependientes]
        for ini, fin in _rangos_contiguos(grados):
            lineas_boundary.append(f"{nid},{ini},{fin}")
    if lineas_boundary:
        lineas.append("*BOUNDARY")
        lineas.extend(lineas_boundary)

    if ecuaciones:
        lineas.append("*EQUATION")
        for terminos in ecuaciones:
            lineas.append(str(len(terminos)))
            lineas.append(", ".join(f"{n},{g},{c:.9g}" for n, g, c in terminos))

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

    lineas.append("*NODE FILE")
    lineas.append("U")
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
    apoyos: Sequence,
    directorio_salida: Path,
    conexiones: Sequence[Conexion] = (),
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
            conexiones,
        )
        rutas[caso_id] = ruta
    return rutas
