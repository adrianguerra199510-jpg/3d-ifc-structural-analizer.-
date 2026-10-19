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
   (eje z) del perfil real — ver ``_seccion_equivalente``. Esta
   aproximación se documenta aquí y en el README; una mejora futura (fuera
   de la Fase 1) sería mapear a ``CIRC``/``PIPE``/``BOX`` según la forma
   real del perfil.

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

3. **Resultados**: CalculiX no ofrece ninguna opción de fuerzas de sección
   (N, V, M, T) para ``*EL PRINT`` (el que escribe el .dat): según el
   manual, los únicos valores de elemento disponibles ahí son esfuerzos,
   deformaciones y energías, nunca fuerzas de sección. La opción
   ``SECTION FORCES`` solo existe para ``*EL FILE``/``*ELEMENT OUTPUT``
   (que escriben el .frd). Por eso el generador pide ``*EL FILE,
   OUTPUT=2D, SECTION FORCES`` y el módulo ``lector_resultados`` parsea el
   .frd resultante (que con ``*EL FILE`` es ASCII puro) en vez del .dat
   para las fuerzas internas. Esto es una desviación deliberada y
   documentada del enunciado original ("leer el .dat"): los
   desplazamientos sí se leen del .dat, pero las fuerzas internas no
   pueden leerse de ahí porque CalculiX no las ofrece en ese archivo.

4. Por la misma razón, las reacciones de apoyo tampoco se leen de
   ``*NODE PRINT, RF``: se comprobó que, para nodos con rotaciones
   restringidas (apoyos empotrados), CalculiX mezcla la reacción de
   momento dentro de la reacción de fuerza en el grado de libertad 1,
   dando valores incorrectos. Las reacciones se derivan, junto con las
   fuerzas internas, de las fuerzas de sección del elemento adyacente al
   nodo de apoyo.

Para dar buena resolución a los diagramas de N/V/M (las fuerzas de sección
se obtienen por punto de integración, esencialmente constantes dentro de
cada elemento corto), cada viga/columna se subdivide en varios elementos
B31 cortos entre cada par de "quiebres" de carga (ver
``N_SUBDIVISIONES_POR_TRAMO``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.cargas import Apoyo, CargaLineal, CargaPuntual
from ..core.modelo import ElementoBarra, PerfilSeccion, Proyecto

Vector3 = Tuple[float, float, float]

#: Cantidad de sub-elementos en que se divide cada tramo de carga continuo
#: (entre dos "quiebres" consecutivos) de cada viga/columna, para mejorar
#: la resolución del diagrama de fuerzas internas.
N_SUBDIVISIONES_POR_TRAMO = 4

#: Densidad ficticia usada únicamente para escalar *DLOAD,GRAV (ver punto 2
#: del docstring del módulo). No representa la densidad real del material.
DENSIDAD_FICTICIA = 1.0

_TOL = 1e-9


def _resta(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cruz(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norma(v: Vector3) -> float:
    return math.sqrt(sum(c * c for c in v))


def _normalizar(v: Vector3) -> Vector3:
    n = _norma(v)
    return (v[0] / n, v[1] / n, v[2] / n)


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
    diagramas de fuerzas internas con buena resolución. ``nodos`` y
    ``elementos`` usan etiquetas enteras (requeridas por CalculiX).
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
    n_subdivisiones: int = N_SUBDIVISIONES_POR_TRAMO,
) -> MallaAnalisis:
    """Construye la malla de análisis, común a todos los casos de carga.

    Usar la MISMA malla para todos los casos es imprescindible para que la
    superposición de resultados (ver combinaciones/motor_superposicion.py)
    pueda comparar nodo a nodo y elemento a elemento entre casos.
    """

    malla = MallaAnalisis()
    contador = {"nodo": 0, "elemento": 0}

    def nuevo_id(clave: str) -> int:
        contador[clave] += 1
        return contador[clave]

    for barra_id, barra in proyecto.todas_las_barras().items():
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
            for k in range(1, n_subdivisiones + 1):
                posiciones.append(s0 + (s1 - s0) * k / n_subdivisiones)

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


def _seccion_equivalente(perfil: PerfilSeccion) -> Tuple[float, float]:
    """Dimensiones (altura, ancho) para ``*BEAM SECTION, SECTION=RECT``.

    Si el perfil ya trae dimensiones reales (sección rectangular, el caso
    típico en vigas/columnas de hormigón), se usan directamente. En caso
    contrario se aproxima con un rectángulo equivalente que conserva el
    área y la inercia fuerte (inercia_z) del perfil real; ver el punto 1
    del docstring del módulo.
    """

    if perfil.altura and perfil.ancho:
        return perfil.altura, perfil.ancho

    area = perfil.area
    inercia_fuerte = perfil.inercia_z if perfil.inercia_z > 0 else perfil.inercia_y
    altura = math.sqrt(12.0 * inercia_fuerte / area)
    ancho = area / altura
    return altura, ancho


def _orientacion_valida(proyecto: Proyecto, barra: ElementoBarra) -> Vector3:
    """Vector n1 (dirección local aproximada, perpendicular al eje de la
    barra) a usar en ``*BEAM SECTION``. Si el ``vector_n1`` configurado en
    el elemento resulta paralelo al eje de la barra (caso típico de
    columnas verticales con el valor por defecto (0,0,1)), se usa un vector
    de respaldo no paralelo.
    """

    a = proyecto.nodos[barra.nodo_inicio]
    b = proyecto.nodos[barra.nodo_fin]
    eje = _normalizar((b.x - a.x, b.y - a.y, b.z - a.z))

    for candidato in (barra.vector_n1, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        if _norma(_cruz(eje, candidato)) > 1e-6:
            return candidato
    return (0.0, 0.0, 1.0)


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


def _agregar_lista_numeros(lineas: List[str], numeros: List[int], por_linea: int = 10) -> None:
    for i in range(0, len(numeros), por_linea):
        lineas.append(",".join(str(n) for n in numeros[i : i + por_linea]))


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

    for barra_id, barra in proyecto.todas_las_barras().items():
        perfil = proyecto.perfiles[barra.perfil]
        altura, ancho = _seccion_equivalente(perfil)
        n1 = _orientacion_valida(proyecto, barra)
        lineas.append(
            f"*BEAM SECTION,ELSET=E_{barra_id},MATERIAL={_etiqueta_ccx(barra.material)},SECTION=RECT"
        )
        lineas.append(f"{altura:.6g},{ancho:.6g}")
        lineas.append(f"{n1[0]:.6f},{n1[1]:.6f},{n1[2]:.6f}")

    apoyos_validos = [a for a in apoyos if a.nodo_id in malla.nodo_ccx_de_nodo_original]
    if apoyos_validos:
        lineas.append("*BOUNDARY")
        for apoyo in apoyos_validos:
            nid = malla.nodo_ccx_de_nodo_original[apoyo.nodo_id]
            for ini, fin in _rangos_contiguos(apoyo.grados_restringidos()):
                lineas.append(f"{nid},{ini},{fin}")

    lineas.append("*NSET,NSET=NALL")
    _agregar_lista_numeros(lineas, sorted(malla.nodos))

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
        barra = proyecto.elemento_barra(barra_id)
        area = proyecto.perfiles[barra.perfil].area
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

    lineas.append("*NODE PRINT,NSET=NALL")
    lineas.append("U")
    lineas.append("*EL FILE,OUTPUT=2D,SECTION FORCES")
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
