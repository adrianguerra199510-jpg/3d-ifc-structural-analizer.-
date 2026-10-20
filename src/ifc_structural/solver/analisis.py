"""Orquestación del análisis completo, independiente de la interfaz.

Une todos los pasos: validar el modelo, preparar las cargas (peso propio
y transferencia de losas), mallar, generar los .inp, ejecutar ccx, leer
los esfuerzos y superponer las combinaciones. La GUI solo llama a
``ejecutar_analisis`` (desde un hilo de trabajo) y muestra el resultado.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..combinaciones import esfuerzos
from ..combinaciones.motor_superposicion import superponer_todas_las_combinaciones
from ..core.asignacion import barras_de_apoyo_area
from ..core.cargas import Apoyo, ApoyoArea, CargaArea, CargaLineal, CargaPuntual, Conexion
from ..core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ..core.modelo import Proyecto
from ..core.transferencia import (
    ErrorTransferencia,
    cargas_area_peso_propio_losas,
    peso_propio_barras,
    transferir_carga_area,
)
from .ejecutor_ccx import ejecutar_ccx_todos_los_casos
from .generador_inp import MallaAnalisis, construir_malla, es_elemento_auxiliar, generar_inp_todos_los_casos
from .recuperacion_perfiles import hay_perfiles_compuestos, recuperar_perfiles
from .lector_resultados import MallaTensiones, leer_desplazamientos_frd, leer_malla_frd, leer_tensiones_frd, recortar_malla

#: Esfuerzo (kPa) a partir del cual se considera que el resultado no es
#: físicamente razonable (100 GPa): señal típica de una estructura inestable.
LIMITE_ESFUERZO_RAZONABLE_KPA = 1e8


class ErrorAnalisis(RuntimeError):
    """El análisis no pudo completarse; el mensaje explica el motivo."""


@dataclass
class ResultadoAnalisis:
    malla: MallaAnalisis
    malla_tensiones: MallaTensiones
    tensiones_por_caso: Dict[str, np.ndarray]
    tensiones_por_combinacion: Dict[str, np.ndarray]
    advertencias: List[str] = field(default_factory=list)
    es_solido: bool = False  # True si viene del análisis como modelo sólido (3D)
    desplazamientos_por_caso: Dict[str, np.ndarray] = field(default_factory=dict)  # (N, 3), en m
    desplazamientos_por_combinacion: Dict[str, np.ndarray] = field(default_factory=dict)

    def resultados_disponibles(self) -> Dict[str, np.ndarray]:
        """Casos y combinaciones, por id, listos para mostrar."""

        todos = dict(self.tensiones_por_caso)
        todos.update(self.tensiones_por_combinacion)
        return todos

    def barra_de_cada_celda(self) -> List[str]:
        """Id de la viga/columna a la que pertenece cada hexaedro de la
        malla de esfuerzos (en el orden de ``malla_tensiones``)."""

        return [self.malla.elementos[int(e)].barra_id for e in self.malla_tensiones.ids_elementos]

    def resumen_por_barra(self, valores_nodales: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """(mínimo, máximo) de un campo nodal sobre los nodos de cada barra."""

        nodos_por_barra: Dict[str, set] = {}
        for barra, conectividad in zip(self.barra_de_cada_celda(), self.malla_tensiones.conectividad):
            nodos_por_barra.setdefault(barra, set()).update(int(i) for i in conectividad)
        return {
            barra: (float(valores_nodales[sorted(nodos)].min()), float(valores_nodales[sorted(nodos)].max()))
            for barra, nodos in nodos_por_barra.items()
        }


def _componentes_conectadas(proyecto: Proyecto, conexiones: Sequence[Conexion] = ()) -> List[Tuple[List[str], set]]:
    """Grupos de barras conectadas entre sí: [(ids de barras, nodos)]. Los
    enlaces entre nudos distintos (conexiones ENLACE) también unen grupos."""

    padre: Dict[str, str] = {}

    def buscar(x: str) -> str:
        while padre.setdefault(x, x) != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    barras = proyecto.todas_las_barras()
    for barra in barras.values():
        padre[buscar(barra.nodo_inicio)] = buscar(barra.nodo_fin)
    for c in conexiones:
        if c.nodo_destino is not None and c.nodo_id in proyecto.nodos and c.nodo_destino in proyecto.nodos:
            padre[buscar(c.nodo_id)] = buscar(c.nodo_destino)

    grupos: Dict[str, Tuple[List[str], set]] = {}
    for id_, barra in barras.items():
        ids, nodos = grupos.setdefault(buscar(barra.nodo_inicio), ([], set()))
        ids.append(id_)
        nodos.update((barra.nodo_inicio, barra.nodo_fin))
    return list(grupos.values())


#: Restricción puntual: (nodos a los que pertenece, coordenadas, banderas ux..rz).
RestriccionPuntual = Tuple[set, Tuple[float, float, float], Tuple[bool, ...]]


def _restricciones_puntuales(proyecto: Proyecto, apoyos: Sequence) -> List[RestriccionPuntual]:
    """Apoyos nodales y de área convertidos en restricciones en puntos: los
    de área se muestrean a lo largo de la barra (extremos y 3 puntos
    intermedios), que basta para el análisis de movimientos de cuerpo rígido."""

    resultado: List[RestriccionPuntual] = []
    for apoyo in apoyos:
        banderas = (apoyo.ux, apoyo.uy, apoyo.uz, apoyo.rx, apoyo.ry, apoyo.rz)
        if isinstance(apoyo, ApoyoArea):
            for barra_id in barras_de_apoyo_area(proyecto, apoyo):
                barra = proyecto.elemento_barra(barra_id)
                a, b = proyecto.nodos[barra.nodo_inicio], proyecto.nodos[barra.nodo_fin]
                for k in range(5):
                    s = apoyo.s_inicio + (apoyo.s_fin - apoyo.s_inicio) * k / 4.0
                    punto = (a.x + s * (b.x - a.x), a.y + s * (b.y - a.y), a.z + s * (b.z - a.z))
                    resultado.append(({barra.nodo_inicio, barra.nodo_fin}, punto, banderas))
        elif apoyo.nodo_id in proyecto.nodos:
            resultado.append(({apoyo.nodo_id}, proyecto.nodos[apoyo.nodo_id].coordenadas, banderas))
    return resultado


def _grados_de_libertad_de_cuerpo_rigido_libres(
    proyecto: Proyecto, nodos: set, restricciones: Sequence[RestriccionPuntual]
) -> int:
    """Cuántos de los 6 movimientos de cuerpo rígido (3 traslaciones y 3
    giros) de un grupo de barras conectadas NO impiden sus apoyos.

    Cada restricción impone una condición lineal sobre el movimiento de
    cuerpo rígido (t, ω): una traslación restringida en el eje ``e`` en un
    punto ``r`` impone ``t·e + ω·(r × e) = 0``; un giro restringido impone
    ``ω·e = 0``. El grupo es estable como cuerpo rígido si esas condiciones
    tienen rango 6. (Los mecanismos internos, por articulaciones, se
    detectan aparte en ``solver.mecanismos``.)
    """

    coordenadas = np.array([proyecto.nodos[n].coordenadas for n in nodos], dtype=float)
    centro = coordenadas.mean(axis=0)
    ejes = np.eye(3)
    filas: List[np.ndarray] = []
    for pertenece, punto, banderas in restricciones:
        if not (pertenece & nodos):
            continue
        r = np.array(punto, dtype=float) - centro
        for i in range(3):
            if banderas[i]:
                filas.append(np.concatenate([ejes[i], np.cross(r, ejes[i])]))
        for i in range(3):
            if banderas[3 + i]:
                filas.append(np.concatenate([np.zeros(3), ejes[i]]))
    if not filas:
        return 6
    return 6 - int(np.linalg.matrix_rank(np.array(filas), tol=1e-9))


def _validar_conexiones(
    proyecto: Proyecto, conexiones: Sequence[Conexion], errores: List[str], advertencias: List[str]
) -> List[Conexion]:
    """Conexiones aplicables; informa de las que no lo son."""

    validas: List[Conexion] = []
    for c in conexiones:
        if c.barra_id is not None:
            barra = proyecto.todas_las_barras().get(c.barra_id)
            if barra is None or c.nodo_id not in (barra.nodo_inicio, barra.nodo_fin):
                advertencias.append(f"La conexión {c.nombre} ya no corresponde a ningún extremo del modelo y se ignora.")
                continue
            otras = [
                b for i, b in proyecto.todas_las_barras().items()
                if i != c.barra_id and c.nodo_id in (b.nodo_inicio, b.nodo_fin)
            ]
            if not otras:
                advertencias.append(
                    f"La conexión {c.nombre}: ninguna otra barra llega al nudo {c.nodo_id}; no hay nada que conectar y se ignora."
                )
                continue
        elif c.nodo_id not in proyecto.nodos or c.nodo_destino not in proyecto.nodos:
            advertencias.append(f"La conexión {c.nombre} usa nudos que ya no existen y se ignora.")
            continue
        validas.append(c)
    return validas


def validar_modelo(
    proyecto: Proyecto,
    casos: Dict[str, CasoCarga],
    combinaciones: Dict[str, Combinacion],
    apoyos: Sequence,
    conexiones: Sequence[Conexion] = (),
) -> Tuple[List[str], List[str]]:
    """Revisa que el modelo se pueda analizar. Devuelve (errores, advertencias):
    los errores impiden el análisis; las advertencias no."""

    errores: List[str] = []
    advertencias: List[str] = []

    barras = proyecto.todas_las_barras()
    if not barras:
        errores.append("El modelo no tiene vigas ni columnas que analizar.")
    if not casos:
        errores.append("No hay casos de carga definidos.")

    nodos_barras = {n for b in barras.values() for n in (b.nodo_inicio, b.nodo_fin)}
    apoyos_validos = []
    for a in apoyos:
        if isinstance(a, ApoyoArea):
            if barras_de_apoyo_area(proyecto, a):
                apoyos_validos.append(a)
        elif a.nodo_id in nodos_barras:
            apoyos_validos.append(a)
    if len(apoyos_validos) < len(apoyos):
        advertencias.append(
            f"{len(apoyos) - len(apoyos_validos)} apoyo(s) no están sobre ninguna viga o columna del modelo "
            "(o la losa no tiene vigas de soporte) y se ignorarán."
        )
    if barras and not apoyos_validos:
        errores.append("No hay apoyos asignados: seleccione la cara extrema de una columna o viga y asigne un apoyo.")

    conexiones_validas = _validar_conexiones(proyecto, conexiones, errores, advertencias)

    if barras and apoyos_validos:
        restricciones = _restricciones_puntuales(proyecto, apoyos_validos)
        nodos_apoyados = {n for pertenece, _, _ in restricciones for n in pertenece}
        for ids, nodos in _componentes_conectadas(proyecto, conexiones_validas):
            nombres = ", ".join(sorted(ids))
            if not (nodos & nodos_apoyados):
                errores.append(
                    f"Hay barras sin conexión a ningún apoyo (la estructura quedaría flotando): {nombres}. "
                    "Revise que sus extremos coincidan con los de las demás barras."
                )
                continue
            libres = _grados_de_libertad_de_cuerpo_rigido_libres(proyecto, nodos, restricciones)
            if libres:
                errores.append(
                    f"La estructura formada por {nombres} es inestable: sus apoyos no impiden {libres} de los 6 "
                    "movimientos de cuerpo rígido (traslaciones o giros). Agregue apoyos o restrinja más grados "
                    "de libertad (p. ej. un empotramiento restringe los 6; dos apoyos articulados alineados "
                    "dejan libre el giro alrededor del eje que los une)."
                )
        if not errores and conexiones_validas:
            from .mecanismos import detectar_mecanismos

            resultado = detectar_mecanismos(proyecto, apoyos_validos, conexiones_validas)
            if resultado is None:
                advertencias.append(
                    "El modelo es grande: no se comprobó si las articulaciones dejan mecanismos internos."
                )
            elif resultado.cantidad:
                barras_txt = ", ".join(resultado.barras) if resultado.barras else "varias barras"
                errores.append(
                    f"Las conexiones y apoyos dejan {resultado.cantidad} mecanismo(s) interno(s) (la estructura se "
                    f"movería sin resistencia), principalmente en: {barras_txt}. Restrinja más grados de libertad, "
                    "transmita más giros en las conexiones o agregue apoyos."
                )

    for losa in proyecto.losas.values():
        if not any(losa.vigas_soporte):
            advertencias.append(
                f"La losa {losa.id} no tiene vigas de soporte detectadas: su peso propio y las cargas "
                "de área sobre ella no se transferirán a las vigas."
            )

    for combinacion in combinaciones.values():
        desconocidos = [c for c in combinacion.factores if c not in casos]
        if desconocidos:
            advertencias.append(
                f"La combinación {combinacion.id} usa casos inexistentes ({', '.join(desconocidos)}); se ignoran."
            )

    return errores, advertencias


def preparar_cargas_lineales(
    proyecto: Proyecto,
    casos: Dict[str, CasoCarga],
    cargas_area_por_caso: Dict[str, List[CargaArea]],
    cargas_lineales_por_caso: Dict[str, List[CargaLineal]],
) -> Tuple[Dict[str, List[CargaLineal]], List[str]]:
    """Cargas lineales finales por caso: las manuales, el peso propio de
    vigas/columnas/losas (en el caso de tipo "D", carga muerta, si existe)
    y la transferencia por áreas tributarias de cada carga de área."""

    advertencias: List[str] = []
    resultado: Dict[str, List[CargaLineal]] = {caso_id: list(c) for caso_id, c in cargas_lineales_por_caso.items()}
    for caso_id in casos:
        resultado.setdefault(caso_id, [])

    def transferir(carga: CargaArea) -> List[CargaLineal]:
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            try:
                lineales = transferir_carga_area(proyecto, carga)
            except ErrorTransferencia as exc:
                advertencias.append(f"No se transfirió la carga {carga.nombre}: {exc}")
                return []
        advertencias.extend(str(a.message) for a in avisos)
        return lineales

    caso_muerta = next((c.id for c in casos.values() if c.tipo == TipoCaso.D), None)
    if caso_muerta:
        resultado[caso_muerta].extend(peso_propio_barras(proyecto, caso_muerta))
        for carga_pp in cargas_area_peso_propio_losas(proyecto, caso_muerta):
            resultado[caso_muerta].extend(transferir(carga_pp))

    for caso_id, cargas in cargas_area_por_caso.items():
        if caso_id not in resultado:
            continue
        for carga in cargas:
            resultado[caso_id].extend(transferir(carga))

    return resultado, list(dict.fromkeys(advertencias))


def ejecutar_analisis(
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
    conexiones: Sequence[Conexion] = (),
) -> ResultadoAnalisis:
    """Ejecuta el análisis completo y devuelve los esfuerzos por caso y
    por combinación. Lanza ``ErrorAnalisis`` con un mensaje explicativo si
    no se puede completar."""

    def avanzar(mensaje: str) -> None:
        if al_avanzar is not None:
            al_avanzar(mensaje)

    avanzar("Validando el modelo...")
    errores, advertencias = validar_modelo(proyecto, casos, combinaciones, apoyos, conexiones)
    if errores:
        raise ErrorAnalisis("\n".join(errores))

    avanzar("Preparando cargas (peso propio y transferencia de losas)...")
    cargas_lineales, avisos_cargas = preparar_cargas_lineales(proyecto, casos, cargas_area_por_caso, cargas_lineales_por_caso)
    advertencias.extend(avisos_cargas)
    cargas_puntuales = {caso_id: list(cargas_puntuales_por_caso.get(caso_id, [])) for caso_id in casos}

    avanzar("Construyendo la malla de análisis...")
    conexiones_validas = _validar_conexiones(proyecto, conexiones, [], [])
    malla = construir_malla(proyecto, cargas_lineales, cargas_puntuales, conexiones=conexiones_validas, apoyos=apoyos)

    avanzar("Generando los archivos de CalculiX...")
    directorio_trabajo.mkdir(parents=True, exist_ok=True)
    rutas_inp = generar_inp_todos_los_casos(
        proyecto, malla, cargas_lineales, cargas_puntuales, apoyos, directorio_trabajo, conexiones_validas
    )
    for ruta in rutas_inp.values():
        for sufijo in (".frd", ".dat"):
            ruta.with_suffix(sufijo).unlink(missing_ok=True)

    ejecuciones = ejecutar_ccx_todos_los_casos(
        rutas_inp,
        ruta_ccx,
        al_avanzar=lambda caso, i, total: avanzar(f"Resolviendo el caso {caso} ({i} de {total}) con CalculiX..."),
    )
    fallidos = {caso: r for caso, r in ejecuciones.items() if not r.exitoso}
    if fallidos:
        detalle = "\n".join(f"Caso {caso}: {r.diagnostico()}" for caso, r in fallidos.items())
        raise ErrorAnalisis("CalculiX no pudo resolver el modelo.\n" + detalle)

    avanzar("Leyendo los esfuerzos...")
    primer_caso = next(iter(ejecuciones))
    malla_tensiones = leer_malla_frd(ejecuciones[primer_caso].ruta_frd)
    tensiones_por_caso: Dict[str, np.ndarray] = {}
    desplazamientos_por_caso: Dict[str, np.ndarray] = {}
    for caso_id, ejecucion in ejecuciones.items():
        malla_caso = leer_malla_frd(ejecucion.ruta_frd)
        if not np.array_equal(malla_caso.ids_nodos, malla_tensiones.ids_nodos):
            raise ErrorAnalisis(f"La malla de resultados del caso {caso_id} no coincide con la de los demás casos.")
        tensiones_por_caso[caso_id] = leer_tensiones_frd(ejecucion.ruta_frd, malla_tensiones)
        desplazamientos_por_caso[caso_id] = leer_desplazamientos_frd(ejecucion.ruta_frd, malla_tensiones)

    # Extremos de barra y cortante de perfiles en I: se rehacen (ver recuperacion_perfiles)
    tensiones_por_caso = {
        caso: recuperar_perfiles(proyecto, malla, malla_tensiones, t) for caso, t in tensiones_por_caso.items()
    }
    if hay_perfiles_compuestos(proyecto):
        advertencias.append(
            "Perfiles en I: los esfuerzos no incluyen el cortante por torsión ni los efectos locales de "
            "placas, soldaduras o pernos (la unión se idealiza por los ejes de las barras)."
        )

    # Los tramos rígidos (enlaces y uniones de las partes de un perfil) no son
    # elementos del modelo: se descartan de la malla y de los esfuerzos.
    mantener = np.array(
        [not es_elemento_auxiliar(malla.elementos[int(e)].barra_id) for e in malla_tensiones.ids_elementos]
    )
    if not mantener.all():
        malla_tensiones, conservados = recortar_malla(malla_tensiones, mantener)
        tensiones_por_caso = {caso: t[conservados] for caso, t in tensiones_por_caso.items()}
        desplazamientos_por_caso = {caso: u[conservados] for caso, u in desplazamientos_por_caso.items()}

    avanzar("Combinando casos de carga...")
    tensiones_por_combinacion = superponer_todas_las_combinaciones(tensiones_por_caso, list(combinaciones.values()))
    desplazamientos_por_combinacion = superponer_todas_las_combinaciones(desplazamientos_por_caso, list(combinaciones.values()))

    maximo = max(
        (float(np.nanmax(esfuerzos.von_mises(t))) for t in tensiones_por_caso.values()),
        default=0.0,
    )
    if not np.isfinite(maximo) or maximo > LIMITE_ESFUERZO_RAZONABLE_KPA:
        advertencias.append(
            "Los esfuerzos obtenidos no son físicamente razonables: es muy probable que la estructura sea "
            "inestable (mecanismo) o que haya barras mal conectadas. Revise los apoyos y la conectividad."
        )

    return ResultadoAnalisis(
        malla=malla,
        malla_tensiones=malla_tensiones,
        tensiones_por_caso=tensiones_por_caso,
        tensiones_por_combinacion=tensiones_por_combinacion,
        advertencias=advertencias,
        desplazamientos_por_caso=desplazamientos_por_caso,
        desplazamientos_por_combinacion=desplazamientos_por_combinacion,
    )
