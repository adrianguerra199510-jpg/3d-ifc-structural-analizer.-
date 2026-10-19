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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..combinaciones import esfuerzos
from ..combinaciones.motor_superposicion import superponer_todas_las_combinaciones
from ..core.cargas import Apoyo, CargaArea, CargaLineal, CargaPuntual
from ..core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ..core.modelo import Proyecto
from ..core.transferencia import (
    ErrorTransferencia,
    cargas_area_peso_propio_losas,
    peso_propio_barras,
    transferir_carga_area,
)
from .ejecutor_ccx import ejecutar_ccx_todos_los_casos
from .generador_inp import MallaAnalisis, construir_malla, generar_inp_todos_los_casos
from .lector_resultados import MallaTensiones, leer_malla_frd, leer_tensiones_frd

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


def _componentes_conectadas(proyecto: Proyecto) -> List[Tuple[List[str], set]]:
    """Grupos de barras conectadas entre sí: [(ids de barras, nodos)]."""

    padre: Dict[str, str] = {}

    def buscar(x: str) -> str:
        while padre.setdefault(x, x) != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    barras = proyecto.todas_las_barras()
    for barra in barras.values():
        padre[buscar(barra.nodo_inicio)] = buscar(barra.nodo_fin)

    grupos: Dict[str, Tuple[List[str], set]] = {}
    for id_, barra in barras.items():
        ids, nodos = grupos.setdefault(buscar(barra.nodo_inicio), ([], set()))
        ids.append(id_)
        nodos.update((barra.nodo_inicio, barra.nodo_fin))
    return list(grupos.values())


def _grados_de_libertad_de_cuerpo_rigido_libres(proyecto: Proyecto, nodos: set, apoyos: List[Apoyo]) -> int:
    """Cuántos de los 6 movimientos de cuerpo rígido (3 traslaciones y 3
    giros) de un grupo de barras conectadas NO impiden sus apoyos.

    Todas las barras se unen rígidamente en los nodos (no hay articulaciones
    internas), así que el grupo solo puede ser inestable como un todo. Cada
    restricción impone una condición lineal sobre el movimiento de cuerpo
    rígido (t, ω): una traslación restringida en el eje ``e`` en un nodo en
    ``r`` impone ``t·e + ω·(r × e) = 0``; un giro restringido impone
    ``ω·e = 0``. El grupo es estable si esas condiciones tienen rango 6.
    """

    coordenadas = np.array([proyecto.nodos[n].coordenadas for n in nodos], dtype=float)
    centro = coordenadas.mean(axis=0)
    ejes = np.eye(3)
    filas: List[np.ndarray] = []
    for apoyo in apoyos:
        if apoyo.nodo_id not in nodos:
            continue
        r = np.array(proyecto.nodos[apoyo.nodo_id].coordenadas, dtype=float) - centro
        for i, restringido in enumerate((apoyo.ux, apoyo.uy, apoyo.uz)):
            if restringido:
                filas.append(np.concatenate([ejes[i], np.cross(r, ejes[i])]))
        for i, restringido in enumerate((apoyo.rx, apoyo.ry, apoyo.rz)):
            if restringido:
                filas.append(np.concatenate([np.zeros(3), ejes[i]]))
    if not filas:
        return 6
    return 6 - int(np.linalg.matrix_rank(np.array(filas), tol=1e-9))


def validar_modelo(
    proyecto: Proyecto,
    casos: Dict[str, CasoCarga],
    combinaciones: Dict[str, Combinacion],
    apoyos: List[Apoyo],
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
    apoyos_validos = [a for a in apoyos if a.nodo_id in nodos_barras]
    if len(apoyos_validos) < len(apoyos):
        advertencias.append(
            f"{len(apoyos) - len(apoyos_validos)} apoyo(s) están en nodos que no pertenecen a ninguna "
            "viga o columna y se ignorarán."
        )
    if barras and not apoyos_validos:
        errores.append("No hay apoyos asignados: seleccione la cara extrema de una columna o viga y asigne un apoyo.")

    if barras and apoyos_validos:
        nodos_apoyados = {a.nodo_id for a in apoyos_validos}
        for ids, nodos in _componentes_conectadas(proyecto):
            nombres = ", ".join(sorted(ids))
            if not (nodos & nodos_apoyados):
                errores.append(
                    f"Hay barras sin conexión a ningún apoyo (la estructura quedaría flotando): {nombres}. "
                    "Revise que sus extremos coincidan con los de las demás barras."
                )
                continue
            libres = _grados_de_libertad_de_cuerpo_rigido_libres(proyecto, nodos, apoyos_validos)
            if libres:
                errores.append(
                    f"La estructura formada por {nombres} es inestable: sus apoyos no impiden {libres} de los 6 "
                    "movimientos de cuerpo rígido (traslaciones o giros). Agregue apoyos o restrinja más grados "
                    "de libertad (p. ej. un empotramiento restringe los 6; dos apoyos articulados alineados "
                    "dejan libre el giro alrededor del eje que los une)."
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
    apoyos: List[Apoyo],
    cargas_area_por_caso: Dict[str, List[CargaArea]],
    cargas_lineales_por_caso: Dict[str, List[CargaLineal]],
    cargas_puntuales_por_caso: Dict[str, List[CargaPuntual]],
    directorio_trabajo: Path,
    ruta_ccx: str = "ccx",
    al_avanzar: Optional[Callable[[str], None]] = None,
) -> ResultadoAnalisis:
    """Ejecuta el análisis completo y devuelve los esfuerzos por caso y
    por combinación. Lanza ``ErrorAnalisis`` con un mensaje explicativo si
    no se puede completar."""

    def avanzar(mensaje: str) -> None:
        if al_avanzar is not None:
            al_avanzar(mensaje)

    avanzar("Validando el modelo...")
    errores, advertencias = validar_modelo(proyecto, casos, combinaciones, apoyos)
    if errores:
        raise ErrorAnalisis("\n".join(errores))

    avanzar("Preparando cargas (peso propio y transferencia de losas)...")
    cargas_lineales, avisos_cargas = preparar_cargas_lineales(proyecto, casos, cargas_area_por_caso, cargas_lineales_por_caso)
    advertencias.extend(avisos_cargas)
    cargas_puntuales = {caso_id: list(cargas_puntuales_por_caso.get(caso_id, [])) for caso_id in casos}

    avanzar("Construyendo la malla de análisis...")
    malla = construir_malla(proyecto, cargas_lineales, cargas_puntuales)

    avanzar("Generando los archivos de CalculiX...")
    directorio_trabajo.mkdir(parents=True, exist_ok=True)
    rutas_inp = generar_inp_todos_los_casos(proyecto, malla, cargas_lineales, cargas_puntuales, apoyos, directorio_trabajo)
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
    for caso_id, ejecucion in ejecuciones.items():
        malla_caso = leer_malla_frd(ejecucion.ruta_frd)
        if not np.array_equal(malla_caso.ids_nodos, malla_tensiones.ids_nodos):
            raise ErrorAnalisis(f"La malla de resultados del caso {caso_id} no coincide con la de los demás casos.")
        tensiones_por_caso[caso_id] = leer_tensiones_frd(ejecucion.ruta_frd, malla_tensiones)

    avanzar("Combinando casos de carga...")
    tensiones_por_combinacion = superponer_todas_las_combinaciones(tensiones_por_caso, list(combinaciones.values()))

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
    )
