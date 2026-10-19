"""Conversión de lo que el usuario selecciona en el visor (elementos o
caras) en apoyos y cargas del modelo.

Todo se traduce a los tipos de carga que ya entiende el solver
(``CargaArea``, ``CargaLineal``, ``CargaPuntual``) y a ``Apoyo``:

=====================================  =====================================
Selección                              Resultado
=====================================  =====================================
Losa (elemento o cara sup./inf.)       ``CargaArea`` (kPa) hacia -Z o, si se
                                       eligió una cara, hacia dentro de ella
Viga/columna completa                  ``CargaLineal`` uniforme (kN/m)
Cara lateral de viga/columna           ``CargaLineal`` = presión × ancho de
                                       la cara, perpendicular a la cara
Cara extrema de viga/columna           ``CargaPuntual`` en ese nodo =
                                       presión × área de la cara, axial
Cara extrema de viga/columna           ``Apoyo`` en el nodo de ese extremo
=====================================  =====================================

Sin dependencias de la interfaz gráfica, para poder probarlo aisladamente.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

from .cargas import Apoyo, ApoyoArea, CargaArea, CargaLineal, CargaPuntual, Conexion, PerfilPresion
from .geometria import Cara, solido_elemento
from .modelo import Proyecto

Vector3 = Tuple[float, float, float]
Carga = Union[CargaArea, CargaLineal, CargaPuntual]

GRAVEDAD: Vector3 = (0.0, 0.0, -1.0)

#: Direcciones globales ofrecidas para cargas que no salen de una cara.
DIRECCIONES = {
    "Gravedad (−Z)": (0.0, 0.0, -1.0),
    "Hacia arriba (+Z)": (0.0, 0.0, 1.0),
    "+X": (1.0, 0.0, 0.0),
    "−X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "−Y": (0.0, -1.0, 0.0),
}


class ErrorAsignacion(ValueError):
    """La selección no admite lo que se pidió; el mensaje explica por qué."""


@dataclass(frozen=True)
class Objetivo:
    """Elemento (o una de sus caras) seleccionado en el visor."""

    tipo: str  # "VIGA", "COLUMNA", "LOSA" o "MURO"
    id: str
    cara: Optional[int] = None  # índice de cara; None = el elemento completo


_ETIQUETA_TIPO = {"VIGA": "Viga", "COLUMNA": "Columna", "LOSA": "Losa", "MURO": "Muro"}


def cara_de(proyecto: Proyecto, objetivo: Objetivo) -> Optional[Cara]:
    if objetivo.cara is None:
        return None
    return solido_elemento(proyecto, objetivo.tipo, objetivo.id).caras[objetivo.cara]


def descripcion(proyecto: Proyecto, objetivo: Objetivo) -> str:
    """Texto legible: «Viga B1», «Losa L1 — cara superior»."""

    base = f"{_ETIQUETA_TIPO[objetivo.tipo]} {objetivo.id}"
    cara = cara_de(proyecto, objetivo)
    return base if cara is None else f"{base} — cara {cara.nombre}"


def _formato(valor: float) -> str:
    return f"{valor:g}"


def _vector(normal) -> Vector3:
    return (float(normal[0]), float(normal[1]), float(normal[2]))


# ---------------------------------------------------------------------------
# Apoyos
# ---------------------------------------------------------------------------


def nodo_para_apoyo(proyecto: Proyecto, objetivo: Objetivo) -> str:
    """Nodo donde va un apoyo asignado sobre la selección."""

    if objetivo.tipo not in ("VIGA", "COLUMNA"):
        raise ErrorAsignacion(
            f"{descripcion(proyecto, objetivo)}: los apoyos se asignan en las caras extremas de vigas y "
            "columnas. Las losas y muros no tienen apoyos propios en la Fase 1 (se apoyan en las vigas)."
        )
    cara = cara_de(proyecto, objetivo)
    if cara is None or cara.nodo_id is None:
        raise ErrorAsignacion(
            f"{descripcion(proyecto, objetivo)}: seleccione la cara extrema (inicial o final) de la "
            "viga o columna donde va el apoyo (modo «Cara»)."
        )
    return cara.nodo_id


# ---------------------------------------------------------------------------
# Cargas
# ---------------------------------------------------------------------------

#: Formas de una carga distribuida sobre una barra (kN/m).
FORMAS_CARGA = {
    "UNIFORME": "Uniforme",
    "TRIANGULAR_CRECIENTE": "Triangular creciente (0 → w)",
    "TRIANGULAR_DECRECIENTE": "Triangular decreciente (w → 0)",
    "TRIANGULAR_SIMETRICA": "Triangular simétrica (0 → w → 0)",
    "TRAPEZOIDAL": "Trapezoidal (w inicial → w final)",
    "TRAPEZOIDAL_SIMETRICA": "Trapezoidal simétrica (rampas y meseta)",
}


@dataclass
class FormaCarga:
    """Cómo varía una carga distribuida a lo largo de la barra.

    ``desde``/``hasta`` (m medidos desde el nodo inicial; ``hasta=None`` =
    longitud completa) acotan el tramo cargado. ``w`` es el valor máximo
    (o el inicial en la trapezoidal), ``w_fin`` el valor final de la
    trapezoidal y ``rampa`` el largo (m) de cada rampa de la simétrica.
    """

    tipo: str = "UNIFORME"
    w_fin: float = 0.0
    desde: float = 0.0
    hasta: Optional[float] = None
    rampa: float = 0.0


def segmentos_de_forma(forma: FormaCarga, w: float, longitud: float) -> List[Tuple[float, float, float, float]]:
    """Trozos lineales (s_inicio, s_fin, w_inicio, w_fin), con s en 0..1, que
    componen la carga. Lanza ``ErrorAsignacion`` si el tramo no es válido."""

    if forma.tipo not in FORMAS_CARGA:
        raise ErrorAsignacion(f"Forma de carga desconocida: {forma.tipo}")
    desde = forma.desde
    hasta = longitud if forma.hasta is None else forma.hasta
    if not (-1e-9 <= desde < hasta <= longitud + 1e-9):
        raise ErrorAsignacion(
            f"El tramo cargado ({desde:g} a {hasta:g} m) debe estar dentro de la barra (0 a {longitud:.4g} m)."
        )
    s0, s1 = max(0.0, desde / longitud), min(1.0, hasta / longitud)
    if forma.tipo == "UNIFORME":
        return [(s0, s1, w, w)]
    if forma.tipo == "TRIANGULAR_CRECIENTE":
        return [(s0, s1, 0.0, w)]
    if forma.tipo == "TRIANGULAR_DECRECIENTE":
        return [(s0, s1, w, 0.0)]
    if forma.tipo == "TRIANGULAR_SIMETRICA":
        sm = (s0 + s1) / 2.0
        return [(s0, sm, 0.0, w), (sm, s1, w, 0.0)]
    if forma.tipo == "TRAPEZOIDAL":
        return [(s0, s1, w, forma.w_fin)]
    # TRAPEZOIDAL_SIMETRICA
    extension = hasta - desde
    if not (0.0 < forma.rampa <= extension / 2.0 + 1e-9):
        raise ErrorAsignacion(
            f"El largo de cada rampa ({forma.rampa:g} m) debe ser mayor que 0 y no superar la mitad del "
            f"tramo cargado ({extension / 2.0:.4g} m)."
        )
    r = forma.rampa / longitud
    if s1 - s0 - 2 * r < 1e-9:
        return [(s0, (s0 + s1) / 2.0, 0.0, w), ((s0 + s1) / 2.0, s1, w, 0.0)]
    return [(s0, s0 + r, 0.0, w), (s0 + r, s1 - r, w, w), (s1 - r, s1, w, 0.0)]


@dataclass
class ParametrosCarga:
    """Lo que el usuario pide en el diálogo de cargas.

    ``tipo``: "PRESION" (kPa sobre una cara o losa), "LINEAL" (kN/m sobre una
    barra), "PUNTUAL" (fuerza en kN), "MOMENTO" (kN·m) o "AUTO" (la que
    corresponda a lo seleccionado: presión en losas y caras, lineal en
    barras completas).

    ``direccion``: vector global de aplicación; ``None`` = la normal de la
    cara hacia dentro, o gravedad si se seleccionó un elemento completo.
    """

    tipo: str = "AUTO"
    presion: float = 0.0
    w: float = 0.0
    fuerza: float = 0.0
    momento: float = 0.0
    posicion: float = 0.0  # m desde el nodo inicial (puntuales sobre barras)
    direccion: Optional[Vector3] = None
    forma: FormaCarga = field(default_factory=FormaCarga)
    perfil_presion: Optional[PerfilPresion] = None


def _unitario(v: Sequence[float]) -> Vector3:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise ErrorAsignacion("La dirección de la carga no puede ser el vector cero.")
    return (v[0] / n, v[1] / n, v[2] / n)


def _direccion_de_carga(parametros: ParametrosCarga, cara: Optional[Cara]) -> Vector3:
    if parametros.direccion is not None:
        return _unitario(parametros.direccion)
    if cara is not None:
        return tuple(-c for c in _vector(cara.normal))  # type: ignore[return-value]
    return GRAVEDAD


def _lineales_de_forma(
    proyecto: Proyecto, objetivo: Objetivo, caso_id: str, id_: str, w: float, parametros: ParametrosCarga,
    direccion: Vector3, texto: str,
) -> List[CargaLineal]:
    if abs(w) < 1e-12:
        raise ErrorAsignacion(f"{descripcion(proyecto, objetivo)}: la carga lineal es cero.")
    longitud = proyecto.longitud_barra(objetivo.id)
    trozos = segmentos_de_forma(parametros.forma, w, longitud)
    nombre_forma = FORMAS_CARGA[parametros.forma.tipo].split(" (")[0].lower()
    cargas = []
    for k, (s0, s1, w0, w1) in enumerate(trozos, start=1):
        cargas.append(
            CargaLineal(
                id=id_ if len(trozos) == 1 else f"{id_}_{k}", caso_id=caso_id, elemento_id=objetivo.id,
                w_inicio=w0, w_fin=w1, s_inicio=s0, s_fin=s1, direccion=direccion,
                nombre=f"{descripcion(proyecto, objetivo)}: {nombre_forma} {texto}"
                + (f" [{k}/{len(trozos)}]" if len(trozos) > 1 else ""),
            )
        )
    return cargas


def cargas_desde_parametros(
    proyecto: Proyecto, objetivo: Objetivo, caso_id: str, id_: str, parametros: ParametrosCarga
) -> List[Carga]:
    """Cargas que resultan de aplicar ``parametros`` sobre la selección."""

    nombre_objetivo = descripcion(proyecto, objetivo)
    cara = cara_de(proyecto, objetivo)
    tipo = parametros.tipo

    if objetivo.tipo == "MURO":
        raise ErrorAsignacion(f"{nombre_objetivo}: las cargas sobre muros no se transfieren en la Fase 1 (son informativos).")

    if tipo == "AUTO":
        tipo = "PRESION" if (objetivo.tipo == "LOSA" or cara is not None) else "LINEAL"

    # -- Presión (kPa) -------------------------------------------------------
    if tipo == "PRESION":
        perfil = parametros.perfil_presion
        valor = parametros.presion if perfil is None else (perfil.valor_a + perfil.valor_b) / 2.0
        if abs(valor) < 1e-12 and (perfil is None or (abs(perfil.valor_a) < 1e-12 and abs(perfil.valor_b) < 1e-12)):
            raise ErrorAsignacion(f"{nombre_objetivo}: la presión es cero.")
        if objetivo.tipo == "LOSA":
            if cara is not None and cara.tipo not in ("SUPERIOR", "INFERIOR"):
                raise ErrorAsignacion(
                    f"{nombre_objetivo}: las cargas de área van sobre la cara superior o inferior de la losa, "
                    "no sobre sus bordes."
                )
            texto = (
                f"{_formato(valor)} kPa"
                if perfil is None
                else f"{_formato(perfil.valor_a)} → {_formato(perfil.valor_b)} kPa"
            )
            return [
                CargaArea(
                    id=id_, caso_id=caso_id, presion=valor, losa_id=objetivo.id,
                    direccion=_direccion_de_carga(parametros, cara), perfil=perfil,
                    nombre=f"{nombre_objetivo}: {texto}",
                )
            ]
        if cara is None:
            raise ErrorAsignacion(
                f"{nombre_objetivo}: para aplicar una presión seleccione una cara (modo «Cara»); "
                "sobre la viga o columna completa use una carga lineal (kN/m)."
            )
        direccion = _direccion_de_carga(parametros, cara)
        if cara.tipo == "LATERAL":
            linea = valor * cara.ancho
            return _lineales_de_forma(
                proyecto, objetivo, caso_id, id_, linea, parametros, direccion,
                f"{_formato(valor)} kPa × {cara.ancho:.3g} m = {linea:.4g} kN/m",
            )
        fuerza = valor * cara.area  # cara extrema: fuerza en el nodo del extremo
        fx, fy, fz = (fuerza * c for c in direccion)
        return [
            CargaPuntual(
                id=id_, caso_id=caso_id, nodo_id=cara.nodo_id, fx=fx, fy=fy, fz=fz,
                nombre=f"{nombre_objetivo}: {_formato(valor)} kPa × {cara.area:.3g} m² = {fuerza:.4g} kN",
            )
        ]

    # -- Cargas sobre barras ---------------------------------------------------
    if objetivo.tipo not in ("VIGA", "COLUMNA"):
        raise ErrorAsignacion(
            f"{nombre_objetivo}: las cargas lineales y puntuales van sobre vigas y columnas; "
            "a las losas se les asigna una presión (kPa)."
        )
    direccion = _direccion_de_carga(parametros, cara)

    if tipo == "LINEAL":
        if cara is not None and cara.tipo not in ("LATERAL",):
            raise ErrorAsignacion(
                f"{nombre_objetivo}: la carga lineal se aplica sobre la barra o sobre una cara lateral, "
                "no sobre un extremo."
            )
        return _lineales_de_forma(
            proyecto, objetivo, caso_id, id_, parametros.w, parametros, direccion, f"{_formato(parametros.w)} kN/m"
        )

    if tipo in ("PUNTUAL", "MOMENTO"):
        magnitud = parametros.fuerza if tipo == "PUNTUAL" else parametros.momento
        unidad = "kN" if tipo == "PUNTUAL" else "kN·m"
        if abs(magnitud) < 1e-12:
            raise ErrorAsignacion(f"{nombre_objetivo}: el valor es cero.")
        vector = tuple(magnitud * c for c in direccion)
        campos = ("fx", "fy", "fz") if tipo == "PUNTUAL" else ("mx", "my", "mz")
        valores = dict(zip(campos, vector))
        barra = proyecto.elemento_barra(objetivo.id)
        if cara is not None:
            if cara.nodo_id is None:
                raise ErrorAsignacion(
                    f"{nombre_objetivo}: una carga puntual va en un punto de la barra (elemento completo) "
                    "o en su cara extrema."
                )
            ubicacion = {"nodo_id": cara.nodo_id}
            donde = f"extremo {cara.nodo_id}"
        else:
            longitud = proyecto.longitud_barra(objetivo.id)
            if not (-1e-9 <= parametros.posicion <= longitud + 1e-9):
                raise ErrorAsignacion(
                    f"{nombre_objetivo}: la posición ({parametros.posicion:g} m) debe estar entre 0 y {longitud:.4g} m."
                )
            fraccion = min(1.0, max(0.0, parametros.posicion / longitud))
            if fraccion < 1e-6:
                ubicacion, donde = {"nodo_id": barra.nodo_inicio}, "inicio"
            elif fraccion > 1.0 - 1e-6:
                ubicacion, donde = {"nodo_id": barra.nodo_fin}, "fin"
            else:
                ubicacion, donde = {"elemento_id": objetivo.id, "posicion": fraccion}, f"a {parametros.posicion:g} m"
        return [
            CargaPuntual(
                id=id_, caso_id=caso_id, **ubicacion, **valores,
                nombre=f"{nombre_objetivo}: {_formato(magnitud)} {unidad} ({donde})",
            )
        ]

    raise ErrorAsignacion(f"Tipo de carga desconocido: {tipo}")


def cargas_para_objetivo(
    proyecto: Proyecto,
    objetivo: Objetivo,
    caso_id: str,
    id_: str,
    presion: float,
    w: float,
    direccion: Vector3 = GRAVEDAD,
) -> List[Carga]:
    """Versión simple: presión en losas y caras (normal a la cara) o carga
    lineal uniforme sobre la barra completa en ``direccion``."""

    parametros = ParametrosCarga(
        tipo="AUTO", presion=presion, w=w, direccion=direccion if objetivo.cara is None else None
    )
    return cargas_desde_parametros(proyecto, objetivo, caso_id, id_, parametros)


def vista_previa_cargas(
    proyecto: Proyecto, objetivos: List[Objetivo], presion, w: float = 0.0, direccion: Vector3 = GRAVEDAD
) -> List[Tuple[bool, str]]:
    """Qué cargas se crearían para cada objetivo, o por qué no se puede.

    ``presion`` puede ser un número (versión simple: ``presion``, ``w`` y
    ``direccion``) o un ``ParametrosCarga``. Devuelve [(es_valido, texto)]
    en el orden de ``objetivos``.
    """

    resultado: List[Tuple[bool, str]] = []
    for objetivo in objetivos:
        try:
            if isinstance(presion, ParametrosCarga):
                cargas = cargas_desde_parametros(proyecto, objetivo, "-", "-", presion)
            else:
                cargas = cargas_para_objetivo(proyecto, objetivo, "-", "-", presion, w, direccion)
            resultado.append((True, "; ".join(c.nombre for c in cargas)))
        except ErrorAsignacion as exc:
            resultado.append((False, str(exc)))
    return resultado


# ---------------------------------------------------------------------------
# Apoyos de área (continuos)
# ---------------------------------------------------------------------------

Restricciones = Tuple[bool, bool, bool, bool, bool, bool]


def restriccion_solo_normal(cara: Optional[Cara]) -> Optional[Restricciones]:
    """Restricción de la dirección normal a la cara (apoyo "de rodillo"), si
    esa normal coincide con un eje global; si no, ``None``."""

    if cara is None:
        return None
    n = _vector(cara.normal)
    for i in range(3):
        if abs(abs(n[i]) - 1.0) < 1e-6:
            banderas = [False] * 6
            banderas[i] = True
            return tuple(banderas)  # type: ignore[return-value]
    return None


def barras_de_apoyo_area(proyecto: Proyecto, apoyo: ApoyoArea) -> List[str]:
    """Barras sobre las que actúa un apoyo de área (la propia barra, o las
    vigas de soporte si se asignó sobre una losa)."""

    if apoyo.elemento_tipo in ("VIGA", "COLUMNA"):
        return [apoyo.elemento_id] if apoyo.elemento_id in proyecto.todas_las_barras() else []
    losa = proyecto.losas.get(apoyo.elemento_id)
    if losa is None:
        return []
    return list(dict.fromkeys(b for b in losa.vigas_soporte if b and b in proyecto.todas_las_barras()))


def apoyo_para_objetivo(
    proyecto: Proyecto,
    objetivo: Objetivo,
    id_: str,
    restricciones: Restricciones,
    tipo: str = "AUTO",
    desde: float = 0.0,
    hasta: Optional[float] = None,
) -> Union[Apoyo, ApoyoArea]:
    """Apoyo que resulta de la selección.

    ``tipo``: "PUNTUAL" (en el nodo de una cara extrema), "AREA"
    (continuo a lo largo de la barra / de las vigas de la losa) o "AUTO"
    (puntual en una cara extrema, de área en cualquier otro caso).
    ``desde``/``hasta`` (m) acotan el tramo de un apoyo de área sobre una barra.
    """

    nombre_objetivo = descripcion(proyecto, objetivo)
    if objetivo.tipo == "MURO":
        raise ErrorAsignacion(f"{nombre_objetivo}: los muros son informativos en la Fase 1 y no admiten apoyos.")
    if not any(restricciones):
        raise ErrorAsignacion("Marque al menos un grado de libertad restringido.")
    cara = cara_de(proyecto, objetivo)
    es_extremo = cara is not None and cara.nodo_id is not None

    if tipo == "AUTO":
        tipo = "PUNTUAL" if es_extremo else "AREA"

    ux, uy, uz, rx, ry, rz = restricciones
    if tipo == "PUNTUAL":
        nodo = nodo_para_apoyo(proyecto, objetivo)
        return Apoyo(id=id_, nodo_id=nodo, ux=ux, uy=uy, uz=uz, rx=rx, ry=ry, rz=rz,
                     nombre=f"{nombre_objetivo} → apoyo en el nodo {nodo}")

    if objetivo.tipo == "LOSA":
        if cara is not None and cara.tipo not in ("SUPERIOR", "INFERIOR"):
            raise ErrorAsignacion(f"{nombre_objetivo}: el apoyo de área de una losa va en su cara superior o inferior.")
        losa = proyecto.losas[objetivo.id]
        if not any(losa.vigas_soporte):
            raise ErrorAsignacion(
                f"{nombre_objetivo}: la losa no tiene vigas de soporte; el apoyo de área se aplica a lo largo de ellas."
            )
        return ApoyoArea(id=id_, elemento_tipo="LOSA", elemento_id=objetivo.id, ux=ux, uy=uy, uz=uz, rx=rx, ry=ry,
                         rz=rz, nombre=f"{nombre_objetivo}: apoyo de área (a lo largo de sus vigas)")

    # Viga o columna: completa o una cara lateral (las caras extremas son puntuales)
    if cara is not None and cara.tipo != "LATERAL":
        raise ErrorAsignacion(f"{nombre_objetivo}: un apoyo de área va sobre una cara lateral o sobre la barra completa.")
    longitud = proyecto.longitud_barra(objetivo.id)
    fin = longitud if hasta is None else hasta
    if not (-1e-9 <= desde < fin <= longitud + 1e-9):
        raise ErrorAsignacion(
            f"{nombre_objetivo}: el tramo apoyado ({desde:g} a {fin:g} m) debe estar dentro de la barra (0 a {longitud:.4g} m)."
        )
    tramo = "" if (desde < 1e-9 and fin > longitud - 1e-9) else f" ({desde:g}–{fin:g} m)"
    return ApoyoArea(
        id=id_, elemento_tipo=objetivo.tipo, elemento_id=objetivo.id, ux=ux, uy=uy, uz=uz, rx=rx, ry=ry, rz=rz,
        s_inicio=max(0.0, desde / longitud), s_fin=min(1.0, fin / longitud),
        nombre=f"{nombre_objetivo}: apoyo de área{tramo}",
    )


# ---------------------------------------------------------------------------
# Conexiones (constraints) entre elementos
# ---------------------------------------------------------------------------

#: Distancia máxima (m) entre extremos de dos barras para proponer un enlace
#: sin que el usuario haya elegido sus caras extremas.
DISTANCIA_MAXIMA_ENLACE = 1.5


def _extremos(proyecto: Proyecto, objetivo: Objetivo) -> List[str]:
    """Nodos candidatos de una selección: el nodo de la cara extrema elegida
    o, si se eligió la barra, sus dos extremos."""

    if objetivo.tipo not in ("VIGA", "COLUMNA"):
        raise ErrorAsignacion(
            f"{descripcion(proyecto, objetivo)}: las conexiones se definen entre vigas y columnas "
            "(las losas se apoyan en sus vigas y los muros son informativos en la Fase 1)."
        )
    cara = cara_de(proyecto, objetivo)
    if cara is not None and cara.nodo_id is not None:
        return [cara.nodo_id]
    barra = proyecto.elemento_barra(objetivo.id)
    return [barra.nodo_inicio, barra.nodo_fin]


def conexion_para_objetivos(
    proyecto: Proyecto, a: Objetivo, b: Objetivo, id_: str, transmitidos: Restricciones, fijar_torsion: bool = True
) -> Conexion:
    """Conexión entre dos barras seleccionadas.

    * Si comparten un nudo: se crea una conexión de EXTREMO en ese nudo (la
      viga —o, si son del mismo tipo, la segunda seleccionada— se separa
      del nudo y se une transmitiendo solo los grados de libertad marcados).
    * Si no comparten nudo: se crea un ENLACE rígido entre sus extremos más
      cercanos (o entre las caras extremas elegidas).
    """

    if a.tipo not in ("VIGA", "COLUMNA") or b.tipo not in ("VIGA", "COLUMNA"):
        _extremos(proyecto, a)
        _extremos(proyecto, b)
    if a.id == b.id:
        raise ErrorAsignacion("Seleccione dos elementos distintos para conectarlos.")
    if not any(transmitidos):
        raise ErrorAsignacion("Marque al menos un grado de libertad que se transmita entre los elementos.")
    if len(set(transmitidos[3:])) != 1:
        raise ErrorAsignacion(
            "Los tres giros se transmiten juntos (unión rígida) o ninguno (articulación): CalculiX no permite "
            "articulaciones parciales en vigas."
        )

    nodos_a, nodos_b = _extremos(proyecto, a), _extremos(proyecto, b)
    ux, uy, uz, rx, ry, rz = transmitidos
    comunes = [n for n in nodos_a if n in nodos_b]
    if comunes:
        if all(transmitidos):
            raise ErrorAsignacion(
                "Estos elementos ya están unidos rígidamente en el nudo que comparten. Para articularlos, "
                "desmarque los giros (rotaciones)."
            )
        nodo = comunes[0]
        # La que se separa del nudo: la viga si la otra es columna; si no, la segunda.
        if a.tipo == "VIGA" and b.tipo == "COLUMNA":
            separada = a
        else:
            separada = b
        otra = a if separada is b else b
        return Conexion(
            id=id_, nodo_id=nodo, barra_id=separada.id, ux=ux, uy=uy, uz=uz, rx=rx, ry=ry, rz=rz,
            fijar_torsion=fijar_torsion, nombre=f"{descripcion(proyecto, separada)} ↔ {descripcion(proyecto, otra)} en {nodo}",
        )

    mejor = min(
        ((na, nb) for na in nodos_a for nb in nodos_b),
        key=lambda par: proyecto.nodos[par[0]].distancia(proyecto.nodos[par[1]]),
    )
    distancia = proyecto.nodos[mejor[0]].distancia(proyecto.nodos[mejor[1]])
    explicitas = len(nodos_a) == 1 and len(nodos_b) == 1
    if distancia > DISTANCIA_MAXIMA_ENLACE and not explicitas:
        raise ErrorAsignacion(
            f"{descripcion(proyecto, a)} y {descripcion(proyecto, b)} no comparten un nudo y sus extremos más "
            f"cercanos están a {distancia:.3g} m. Seleccione en modo «Cara» las caras extremas que quiere unir."
        )
    return Conexion(
        id=id_, nodo_id=mejor[0], nodo_destino=mejor[1], ux=ux, uy=uy, uz=uz, rx=rx, ry=ry, rz=rz,
        nombre=f"{descripcion(proyecto, a)} ↔ {descripcion(proyecto, b)}: enlace rígido de {distancia:.3g} m",
    )


# ---------------------------------------------------------------------------
# Materiales
# ---------------------------------------------------------------------------


def asignar_material(proyecto: Proyecto, objetivos: Sequence[Objetivo], nombre_material: str) -> List[str]:
    """Asigna un material a los elementos seleccionados (vigas, columnas,
    losas y muros). Devuelve las descripciones de los elementos cambiados."""

    if nombre_material not in proyecto.materiales:
        raise ErrorAsignacion(f"El material «{nombre_material}» no existe.")
    cambiados: List[str] = []
    for objetivo in dict.fromkeys((o.tipo, o.id) for o in objetivos):
        tipo, id_ = objetivo
        coleccion = {"VIGA": proyecto.vigas, "COLUMNA": proyecto.columnas, "LOSA": proyecto.losas, "MURO": proyecto.muros}[tipo]
        elemento = coleccion[id_]
        elemento.material = nombre_material
        cambiados.append(descripcion(proyecto, Objetivo(tipo, id_)))
    return cambiados


