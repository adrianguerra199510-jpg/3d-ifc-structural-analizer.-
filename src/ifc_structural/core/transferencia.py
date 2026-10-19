"""Transferencia de cargas de área a cargas lineales sobre vigas.

Ver README.md ("Transferencia de cargas de área → vigas") para la
justificación de usar áreas tributarias con líneas de reparto a 45°, en
lugar de líneas de fluencia, para losas en dos direcciones.

Resumen del método:

- Losa en una dirección: la carga se reparte por mitades sobre las dos
  vigas que delimitan la luz de la losa. Carga lineal uniforme sobre cada
  viga = presión × (luz / 2).

- Losa en dos direcciones, panel rectangular: para cada lado del
  rectángulo, con longitud propia ``Le`` y profundidad perpendicular del
  panel ``Lp`` (longitud de los lados adyacentes), el ancho tributario a lo
  largo del lado, parametrizado por ``s`` en ``[0, Le]`` desde un extremo,
  es:

      ancho(s) = min(s, Le - s, Lp / 2)

  Esta es la expresión analítica exacta de la partición de un rectángulo
  por las bisectrices a 45° trazadas desde cada esquina. Da un perfil
  triangular cuando ``Le <= Lp`` (lado corto) y trapezoidal cuando
  ``Le > Lp`` (lado largo, con meseta en ``Lp / 2``).

- Losa en dos direcciones, panel no rectangular: no existe una fórmula
  cerrada simple y una partición numérica exacta excede el alcance de la
  Fase 1. Se usa un reparto proporcional a la longitud de cada lado
  (carga uniforme por viga), que se documenta explícitamente como
  aproximación en los resultados.

Las cargas resultantes siempre se devuelven como objetos ``CargaLineal``
(posiblemente varios segmentos por viga, con ``s_inicio``/``s_fin``, para
representar perfiles trapezoidales/triangulares), listas para usarse en el
generador de .inp o para mostrarse en el visor 3D como "cargas lineales
equivalentes".
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .cargas import CargaArea, CargaLineal, SistemaReferencia
from .modelo import DireccionLosa, Losa, Proyecto

Vector3 = Tuple[float, float, float]

_TOL = 1e-6


def _resta(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cruz(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _punto(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norma(v: Vector3) -> float:
    return math.sqrt(_punto(v, v))


def _normalizar(v: Vector3) -> Vector3:
    n = _norma(v)
    if n < _TOL:
        raise ValueError("No se puede normalizar un vector de longitud ~0")
    return (v[0] / n, v[1] / n, v[2] / n)


class ErrorTransferencia(ValueError):
    """Error de datos al transferir una carga de área (losa mal configurada)."""


def _coordenadas_contorno(proyecto: Proyecto, losa: Losa) -> List[Vector3]:
    return [proyecto.nodos[nid].coordenadas for nid in losa.nodos_contorno]


def _longitud_lado(puntos: List[Vector3], i: int) -> float:
    n = len(puntos)
    return _norma(_resta(puntos[(i + 1) % n], puntos[i]))


def _validar_losa(losa: Losa) -> None:
    n = len(losa.nodos_contorno)
    if n < 3:
        raise ErrorTransferencia(f"La losa {losa.id} necesita al menos 3 nodos de contorno")
    if len(losa.vigas_soporte) != n:
        raise ErrorTransferencia(
            f"La losa {losa.id} tiene {n} lados pero {len(losa.vigas_soporte)} "
            "vigas de soporte asignadas; deben coincidir en cantidad y orden."
        )
    if not any(losa.vigas_soporte):
        raise ErrorTransferencia(
            f"La losa {losa.id} no tiene ninguna viga de soporte asignada; "
            "no se puede transferir la carga de área."
        )


def _es_rectangulo(puntos: List[Vector3]) -> bool:
    if len(puntos) != 4:
        return False
    lados = [_resta(puntos[(i + 1) % 4], puntos[i]) for i in range(4)]
    longitudes = [_norma(l) for l in lados]
    # Lados opuestos de igual longitud.
    if abs(longitudes[0] - longitudes[2]) > 1e-3 or abs(longitudes[1] - longitudes[3]) > 1e-3:
        return False
    # Ángulos rectos entre lados consecutivos.
    for i in range(4):
        a = lados[i]
        b = lados[(i + 1) % 4]
        if any(l < _TOL for l in (longitudes[i], longitudes[(i + 1) % 4])):
            return False
        coseno = _punto(a, b) / (longitudes[i] * longitudes[(i + 1) % 4])
        if abs(coseno) > 1e-3:
            return False
    return True


def _nombre_segmento(carga: CargaArea, viga_id: str, indice: int) -> str:
    return f"{carga.id}_{viga_id}_seg{indice}"


def _segmentos_desde_perfil(
    carga: CargaArea,
    viga_id: str,
    estaciones: List[Tuple[float, float]],
    longitud_lado: float,
    invertir: bool,
) -> List[CargaLineal]:
    """Convierte una lista de puntos (s, ancho) de un perfil lineal a trozos
    en objetos CargaLineal con s_inicio/s_fin normalizados a [0, 1].

    ``invertir`` se usa cuando el sentido de recorrido del lado de la losa
    (desde nodos_contorno[i] a nodos_contorno[i+1]) es opuesto al sentido
    nodo_inicio -> nodo_fin de la viga real.
    """

    segmentos: List[CargaLineal] = []
    for idx in range(len(estaciones) - 1):
        s0, w0 = estaciones[idx]
        s1, w1 = estaciones[idx + 1]
        if abs(w0) < 1e-9 and abs(w1) < 1e-9:
            continue
        frac0 = s0 / longitud_lado
        frac1 = s1 / longitud_lado
        presion_a_carga0 = carga.presion * w0
        presion_a_carga1 = carga.presion * w1
        if invertir:
            frac0, frac1 = 1.0 - frac0, 1.0 - frac1
            presion_a_carga0, presion_a_carga1 = presion_a_carga1, presion_a_carga0
        s_inicio, s_fin = sorted((frac0, frac1))
        w_inicio, w_fin = (presion_a_carga0, presion_a_carga1) if frac0 <= frac1 else (
            presion_a_carga1,
            presion_a_carga0,
        )
        segmentos.append(
            CargaLineal(
                id=_nombre_segmento(carga, viga_id, idx),
                caso_id=carga.caso_id,
                elemento_id=viga_id,
                w_inicio=w_inicio,
                w_fin=w_fin,
                s_inicio=s_inicio,
                s_fin=s_fin,
                direccion=carga.direccion,
                sistema=SistemaReferencia.GLOBAL,
                origen="TRANSFERENCIA_LOSA",
                losa_origen_id=carga.losa_id,
            )
        )
    return segmentos


def _sentido_invertido(proyecto: Proyecto, viga_id: str, nodo_inicio_lado: str) -> bool:
    """True si el nodo_inicio de la viga NO coincide con el inicio del lado
    de la losa (es decir, viga y lado se recorren en sentidos opuestos)."""

    barra = proyecto.elemento_barra(viga_id)
    return barra.nodo_inicio != nodo_inicio_lado


def _transferir_una_direccion(proyecto: Proyecto, losa: Losa, carga: CargaArea) -> List[CargaLineal]:
    puntos = _coordenadas_contorno(proyecto, losa)
    n = len(puntos)
    lados_con_viga = [i for i in range(n) if losa.vigas_soporte[i]]
    if len(lados_con_viga) != 2:
        raise ErrorTransferencia(
            f"La losa {losa.id} está marcada como de una dirección pero tiene "
            f"{len(lados_con_viga)} lados con viga asignada (se esperaban 2: las "
            "vigas que delimitan la luz de la losa)."
        )
    i_a, i_b = lados_con_viga
    centro_a = _punto_medio(puntos, i_a)
    centro_b = _punto_medio(puntos, i_b)
    luz = _norma(_resta(centro_b, centro_a))
    if luz < _TOL:
        raise ErrorTransferencia(f"La losa {losa.id} tiene luz nula entre sus vigas de soporte")

    ancho_tributario = luz / 2.0
    resultado: List[CargaLineal] = []
    for i in (i_a, i_b):
        viga_id = losa.vigas_soporte[i]
        assert viga_id is not None
        w = carga.presion * ancho_tributario
        invertir = _sentido_invertido(proyecto, viga_id, losa.nodos_contorno[i])
        resultado.append(
            CargaLineal(
                id=_nombre_segmento(carga, viga_id, 0),
                caso_id=carga.caso_id,
                elemento_id=viga_id,
                w_inicio=w,
                w_fin=w,
                s_inicio=0.0,
                s_fin=1.0,
                direccion=carga.direccion,
                origen="TRANSFERENCIA_LOSA",
                losa_origen_id=losa.id,
            )
        )
        _ = invertir  # uniforme: el sentido no afecta el resultado
    return resultado


def _punto_medio(puntos: List[Vector3], i: int) -> Vector3:
    n = len(puntos)
    a, b = puntos[i], puntos[(i + 1) % n]
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def _transferir_rectangular_dos_direcciones(
    proyecto: Proyecto, losa: Losa, carga: CargaArea
) -> List[CargaLineal]:
    puntos = _coordenadas_contorno(proyecto, losa)
    longitudes = [_longitud_lado(puntos, i) for i in range(4)]

    resultado: List[CargaLineal] = []
    for i in range(4):
        viga_id = losa.vigas_soporte[i]
        if not viga_id:
            continue
        le = longitudes[i]
        lp = longitudes[(i + 1) % 4]  # longitud del lado adyacente = profundidad del panel
        if le < _TOL:
            continue

        # Estaciones del perfil ancho(s) = min(s, le - s, lp/2), muestreado en
        # sus puntos de quiebre exactos (triángulo o trapecio).
        s_pico = min(le / 2.0, lp / 2.0)
        if s_pico >= le / 2.0 - 1e-9:
            # Perfil triangular puro (lado corto): 0 -> le/2 -> le
            estaciones = [(0.0, 0.0), (le / 2.0, le / 2.0), (le, 0.0)]
        else:
            # Perfil trapezoidal (lado largo): 0 -> lp/2 -> meseta -> lp/2 -> 0
            estaciones = [
                (0.0, 0.0),
                (lp / 2.0, lp / 2.0),
                (le - lp / 2.0, lp / 2.0),
                (le, 0.0),
            ]

        invertir = _sentido_invertido(proyecto, viga_id, losa.nodos_contorno[i])
        resultado.extend(_segmentos_desde_perfil(carga, viga_id, estaciones, le, invertir))
    return resultado


def _transferir_proporcional(proyecto: Proyecto, losa: Losa, carga: CargaArea) -> List[CargaLineal]:
    """Reparto aproximado para paneles no rectangulares: carga uniforme en
    cada viga, proporcional a la fracción del perímetro tributario que le
    corresponde a su lado, de modo que la carga total resultante coincide
    con presión × área de la losa.
    """

    import warnings

    puntos = _coordenadas_contorno(proyecto, losa)
    n = len(puntos)
    area_losa = _area_poligono(puntos)
    perimetro_con_viga = sum(
        _longitud_lado(puntos, i) for i in range(n) if losa.vigas_soporte[i]
    )
    if perimetro_con_viga < _TOL:
        raise ErrorTransferencia(f"La losa {losa.id} no tiene ningún lado con viga asignada")

    warnings.warn(
        f"La losa {losa.id} no es un rectángulo: se usa un reparto "
        "proporcional a la longitud de cada lado como aproximación de las "
        "áreas tributarias (en vez del método exacto a 45°).",
        stacklevel=2,
    )

    carga_total = carga.presion * area_losa
    resultado: List[CargaLineal] = []
    for i in range(n):
        viga_id = losa.vigas_soporte[i]
        if not viga_id:
            continue
        le = _longitud_lado(puntos, i)
        fraccion = le / perimetro_con_viga
        carga_total_viga = carga_total * fraccion
        w = carga_total_viga / le
        resultado.append(
            CargaLineal(
                id=_nombre_segmento(carga, viga_id, 0),
                caso_id=carga.caso_id,
                elemento_id=viga_id,
                w_inicio=w,
                w_fin=w,
                s_inicio=0.0,
                s_fin=1.0,
                direccion=carga.direccion,
                origen="TRANSFERENCIA_LOSA",
                losa_origen_id=losa.id,
            )
        )
    return resultado


def _area_poligono(puntos: List[Vector3]) -> float:
    """Área de un polígono plano en 3D, vía la magnitud de la suma de
    productos cruzados (fórmula del "shoelace" generalizada a 3D)."""

    n = len(puntos)
    origen = puntos[0]
    acumulado = (0.0, 0.0, 0.0)
    for i in range(1, n - 1):
        a = _resta(puntos[i], origen)
        b = _resta(puntos[i + 1], origen)
        c = _cruz(a, b)
        acumulado = (acumulado[0] + c[0], acumulado[1] + c[1], acumulado[2] + c[2])
    return _norma(acumulado) / 2.0


def transferir_carga_area(proyecto: Proyecto, carga: CargaArea) -> List[CargaLineal]:
    """Convierte una CargaArea en una lista de CargaLineal equivalentes sobre
    las vigas de soporte de la losa.

    Las cargas de área sobre muros no se transfieren automáticamente en la
    Fase 1 (los muros son informativos); se devuelve una lista vacía.
    """

    if carga.muro_id is not None:
        return []

    assert carga.losa_id is not None
    losa = proyecto.losas[carga.losa_id]
    _validar_losa(losa)

    if losa.direccion == DireccionLosa.UNA_DIRECCION:
        return _transferir_una_direccion(proyecto, losa, carga)

    puntos = _coordenadas_contorno(proyecto, losa)
    if len(losa.nodos_contorno) == 4 and _es_rectangulo(puntos):
        return _transferir_rectangular_dos_direcciones(proyecto, losa, carga)

    return _transferir_proporcional(proyecto, losa, carga)


# ---------------------------------------------------------------------------
# Peso propio
# ---------------------------------------------------------------------------


def peso_propio_barras(proyecto: Proyecto, caso_id: str) -> List[CargaLineal]:
    """Carga lineal uniforme de peso propio para cada viga y columna.

    peso_propio [kN/m] = densidad del material [kN/m3] x área de la sección [m2]
    aplicado en la dirección global -Z (gravedad).
    """

    cargas: List[CargaLineal] = []
    for id_, barra in proyecto.todas_las_barras().items():
        material = proyecto.materiales[barra.material]
        perfil = proyecto.perfiles[barra.perfil]
        w = material.densidad * perfil.area
        cargas.append(
            CargaLineal(
                id=f"PP_{id_}",
                caso_id=caso_id,
                elemento_id=id_,
                w_inicio=w,
                w_fin=w,
                s_inicio=0.0,
                s_fin=1.0,
                direccion=(0.0, 0.0, -1.0),
                origen="MANUAL",
                nombre=f"Peso propio {id_}",
            )
        )
    return cargas


def cargas_area_peso_propio_losas(proyecto: Proyecto, caso_id: str) -> List[CargaArea]:
    """Genera una CargaArea de peso propio (presión = densidad x espesor)
    para cada losa, lista para transferirse con transferir_carga_area().
    """

    cargas: List[CargaArea] = []
    for id_, losa in proyecto.losas.items():
        material = proyecto.materiales[losa.material]
        presion = material.densidad * losa.espesor
        cargas.append(
            CargaArea(
                id=f"PP_{id_}",
                caso_id=caso_id,
                presion=presion,
                losa_id=id_,
                direccion=(0.0, 0.0, -1.0),
                origen="PESO_PROPIO",
                nombre=f"Peso propio {id_}",
            )
        )
    return cargas
