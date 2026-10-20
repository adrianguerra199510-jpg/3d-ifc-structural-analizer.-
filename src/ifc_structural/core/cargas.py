"""Cargas y apoyos.

Unidades: fuerzas puntuales en kN, cargas lineales en kN/m, cargas de área
en kPa (kN/m2), momentos en kN·m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

Vector3 = Tuple[float, float, float]


class SistemaReferencia(str, Enum):
    GLOBAL = "GLOBAL"
    LOCAL = "LOCAL"


@dataclass
class Apoyo:
    """Restricción de desplazamiento/rotación en un nodo.

    Cada bandera indica si ese grado de libertad está restringido (True) o
    libre (False). ``ux, uy, uz`` son traslaciones; ``rx, ry, rz`` rotaciones
    (estas últimas solo aplican a nodos que pertenecen a elementos tipo
    barra, ya que los elementos de losa en modo transferencia no tienen
    grados de libertad rotacionales propios en Fase 1).
    """

    id: str
    nodo_id: str
    ux: bool = True
    uy: bool = True
    uz: bool = True
    rx: bool = True
    ry: bool = True
    rz: bool = True
    nombre: str = ""

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id

    def grados_restringidos(self) -> List[int]:
        """Números de grado de libertad (1-6, convención CalculiX) restringidos."""

        banderas = [self.ux, self.uy, self.uz, self.rx, self.ry, self.rz]
        return [i + 1 for i, restringido in enumerate(banderas) if restringido]


@dataclass
class ApoyoArea:
    """Apoyo continuo ("de área") sobre un elemento: restringe los grados de
    libertad elegidos en TODOS los puntos de la barra (o del tramo
    ``s_inicio``..``s_fin``, fracciones 0..1 desde su nodo inicial).

    * Sobre una viga o columna (elemento completo o una de sus caras
      laterales): la barra queda apoyada a lo largo, como sobre un muro o el
      terreno.
    * Sobre una losa (elemento o cara superior/inferior): la losa no se
      malla en la Fase 1 (se apoya en sus vigas), así que el apoyo se aplica
      a lo largo de las vigas de soporte de la losa.
    """

    id: str
    elemento_tipo: str  # "VIGA", "COLUMNA" o "LOSA"
    elemento_id: str
    ux: bool = True
    uy: bool = True
    uz: bool = True
    rx: bool = False
    ry: bool = False
    rz: bool = False
    s_inicio: float = 0.0
    s_fin: float = 1.0
    nombre: str = ""

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id
        if not (0.0 <= self.s_inicio < self.s_fin <= 1.0 + 1e-9):
            raise ValueError("El tramo del apoyo debe cumplir 0 <= s_inicio < s_fin <= 1")

    def grados_restringidos(self) -> List[int]:
        banderas = [self.ux, self.uy, self.uz, self.rx, self.ry, self.rz]
        return [i + 1 for i, restringido in enumerate(banderas) if restringido]


@dataclass
class ApoyoSuperficie:
    """Apoyo sobre una superficie cualquiera (una cara de un sólido o de una barra, losa o muro).

    Restringe los grados de libertad elegidos en todos los nodos de la superficie. Con las tres
    traslaciones restringidas la superficie queda empotrada; con solo una (p. ej. Uz) es un apoyo
    deslizante sobre el terreno o sobre otra pieza. Los giros (Rx, Ry, Rz) hacen que la superficie
    se mueva como cuerpo rígido con esos giros impedidos. Solo se usa en el análisis 3D.
    """

    id: str
    elemento_tipo: str  # "SOLIDO", "VIGA", "COLUMNA", "LOSA" o "MURO"
    elemento_id: str
    cara: int
    ux: bool = True
    uy: bool = True
    uz: bool = True
    rx: bool = False
    ry: bool = False
    rz: bool = False
    nombre: str = ""
    nodo_id: Optional[str] = None  # sin uso: lo leen por compatibilidad los apoyos nodales del análisis de barras

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id

    def grados_restringidos(self) -> List[int]:
        banderas = [self.ux, self.uy, self.uz, self.rx, self.ry, self.rz]
        return [i + 1 for i, restringido in enumerate(banderas) if restringido]


@dataclass
class CargaSuperficie:
    """Carga sobre una superficie cualquiera, repartida por toda su área. Solo se usa en el análisis 3D.

    * ``PRESION`` (kPa): ``valor`` × área. Sin ``direccion`` actúa normal a la superficie, hacia dentro
      (compresión si es positiva); con ``direccion`` es una tracción con ese sentido.
    * ``FUERZA`` (kN): fuerza total ``valor`` en ``direccion`` repartida uniformemente.
    * ``MOMENTO`` (kN·m): momento total ``valor`` alrededor del eje ``direccion`` (regla de la mano
      derecha) respecto del centro de la superficie, repartido como fuerzas proporcionales a la distancia.
    """

    id: str
    caso_id: str
    elemento_tipo: str
    elemento_id: str
    cara: int
    tipo: str  # "PRESION", "FUERZA" o "MOMENTO"
    valor: float
    direccion: Optional[Vector3] = None
    nombre: str = ""

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id
        if self.tipo not in ("PRESION", "FUERZA", "MOMENTO"):
            raise ValueError(f"Tipo de carga de superficie desconocido: {self.tipo}")
        if self.tipo in ("FUERZA", "MOMENTO") and self.direccion is None:
            raise ValueError("Una fuerza o un momento sobre una superficie necesita una dirección")


@dataclass
class Conexion:
    """Restricción (constraint) entre dos elementos sólidos.

    Dos tipos, según el caso:

    * **EXTREMO** (``barra_id`` indicado, ``nodo_destino`` vacío): el extremo
      de ``barra_id`` que está en ``nodo_id`` se separa del nudo y se vuelve
      a unir con él transmitiendo SOLO las traslaciones marcadas (``ux, uy,
      uz``) y liberando los tres giros: una articulación (rótula) esférica.
      Con ``fijar_torsion`` (por defecto) se impide además el giro del extremo
      alrededor del eje de la barra (la torsión), como hacen los modelos
      habituales de articulación viga-columna; si no, una barra articulada
      por ambos extremos puede girar sobre su propio eje (mecanismo).
    * **ENLACE** (``nodo_destino`` indicado): une dos nudos distintos. Con los
      seis grados transmitidos es un enlace rígido (brazo infinitamente
      rígido: el nudo destino se mueve como sólido rígido con el origen); con
      solo traslaciones, un enlace articulado (las traslaciones marcadas son
      iguales en ambos nudos).

    Limitación de CalculiX (comprobada con la versión 2.21): las ecuaciones
    sobre los giros de las vigas B31 no se aplican, de modo que los giros se
    transmiten todos (unión continua) o ninguno (articulación); no hay
    articulaciones parciales (p. ej. liberar solo un giro).
    """

    id: str
    nodo_id: str
    barra_id: Optional[str] = None
    nodo_destino: Optional[str] = None
    ux: bool = True
    uy: bool = True
    uz: bool = True
    rx: bool = False
    ry: bool = False
    rz: bool = False
    fijar_torsion: bool = True
    nombre: str = ""

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id
        if (self.barra_id is None) == (self.nodo_destino is None):
            raise ValueError("Conexion debe indicar 'barra_id' (extremo) o 'nodo_destino' (enlace), no ambos ni ninguno")
        if len({self.rx, self.ry, self.rz}) != 1:
            raise ValueError("Los tres giros se transmiten juntos o no se transmite ninguno (limitación de CalculiX)")

    @property
    def tipo(self) -> str:
        return "EXTREMO" if self.barra_id is not None else "ENLACE"

    @property
    def transmite_giros(self) -> bool:
        return self.rx and self.ry and self.rz

    @property
    def es_enlace_rigido(self) -> bool:
        """Enlace con los seis grados transmitidos (se modela con un tramo rígido)."""

        return self.nodo_destino is not None and self.transmite_giros and self.ux and self.uy and self.uz

    def grados_transmitidos(self) -> List[int]:
        banderas = [self.ux, self.uy, self.uz, self.rx, self.ry, self.rz]
        return [i + 1 for i, activo in enumerate(banderas) if activo]

    def traslaciones_transmitidas(self) -> List[int]:
        return [g for g in self.grados_transmitidos() if g <= 3]


@dataclass
class CargaPuntual:
    """Fuerza y/o momento puntual, en el sistema global.

    Puede aplicarse directamente sobre un nodo existente (``nodo_id``) o
    sobre un punto intermedio de una viga/columna (``elemento_id`` +
    ``posicion``, fracción 0..1 desde su nodo de inicio). En este segundo
    caso, el generador de .inp inserta un nodo real en esa posición al
    construir la malla de análisis.
    """

    id: str
    caso_id: str
    nodo_id: Optional[str] = None
    elemento_id: Optional[str] = None
    posicion: Optional[float] = None
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    nombre: str = ""

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id
        sobre_nodo = self.nodo_id is not None
        sobre_elemento = self.elemento_id is not None and self.posicion is not None
        if sobre_nodo == sobre_elemento:
            raise ValueError(
                "CargaPuntual debe indicar exactamente una ubicación: "
                "'nodo_id', o bien 'elemento_id' + 'posicion'."
            )
        if sobre_elemento and not (0.0 <= self.posicion <= 1.0):
            raise ValueError("'posicion' debe estar entre 0.0 y 1.0")


@dataclass
class CargaLineal:
    """Carga distribuida sobre una viga o columna, en el sistema global.

    En el caso general (carga transferida desde una losa) la intensidad
    puede variar linealmente a lo largo del elemento entre ``w_inicio`` y
    ``w_fin`` (en kN/m); para una carga uniforme simple ambos valores son
    iguales. La dirección de la carga es el vector unitario
    ``direccion`` (por defecto, gravedad: -Z global).
    """

    id: str
    caso_id: str
    elemento_id: str
    w_inicio: float
    w_fin: float
    s_inicio: float = 0.0  # posición relativa (0..1) sobre el elemento, desde nodo_inicio
    s_fin: float = 1.0
    direccion: Vector3 = (0.0, 0.0, -1.0)
    sistema: SistemaReferencia = SistemaReferencia.GLOBAL
    nombre: str = ""
    origen: str = "MANUAL"  # "MANUAL" o "TRANSFERENCIA_LOSA"
    losa_origen_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id

    @property
    def es_uniforme(self) -> bool:
        return abs(self.w_inicio - self.w_fin) < 1e-9

    @property
    def es_parcial(self) -> bool:
        return self.s_inicio > 1e-9 or self.s_fin < 1.0 - 1e-9


@dataclass
class PerfilPresion:
    """Variación lineal de una presión a lo largo de un eje global: vale
    ``valor_a`` en la coordenada ``coord_a`` y ``valor_b`` en ``coord_b``
    (constante fuera de ese intervalo). Con ``valor_a = 0`` es una carga
    triangular; con ambos distintos de cero, trapezoidal."""

    eje: int  # 0 = X, 1 = Y, 2 = Z
    coord_a: float
    valor_a: float
    coord_b: float
    valor_b: float

    def valor_en(self, punto: Vector3) -> float:
        if abs(self.coord_b - self.coord_a) < 1e-12:
            return self.valor_a
        t = (punto[self.eje] - self.coord_a) / (self.coord_b - self.coord_a)
        t = min(1.0, max(0.0, t))
        return self.valor_a + t * (self.valor_b - self.valor_a)


@dataclass
class CargaArea:
    """Presión uniforme sobre una losa o muro, en kPa (kN/m2).

    ``direccion`` es el vector unitario de aplicación de la presión
    (por defecto, gravedad: -Z global, típico de cargas de piso/techo).
    La presión puede ser uniforme (``presion``) o variar linealmente a lo
    largo de un eje global (``perfil``: carga triangular o trapezoidal); en
    ese caso ``presion`` es solo el valor medio, a título informativo.
    """

    id: str
    caso_id: str
    presion: float
    losa_id: Optional[str] = None
    muro_id: Optional[str] = None
    direccion: Vector3 = (0.0, 0.0, -1.0)
    nombre: str = ""
    origen: str = "MANUAL"  # "MANUAL" o "PESO_PROPIO"
    perfil: Optional[PerfilPresion] = None  # None = presión uniforme

    def presion_en(self, punto: Vector3) -> float:
        """Presión (kPa) en un punto del espacio (uniforme o según ``perfil``)."""

        return self.presion if self.perfil is None else self.perfil.valor_en(punto)

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id
        if not self.losa_id and not self.muro_id:
            raise ValueError("CargaArea debe referenciar una losa o un muro")


TIPOS_UNION = ("SOLDADURA", "CUERPO_RIGIDO", "IGUALAR", "ARTICULACION")


@dataclass
class UnionSuperficie:
    """Constraint entre dos superficies (como los joint constraints de SAP), solo en el análisis 3D.

    * ``SOLDADURA``: las dos superficies quedan unidas punto a punto (``*TIE``), aunque haya una holgura.
    * ``CUERPO_RIGIDO``: las dos superficies se mueven juntas como un único cuerpo rígido.
    * ``IGUALAR``: los puntos de una superficie igualan el movimiento del punto más cercano de la otra
      en los grados marcados (Ux, Uy, Uz); los demás quedan libres (deslizamiento).
    * ``ARTICULACION``: cada superficie es rígida y sus centros se igualan en las traslaciones; los giros
      marcados (Rx, Ry, Rz) también se igualan y los demás quedan libres (rótula).
    """

    id: str
    tipo: str
    a_tipo: str
    a_id: str
    a_cara: int
    b_tipo: str
    b_id: str
    b_cara: int
    ux: bool = True
    uy: bool = True
    uz: bool = True
    rx: bool = False
    ry: bool = False
    rz: bool = False
    nombre: str = ""

    def grados(self) -> List[int]:
        """Grados de libertad igualados (1-6, convención CalculiX)."""

        banderas = [self.ux, self.uy, self.uz, self.rx, self.ry, self.rz]
        return [i + 1 for i, activo in enumerate(banderas) if activo]
