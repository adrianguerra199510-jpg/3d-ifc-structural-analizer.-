"""Modelo de dominio del proyecto estructural.

Representa, en memoria, el modelo estructural simplificado que se obtiene al
importar un IFC (o que se construye manualmente): nodos, materiales,
perfiles/secciones y elementos (vigas, columnas, losas, muros).

Este modelo es independiente de ifcopenshell, de CalculiX y de la interfaz
gráfica: es el núcleo que las demás capas (importador IFC, generador de
.inp, GUI) leen y escriben.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

Vector3 = Tuple[float, float, float]


class TipoElementoBarra(str, Enum):
    VIGA = "VIGA"
    COLUMNA = "COLUMNA"


class ModoAnalisisLosa(str, Enum):
    """Cómo se analiza una losa.

    TRANSFERENCIA: la carga de área se convierte en cargas lineales sobre
        las vigas de apoyo mediante áreas tributarias (modo por defecto en
        la Fase 1).
    SHELL: la losa se malla con elementos de cáscara (S4/S3) y la presión se
        aplica directamente sobre la cáscara (Fase 2, no implementado aún,
        pero contemplado en la arquitectura desde el inicio).
    """

    TRANSFERENCIA = "TRANSFERENCIA"
    SHELL = "SHELL"


class DireccionLosa(str, Enum):
    """Dirección de trabajo de una losa para la transferencia de cargas."""

    UNA_DIRECCION = "UNA_DIRECCION"
    DOS_DIRECCIONES = "DOS_DIRECCIONES"


@dataclass
class Nodo:
    """Punto en el espacio global, en metros."""

    id: str
    x: float
    y: float
    z: float

    @property
    def coordenadas(self) -> Vector3:
        return (self.x, self.y, self.z)

    def distancia(self, otro: "Nodo") -> float:
        return (
            (self.x - otro.x) ** 2 + (self.y - otro.y) ** 2 + (self.z - otro.z) ** 2
        ) ** 0.5


@dataclass
class Material:
    """Material elástico lineal.

    Unidades: módulo de elasticidad y de corte en kPa (kN/m2), densidad en
    kN/m3 (peso específico, no densidad de masa), de modo que
    peso_propio = densidad * área_o_espesor se obtiene directamente en kN/m
    o kN/m2 sin conversiones adicionales.
    """

    nombre: str
    modulo_elasticidad: float  # E, en kPa
    modulo_poisson: float = 0.2
    densidad: float = 24.0  # peso específico, en kN/m3 (24 kN/m3 ~ hormigón)
    ifc_guid: Optional[str] = None

    @property
    def modulo_corte(self) -> float:
        """G = E / (2*(1+nu)), en kPa."""

        return self.modulo_elasticidad / (2.0 * (1.0 + self.modulo_poisson))


@dataclass
class PerfilSeccion:
    """Sección transversal de un elemento tipo barra (viga o columna).

    Todas las propiedades geométricas en unidades consistentes con metros
    (área en m2, inercias en m4). ``inercia_z`` es la inercia de la flexión
    "fuerte" (alrededor del eje perpendicular a la altura) e ``inercia_y`` la
    "débil".

    ``forma`` indica cómo se modela la sección en el análisis:

    * ``"RECT"``: rectángulo macizo ``altura`` × ``ancho`` (o el rectángulo
      equivalente si no se conocen).
    * ``"I"``: perfil doble T (``altura`` = peralte total, ``ancho`` = ancho
      de las alas) con ``espesor_ala`` y ``espesor_alma``; se analiza con su
      forma real (ver ``core.geometria.partes_seccion``).
    """

    nombre: str
    area: float
    inercia_y: float
    inercia_z: float
    inercia_torsional: float
    altura: Optional[float] = None
    ancho: Optional[float] = None
    ifc_guid: Optional[str] = None
    forma: str = "RECT"
    espesor_ala: Optional[float] = None
    espesor_alma: Optional[float] = None


@dataclass
class ElementoBarra:
    """Elemento lineal (viga o columna), modelado en CalculiX con B31."""

    id: str
    tipo: TipoElementoBarra
    nodo_inicio: str
    nodo_fin: str
    material: str  # nombre del Material en Proyecto.materiales
    perfil: str  # nombre del PerfilSeccion en Proyecto.perfiles
    nombre: str = ""
    vector_n1: Vector3 = (0.0, 0.0, 1.0)  # orientación local de la sección
    ifc_guid: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id


@dataclass
class Losa:
    """Losa (placa horizontal o inclinada), definida por un contorno de nodos.

    ``vigas_soporte`` define qué viga soporta cada lado del contorno: debe
    tener exactamente la misma longitud que ``nodos_contorno``, y
    ``vigas_soporte[i]`` es el id de la viga que soporta el lado entre
    ``nodos_contorno[i]`` y ``nodos_contorno[(i+1) % n]``. Un lado sin viga
    tributaria asignada (p. ej. apoyado en un muro, o losa en una dirección
    donde ese lado no recibe carga transferida) se marca con ``None``.
    """

    id: str
    nodos_contorno: List[str]
    espesor: float
    material: str
    nombre: str = ""
    modo_analisis: ModoAnalisisLosa = ModoAnalisisLosa.TRANSFERENCIA
    direccion: DireccionLosa = DireccionLosa.DOS_DIRECCIONES
    vigas_soporte: List[Optional[str]] = field(default_factory=list)
    ifc_guid: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id


@dataclass
class Muro:
    """Muro, definido por un contorno de nodos. Informativo en la Fase 1."""

    id: str
    nodos_contorno: List[str]
    espesor: float
    material: str
    nombre: str = ""
    ifc_guid: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.nombre:
            self.nombre = self.id


class _ContadorId:
    """Generador simple de IDs incrementales con un prefijo."""

    def __init__(self) -> None:
        self._contadores: Dict[str, int] = {}

    def siguiente(self, prefijo: str) -> str:
        n = self._contadores.get(prefijo, 0) + 1
        self._contadores[prefijo] = n
        return f"{prefijo}{n}"


@dataclass
class Proyecto:
    """Contenedor principal del modelo estructural y de análisis."""

    nombre: str = "Proyecto sin nombre"
    nodos: Dict[str, Nodo] = field(default_factory=dict)
    materiales: Dict[str, Material] = field(default_factory=dict)
    perfiles: Dict[str, PerfilSeccion] = field(default_factory=dict)
    vigas: Dict[str, ElementoBarra] = field(default_factory=dict)
    columnas: Dict[str, ElementoBarra] = field(default_factory=dict)
    losas: Dict[str, Losa] = field(default_factory=dict)
    muros: Dict[str, Muro] = field(default_factory=dict)

    #: Avisos que deja la importación del IFC (no se guardan en el proyecto).
    avisos: List[str] = field(default_factory=list, repr=False, compare=False)

    _contador: _ContadorId = field(default_factory=_ContadorId, repr=False)

    # -- Nodos -----------------------------------------------------------
    def agregar_nodo(self, x: float, y: float, z: float, id_: Optional[str] = None) -> Nodo:
        id_ = id_ or self._contador.siguiente("N")
        nodo = Nodo(id=id_, x=x, y=y, z=z)
        self.nodos[id_] = nodo
        return nodo

    def nodo_o_existente(self, x: float, y: float, z: float, tolerancia: float = 1e-4) -> Nodo:
        """Reutiliza un nodo existente si cae dentro de la tolerancia, o crea uno nuevo.

        Es habitual que, al importar un IFC, varios elementos compartan el
        mismo punto geométrico (p. ej. el extremo de una viga coincide con
        una columna). Sin esta deduplicación, el modelo de análisis quedaría
        con elementos desconectados.
        """

        for nodo in self.nodos.values():
            if (
                abs(nodo.x - x) <= tolerancia
                and abs(nodo.y - y) <= tolerancia
                and abs(nodo.z - z) <= tolerancia
            ):
                return nodo
        return self.agregar_nodo(x, y, z)

    # -- Materiales y perfiles --------------------------------------------
    def agregar_material(self, material: Material) -> Material:
        self.materiales[material.nombre] = material
        return material

    def agregar_perfil(self, perfil: PerfilSeccion) -> PerfilSeccion:
        self.perfiles[perfil.nombre] = perfil
        return perfil

    # -- Elementos ---------------------------------------------------------
    def agregar_viga(
        self,
        nodo_inicio: str,
        nodo_fin: str,
        material: str,
        perfil: str,
        id_: Optional[str] = None,
        **kwargs,
    ) -> ElementoBarra:
        id_ = id_ or self._contador.siguiente("B")
        viga = ElementoBarra(
            id=id_,
            tipo=TipoElementoBarra.VIGA,
            nodo_inicio=nodo_inicio,
            nodo_fin=nodo_fin,
            material=material,
            perfil=perfil,
            **kwargs,
        )
        self.vigas[id_] = viga
        return viga

    def agregar_columna(
        self,
        nodo_inicio: str,
        nodo_fin: str,
        material: str,
        perfil: str,
        id_: Optional[str] = None,
        **kwargs,
    ) -> ElementoBarra:
        id_ = id_ or self._contador.siguiente("C")
        columna = ElementoBarra(
            id=id_,
            tipo=TipoElementoBarra.COLUMNA,
            nodo_inicio=nodo_inicio,
            nodo_fin=nodo_fin,
            material=material,
            perfil=perfil,
            **kwargs,
        )
        self.columnas[id_] = columna
        return columna

    def agregar_losa(
        self,
        nodos_contorno: List[str],
        espesor: float,
        material: str,
        id_: Optional[str] = None,
        **kwargs,
    ) -> Losa:
        id_ = id_ or self._contador.siguiente("L")
        losa = Losa(id=id_, nodos_contorno=nodos_contorno, espesor=espesor, material=material, **kwargs)
        self.losas[id_] = losa
        return losa

    def agregar_muro(
        self,
        nodos_contorno: List[str],
        espesor: float,
        material: str,
        id_: Optional[str] = None,
        **kwargs,
    ) -> Muro:
        id_ = id_ or self._contador.siguiente("M")
        muro = Muro(id=id_, nodos_contorno=nodos_contorno, espesor=espesor, material=material, **kwargs)
        self.muros[id_] = muro
        return muro

    # -- Utilidades ---------------------------------------------------------
    def elemento_barra(self, id_: str) -> ElementoBarra:
        if id_ in self.vigas:
            return self.vigas[id_]
        if id_ in self.columnas:
            return self.columnas[id_]
        raise KeyError(f"No existe ninguna viga o columna con id '{id_}'")

    def todas_las_barras(self) -> Dict[str, ElementoBarra]:
        return {**self.vigas, **self.columnas}

    def longitud_barra(self, id_: str) -> float:
        barra = self.elemento_barra(id_)
        return self.nodos[barra.nodo_inicio].distancia(self.nodos[barra.nodo_fin])
