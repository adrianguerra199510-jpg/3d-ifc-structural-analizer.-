"""Importador de modelos IFC (p. ej. exportados desde Revit) con ifcopenshell.

Extrae vigas, columnas, losas y muros con su geometría, perfil y material,
y construye un ``core.modelo.Proyecto``.

Estrategia de extracción de geometría
--------------------------------------
No se depende de que el IFC tenga una representación de tipo "Axis" bien
formada (en archivos exportados desde distintas herramientas esto varía
mucho, y no siempre está presente). En su lugar, se malla la geometría real
de cada elemento con ``ifcopenshell.geom`` y se calcula su caja orientada
(OBB) mediante PCA (análisis de componentes principales) sobre la nube de
vértices:

- Para vigas y columnas, se asume que la dimensión más larga de la OBB es
  el eje estructural del elemento; los nodos de inicio/fin son los centros
  de las caras extremas de la OBB en esa dirección, y las otras dos
  dimensiones de la OBB se usan como alto/ancho de una sección rectangular
  aproximada (solo si no se pudo obtener un perfil real, ver más abajo).
- Para losas y muros, se asume que la dimensión más corta de la OBB es el
  espesor; el contorno se obtiene proyectando los vértices sobre el plano
  definido por las otras dos direcciones principales y calculando su
  envolvente convexa (convex hull, con shapely).

Esta heurística es robusta ante variaciones de modelado, a costa de no
distinguir plantas no convexas o con huecos (fuera de alcance de la Fase 1;
una losa con huecos se importa con el contorno de su envolvente convexa,
lo que sobrestima su área — una limitación conocida, a mejorar en una fase
posterior con un recorte de polígono más preciso).

Perfil y material reales
--------------------------
Cuando el elemento tiene asociado un ``IfcMaterialProfileSetUsage`` (lo
habitual en vigas/columnas de acero exportadas como elementos
estructurales), se usa el perfil real (``IfcRectangleProfileDef`` o
``IfcCircleProfileDef``; otras formas de perfil quedan fuera de la Fase 1 y
se aproximan con la OBB). El material se busca en las propiedades
mecánicas del IFC (``Pset_MaterialMechanical`` / ``IfcMechanicalMaterialPropertiesValues``
según la versión del esquema); si no están presentes (frecuente en
exportaciones de Revit sin ese detalle), se usan valores por defecto de
hormigón, dejando constancia con una advertencia para que el usuario los
revise en la aplicación.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as util_element
import ifcopenshell.util.unit as util_unit

from ..core.conectividad import conectar_nodos
from ..core.modelo import DireccionLosa, Material, ModoAnalisisLosa, PerfilSeccion, Proyecto
from ..core.vigas_soporte import asignar_vigas_soporte

logger = logging.getLogger(__name__)

#: Valores por defecto (hormigón armado típico) para cuando el IFC no trae
#: propiedades mecánicas de material.
E_POR_DEFECTO_KPA = 2.5e7
POISSON_POR_DEFECTO = 0.2
DENSIDAD_POR_DEFECTO_KN_M3 = 24.0


class ErrorImportacionIfc(RuntimeError):
    pass


def _ajustes_geometria() -> "ifcopenshell.geom.settings":
    ajustes = ifcopenshell.geom.settings()
    ajustes.set("use-world-coords", True)
    return ajustes


def _vertices_elemento(elemento, ajustes, factor_longitud: float) -> Optional[np.ndarray]:
    try:
        forma = ifcopenshell.geom.create_shape(ajustes, elemento)
    except Exception as exc:  # ifcopenshell lanza distintas excepciones según el caso
        logger.warning("No se pudo obtener la geometría de %s (%s): %s", elemento.is_a(), elemento.GlobalId, exc)
        return None
    verts = np.array(forma.geometry.verts, dtype=float).reshape(-1, 3)
    return verts * factor_longitud


def _pca(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA sobre la nube de puntos: devuelve (centro, ejes, mínimos, máximos).

    ``ejes`` es una matriz 3x3 cuyas columnas son los ejes principales,
    ordenados de mayor a menor varianza. ``mínimos``/``máximos`` son las
    extensiones (en las coordenadas de esos ejes) de la nube de puntos.
    """

    centro = vertices.mean(axis=0)
    centrados = vertices - centro
    covarianza = np.cov(centrados.T)
    valores, vectores = np.linalg.eigh(covarianza)
    orden = np.argsort(valores)[::-1]
    ejes = vectores[:, orden]
    proyecciones = centrados @ ejes
    return centro, ejes, proyecciones.min(axis=0), proyecciones.max(axis=0)


def _nombre_material(material_ifc) -> str:
    nombre = getattr(material_ifc, "Name", None)
    return nombre or "SIN_NOMBRE"


def _propiedades_mecanicas(material_ifc) -> Optional[Tuple[float, float, float]]:
    """Busca (E [Pa->kPa], poisson, densidad [kg/m3->kN/m3]) en las
    propiedades asociadas al material IFC. Devuelve None si no se
    encuentran (caso frecuente en exportaciones sin ese detalle)."""

    try:
        psets = util_element.get_psets(material_ifc)
    except Exception:
        psets = {}

    for nombre_pset, propiedades in psets.items():
        if "Mechanical" not in nombre_pset:
            continue
        e = propiedades.get("YoungModulus")
        nu = propiedades.get("PoissonRatio")
        rho = propiedades.get("MassDensity")
        if e is not None:
            e_kpa = float(e) / 1000.0  # Pa -> kPa
            nu_val = float(nu) if nu is not None else POISSON_POR_DEFECTO
            densidad_kn_m3 = float(rho) * 9.80665 / 1000.0 if rho is not None else DENSIDAD_POR_DEFECTO_KN_M3
            return e_kpa, nu_val, densidad_kn_m3
    return None


def _obtener_o_crear_material(proyecto: Proyecto, elemento) -> str:
    material_ifc = util_element.get_material(elemento)
    if material_ifc is None:
        nombre = "HORMIGON_DEFECTO"
        if nombre not in proyecto.materiales:
            logger.warning(
                "El elemento %s (%s) no tiene material asignado en el IFC; "
                "se usa un hormigón por defecto.",
                elemento.is_a(),
                elemento.GlobalId,
            )
            proyecto.agregar_material(
                Material(
                    nombre=nombre,
                    modulo_elasticidad=E_POR_DEFECTO_KPA,
                    modulo_poisson=POISSON_POR_DEFECTO,
                    densidad=DENSIDAD_POR_DEFECTO_KN_M3,
                )
            )
        return nombre

    # IfcMaterialProfileSetUsage / IfcMaterialLayerSetUsage envuelven al
    # IfcMaterial real; nos interesa el material de base para las
    # propiedades mecánicas.
    material_base = material_ifc
    if material_ifc.is_a("IfcMaterialProfileSetUsage"):
        perfiles = material_ifc.ForProfileSet.MaterialProfiles
        material_base = perfiles[0].Material if perfiles else material_ifc
    elif material_ifc.is_a("IfcMaterialLayerSetUsage"):
        capas = material_ifc.ForLayerSet.MaterialLayers
        material_base = capas[0].Material if capas else material_ifc

    nombre = _nombre_material(material_base)
    if nombre in proyecto.materiales:
        return nombre

    propiedades = _propiedades_mecanicas(material_base)
    if propiedades is None:
        logger.warning(
            "El material '%s' no tiene propiedades mecánicas en el IFC; "
            "se usan valores por defecto de hormigón (revisar en la aplicación).",
            nombre,
        )
        e_kpa, nu_val, densidad = E_POR_DEFECTO_KPA, POISSON_POR_DEFECTO, DENSIDAD_POR_DEFECTO_KN_M3
    else:
        e_kpa, nu_val, densidad = propiedades

    proyecto.agregar_material(
        Material(
            nombre=nombre,
            modulo_elasticidad=e_kpa,
            modulo_poisson=nu_val,
            densidad=densidad,
            ifc_guid=getattr(material_base, "GlobalId", None),
        )
    )
    return nombre


def _perfil_real_desde_material(elemento) -> Optional[Tuple[str, float, float, float, float, Optional[float], Optional[float]]]:
    """Intenta leer un perfil real (rectángulo o círculo) desde
    IfcMaterialProfileSetUsage. Devuelve
    (nombre, area, inercia_y, inercia_z, inercia_torsional, altura, ancho)
    o None si no hay perfil utilizable."""

    material_ifc = util_element.get_material(elemento)
    if material_ifc is None or not material_ifc.is_a("IfcMaterialProfileSetUsage"):
        return None
    perfiles = material_ifc.ForProfileSet.MaterialProfiles
    if not perfiles:
        return None
    perfil_ifc = perfiles[0].Profile
    nombre = perfil_ifc.ProfileName or perfil_ifc.is_a()

    if perfil_ifc.is_a("IfcRectangleProfileDef"):
        ancho, altura = float(perfil_ifc.XDim), float(perfil_ifc.YDim)
        area = ancho * altura
        inercia_y = ancho * altura**3 / 12.0
        inercia_z = altura * ancho**3 / 12.0
        inercia_torsional = _torsion_rectangular(ancho, altura)
        return nombre, area, inercia_y, inercia_z, inercia_torsional, altura, ancho

    if perfil_ifc.is_a("IfcCircleProfileDef"):
        radio = float(perfil_ifc.Radius)
        area = 3.141592653589793 * radio**2
        inercia = 3.141592653589793 * radio**4 / 4.0
        inercia_torsional = 3.141592653589793 * radio**4 / 2.0
        return nombre, area, inercia, inercia, inercia_torsional, 2 * radio, 2 * radio

    return None  # otras formas (I, L, etc.): fuera de alcance de la Fase 1


def _torsion_rectangular(ancho: float, altura: float) -> float:
    """Constante de torsión de Saint-Venant para una sección rectangular
    maciza, con la fórmula aproximada usual (b = lado menor, h = lado mayor)."""

    b, h = min(ancho, altura), max(ancho, altura)
    if b < 1e-9:
        return 1e-9
    relacion = h / b
    beta = 1.0 / 3.0 - 0.21 * b / h * (1 - (b / h) ** 4 / 12.0) if h > 0 else 1.0 / 3.0
    return max(beta * b**3 * h, 1e-9)


def _elegir_orientacion(ejes: np.ndarray, minimos: np.ndarray, maximos: np.ndarray, altura_perfil: Optional[float]):
    """Elige hacia dónde apunta la altura de la sección (``vector_n1``).

    Las dos direcciones transversales de la caja orientada son ``ejes[:, 1]``
    y ``ejes[:, 2]``. Si se conoce la altura real del perfil, se usa la
    dirección cuya extensión medida es más parecida; si no (o si la sección
    es cuadrada), en vigas se toma la más vertical y en columnas la de mayor
    extensión. Devuelve (n1, altura, ancho) medidos sobre la geometría.
    """

    direcciones = [ejes[:, 1], ejes[:, 2]]
    extensiones = [float(maximos[1] - minimos[1]), float(maximos[2] - minimos[2])]
    if altura_perfil is not None and abs(extensiones[0] - extensiones[1]) > 1e-3:
        k = 0 if abs(extensiones[0] - altura_perfil) <= abs(extensiones[1] - altura_perfil) else 1
    elif abs(ejes[2, 0]) > 0.7:  # columna (eje principal casi vertical)
        k = int(np.argmax(extensiones))
    else:  # viga: la dirección transversal más vertical
        k = 0 if abs(direcciones[0][2]) >= abs(direcciones[1][2]) else 1
    n1 = direcciones[k] if direcciones[k][2] >= 0 else -direcciones[k]
    return tuple(float(c) for c in n1), extensiones[k], extensiones[1 - k]


def _obtener_o_crear_perfil(proyecto: Proyecto, elemento, vertices: np.ndarray):
    """Devuelve (nombre del perfil, vector_n1 de la barra)."""

    _, ejes, minimos, maximos = _pca(vertices)
    real = _perfil_real_desde_material(elemento)
    if real is not None:
        nombre, area, iy, iz, it, altura, ancho = real
        n1, _, _ = _elegir_orientacion(ejes, minimos, maximos, altura)
        if nombre not in proyecto.perfiles:
            proyecto.agregar_perfil(
                PerfilSeccion(
                    nombre=nombre, area=area, inercia_y=iy, inercia_z=iz,
                    inercia_torsional=it, altura=altura, ancho=ancho,
                )
            )
        return nombre, n1

    # Sin perfil real: aproximar con la caja orientada de la geometría.
    n1, altura, ancho = _elegir_orientacion(ejes, minimos, maximos, None)
    nombre = f"APROX_{elemento.GlobalId}"
    proyecto.agregar_perfil(
        PerfilSeccion(
            nombre=nombre,
            area=ancho * altura,
            inercia_y=altura * ancho**3 / 12.0,
            inercia_z=ancho * altura**3 / 12.0,
            inercia_torsional=_torsion_rectangular(ancho, altura),
            altura=altura,
            ancho=ancho,
        )
    )
    return nombre, n1


def _limpiar(punto: np.ndarray) -> Tuple[float, float, float]:
    """Coordenadas como floats de Python redondeados al micrómetro (sin -0.0)."""

    return tuple(round(float(c), 6) + 0.0 for c in punto)


def _importar_barra(proyecto: Proyecto, elemento, ajustes, factor_longitud: float, tipo: str) -> None:
    vertices = _vertices_elemento(elemento, ajustes, factor_longitud)
    if vertices is None or len(vertices) < 2:
        logger.warning("Se omite %s %s por no tener geometría utilizable", tipo, elemento.GlobalId)
        return

    centro, ejes, minimos, maximos = _pca(vertices)
    eje_principal = ejes[:, 0]
    punto_inicio = centro + eje_principal * minimos[0]
    punto_fin = centro + eje_principal * maximos[0]
    # El sentido del eje principal de la PCA es arbitrario: las columnas se
    # orientan siempre de la base (inicio) al tope (fin).
    if tipo != "VIGA" and punto_fin[2] < punto_inicio[2]:
        punto_inicio, punto_fin = punto_fin, punto_inicio

    nodo_inicio = proyecto.nodo_o_existente(*_limpiar(punto_inicio))
    nodo_fin = proyecto.nodo_o_existente(*_limpiar(punto_fin))

    material = _obtener_o_crear_material(proyecto, elemento)
    perfil, n1 = _obtener_o_crear_perfil(proyecto, elemento, vertices)

    nombre = elemento.Name or elemento.GlobalId
    if tipo == "VIGA":
        proyecto.agregar_viga(
            nodo_inicio.id, nodo_fin.id, material, perfil, nombre=nombre, ifc_guid=elemento.GlobalId,
            vector_n1=n1,
        )
    else:
        proyecto.agregar_columna(
            nodo_inicio.id, nodo_fin.id, material, perfil, nombre=nombre, ifc_guid=elemento.GlobalId,
            vector_n1=n1,
        )


def _importar_losa(proyecto: Proyecto, elemento, ajustes, factor_longitud: float) -> None:
    vertices = _vertices_elemento(elemento, ajustes, factor_longitud)
    if vertices is None or len(vertices) < 3:
        logger.warning("Se omite losa %s por no tener geometría utilizable", elemento.GlobalId)
        return

    nombre, espesor, material, nodos_contorno_ids = _procesar_elemento_planar(proyecto, elemento, vertices, "LOSA")
    if nodos_contorno_ids is None:
        return

    proyecto.agregar_losa(
        nodos_contorno=nodos_contorno_ids,
        espesor=espesor,
        material=material,
        nombre=nombre,
        modo_analisis=ModoAnalisisLosa.TRANSFERENCIA,
        direccion=DireccionLosa.DOS_DIRECCIONES,
        vigas_soporte=[None] * len(nodos_contorno_ids),
        ifc_guid=elemento.GlobalId,
    )


def _importar_muro(proyecto: Proyecto, elemento, ajustes, factor_longitud: float) -> None:
    vertices = _vertices_elemento(elemento, ajustes, factor_longitud)
    if vertices is None or len(vertices) < 3:
        logger.warning("Se omite muro %s por no tener geometría utilizable", elemento.GlobalId)
        return

    nombre, espesor, material, nodos_contorno_ids = _procesar_elemento_planar(proyecto, elemento, vertices, "MURO")
    if nodos_contorno_ids is None:
        return

    proyecto.agregar_muro(
        nodos_contorno=nodos_contorno_ids,
        espesor=espesor,
        material=material,
        nombre=nombre,
        ifc_guid=elemento.GlobalId,
    )


def _procesar_elemento_planar(proyecto: Proyecto, elemento, vertices: np.ndarray, tipo: str):
    """Lógica común a losas y muros: obtiene espesor (dimensión más corta
    de la OBB) y el contorno (envolvente convexa proyectada en el plano de
    las otras dos direcciones principales)."""

    import shapely.geometry as geom_shapely

    centro, ejes, minimos, maximos = _pca(vertices)
    extensiones = maximos - minimos
    indice_espesor = int(np.argmin(extensiones))
    espesor = float(extensiones[indice_espesor])

    indices_plano = [i for i in range(3) if i != indice_espesor]
    centrados = vertices - centro
    u = centrados @ ejes[:, indices_plano[0]]
    v = centrados @ ejes[:, indices_plano[1]]

    puntos_2d = list(zip(u.tolist(), v.tolist()))
    if len(puntos_2d) < 3:
        return None, None, None, None
    envolvente = geom_shapely.MultiPoint(puntos_2d).convex_hull
    if envolvente.is_empty or envolvente.geom_type != "Polygon":
        logger.warning("No se pudo calcular el contorno de %s %s", tipo, elemento.GlobalId)
        return None, None, None, None

    coords_2d = list(envolvente.exterior.coords)[:-1]  # el último repite al primero

    eje_u = ejes[:, indices_plano[0]]
    eje_v = ejes[:, indices_plano[1]]
    nodos_contorno_ids: List[str] = []
    for pu, pv in coords_2d:
        punto_3d = centro + eje_u * pu + eje_v * pv
        nodo = proyecto.nodo_o_existente(*_limpiar(punto_3d))
        nodos_contorno_ids.append(nodo.id)

    material = _obtener_o_crear_material(proyecto, elemento)
    nombre = elemento.Name or elemento.GlobalId
    return nombre, espesor, material, nodos_contorno_ids


def importar_ifc(ruta_ifc: str) -> Proyecto:
    """Punto de entrada principal: importa un archivo IFC y devuelve un Proyecto."""

    try:
        modelo_ifc = ifcopenshell.open(ruta_ifc)
    except Exception as exc:
        raise ErrorImportacionIfc(f"No se pudo abrir el archivo IFC '{ruta_ifc}': {exc}") from exc

    factor_longitud = util_unit.calculate_unit_scale(modelo_ifc)
    ajustes = _ajustes_geometria()

    proyecto = Proyecto(nombre=Path(ruta_ifc).stem)

    for elemento in modelo_ifc.by_type("IfcBeam"):
        _importar_barra(proyecto, elemento, ajustes, factor_longitud, "VIGA")
    for elemento in modelo_ifc.by_type("IfcColumn"):
        _importar_barra(proyecto, elemento, ajustes, factor_longitud, "COLUMNA")
    unidos = conectar_nodos(proyecto)
    if unidos:
        logger.info("Se unieron %d extremos de barras que casi coincidían (idealización por ejes).", unidos)
    for elemento in modelo_ifc.by_type("IfcSlab"):
        _importar_losa(proyecto, elemento, ajustes, factor_longitud)
    for elemento in modelo_ifc.by_type("IfcWall"):
        _importar_muro(proyecto, elemento, ajustes, factor_longitud)
    asignar_vigas_soporte(proyecto)

    logger.info(
        "Importación completa: %d vigas, %d columnas, %d losas, %d muros.",
        len(proyecto.vigas), len(proyecto.columnas), len(proyecto.losas), len(proyecto.muros),
    )
    return proyecto
