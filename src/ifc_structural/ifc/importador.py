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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as util_element
import ifcopenshell.util.placement as util_placement
import ifcopenshell.util.unit as util_unit

from ..core.conectividad import conectar_nodos, conectar_uniones_en_t
from ..core.modelo import DireccionLosa, Material, ModoAnalisisLosa, PerfilSeccion, Proyecto
from ..core.solidos import SolidoGeneral, partes_conexas, soldar_vertices
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


def _vertices_elemento(elemento, ajustes, factor_longitud: float, proyecto: Optional[Proyecto] = None) -> Optional[np.ndarray]:
    try:
        forma = ifcopenshell.geom.create_shape(ajustes, elemento)
    except Exception as exc:  # ifcopenshell lanza distintas excepciones según el caso
        logger.warning("No se pudo obtener la geometría de %s (%s): %s", elemento.is_a(), elemento.GlobalId, exc)
        return None
    # ifcopenshell entrega la geometría ya en metros, sea cual sea la unidad del archivo (mm, cm, pies...):
    # ``factor_longitud`` solo sirve para los valores crudos del IFC (dimensiones de perfiles, por ejemplo).
    vertices = np.array(forma.geometry.verts, dtype=float).reshape(-1, 3)
    return _solo_finitos(vertices, elemento, proyecto)


def _solo_finitos(vertices: np.ndarray, elemento, proyecto: Optional[Proyecto]) -> Optional[np.ndarray]:
    """``vertices`` sin los puntos con coordenadas no finitas; ``None`` (con aviso) si no queda geometría o es nula."""

    buenos = np.isfinite(vertices).all(axis=1)
    if buenos.all():
        return vertices
    nombre = getattr(elemento, "Name", None) or elemento.GlobalId
    logger.warning("%s %s tiene vértices con coordenadas no válidas", elemento.is_a(), elemento.GlobalId)
    if proyecto is not None:
        proyecto.avisos.append(f"{nombre} ({elemento.is_a()}): su geometría tiene coordenadas no válidas (NaN); "
                               + ("se omitieron los puntos erróneos." if buenos.sum() >= 3 else "se omitió el elemento."))
    return vertices[buenos] if buenos.sum() >= 3 else None


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


#: Acero estructural por defecto (E = 210 GPa, ν = 0,3, γ = 78,5 kN/m³) para materiales sin propiedades cuyo nombre,
#: o el del elemento o de su tipo, delata que son de acero (p. ej. «Muro básico:Acero T18 mm»).
_ES_ACERO = re.compile(r"\b(acero|steel|inox|s\s?235|s\s?275|s\s?355|a36|a572|a992)\b", re.IGNORECASE)
E_ACERO_KPA, POISSON_ACERO, DENSIDAD_ACERO_KN_M3 = 2.1e8, 0.3, 78.5


def _nombres_del_elemento(elemento, material_ifc=None) -> str:
    """Nombre del material, del elemento, de su tipo y de su clase: de ahí se adivina el material si el IFC no lo dice."""

    nombres = [getattr(elemento, "Name", None), getattr(elemento, "ObjectType", None), getattr(material_ifc, "Name", None)]
    try:
        tipo = util_element.get_type(elemento)
        if tipo is not None:
            nombres.append(getattr(tipo, "Name", None))
    except Exception:
        pass
    return " ".join(n for n in nombres if n)


def _obtener_o_crear_material(proyecto: Proyecto, elemento) -> str:
    material_ifc = util_element.get_material(elemento)
    if material_ifc is None and _ES_ACERO.search(_nombres_del_elemento(elemento)):
        if "ACERO_POR_NOMBRE" not in proyecto.materiales:
            proyecto.avisos.append("Los elementos sin material cuyo nombre indica acero se importaron como acero "
                                   "(E = 210 GPa, ν = 0,3, γ = 78,5 kN/m³): revise el material.")
            proyecto.agregar_material(Material(nombre="ACERO_POR_NOMBRE", modulo_elasticidad=E_ACERO_KPA,
                                               modulo_poisson=POISSON_ACERO, densidad=DENSIDAD_ACERO_KN_M3))
        return "ACERO_POR_NOMBRE"
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
    if propiedades is None and _ES_ACERO.search(_nombres_del_elemento(elemento, material_base)):
        propiedades = (E_ACERO_KPA, POISSON_ACERO, DENSIDAD_ACERO_KN_M3)  # el nombre dice acero
        proyecto.avisos.append(f"El material «{nombre}» no tiene propiedades en el IFC pero su nombre indica acero: "
                               "se usó acero estructural (E = 210 GPa, ν = 0,3, γ = 78,5 kN/m³); revíselo.")
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


def _perfil_ifc_del_elemento(elemento):
    """Perfil IFC del elemento: el del conjunto de perfiles del material o, si
    no lo hay, el de su primera extrusión (``IfcExtrudedAreaSolid``)."""

    material_ifc = util_element.get_material(elemento)
    if material_ifc is not None and material_ifc.is_a("IfcMaterialProfileSetUsage"):
        perfiles = material_ifc.ForProfileSet.MaterialProfiles
        if perfiles and perfiles[0].Profile is not None:
            return perfiles[0].Profile
    extrusion = _extrusion_del_elemento(elemento)
    return extrusion.SweptArea if extrusion is not None else None


def _extrusion_del_elemento(elemento):
    representacion = getattr(elemento, "Representation", None)
    if representacion is None:
        return None
    for rep in representacion.Representations:
        for item in rep.Items:
            if item.is_a("IfcExtrudedAreaSolid"):
                return item
    return None


def _perfil_real(perfil_ifc, factor_longitud: float) -> Optional[PerfilSeccion]:
    """Propiedades de un perfil IFC conocido (rectángulo, círculo o doble T),
    en metros; None si la forma no está soportada."""

    if perfil_ifc is None:
        return None
    nombre = perfil_ifc.ProfileName or perfil_ifc.is_a()
    f = factor_longitud

    if perfil_ifc.is_a("IfcRectangleProfileDef") and not perfil_ifc.is_a("IfcRectangleHollowProfileDef"):
        ancho, altura = float(perfil_ifc.XDim) * f, float(perfil_ifc.YDim) * f
        return PerfilSeccion(
            nombre=nombre, area=ancho * altura, inercia_y=altura * ancho**3 / 12.0,
            inercia_z=ancho * altura**3 / 12.0, inercia_torsional=_torsion_rectangular(ancho, altura),
            altura=altura, ancho=ancho,
        )

    if perfil_ifc.is_a("IfcCircleProfileDef") and not perfil_ifc.is_a("IfcCircleHollowProfileDef"):
        # ccx 2.21 no expande vigas B31 circulares: se analiza con el cuadrado
        # de igual inercia (área real aparte, para el peso propio).
        radio = float(perfil_ifc.Radius) * f
        inercia = np.pi * radio**4 / 4.0
        lado = (12.0 * inercia) ** 0.25
        return PerfilSeccion(
            nombre=nombre, area=np.pi * radio**2, inercia_y=inercia, inercia_z=inercia,
            inercia_torsional=np.pi * radio**4 / 2.0, altura=lado, ancho=lado,
        )

    if perfil_ifc.is_a("IfcIShapeProfileDef") and not perfil_ifc.is_a("IfcAsymmetricIShapeProfileDef"):
        ancho, altura = float(perfil_ifc.OverallWidth) * f, float(perfil_ifc.OverallDepth) * f
        alma, ala = float(perfil_ifc.WebThickness) * f, float(perfil_ifc.FlangeThickness) * f
        if min(ancho, altura, alma, ala) <= 0 or 2 * ala >= altura or alma >= ancho:
            return None
        interior = altura - 2 * ala
        return PerfilSeccion(
            nombre=nombre,
            area=2 * ancho * ala + interior * alma,
            inercia_y=2 * ala * ancho**3 / 12.0 + interior * alma**3 / 12.0,
            inercia_z=ancho * altura**3 / 12.0 - (ancho - alma) * interior**3 / 12.0,
            inercia_torsional=(2 * ancho * ala**3 + interior * alma**3) / 3.0,
            altura=altura, ancho=ancho, forma="I", espesor_ala=ala, espesor_alma=alma,
        )
    return None


def _direccion_peralte(elemento) -> Optional[np.ndarray]:
    """Dirección global (unitaria) del eje Y del perfil de la extrusión, que
    para un perfil en I o rectangular es el sentido del peralte."""

    extrusion = _extrusion_del_elemento(elemento)
    if extrusion is None or getattr(elemento, "ObjectPlacement", None) is None:
        return None
    try:
        matriz = util_placement.get_local_placement(elemento.ObjectPlacement)
        if extrusion.Position is not None:
            matriz = matriz @ util_placement.get_axis2placement(extrusion.Position)
        y_local = np.array([0.0, 1.0, 0.0])
        posicion2d = getattr(extrusion.SweptArea, "Position", None)
        if posicion2d is not None and posicion2d.RefDirection is not None:
            x2d = np.array(posicion2d.RefDirection.DirectionRatios[:2], dtype=float)
            x2d /= np.linalg.norm(x2d)
            y_local = np.array([-x2d[1], x2d[0], 0.0])
        direccion = matriz[:3, :3] @ y_local
    except Exception as exc:  # geometría de ubicación atípica: se usa la heurística de la OBB
        logger.debug("No se pudo leer la ubicación del perfil: %s", exc)
        return None
    norma = np.linalg.norm(direccion)
    return direccion / norma if norma > 1e-9 else None


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


def _obtener_o_crear_perfil(proyecto: Proyecto, elemento, vertices: np.ndarray, factor_longitud: float, eje: np.ndarray):
    """Devuelve (nombre del perfil, vector_n1 de la barra)."""

    _, ejes, minimos, maximos = _pca(vertices)
    real = _perfil_real(_perfil_ifc_del_elemento(elemento), factor_longitud)
    if real is not None:
        n1, _, _ = _elegir_orientacion(ejes, minimos, maximos, real.altura)
        # El peralte del perfil, tal como está colocado en el IFC, es más fiable que la caja
        # orientada (imprescindible en perfiles casi cuadrados como un HEB).
        peralte = _direccion_peralte(elemento)
        if peralte is not None:
            peralte = peralte - np.dot(peralte, eje) * eje
            if np.linalg.norm(peralte) > 0.5:
                peralte = peralte / np.linalg.norm(peralte)
                n1 = tuple(float(c) for c in (peralte if peralte[2] >= 0 else -peralte))
        if real.nombre not in proyecto.perfiles:
            proyecto.agregar_perfil(real)
        return real.nombre, n1

    # Sin perfil real: aproximar con la caja orientada de la geometría.
    perfil_ifc = _perfil_ifc_del_elemento(elemento)
    if perfil_ifc is not None:
        proyecto.avisos.append(
            f"{elemento.Name or elemento.GlobalId}: el perfil {perfil_ifc.is_a()} no está soportado; "
            "se aproxima con un rectángulo macizo (rigidez y esfuerzos poco fiables)."
        )
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
    vertices = _vertices_elemento(elemento, ajustes, factor_longitud, proyecto)
    if vertices is None or len(vertices) < 2:
        logger.warning("Se omite %s %s por no tener geometría utilizable", tipo, elemento.GlobalId)
        return

    centro, ejes, minimos, maximos = _pca(vertices)
    # Una placa (rigidizador, cartela, platina) clasificada como viga o columna no es una barra: su sección es una
    # lámina (espesor < 8 % del ancho) y idealizarla por el eje no representa nada. Se deja para la importación de
    # sólidos, que conserva su geometría real.
    extension = np.sort(maximos - minimos)
    if extension[1] > 0 and extension[0] < 0.08 * extension[1]:
        proyecto.avisos.append(f"{elemento.Name or elemento.GlobalId}: es una placa delgada y se importó como sólido "
                               "(no como barra).")
        return
    eje_principal = ejes[:, 0]
    punto_inicio = centro + eje_principal * minimos[0]
    punto_fin = centro + eje_principal * maximos[0]
    if tipo == "MIEMBRO":  # IfcMember: columna si es casi vertical, si no viga
        tipo = "COLUMNA" if abs(eje_principal[2]) > 0.7 else "VIGA"
    # El sentido del eje principal de la PCA es arbitrario: las columnas se
    # orientan siempre de la base (inicio) al tope (fin).
    if tipo != "VIGA" and punto_fin[2] < punto_inicio[2]:
        punto_inicio, punto_fin = punto_fin, punto_inicio

    if not (np.isfinite(punto_inicio).all() and np.isfinite(punto_fin).all()) or np.linalg.norm(punto_fin - punto_inicio) < 1e-5:
        proyecto.avisos.append(f"{elemento.Name or elemento.GlobalId}: su eje tiene longitud cero o coordenadas no válidas; "
                               "se omitió la barra.")
        return
    nodo_inicio = proyecto.nodo_o_existente(*_limpiar(punto_inicio))
    nodo_fin = proyecto.nodo_o_existente(*_limpiar(punto_fin))

    material = _obtener_o_crear_material(proyecto, elemento)
    perfil, n1 = _obtener_o_crear_perfil(proyecto, elemento, vertices, factor_longitud, eje_principal)

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
    vertices = _vertices_elemento(elemento, ajustes, factor_longitud, proyecto)
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
    vertices = _vertices_elemento(elemento, ajustes, factor_longitud, proyecto)
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


#: Tipos de IFC que no son piezas físicas (huecos y elementos virtuales).
_TIPOS_SIN_SOLIDO = ("IfcOpeningElement", "IfcVirtualElement")
#: Máximo de sólidos genéricos que se importan de un archivo.
MAXIMO_DE_SOLIDOS = 400


#: Máximo de piezas sueltas que se conservan de un mismo elemento IFC (las de mayor superficie).
MAXIMO_DE_PIEZAS_POR_ELEMENTO = 300


def importar_solidos_de_elemento(proyecto: Proyecto, elemento, ajustes, factor_longitud: float) -> List[SolidoGeneral]:
    """Lee la geometría real de un elemento IFC (cualquiera) y la agrega como sólidos con sus superficies.

    Un mismo elemento puede contener varias piezas sin conexión (un equipo exportado como un solo
    ``IfcBuildingElementProxy`` con patas, pernos y engranajes): cada una entra como sólido independiente, para poder
    asignarles materiales, apoyos y cargas por separado y detectar los contactos entre ellas."""

    try:
        forma = ifcopenshell.geom.create_shape(ajustes, elemento)
    except Exception as exc:  # ifcopenshell lanza distintas excepciones según el caso
        logger.debug("Sin geometría en %s %s: %s", elemento.is_a(), elemento.GlobalId, exc)
        return []
    vertices = np.array(forma.geometry.verts, dtype=float).reshape(-1, 3)  # ya en metros
    triangulos = np.array(forma.geometry.faces, dtype=np.int64).reshape(-1, 3)
    if not np.isfinite(vertices).all():
        proyecto.avisos.append(f"{elemento.Name or elemento.GlobalId} ({elemento.is_a()}): su geometría tiene coordenadas "
                               "no válidas (NaN); se omitió el elemento.")
        return []
    if len(vertices) < 4 or len(triangulos) < 4:
        return []
    vertices, triangulos = soldar_vertices(vertices, triangulos)
    if len(triangulos) < 4:
        return []
    material = _obtener_o_crear_material(proyecto, elemento)
    nombre = elemento.Name or elemento.GlobalId

    pieza = partes_conexas(triangulos, len(vertices))
    numeros, cuenta = np.unique(pieza, return_counts=True)
    grupos = [triangulos[pieza == n] for n in numeros[np.argsort(-cuenta)]]
    grupos = [g for g in grupos if len(g) >= 4]
    if len(grupos) > MAXIMO_DE_PIEZAS_POR_ELEMENTO:
        proyecto.avisos.append(f"{nombre}: tiene {len(grupos)} piezas sueltas; se importan las {MAXIMO_DE_PIEZAS_POR_ELEMENTO} mayores.")
        grupos = grupos[:MAXIMO_DE_PIEZAS_POR_ELEMENTO]
    solidos: List[SolidoGeneral] = []
    for k, grupo in enumerate(grupos, start=1):
        usados, inverso = np.unique(grupo, return_inverse=True)
        solido = SolidoGeneral(
            id=proyecto.siguiente_id("S"), nombre=nombre if len(grupos) == 1 else f"{nombre} · pieza {k}", tipo_ifc=elemento.is_a(),
            material=material, vertices=vertices[usados], triangulos=inverso.reshape(-1, 3), ifc_guid=elemento.GlobalId,
        )
        if solido.volumen < 1e-12:
            continue
        proyecto.solidos[solido.id] = solido
        solidos.append(solido)
    if len(solidos) > 1:
        proyecto.avisos.append(f"{nombre}: contiene {len(solidos)} piezas sin conexión entre sí; se importaron como sólidos "
                               "independientes (use «Conectar» para unirlas por donde se tocan).")
    return solidos


def _importar_solidos(modelo_ifc, proyecto: Proyecto, ajustes, factor_longitud: float) -> None:
    """Todo elemento que no sea viga, columna, losa o muro entra como sólido de forma libre:
    placas, pernos, zapatas, escaleras, piezas especiales..."""

    ya_importados = {e.ifc_guid for e in proyecto.todas_las_barras().values()}
    ya_importados |= {e.ifc_guid for e in list(proyecto.losas.values()) + list(proyecto.muros.values())}
    candidatos = [e for e in modelo_ifc.by_type("IfcElement")
                  if e.GlobalId not in ya_importados and not any(e.is_a(t) for t in _TIPOS_SIN_SOLIDO)]
    if len(candidatos) > MAXIMO_DE_SOLIDOS:
        proyecto.avisos.append(
            f"El archivo tiene {len(candidatos)} elementos que no son vigas, columnas, losas ni muros: se importan "
            f"solo los primeros {MAXIMO_DE_SOLIDOS} como sólidos.")
        candidatos = candidatos[:MAXIMO_DE_SOLIDOS]
    por_tipo: Dict[str, int] = {}
    abiertos = 0
    for elemento in candidatos:
        for solido in importar_solidos_de_elemento(proyecto, elemento, ajustes, factor_longitud):
            por_tipo[elemento.is_a()] = por_tipo.get(elemento.is_a(), 0) + 1
            if not solido.cerrado:
                abiertos += 1
    if por_tipo:
        detalle = ", ".join(f"{n} {t.replace('Ifc', '')}" for t, n in sorted(por_tipo.items()))
        proyecto.avisos.append(
            f"Se importaron como sólidos (geometría real, con superficies para asignar apoyos y cargas): {detalle}. "
            "Solo se analizan con ANALIZAR 3D.")
    if abiertos:
        proyecto.avisos.append(
            f"{abiertos} sólido(s) tienen la malla abierta o con errores: se pueden ver y seleccionar, pero no se podrán "
            "mallar en el análisis 3D.")


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
    for elemento in modelo_ifc.by_type("IfcMember"):
        _importar_barra(proyecto, elemento, ajustes, factor_longitud, "MIEMBRO")
    unidos = conectar_nodos(proyecto)
    if unidos:
        logger.info("Se unieron %d extremos de barras que casi coincidían (idealización por ejes).", unidos)
    partidas = conectar_uniones_en_t(proyecto)
    if partidas:
        logger.info("Se partieron %d barras donde otras llegan a su eje (uniones en T).", partidas)
    for elemento in modelo_ifc.by_type("IfcSlab"):
        _importar_losa(proyecto, elemento, ajustes, factor_longitud)
    for elemento in modelo_ifc.by_type("IfcWall"):
        _importar_muro(proyecto, elemento, ajustes, factor_longitud)
    asignar_vigas_soporte(proyecto)
    _importar_solidos(modelo_ifc, proyecto, ajustes, factor_longitud)
    proyecto.ruta_ifc = str(ruta_ifc)

    logger.info(
        "Importación completa: %d vigas, %d columnas, %d losas, %d muros, %d sólidos.",
        len(proyecto.vigas), len(proyecto.columnas), len(proyecto.losas), len(proyecto.muros), len(proyecto.solidos),
    )
    return proyecto
