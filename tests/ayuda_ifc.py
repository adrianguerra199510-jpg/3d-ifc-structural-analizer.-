"""Construcción de un archivo IFC mínimo y sintético para probar el
importador sin depender de un archivo real exportado de Revit.

No forma parte del código de la aplicación: es solo infraestructura de
pruebas. Crea:

- Una columna vertical (sin perfil real en el IFC ni propiedades
  mecánicas: ejercita el camino de respaldo por geometría/OBB y los
  valores de material por defecto).
- Una viga horizontal con un ``IfcMaterialProfileSetUsage`` real
  (perfil rectangular) y material con propiedades mecánicas explícitas.
- Una losa horizontal con material con propiedades mecánicas explícitas.
"""

from __future__ import annotations

import ifcopenshell
import ifcopenshell.guid as guid


def _placement(model, x=0.0, y=0.0, z=0.0, axis=None, ref_direction=None):
    posicion_kwargs = {"Location": model.create_entity("IfcCartesianPoint", Coordinates=(x, y, z))}
    if axis is not None:
        posicion_kwargs["Axis"] = model.create_entity("IfcDirection", DirectionRatios=axis)
    if ref_direction is not None:
        posicion_kwargs["RefDirection"] = model.create_entity("IfcDirection", DirectionRatios=ref_direction)
    return model.create_entity(
        "IfcLocalPlacement",
        RelativePlacement=model.create_entity("IfcAxis2Placement3D", **posicion_kwargs),
    )


def _propiedad_real(model, nombre, valor):
    return model.create_entity(
        "IfcPropertySingleValue", Name=nombre, NominalValue=model.create_entity("IfcReal", valor)
    )


def _material_con_propiedades(model, nombre, e_pa, poisson, densidad_kg_m3):
    material = model.create_entity("IfcMaterial", Name=nombre)
    model.create_entity(
        "IfcMaterialProperties",
        Name="Pset_MaterialMechanical",
        Properties=[
            _propiedad_real(model, "YoungModulus", e_pa),
            _propiedad_real(model, "PoissonRatio", poisson),
            _propiedad_real(model, "MassDensity", densidad_kg_m3),
        ],
        Material=material,
    )
    return material


def _solido_extruido(model, context, ancho, alto, profundidad, nombre_perfil=None):
    perfil = model.create_entity(
        "IfcRectangleProfileDef", ProfileType="AREA", ProfileName=nombre_perfil, XDim=ancho, YDim=alto
    )
    posicion = model.create_entity(
        "IfcAxis2Placement3D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    )
    solido = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=perfil,
        Position=posicion,
        ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
        Depth=profundidad,
    )
    representacion = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solido],
    )
    return perfil, model.create_entity("IfcProductDefinitionShape", Representations=[representacion])


def crear_ifc_prueba(ruta: str) -> None:
    model = ifcopenshell.file(schema="IFC4")

    longitud = model.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    unidades = model.create_entity("IfcUnitAssignment", Units=[longitud])
    origen = model.create_entity("IfcAxis2Placement3D", Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)))
    contexto = model.create_entity(
        "IfcGeometricRepresentationContext",
        ContextType="Model",
        CoordinateSpaceDimension=3,
        Precision=1e-5,
        WorldCoordinateSystem=origen,
    )
    proyecto = model.create_entity(
        "IfcProject", GlobalId=guid.new(), Name="Proyecto de prueba", UnitsInContext=unidades, RepresentationContexts=[contexto]
    )

    sitio = model.create_entity("IfcSite", GlobalId=guid.new(), Name="Sitio", ObjectPlacement=_placement(model))
    edificio = model.create_entity("IfcBuilding", GlobalId=guid.new(), Name="Edificio", ObjectPlacement=_placement(model))
    piso = model.create_entity("IfcBuildingStorey", GlobalId=guid.new(), Name="Piso 1", ObjectPlacement=_placement(model))
    model.create_entity("IfcRelAggregates", GlobalId=guid.new(), RelatingObject=proyecto, RelatedObjects=[sitio])
    model.create_entity("IfcRelAggregates", GlobalId=guid.new(), RelatingObject=sitio, RelatedObjects=[edificio])
    model.create_entity("IfcRelAggregates", GlobalId=guid.new(), RelatingObject=edificio, RelatedObjects=[piso])

    # --- Columna: 0.4x0.4, 3 m de altura, SIN perfil real ni propiedades
    # mecánicas (ejercita los caminos de respaldo del importador).
    material_columna = model.create_entity("IfcMaterial", Name="HORMIGON_COLUMNA")
    _, forma_columna = _solido_extruido(model, contexto, 0.4, 0.4, 3.0)
    columna = model.create_entity(
        "IfcColumn", GlobalId=guid.new(), Name="C1", ObjectPlacement=_placement(model, 0.0, 0.0, 0.0), Representation=forma_columna
    )
    model.create_entity("IfcRelAssociatesMaterial", GlobalId=guid.new(), RelatingMaterial=material_columna, RelatedObjects=[columna])
    model.create_entity("IfcRelContainedInSpatialStructure", GlobalId=guid.new(), RelatingStructure=piso, RelatedElements=[columna])

    # --- Viga: 0.3x0.5, 4 m de longitud, a lo largo de X, en z=3. CON
    # perfil real (IfcMaterialProfileSetUsage) y propiedades mecánicas.
    material_viga = _material_con_propiedades(model, "HORMIGON_VIGA", 2.5e10, 0.2, 2400.0)
    perfil_viga, forma_viga = _solido_extruido(model, contexto, 0.3, 0.5, 4.0, nombre_perfil="R30X50")
    perfil_material = model.create_entity("IfcMaterialProfile", Profile=perfil_viga, Material=material_viga)
    conjunto_perfiles = model.create_entity("IfcMaterialProfileSet", MaterialProfiles=[perfil_material])
    uso_perfil = model.create_entity("IfcMaterialProfileSetUsage", ForProfileSet=conjunto_perfiles, CardinalPoint=5)
    viga = model.create_entity(
        "IfcBeam",
        GlobalId=guid.new(),
        Name="B1",
        ObjectPlacement=_placement(model, 0.0, 0.0, 3.0, axis=(1.0, 0.0, 0.0), ref_direction=(0.0, 0.0, 1.0)),
        Representation=forma_viga,
    )
    model.create_entity("IfcRelAssociatesMaterial", GlobalId=guid.new(), RelatingMaterial=uso_perfil, RelatedObjects=[viga])
    model.create_entity("IfcRelContainedInSpatialStructure", GlobalId=guid.new(), RelatingStructure=piso, RelatedElements=[viga])

    # --- Losa: 4 x 6 m, espesor 0.15 m, en z=3. CON propiedades mecánicas.
    material_losa = _material_con_propiedades(model, "HORMIGON_LOSA", 2.5e10, 0.2, 2400.0)
    _, forma_losa = _solido_extruido(model, contexto, 4.0, 6.0, 0.15)
    losa = model.create_entity(
        "IfcSlab", GlobalId=guid.new(), Name="L1", ObjectPlacement=_placement(model, 0.0, 0.0, 3.0), Representation=forma_losa
    )
    model.create_entity("IfcRelAssociatesMaterial", GlobalId=guid.new(), RelatingMaterial=material_losa, RelatedObjects=[losa])
    model.create_entity("IfcRelContainedInSpatialStructure", GlobalId=guid.new(), RelatingStructure=piso, RelatedElements=[losa])

    model.write(ruta)
