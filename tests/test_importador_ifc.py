"""Pruebas del importador IFC, contra un archivo sintético generado con
ifcopenshell (ver ayuda_ifc.py), ya que no se dispone de un archivo real
exportado de Revit en este entorno."""

from pathlib import Path

import pytest

from ifc_structural.ifc.importador import importar_ifc

from .ayuda_ifc import crear_ifc_portico, crear_ifc_prueba


@pytest.fixture()
def ruta_ifc_prueba(tmp_path: Path) -> Path:
    ruta = tmp_path / "prueba.ifc"
    crear_ifc_prueba(str(ruta))
    return ruta


def test_importa_columna_viga_y_losa(ruta_ifc_prueba: Path):
    proyecto = importar_ifc(str(ruta_ifc_prueba), modo="barras")

    assert len(proyecto.columnas) == 1
    assert len(proyecto.vigas) == 1
    assert len(proyecto.losas) == 1
    assert len(proyecto.muros) == 0


def test_columna_sin_perfil_real_usa_respaldo_por_geometria(ruta_ifc_prueba: Path):
    proyecto = importar_ifc(str(ruta_ifc_prueba), modo="barras")

    columna = next(iter(proyecto.columnas.values()))
    n1 = proyecto.nodos[columna.nodo_inicio]
    n2 = proyecto.nodos[columna.nodo_fin]
    longitud = n1.distancia(n2)
    assert longitud == pytest.approx(3.0, abs=1e-3)

    perfil = proyecto.perfiles[columna.perfil]
    assert perfil.area == pytest.approx(0.4 * 0.4, rel=0.05)

    material = proyecto.materiales[columna.material]
    # Sin Pset_MaterialMechanical en el IFC -> valores por defecto de hormigón.
    assert material.modulo_elasticidad == pytest.approx(2.5e7)
    assert material.densidad == pytest.approx(24.0)


def test_viga_con_perfil_real_usa_dimensiones_exactas(ruta_ifc_prueba: Path):
    proyecto = importar_ifc(str(ruta_ifc_prueba), modo="barras")

    viga = next(iter(proyecto.vigas.values()))
    n1 = proyecto.nodos[viga.nodo_inicio]
    n2 = proyecto.nodos[viga.nodo_fin]
    longitud = n1.distancia(n2)
    assert longitud == pytest.approx(4.0, abs=1e-3)
    # La viga debe quedar en z=3 (un piso elevado), no en el origen.
    assert n1.z == pytest.approx(3.0, abs=1e-3)
    assert n2.z == pytest.approx(3.0, abs=1e-3)

    # La altura del perfil (0,5 m) apunta hacia donde mide 0,5 m en el modelo:
    # en este IFC la viga está "acostada" (0,5 m en Y, 0,3 m en Z).
    assert abs(viga.vector_n1[1]) == pytest.approx(1.0, abs=1e-6)

    perfil = proyecto.perfiles[viga.perfil]
    assert perfil.area == pytest.approx(0.3 * 0.5, rel=1e-6)
    assert perfil.altura == pytest.approx(0.5, rel=1e-6)
    assert perfil.ancho == pytest.approx(0.3, rel=1e-6)

    material = proyecto.materiales[viga.material]
    # 2.5e10 Pa -> 2.5e7 kPa
    assert material.modulo_elasticidad == pytest.approx(2.5e7, rel=1e-6)
    # 2400 kg/m3 -> kN/m3
    assert material.densidad == pytest.approx(2400.0 * 9.80665 / 1000.0, rel=1e-6)


def test_losa_tiene_contorno_espesor_y_area_correctos(ruta_ifc_prueba: Path):
    proyecto = importar_ifc(str(ruta_ifc_prueba), modo="barras")

    losa = next(iter(proyecto.losas.values()))
    assert losa.espesor == pytest.approx(0.15, rel=1e-3)
    assert len(losa.nodos_contorno) == 4

    puntos = [proyecto.nodos[nid].coordenadas for nid in losa.nodos_contorno]
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    assert max(xs) - min(xs) == pytest.approx(4.0, rel=0.02)
    assert max(ys) - min(ys) == pytest.approx(6.0, rel=0.02)


@pytest.fixture()
def proyecto_portico(tmp_path: Path):
    ruta = tmp_path / "portico.ifc"
    crear_ifc_portico(str(ruta))
    return importar_ifc(str(ruta), modo="barras")


def test_portico_las_vigas_se_conectan_a_los_ejes_de_las_columnas(proyecto_portico):
    p = proyecto_portico
    assert (len(p.columnas), len(p.vigas), len(p.losas)) == (4, 4, 1)

    # Cada extremo de viga debe ser exactamente el tope de una columna.
    topes = {c.nodo_fin for c in p.columnas.values()}
    extremos_de_vigas = {n for v in p.vigas.values() for n in (v.nodo_inicio, v.nodo_fin)}
    assert extremos_de_vigas == topes
    for v in p.vigas.values():
        assert p.longitud_barra(v.id) in (pytest.approx(4.0, abs=1e-3), pytest.approx(6.0, abs=1e-3))


def test_portico_las_columnas_van_de_la_base_al_tope(proyecto_portico):
    p = proyecto_portico
    for c in p.columnas.values():
        assert p.nodos[c.nodo_inicio].z == pytest.approx(0.0, abs=1e-6)
        assert p.nodos[c.nodo_fin].z == pytest.approx(3.0, abs=1e-6)


def test_portico_la_altura_de_las_vigas_queda_en_vertical(proyecto_portico):
    p = proyecto_portico
    for v in p.vigas.values():
        assert abs(v.vector_n1[2]) == pytest.approx(1.0, abs=1e-6)
        assert p.perfiles[v.perfil].altura == pytest.approx(0.5)


def test_portico_detecta_las_vigas_de_soporte_de_la_losa(proyecto_portico):
    losa = next(iter(proyecto_portico.losas.values()))

    assert all(losa.vigas_soporte), losa.vigas_soporte
    assert sorted(losa.vigas_soporte) == sorted(proyecto_portico.vigas)
    assert losa.espesor == pytest.approx(0.2, abs=1e-3)


def test_un_ifc_en_milimetros_se_importa_en_metros_y_las_placas_no_son_barras(tmp_path):
    """Regresión: la geometría de ifcopenshell ya viene en metros; escalarla otra vez con la unidad del archivo
    dejaba el modelo 1000 veces más pequeño y rompía el análisis (NaN)."""

    from .ayuda_ifc import crear_ifc_viga_armada_en_milimetros

    ruta = tmp_path / "viga_mm.ifc"
    crear_ifc_viga_armada_en_milimetros(str(ruta))
    p = importar_ifc(str(ruta), modo="barras")
    assert len(p.vigas) == 1 and not p.columnas  # los rigidizadores no son barras
    viga = next(iter(p.vigas.values()))
    a, b = p.nodos[viga.nodo_inicio], p.nodos[viga.nodo_fin]
    assert abs(b.x - a.x) == pytest.approx(6.0, abs=1e-3)  # 6000 mm = 6 m
    assert p.perfiles[viga.perfil].altura == pytest.approx(0.6)
    assert len(p.solidos) == 12  # 11 rigidizadores y la placa superior, con su geometría real
    extension = max(s.extension.max() for s in p.solidos.values())
    assert extension == pytest.approx(6.0, abs=1e-3)
    material = p.materiales[next(iter(p.materiales))]
    assert material.modulo_elasticidad == pytest.approx(2.1e8)  # «Acero» en el nombre: acero estructural
    from ifc_structural.solido.miembros import barras_de_longitud_cero

    assert barras_de_longitud_cero(p) == []


# --- perfiles que Revit exporta dentro de recortes booleanos y de IfcMappedItem, o que la app no idealiza como barra ---

@pytest.fixture()
def proyecto_perfiles_especiales(tmp_path: Path):
    from .ayuda_ifc import crear_ifc_perfiles_especiales

    ruta = tmp_path / "perfiles_especiales.ifc"
    crear_ifc_perfiles_especiales(str(ruta))
    return importar_ifc(str(ruta), modo="barras")


def _barra(proyecto, nombre):
    return next(b for b in proyecto.todas_las_barras().values() if b.nombre == nombre)


def test_la_viga_recortada_conserva_su_perfil_en_i_y_su_longitud_recortada(proyecto_perfiles_especiales):
    p = proyecto_perfiles_especiales
    viga = _barra(p, "VIGA_RECORTADA")
    perfil = p.perfiles[viga.perfil]
    assert perfil.forma == "I" and perfil.nombre == "IPE300"
    assert perfil.altura == pytest.approx(0.3) and perfil.ancho == pytest.approx(0.15)
    assert p.nodos[viga.nodo_inicio].distancia(p.nodos[viga.nodo_fin]) == pytest.approx(2.5, abs=1e-3)


def test_el_perfil_en_i_dentro_de_un_mapped_item_se_lee(proyecto_perfiles_especiales):
    p = proyecto_perfiles_especiales
    columna = _barra(p, "COLUMNA_I_MAPEADA")
    perfil = p.perfiles[columna.perfil]
    assert perfil.forma == "I" and perfil.nombre == "HEB200"
    assert perfil.espesor_ala == pytest.approx(0.015) and perfil.espesor_alma == pytest.approx(0.009)


@pytest.mark.parametrize("nombre, area", [
    ("TUBO_MAPEADO", 3.141592653589793 * (0.2**2 - 0.19**2)),  # tubo de R = 0,2 y e = 0,01 m
    ("SHS_DIRECTO", 0.2**2 - 0.18**2),  # cajón cuadrado de 0,2 m y e = 0,01 m
])
def test_los_perfiles_huecos_no_soportados_se_importan_como_solidos_reales(proyecto_perfiles_especiales, nombre, area):
    p = proyecto_perfiles_especiales
    assert nombre not in {b.nombre for b in p.todas_las_barras().values()}
    assert not [n for n in p.perfiles if n.startswith("APROX_")]
    solidos = [s for s in p.solidos.values() if s.nombre == nombre]
    assert len(solidos) == 1
    assert solidos[0].volumen == pytest.approx(area * 3.0, rel=0.03)  # la sección hueca real, no un cuadrado macizo


def test_se_avisa_de_los_perfiles_que_pasaron_a_solido(proyecto_perfiles_especiales):
    avisos = " ".join(proyecto_perfiles_especiales.avisos)
    assert "IfcCircleHollowProfileDef" in avisos and "IfcRectangleHollowProfileDef" in avisos
    assert "sólido" in avisos


def test_la_opcion_aproximar_mantiene_el_rectangulo_macizo_de_antes(tmp_path):
    from .ayuda_ifc import crear_ifc_perfiles_especiales

    ruta = tmp_path / "perfiles_especiales.ifc"
    crear_ifc_perfiles_especiales(str(ruta))
    p = importar_ifc(str(ruta), modo="barras", perfiles_no_soportados="aproximar")
    assert {"SHS_DIRECTO", "TUBO_MAPEADO"} <= {b.nombre for b in p.todas_las_barras().values()}
    assert len([n for n in p.perfiles if n.startswith("APROX_")]) == 2
    assert not p.solidos


# --- modo por defecto: todo entra como sólido, con la geometría exacta del IFC ---

def _geometria_cruda_del_ifc(ruta):
    """Vértices en coordenadas globales (metros) de cada elemento tal como los entrega ifcopenshell, sin ningún
    tratamiento posterior: la referencia contra la que se compara lo que importa la aplicación."""

    import ifcopenshell
    import ifcopenshell.geom
    import numpy as np

    modelo = ifcopenshell.open(str(ruta))
    ajustes = ifcopenshell.geom.settings()
    ajustes.set("use-world-coords", True)
    crudos = {}
    for e in modelo.by_type("IfcElement"):
        if e.is_a("IfcOpeningElement"):
            continue
        crudos[e.GlobalId] = np.array(ifcopenshell.geom.create_shape(ajustes, e).geometry.verts).reshape(-1, 3)
    return crudos


def test_por_defecto_vigas_columnas_y_losas_entran_como_solidos(ruta_ifc_prueba: Path):
    p = importar_ifc(str(ruta_ifc_prueba))

    assert not p.vigas and not p.columnas and not p.losas and not p.muros
    assert not p.nodos and not p.perfiles  # sin idealización: ni nodos ni perfiles de barra
    assert sorted(s.tipo_ifc for s in p.solidos.values()) == ["IfcBeam", "IfcColumn", "IfcSlab"]
    assert all(s.cerrado and s.volumen > 0 for s in p.solidos.values())


def test_los_solidos_importados_no_mueven_la_geometria_del_ifc(tmp_path):
    """El pórtico tiene las vigas terminando en las caras de las columnas: la idealización por barras las movía a los
    ejes. Como sólidos, cada vértice debe quedar donde lo pone el IFC (tolerancia de la soldadura de vértices, 1 µm)."""

    import numpy as np
    from scipy.spatial import cKDTree

    from .ayuda_ifc import crear_ifc_portico

    ruta = tmp_path / "portico.ifc"
    crear_ifc_portico(str(ruta))
    p = importar_ifc(str(ruta))
    crudos = _geometria_cruda_del_ifc(ruta)

    assert {s.ifc_guid for s in p.solidos.values()} == set(crudos)  # no se pierde ningún elemento
    for guid, original in crudos.items():
        importado = np.vstack([s.vertices for s in p.solidos.values() if s.ifc_guid == guid])
        assert cKDTree(original).query(importado)[0].max() < 2e-6  # ningún vértice importado se movió
        assert cKDTree(importado).query(original)[0].max() < 2e-6  # ni se perdió ninguno del IFC
        assert np.allclose(importado.min(axis=0), original.min(axis=0), atol=2e-6)
        assert np.allclose(importado.max(axis=0), original.max(axis=0), atol=2e-6)


def test_un_ifc_en_milimetros_como_solidos_queda_en_metros(tmp_path):
    from .ayuda_ifc import crear_ifc_viga_armada_en_milimetros

    ruta = tmp_path / "viga_mm.ifc"
    crear_ifc_viga_armada_en_milimetros(str(ruta))
    p = importar_ifc(str(ruta))
    assert not p.vigas and len(p.solidos) == 13  # la viga, 11 rigidizadores y la placa superior
    assert max(s.extension.max() for s in p.solidos.values()) == pytest.approx(6.0, abs=1e-3)


def test_en_modo_solidos_no_se_descartan_elementos_por_el_limite(ruta_ifc_prueba: Path, monkeypatch):
    from ifc_structural.ifc import importador

    monkeypatch.setattr(importador, "MAXIMO_DE_SOLIDOS", 2)
    assert len(importar_ifc(str(ruta_ifc_prueba)).solidos) == 3  # el límite solo protege el modo de barras


def test_el_modo_barras_sigue_disponible_y_el_modo_invalido_se_rechaza(ruta_ifc_prueba: Path):
    p = importar_ifc(str(ruta_ifc_prueba), modo="barras")
    assert len(p.columnas) == 1 and len(p.vigas) == 1 and len(p.losas) == 1 and not p.solidos
    with pytest.raises(ValueError):
        importar_ifc(str(ruta_ifc_prueba), modo="otro")
