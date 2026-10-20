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
    proyecto = importar_ifc(str(ruta_ifc_prueba))

    assert len(proyecto.columnas) == 1
    assert len(proyecto.vigas) == 1
    assert len(proyecto.losas) == 1
    assert len(proyecto.muros) == 0


def test_columna_sin_perfil_real_usa_respaldo_por_geometria(ruta_ifc_prueba: Path):
    proyecto = importar_ifc(str(ruta_ifc_prueba))

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
    proyecto = importar_ifc(str(ruta_ifc_prueba))

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
    proyecto = importar_ifc(str(ruta_ifc_prueba))

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
    return importar_ifc(str(ruta))


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
    p = importar_ifc(str(ruta))
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
