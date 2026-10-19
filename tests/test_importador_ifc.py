"""Pruebas del importador IFC, contra un archivo sintético generado con
ifcopenshell (ver ayuda_ifc.py), ya que no se dispone de un archivo real
exportado de Revit en este entorno."""

from pathlib import Path

import pytest

from ifc_structural.ifc.importador import importar_ifc

from .ayuda_ifc import crear_ifc_prueba


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
