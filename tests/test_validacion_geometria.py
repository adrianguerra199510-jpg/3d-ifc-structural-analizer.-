"""Geometría no finita (NaN): se detecta y se nombra el elemento en lugar de fallar con «cannot convert float NaN»."""

from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.validacion import exigir_geometria_valida, problemas_de_geometria
from ifc_structural.ifc import importador
from ifc_structural.ifc.importador import importar_ifc
from ifc_structural.solido.analisis_solido import estimar, tamano_sugerido
from ifc_structural.solido.deteccion import detectar_contactos

from .ayuda_ifc import crear_ifc_portico


@pytest.fixture
def portico(tmp_path: Path):
    ruta = tmp_path / "p.ifc"
    crear_ifc_portico(str(ruta))
    return importar_ifc(str(ruta))


def test_modelo_sano_no_tiene_problemas(portico):
    assert problemas_de_geometria(portico) == []
    exigir_geometria_valida(portico)


def test_nodo_con_nan_se_nombra_y_bloquea_el_mallado(portico):
    nodo = next(iter(portico.nodos.values()))
    nodo.x = float("nan")
    texto = "\n".join(problemas_de_geometria(portico))
    assert f"nodo {nodo.id}" in texto and ("viga" in texto or "columna" in texto)
    for llamada in (lambda: detectar_contactos(portico, 0.01), lambda: estimar(portico, 0.1), lambda: tamano_sugerido(portico)):
        with pytest.raises(ValueError, match="no es válida"):
            llamada()


def test_espesor_con_nan_de_una_losa(portico):
    losa = next(iter(portico.losas.values()))
    losa.espesor = float("nan")
    assert any(f"losa {losa.id}" in p and "espesor" in p for p in problemas_de_geometria(portico))
    with pytest.raises(ValueError, match=losa.id):
        estimar(portico, 0.1)


def test_perfil_con_propiedades_nan(portico):
    perfil = next(iter(portico.perfiles.values()))
    perfil.area = float("nan")
    assert any("perfil" in p for p in problemas_de_geometria(portico))


def test_vertices_no_finitos_en_la_importacion():
    class E:
        Name = "Placa X"
        GlobalId = "G1"

        def is_a(self):
            return "IfcPlate"

    class P:
        avisos = []

    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [np.nan, 0, 1.0]])
    limpio = importador._solo_finitos(v, E(), P)
    assert len(limpio) == 4 and np.isfinite(limpio).all() and "Placa X" in P.avisos[0]
    assert importador._solo_finitos(v[[0, 4, 4]], E(), P) is None
    assert len(importador._solo_finitos(v[:4], E(), P)) == 4


def test_ubicacion_del_error_indica_archivo_y_linea():
    pytest.importorskip("PySide6")
    from ifc_structural.gui.trabajador_analisis import ubicacion_del_error

    try:
        int(float("nan"))
    except ValueError as exc:
        assert "test_validacion_geometria.py" in ubicacion_del_error(exc)
