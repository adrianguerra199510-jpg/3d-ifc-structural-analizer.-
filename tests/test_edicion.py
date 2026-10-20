"""Eliminar elementos del modelo con todo lo que depende de ellos."""

from pathlib import Path

from ifc_structural.core.cargas import ApoyoArea, CargaArea, CargaLineal, CargaPuntual, Conexion
from ifc_structural.core.edicion import eliminar_elementos
from ifc_structural.ifc.importador import importar_ifc

from .ayuda_ifc import crear_ifc_portico


def _portico(tmp_path: Path):
    ruta = tmp_path / "p.ifc"
    crear_ifc_portico(str(ruta))
    return importar_ifc(str(ruta))


def test_eliminar_una_viga_quita_sus_cargas_y_lo_que_depende_de_ella(tmp_path: Path):
    p = _portico(tmp_path)
    viga = next(iter(p.vigas))
    nodos_antes = len(p.nodos)
    lineales = {"D": [CargaLineal(id="CL1", caso_id="D", elemento_id=viga, w_inicio=5.0, w_fin=5.0, direccion=(0, 0, -1))]}
    puntuales = {"D": [CargaPuntual(id="CP1", caso_id="D", elemento_id=viga, posicion=0.5, fz=-3.0)]}
    apoyos, conexiones, r = eliminar_elementos(p, [("VIGA", viga)], [], [], {}, lineales, puntuales, {})
    assert viga not in p.vigas and r.elementos == [("VIGA", viga)]
    assert lineales["D"] == [] and puntuales["D"] == [] and r.cargas == 2
    assert all(viga not in (l.vigas_soporte or []) for l in p.losas.values())  # la losa ya no la usa de soporte
    assert len(p.nodos) <= nodos_antes


def test_eliminar_una_losa_quita_sus_cargas_de_area(tmp_path: Path):
    p = _portico(tmp_path)
    losa = next(iter(p.losas))
    area = {"D": [CargaArea(id="CA1", caso_id="D", losa_id=losa, presion=5.0)]}
    _, _, r = eliminar_elementos(p, [("LOSA", losa)], [], [], area, {}, {}, {})
    assert losa not in p.losas and area["D"] == [] and r.cargas == 1


def test_eliminar_una_columna_quita_los_apoyos_de_sus_nodos_sin_uso(tmp_path: Path):
    p = _portico(tmp_path)
    columna = next(iter(p.columnas))
    barra = p.columnas[columna]
    from ifc_structural.core.cargas import Apoyo

    base = barra.nodo_inicio
    apoyos, _, r = eliminar_elementos(p, [("COLUMNA", columna)], [Apoyo(id="AP1", nodo_id=base)], [], {}, {}, {}, {})
    assert columna not in p.columnas
    assert apoyos == [] and r.apoyos == 1 and base not in p.nodos
