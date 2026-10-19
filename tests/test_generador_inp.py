"""Pruebas del generador de .inp (sin invocar a ccx)."""

from pathlib import Path

import pytest

from ifc_structural.core.cargas import Apoyo, CargaLineal, CargaPuntual
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.solver.generador_inp import construir_malla, generar_inp_caso


def _proyecto_viga_simple() -> Proyecto:
    p = Proyecto("Prueba generador")
    p.agregar_material(Material(nombre="HORMIGON", modulo_elasticidad=2.5e7, densidad=24.0))
    p.agregar_perfil(
        PerfilSeccion(
            nombre="V30X50",
            area=0.15,
            inercia_y=3.125e-4,
            inercia_z=1.125e-3,
            inercia_torsional=1e-4,
            altura=0.5,
            ancho=0.3,
        )
    )
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(4, 0, 0, "N2")
    p.agregar_viga(n1.id, n2.id, "HORMIGON", "V30X50", id_="B1")
    return p


def test_construir_malla_subdivide_segun_quiebres_de_carga():
    p = _proyecto_viga_simple()
    cargas_lineales = {
        "D": [
            CargaLineal(id="CL1", caso_id="D", elemento_id="B1", w_inicio=3.6, w_fin=3.6, s_inicio=0.0, s_fin=0.5),
            CargaLineal(id="CL2", caso_id="D", elemento_id="B1", w_inicio=5.0, w_fin=2.0, s_inicio=0.5, s_fin=1.0),
        ]
    }
    cargas_puntuales = {"D": [CargaPuntual(id="CP1", caso_id="D", elemento_id="B1", posicion=0.25, fz=-5.0)]}

    malla = construir_malla(p, cargas_lineales, cargas_puntuales, longitud_elemento=1.0)

    # Quiebres esperados: 0, 0.25 (carga puntual), 0.5 (cambio de carga lineal), 1.0
    # -> 3 tramos de 1 m, 1 m y 2 m; con elementos de 1 m como máximo y mínimo 2
    # por tramo => 2 + 2 + 2 = 6 elementos.
    assert len(malla.elementos_por_barra["B1"]) == 6
    assert len(malla.orden_nodos_por_barra["B1"]) == 7

    # El nodo en s=0.25 debe existir (para aplicar la carga puntual ahí).
    assert ("B1", 0.25) in malla.nodo_ccx_de_posicion

    # Los nodos extremos deben coincidir con los nodos reales del proyecto.
    assert malla.nodo_ccx_de_nodo_original["N1"] == malla.orden_nodos_por_barra["B1"][0]
    assert malla.nodo_ccx_de_nodo_original["N2"] == malla.orden_nodos_por_barra["B1"][-1]


def test_generar_inp_contiene_las_secciones_esperadas(tmp_path: Path):
    p = _proyecto_viga_simple()
    apoyo = Apoyo(id="AP1", nodo_id="N1")
    cargas_lineales = {"D": []}
    cargas_puntuales = {"D": [CargaPuntual(id="CP1", caso_id="D", nodo_id="N2", fz=-10.0)]}

    malla = construir_malla(p, cargas_lineales, cargas_puntuales)
    ruta = generar_inp_caso(p, malla, "D", cargas_lineales["D"], cargas_puntuales["D"], [apoyo], tmp_path / "D.inp")

    texto = ruta.read_text(encoding="utf-8")
    assert "*NODE" in texto
    assert "*ELEMENT,TYPE=B31" in texto
    assert "*MATERIAL,NAME=HORMIGON" in texto
    assert "SECTION=RECT" in texto
    assert "*BOUNDARY" in texto
    assert "*CLOAD" in texto
    assert "*STEP" in texto
    # Se piden esfuerzos (malla expandida 3D), no fuerzas de sección.
    assert "*EL FILE\nS\n" in texto
    assert "SECTION FORCES" not in texto
    assert "*NODE PRINT" not in texto
    assert "*END STEP" in texto
    # No debe usarse SECTION=GENERAL (no funciona para B31 en ccx 2.21).
    assert "GENERAL" not in texto


def test_seccion_equivalente_para_perfil_sin_dimensiones(tmp_path: Path):
    p = Proyecto("Prueba perfil sin dimensiones")
    p.agregar_material(Material(nombre="ACERO", modulo_elasticidad=2.0e8, densidad=78.5))
    p.agregar_perfil(
        PerfilSeccion(nombre="IPE200", area=0.00285, inercia_y=1.42e-6, inercia_z=1.94e-5, inercia_torsional=6e-8)
    )
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(3, 0, 0, "N2")
    p.agregar_viga(n1.id, n2.id, "ACERO", "IPE200", id_="B1")
    apoyo = Apoyo(id="AP1", nodo_id="N1")

    cargas_lineales = {"D": []}
    cargas_puntuales = {"D": []}
    malla = construir_malla(p, cargas_lineales, cargas_puntuales)
    ruta = generar_inp_caso(p, malla, "D", [], [], [apoyo], tmp_path / "D.inp")
    texto = ruta.read_text(encoding="utf-8")
    assert "SECTION=RECT" in texto
