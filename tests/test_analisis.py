"""Pruebas de la orquestación del análisis: validación previa, preparación
de cargas y un pórtico completo importado de un IFC (con ccx real)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.combinaciones import esfuerzos
from ifc_structural.core.cargas import Apoyo, CargaArea
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.ifc.importador import importar_ifc
from ifc_structural.solver.analisis import (
    ErrorAnalisis,
    ejecutar_analisis,
    preparar_cargas_lineales,
    validar_modelo,
)

from .ayuda_ifc import crear_ifc_portico

CASOS = {"D": CasoCarga("D", "Carga muerta", TipoCaso.D), "L": CasoCarga("L", "Carga viva", TipoCaso.L)}
COMBINACIONES = {
    "C1": Combinacion("C1", "1.4D", {"D": 1.4}),
    "C2": Combinacion("C2", "1.2D+1.6L", {"D": 1.2, "L": 1.6}),
}


def _dos_vigas_separadas() -> Proyecto:
    p = Proyecto("Validación")
    p.agregar_material(Material(nombre="H", modulo_elasticidad=2.5e7))
    p.agregar_perfil(PerfilSeccion(nombre="V", area=0.15, inercia_y=1e-4, inercia_z=1e-3, inercia_torsional=1e-4,
                                   altura=0.5, ancho=0.3))
    for id_, (x, y, z) in {"A": (0, 0, 0), "B": (4, 0, 0), "C": (10, 0, 0), "D": (14, 0, 0)}.items():
        p.agregar_nodo(x, y, z, id_)
    p.agregar_viga("A", "B", "H", "V", id_="B1")
    p.agregar_viga("C", "D", "H", "V", id_="B2")  # sin conexión con B1
    return p


def test_sin_apoyos_es_un_error():
    errores, _ = validar_modelo(_dos_vigas_separadas(), CASOS, COMBINACIONES, [])
    assert any("apoyos" in e for e in errores)


def test_detecta_barras_que_flotan_sin_conexion_a_ningun_apoyo():
    p = _dos_vigas_separadas()

    errores, _ = validar_modelo(p, CASOS, COMBINACIONES, [Apoyo(id="AP1", nodo_id="A")])

    flotantes = [e for e in errores if "sin conexión" in e]
    assert len(flotantes) == 1
    assert "B2" in flotantes[0] and "B1" not in flotantes[0]


def test_sin_casos_es_un_error():
    errores, _ = validar_modelo(_dos_vigas_separadas(), {}, {}, [Apoyo(id="AP1", nodo_id="A")])
    assert any("casos de carga" in e for e in errores)


def test_apoyo_en_nodo_que_no_es_de_ninguna_barra_se_advierte():
    p = _dos_vigas_separadas()
    p.agregar_nodo(50, 50, 0, "HUERFANO")

    _, advertencias = validar_modelo(p, CASOS, COMBINACIONES, [Apoyo(id="X", nodo_id="HUERFANO"), Apoyo(id="Y", nodo_id="A")])

    assert any("ignorarán" in a for a in advertencias)


def test_losa_sin_vigas_de_soporte_se_advierte_y_su_carga_no_aborta_el_analisis():
    p = _dos_vigas_separadas()
    contorno = [p.agregar_nodo(x, y, 3.0).id for x, y in ((0, 0), (4, 0), (4, 6), (0, 6))]
    losa = p.agregar_losa(contorno, espesor=0.2, material="H", id_="L1")  # sin vigas de soporte
    assert not any(losa.vigas_soporte)

    _, advertencias = validar_modelo(p, CASOS, COMBINACIONES, [Apoyo(id="AP1", nodo_id="A")])
    assert any("L1" in a and "vigas de soporte" in a for a in advertencias)

    cargas, avisos = preparar_cargas_lineales(
        p, CASOS, {"L": [CargaArea(id="CA", caso_id="L", presion=5.0, losa_id="L1")]}, {}
    )
    assert any("no se transfirió" in a.lower() for a in avisos)
    assert cargas["L"] == []  # la carga de área no se pudo transferir
    assert len(cargas["D"]) == 2  # pero el peso propio de las dos vigas sí está


pytestmark_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


@pytestmark_ccx
def test_portico_importado_de_un_ifc_se_analiza_de_punta_a_punta(tmp_path: Path):
    ruta = tmp_path / "portico.ifc"
    crear_ifc_portico(str(ruta))
    p = importar_ifc(str(ruta))

    apoyos = [Apoyo(id=f"AP{i}", nodo_id=c.nodo_inicio) for i, c in enumerate(p.columnas.values(), start=1)]
    losa = next(iter(p.losas.values()))
    resultado = ejecutar_analisis(
        p, CASOS, COMBINACIONES, apoyos,
        {"L": [CargaArea(id="CA1", caso_id="L", presion=5.0, losa_id=losa.id)]}, {}, {},
        tmp_path / "trabajo",
    )

    assert resultado.advertencias == []
    assert set(resultado.tensiones_por_caso) == {"D", "L"}
    assert set(resultado.tensiones_por_combinacion) == {"C1", "C2"}

    # La carga viva de 5 kPa sobre la losa produce esfuerzos apreciables y
    # finitos; la combinación mayorada supera al caso por separado.
    vm_l = esfuerzos.von_mises(resultado.tensiones_por_caso["L"])
    vm_c2 = esfuerzos.von_mises(resultado.tensiones_por_combinacion["C2"])
    assert np.isfinite(vm_c2).all()
    assert vm_l.max() > 100.0  # kPa
    assert vm_c2.max() > vm_l.max()

    # Cada hexaedro de la malla de esfuerzos pertenece a una barra del modelo.
    barras_en_resultados = {resultado.malla.elementos[int(e)].barra_id for e in resultado.malla_tensiones.ids_elementos}
    assert barras_en_resultados == set(p.vigas) | set(p.columnas)


def _viga_simple() -> Proyecto:
    p = Proyecto("Estabilidad")
    p.agregar_material(Material(nombre="H", modulo_elasticidad=2.5e7))
    p.agregar_perfil(PerfilSeccion(nombre="V", area=0.15, inercia_y=1e-4, inercia_z=1e-3, inercia_torsional=1e-4,
                                   altura=0.5, ancho=0.3))
    p.agregar_nodo(0, 0, 0, "A")
    p.agregar_nodo(4, 0, 0, "B")
    p.agregar_viga("A", "B", "H", "V", id_="B1")
    return p


def _errores_de_estabilidad(p: Proyecto, apoyos):
    errores, _ = validar_modelo(p, CASOS, COMBINACIONES, apoyos)
    return [e for e in errores if "inestable" in e]


def test_un_empotramiento_es_estable():
    assert _errores_de_estabilidad(_viga_simple(), [Apoyo(id="E", nodo_id="A")]) == []


def test_un_solo_pasador_es_un_mecanismo():
    """ccx NO avisa de esto: 'resuelve' y devuelve esfuerzos falsos pero
    verosímiles, por eso se verifica la estabilidad antes de resolver."""

    pasador = Apoyo(id="P", nodo_id="A", rx=False, ry=False, rz=False)
    errores = _errores_de_estabilidad(_viga_simple(), [pasador])
    assert len(errores) == 1 and "3 de los 6" in errores[0]


def test_dos_pasadores_alineados_dejan_libre_el_giro_alrededor_del_eje():
    libre = dict(rx=False, ry=False, rz=False)
    errores = _errores_de_estabilidad(
        _viga_simple(), [Apoyo(id="P1", nodo_id="A", **libre), Apoyo(id="P2", nodo_id="B", **libre)]
    )
    assert len(errores) == 1 and "1 de los 6" in errores[0]


def test_dos_pasadores_mas_un_giro_de_torsion_restringido_son_estables():
    p = _viga_simple()
    apoyos = [
        Apoyo(id="P1", nodo_id="A", rx=True, ry=False, rz=False),  # restringe el giro en X (torsión)
        Apoyo(id="P2", nodo_id="B", rx=False, ry=False, rz=False),
    ]
    assert _errores_de_estabilidad(p, apoyos) == []


def test_apoyos_deslizantes_que_no_impiden_una_traslacion_son_inestables():
    # Todos los giros y dos traslaciones restringidas, pero la traslación X libre.
    apoyo = Apoyo(id="D", nodo_id="A", ux=False)
    errores = _errores_de_estabilidad(_viga_simple(), [apoyo])
    assert len(errores) == 1 and "1 de los 6" in errores[0]
