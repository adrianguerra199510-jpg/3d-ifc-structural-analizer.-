"""Pruebas de la conversión de selecciones (elementos y caras) en apoyos y cargas."""

import pytest

from ifc_structural.core.asignacion import (
    ErrorAsignacion,
    Objetivo,
    cargas_para_objetivo,
    descripcion,
    nodo_para_apoyo,
    vista_previa_cargas,
)
from ifc_structural.core.cargas import CargaArea, CargaLineal, CargaPuntual
from ifc_structural.core.geometria import solido_elemento
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto

@pytest.fixture()
def p() -> Proyecto:
    p = Proyecto("Asignación")
    p.agregar_material(Material(nombre="H", modulo_elasticidad=2.5e7))
    p.agregar_perfil(PerfilSeccion(nombre="V", area=0.15, inercia_y=1e-4, inercia_z=1e-3, inercia_torsional=1e-4,
                                   altura=0.5, ancho=0.3))
    p.agregar_nodo(0, 0, 3, "N1")
    p.agregar_nodo(4, 0, 3, "N2")
    p.agregar_nodo(0, 0, 0, "N0")
    p.agregar_viga("N1", "N2", "H", "V", id_="B1")
    p.agregar_columna("N0", "N1", "H", "V", id_="C1")
    contorno = [p.agregar_nodo(x, y, 3.0).id for x, y in ((0, 0), (4, 0), (4, 6), (0, 6))]
    p.agregar_losa(contorno, espesor=0.2, material="H", id_="L1")
    p.agregar_muro([p.agregar_nodo(x, 8.0, z).id for x, z in ((0, 0), (4, 0), (4, 3), (0, 3))], espesor=0.2, material="H", id_="M1")
    return p


def _cara(p: Proyecto, tipo_elemento: str, id_: str, **filtro) -> Objetivo:
    for cara in solido_elemento(p, tipo_elemento, id_).caras:
        if all(getattr(cara, k) == v for k, v in filtro.items()):
            return Objetivo(tipo_elemento, id_, cara.indice)
    raise AssertionError(f"no hay cara {filtro}")


# -- Apoyos ------------------------------------------------------------------


def test_apoyo_en_la_cara_extrema_de_una_columna_va_en_su_nodo(p):
    assert nodo_para_apoyo(p, _cara(p, "COLUMNA", "C1", tipo="EXTREMO_INICIO")) == "N0"  # base
    assert nodo_para_apoyo(p, _cara(p, "COLUMNA", "C1", tipo="EXTREMO_FIN")) == "N1"  # tope
    assert nodo_para_apoyo(p, _cara(p, "VIGA", "B1", tipo="EXTREMO_FIN")) == "N2"


@pytest.mark.parametrize(
    "objetivo",
    [Objetivo("COLUMNA", "C1"), Objetivo("LOSA", "L1"), Objetivo("MURO", "M1")],
)
def test_apoyo_sin_cara_extrema_explica_que_hacer(p, objetivo):
    with pytest.raises(ErrorAsignacion):
        nodo_para_apoyo(p, objetivo)


def test_apoyo_en_cara_lateral_de_una_viga_no_vale(p):
    with pytest.raises(ErrorAsignacion, match="cara extrema"):
        nodo_para_apoyo(p, _cara(p, "VIGA", "B1", nombre="superior"))


# -- Cargas de área sobre losas -----------------------------------------------


def test_losa_completa_recibe_carga_de_area_hacia_la_direccion_elegida(p):
    [carga] = cargas_para_objetivo(p, Objetivo("LOSA", "L1"), "L", "C1", presion=5.0, w=0.0, direccion=(0, 0, -1))

    assert isinstance(carga, CargaArea)
    assert (carga.losa_id, carga.presion, carga.caso_id) == ("L1", 5.0, "L")
    assert carga.direccion == (0.0, 0.0, -1.0)


def test_cara_superior_de_losa_empuja_hacia_abajo(p):
    [carga] = cargas_para_objetivo(p, _cara(p, "LOSA", "L1", tipo="SUPERIOR"), "L", "C1", 5.0, 0.0)
    assert carga.direccion == pytest.approx((0, 0, -1))


def test_cara_inferior_de_losa_empuja_hacia_arriba(p):
    [carga] = cargas_para_objetivo(p, _cara(p, "LOSA", "L1", tipo="INFERIOR"), "L", "C1", 5.0, 0.0)
    assert carga.direccion == pytest.approx((0, 0, 1))


def test_borde_de_losa_no_admite_carga(p):
    with pytest.raises(ErrorAsignacion, match="cara superior o inferior"):
        cargas_para_objetivo(p, _cara(p, "LOSA", "L1", tipo="BORDE"), "L", "C1", 5.0, 0.0)


def test_muro_no_admite_cargas_en_la_fase_1(p):
    with pytest.raises(ErrorAsignacion, match="muros"):
        cargas_para_objetivo(p, Objetivo("MURO", "M1"), "L", "C1", 5.0, 0.0)


def test_presion_cero_se_rechaza(p):
    with pytest.raises(ErrorAsignacion, match="cero"):
        cargas_para_objetivo(p, Objetivo("LOSA", "L1"), "L", "C1", 0.0, 10.0)


# -- Cargas sobre vigas y columnas -----------------------------------------------


def test_viga_completa_recibe_carga_lineal_uniforme(p):
    [carga] = cargas_para_objetivo(p, Objetivo("VIGA", "B1"), "L", "C1", presion=0.0, w=12.0, direccion=(0, 0, -1))

    assert isinstance(carga, CargaLineal)
    assert (carga.elemento_id, carga.w_inicio, carga.w_fin) == ("B1", 12.0, 12.0)
    assert carga.es_uniforme and not carga.es_parcial


def test_presion_sobre_la_cara_superior_de_una_viga_es_presion_por_ancho(p):
    [carga] = cargas_para_objetivo(p, _cara(p, "VIGA", "B1", nombre="superior"), "L", "C1", presion=10.0, w=0.0)

    assert isinstance(carga, CargaLineal)
    assert carga.w_inicio == pytest.approx(10.0 * 0.3)  # kPa × 0,3 m de ancho
    assert carga.direccion == pytest.approx((0, 0, -1))


def test_presion_sobre_la_cara_inferior_de_una_viga_empuja_hacia_arriba(p):
    [carga] = cargas_para_objetivo(p, _cara(p, "VIGA", "B1", nombre="inferior"), "L", "C1", presion=10.0, w=0.0)
    assert carga.direccion == pytest.approx((0, 0, 1))


def test_cara_extrema_de_columna_recibe_fuerza_axial_en_su_nodo(p):
    tope = _cara(p, "COLUMNA", "C1", tipo="EXTREMO_FIN")
    [carga] = cargas_para_objetivo(p, tope, "L", "C1", presion=100.0, w=0.0)

    assert isinstance(carga, CargaPuntual)
    assert carga.nodo_id == "N1"
    assert carga.fz == pytest.approx(-100.0 * 0.5 * 0.3)  # presión × área de la cara, hacia abajo
    assert (carga.fx, carga.fy) == pytest.approx((0.0, 0.0))


def test_las_cargas_creadas_son_validas_para_el_solver(p):
    # Todas deben poder construirse y tener un caso asignado.
    for objetivo in (Objetivo("LOSA", "L1"), Objetivo("VIGA", "B1"), _cara(p, "VIGA", "B1", nombre="superior")):
        for carga in cargas_para_objetivo(p, objetivo, "D", "X", 5.0, 5.0):
            assert carga.caso_id == "D"


# -- Descripciones y vista previa ------------------------------------------------


def test_descripcion_incluye_el_elemento_y_la_cara(p):
    assert descripcion(p, Objetivo("VIGA", "B1")) == "Viga B1"
    assert descripcion(p, _cara(p, "LOSA", "L1", tipo="SUPERIOR")) == "Losa L1 — cara superior"


def test_vista_previa_marca_los_objetivos_que_no_se_pueden_cargar(p):
    objetivos = [Objetivo("LOSA", "L1"), Objetivo("MURO", "M1"), Objetivo("VIGA", "B1")]

    vista = vista_previa_cargas(p, objetivos, presion=5.0, w=10.0)

    assert [valido for valido, _ in vista] == [True, False, True]
    assert "5 kPa" in vista[0][1] and "10 kN/m" in vista[2][1]
    assert "muros" in vista[1][1]
