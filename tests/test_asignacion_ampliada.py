"""Formas de carga, direcciones, apoyos de área, conexiones y materiales."""

import pytest

from ifc_structural.core.asignacion import (
    ErrorAsignacion,
    FormaCarga,
    Objetivo,
    ParametrosCarga,
    apoyo_para_objetivo,
    asignar_material,
    barras_de_apoyo_area,
    cargas_desde_parametros,
    conexion_para_objetivos,
    restriccion_solo_normal,
    segmentos_de_forma,
    vista_previa_cargas,
)
from ifc_structural.core.cargas import Apoyo, ApoyoArea, CargaArea, CargaLineal, CargaPuntual, PerfilPresion
from ifc_structural.core.geometria import solido_elemento
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.core.persistencia import cargar_proyecto, guardar_proyecto

from .test_asignacion import _cara, p  # noqa: F401  (fixture del proyecto de ejemplo)


def _integral(cargas):
    """Carga total (kN) de una lista de CargaLineal sobre B1 (4 m)."""
    return sum(0.5 * (c.w_inicio + c.w_fin) * (c.s_fin - c.s_inicio) * 4.0 for c in cargas)


@pytest.mark.parametrize(
    "forma, w, esperado",
    [
        (FormaCarga("UNIFORME"), 10.0, 40.0),
        (FormaCarga("TRIANGULAR_CRECIENTE"), 10.0, 20.0),
        (FormaCarga("TRIANGULAR_DECRECIENTE"), 10.0, 20.0),
        (FormaCarga("TRIANGULAR_SIMETRICA"), 10.0, 20.0),
        (FormaCarga("TRAPEZOIDAL", w_fin=30.0), 10.0, 80.0),
        (FormaCarga("TRAPEZOIDAL_SIMETRICA", rampa=1.0), 10.0, 30.0),
        (FormaCarga("UNIFORME", desde=1.0, hasta=3.0), 10.0, 20.0),
    ],
)
def test_resultante_de_cada_forma(p, forma, w, esperado):
    cargas = cargas_desde_parametros(p, Objetivo("VIGA", "B1"), "D", "C", ParametrosCarga(tipo="LINEAL", w=w, forma=forma))
    assert _integral(cargas) == pytest.approx(esperado)


def test_trapezoidal_simetrica_tiene_rampas_y_meseta():
    trozos = segmentos_de_forma(FormaCarga("TRAPEZOIDAL_SIMETRICA", rampa=1.0), 10.0, 4.0)
    assert trozos == [(0.0, 0.25, 0.0, 10.0), (0.25, 0.75, 10.0, 10.0), (0.75, 1.0, 10.0, 0.0)]


def test_tramo_cargado_fuera_de_la_barra_se_rechaza(p):
    with pytest.raises(ErrorAsignacion, match="tramo"):
        cargas_desde_parametros(p, Objetivo("VIGA", "B1"), "D", "C",
                                ParametrosCarga(tipo="LINEAL", w=5, forma=FormaCarga(desde=1.0, hasta=9.0)))
    with pytest.raises(ErrorAsignacion, match="rampa"):
        cargas_desde_parametros(p, Objetivo("VIGA", "B1"), "D", "C",
                                ParametrosCarga(tipo="LINEAL", w=5, forma=FormaCarga("TRAPEZOIDAL_SIMETRICA", rampa=3.0)))


def test_direccion_elegida_en_carga_lineal_y_en_cara_de_losa(p):
    [c] = cargas_desde_parametros(p, Objetivo("VIGA", "B1"), "D", "C", ParametrosCarga(tipo="LINEAL", w=5, direccion=(1, 0, 0)))
    assert c.direccion == (1.0, 0.0, 0.0)
    cara = _cara(p, "LOSA", "L1", tipo="SUPERIOR")
    [a] = cargas_desde_parametros(p, cara, "D", "C", ParametrosCarga(tipo="PRESION", presion=5, direccion=(0, 1, 0)))
    assert a.direccion == (0.0, 1.0, 0.0)  # la dirección elegida manda sobre la normal de la cara
    [b] = cargas_desde_parametros(p, cara, "D", "C", ParametrosCarga(tipo="PRESION", presion=5))
    assert b.direccion[2] < 0  # por defecto: hacia dentro de la cara superior


def test_presion_variable_sobre_losa_guarda_el_perfil(p):
    perfil = PerfilPresion(eje=1, coord_a=0.0, valor_a=0.0, coord_b=6.0, valor_b=10.0)
    [a] = cargas_desde_parametros(p, Objetivo("LOSA", "L1"), "D", "C", ParametrosCarga(tipo="PRESION", perfil_presion=perfil))
    assert isinstance(a, CargaArea) and a.perfil == perfil
    assert a.presion_en((0, 3, 3)) == pytest.approx(5.0)
    assert "0 → 10" in a.nombre


def test_carga_puntual_y_momento_sobre_barra(p):
    [f] = cargas_desde_parametros(p, Objetivo("VIGA", "B1"), "D", "C", ParametrosCarga(tipo="PUNTUAL", fuerza=12, posicion=1.0, direccion=(0, 0, -1)))
    assert isinstance(f, CargaPuntual) and f.elemento_id == "B1" and f.posicion == pytest.approx(0.25) and f.fz == pytest.approx(-12)
    [extremo] = cargas_desde_parametros(p, Objetivo("VIGA", "B1"), "D", "C", ParametrosCarga(tipo="PUNTUAL", fuerza=5, posicion=4.0))
    assert extremo.nodo_id == "N2"  # en el extremo se aplica sobre el nodo
    [m] = cargas_desde_parametros(p, Objetivo("VIGA", "B1"), "D", "C", ParametrosCarga(tipo="MOMENTO", momento=8, posicion=2.0, direccion=(0, 1, 0)))
    assert m.my == pytest.approx(8) and m.fz == 0
    with pytest.raises(ErrorAsignacion, match="posición"):
        cargas_desde_parametros(p, Objetivo("VIGA", "B1"), "D", "C", ParametrosCarga(tipo="PUNTUAL", fuerza=1, posicion=9.0))


def test_cara_lateral_con_presion_triangular(p):
    lateral = _cara(p, "VIGA", "B1", tipo="LATERAL")
    cargas = cargas_desde_parametros(p, lateral, "D", "C", ParametrosCarga(tipo="PRESION", presion=10, forma=FormaCarga("TRIANGULAR_CRECIENTE")))
    ancho = solido_elemento(p, "VIGA", "B1").caras[lateral.cara].ancho
    assert _integral(cargas) == pytest.approx(0.5 * 10 * ancho * 4.0)


def test_vista_previa_acepta_parametros(p):
    vista = vista_previa_cargas(p, [Objetivo("VIGA", "B1"), Objetivo("MURO", "M1")], ParametrosCarga(tipo="LINEAL", w=5))
    assert vista[0][0] and not vista[1][0]


# -- Apoyos de área ----------------------------------------------------------


def test_apoyo_puntual_en_extremo_y_de_area_en_el_resto(p):
    base = _cara(p, "COLUMNA", "C1", tipo="EXTREMO_INICIO")
    assert isinstance(apoyo_para_objetivo(p, base, "A1", (True,) * 6), Apoyo)
    area = apoyo_para_objetivo(p, Objetivo("VIGA", "B1"), "A2", (False, False, True, False, False, False))
    assert isinstance(area, ApoyoArea) and area.grados_restringidos() == [3] and area.s_fin == 1.0
    tramo = apoyo_para_objetivo(p, Objetivo("VIGA", "B1"), "A3", (True,) * 3 + (False,) * 3, desde=1.0, hasta=3.0)
    assert (tramo.s_inicio, tramo.s_fin) == (0.25, 0.75)


def test_apoyo_de_area_en_losa_va_en_sus_vigas(p):
    p.losas["L1"].vigas_soporte = ["B1", None, None, None]
    apoyo = apoyo_para_objetivo(p, _cara(p, "LOSA", "L1", tipo="INFERIOR"), "A", (False, False, True, False, False, False))
    assert apoyo.elemento_tipo == "LOSA" and barras_de_apoyo_area(p, apoyo) == ["B1"]
    p.losas["L1"].vigas_soporte = [None] * 4
    with pytest.raises(ErrorAsignacion, match="vigas de soporte"):
        apoyo_para_objetivo(p, Objetivo("LOSA", "L1"), "A", (True,) * 6)


@pytest.mark.parametrize("objetivo, restricciones", [(Objetivo("MURO", "M1"), (True,) * 6), (Objetivo("VIGA", "B1"), (False,) * 6)])
def test_apoyos_invalidos(p, objetivo, restricciones):
    with pytest.raises(ErrorAsignacion):
        apoyo_para_objetivo(p, objetivo, "A", restricciones)


def test_restriccion_solo_normal_a_la_cara(p):
    superior = solido_elemento(p, "LOSA", "L1").caras[1]
    assert restriccion_solo_normal(superior) == (False, False, True, False, False, False)
    assert restriccion_solo_normal(None) is None


# -- Conexiones ---------------------------------------------------------------


def test_conexion_en_nudo_compartido_separa_la_viga(p):
    c = conexion_para_objetivos(p, Objetivo("COLUMNA", "C1"), Objetivo("VIGA", "B1"), "X", (True, True, True, False, False, False))
    assert (c.tipo, c.barra_id, c.nodo_id) == ("EXTREMO", "B1", "N1")
    c2 = conexion_para_objetivos(p, Objetivo("VIGA", "B1"), Objetivo("COLUMNA", "C1"), "X", (True, True, True, False, False, False))
    assert c2.barra_id == "B1"  # la viga es la que se articula, sin importar el orden de selección


def test_conexion_rigida_en_nudo_compartido_se_explica(p):
    with pytest.raises(ErrorAsignacion, match="ya están unidos"):
        conexion_para_objetivos(p, Objetivo("COLUMNA", "C1"), Objetivo("VIGA", "B1"), "X", (True,) * 6)


def test_giros_parciales_no_se_admiten(p):
    with pytest.raises(ErrorAsignacion, match="juntos"):
        conexion_para_objetivos(p, Objetivo("COLUMNA", "C1"), Objetivo("VIGA", "B1"), "X", (True, True, True, True, False, False))


def test_enlace_entre_barras_sin_nudo_comun(p):
    p.agregar_nodo(0.5, 0, 3, "NA"); p.agregar_nodo(4.5, 0, 3, "NB")
    p.agregar_viga("NA", "NB", "H", "V", id_="B9")
    c = conexion_para_objetivos(p, Objetivo("VIGA", "B1"), Objetivo("VIGA", "B9"), "E", (True,) * 6)
    assert c.tipo == "ENLACE" and c.es_enlace_rigido and {c.nodo_id, c.nodo_destino} == {"N1", "NA"}
    p.agregar_nodo(20, 0, 3, "NF"); p.agregar_nodo(24, 0, 3, "NG")
    p.agregar_viga("NF", "NG", "H", "V", id_="B10")
    with pytest.raises(ErrorAsignacion, match="no comparten"):
        conexion_para_objetivos(p, Objetivo("VIGA", "B1"), Objetivo("VIGA", "B10"), "E", (True,) * 6)


def test_conexion_con_losa_se_rechaza(p):
    with pytest.raises(ErrorAsignacion, match="vigas y columnas"):
        conexion_para_objetivos(p, Objetivo("LOSA", "L1"), Objetivo("VIGA", "B1"), "X", (True,) * 3 + (False,) * 3)


# -- Materiales ---------------------------------------------------------------


def test_asignar_material_a_varios_elementos(p):
    p.agregar_material(Material(nombre="ACERO", modulo_elasticidad=2.0e8, modulo_poisson=0.3, densidad=78.5))
    cambiados = asignar_material(p, [Objetivo("VIGA", "B1"), Objetivo("COLUMNA", "C1", 0), Objetivo("LOSA", "L1")], "ACERO")
    assert len(cambiados) == 3
    assert p.vigas["B1"].material == p.columnas["C1"].material == p.losas["L1"].material == "ACERO"
    with pytest.raises(ErrorAsignacion, match="no existe"):
        asignar_material(p, [Objetivo("VIGA", "B1")], "INEXISTENTE")


# -- Persistencia ---------------------------------------------------------------


def test_persistencia_de_apoyos_de_area_conexiones_y_perfil(p, tmp_path):
    from ifc_structural.core.cargas import Conexion

    area = ApoyoArea(id="A", elemento_tipo="VIGA", elemento_id="B1", uz=True, s_inicio=0.25, s_fin=0.75)
    conexion = Conexion(id="X", nodo_id="N1", barra_id="B1")
    carga = CargaArea("Q", "D", 5.0, losa_id="L1", perfil=PerfilPresion(1, 0, 0, 6, 10))
    ruta = tmp_path / "p.json"
    guardar_proyecto(ruta, p, [Apoyo(id="A0", nodo_id="N0"), area], {}, {}, {"D": [carga]}, {}, {}, [conexion])
    proyecto, apoyos, _, _, areas, _, _, conexiones = cargar_proyecto(ruta)
    assert isinstance(apoyos[0], Apoyo) and apoyos[1] == area
    assert conexiones == [conexion]
    assert areas["D"][0].perfil == PerfilPresion(1, 0, 0, 6, 10)
