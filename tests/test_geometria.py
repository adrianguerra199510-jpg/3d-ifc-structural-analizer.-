"""Pruebas de la geometría compartida: ejes de sección y sólidos con caras."""

import math

import numpy as np
import pytest

from ifc_structural.core.geometria import (
    dimensiones_seccion,
    ejes_seccion,
    solido_barra,
    solido_plano,
)
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto

S = 1.0 / math.sqrt(2.0)


def _proyecto() -> Proyecto:
    p = Proyecto("Geometría")
    p.agregar_material(Material(nombre="H", modulo_elasticidad=2.5e7))
    p.agregar_perfil(
        PerfilSeccion(nombre="V", area=0.15, inercia_y=3.125e-4, inercia_z=1.125e-3, inercia_torsional=1e-4,
                      altura=0.5, ancho=0.3)
    )
    return p


def _barra(p: Proyecto, a, b, **kwargs):
    na = p.nodo_o_existente(*a)
    nb = p.nodo_o_existente(*b)
    return p.agregar_viga(na.id, nb.id, "H", "V", **kwargs)


def test_viga_horizontal_tiene_la_altura_en_vertical():
    p = _proyecto()
    ejes = ejes_seccion(p, _barra(p, (0, 0, 0), (4, 0, 0)))

    assert ejes.t == pytest.approx([1, 0, 0])
    assert abs(ejes.e1[2]) == pytest.approx(1.0)  # e1 vertical
    assert ejes.dim1 == pytest.approx(0.5)  # altura a lo largo de e1
    assert ejes.dim2 == pytest.approx(0.3)
    # Terna ortonormal.
    assert np.dot(ejes.t, ejes.e1) == pytest.approx(0.0)
    assert np.dot(ejes.t, ejes.e2) == pytest.approx(0.0)
    assert np.dot(ejes.e1, ejes.e2) == pytest.approx(0.0)


def test_columna_vertical_usa_un_eje_de_respaldo_perpendicular():
    p = _proyecto()
    ejes = ejes_seccion(p, _barra(p, (0, 0, 0), (0, 0, 3)))

    # n1 por defecto (0,0,1) es paralelo al eje: debe usarse (1,0,0).
    assert abs(ejes.e1[0]) == pytest.approx(1.0)
    assert np.dot(ejes.t, ejes.e1) == pytest.approx(0.0)


def test_el_vector_n1_define_hacia_donde_apunta_la_altura():
    p = _proyecto()
    ejes = ejes_seccion(p, _barra(p, (0, 0, 0), (0, 0, 3), vector_n1=(0, 1, 0)))

    assert abs(ejes.e1[1]) == pytest.approx(1.0)
    assert ejes.dim1 == pytest.approx(0.5)


def test_diagonal_de_45_grados_exactos_intercambia_ejes_sin_cambiar_la_seccion_fisica():
    p = _proyecto()
    # Diagonal de 45° en el plano XZ con la altura en el plano: configuración
    # que hace fallar la expansión de ccx 2.21 si no se intercambian los ejes.
    barra = _barra(p, (0, 0, 0), (3, 0, 3), vector_n1=(-S, 0, S))
    ejes = ejes_seccion(p, barra)

    assert abs(ejes.e1[1]) == pytest.approx(1.0)  # ahora e1 es el eje Y global
    assert ejes.dim1 == pytest.approx(0.3)  # el ancho pasó a ser dim1
    assert ejes.dim2 == pytest.approx(0.5)
    # La dimensión de 0.5 m sigue medida en el plano XZ (misma sección física).
    assert abs(ejes.e2[1]) == pytest.approx(0.0, abs=1e-9)


def test_dimensiones_de_perfil_sin_dimensiones_conserva_area_e_inercia():
    perfil = PerfilSeccion(nombre="IPE", area=0.00285, inercia_y=1.42e-6, inercia_z=1.94e-5, inercia_torsional=6e-8)
    altura, ancho = dimensiones_seccion(perfil)

    assert altura * ancho == pytest.approx(perfil.area)
    assert ancho * altura**3 / 12.0 == pytest.approx(perfil.inercia_z)


def test_solido_barra_tiene_seis_caras_con_areas_y_normales_correctas():
    p = _proyecto()
    barra = _barra(p, (0, 0, 0), (4, 0, 0))
    solido = solido_barra(p, barra)

    assert len(solido.caras) == 6
    assert solido.puntos.shape == (8, 3)

    inicio, fin = solido.caras[0], solido.caras[1]
    assert inicio.tipo == "EXTREMO_INICIO" and fin.tipo == "EXTREMO_FIN"
    assert inicio.normal == pytest.approx([-1, 0, 0])
    assert fin.normal == pytest.approx([1, 0, 0])
    assert inicio.nodo_id == barra.nodo_inicio and fin.nodo_id == barra.nodo_fin
    assert inicio.area == pytest.approx(0.5 * 0.3)

    superior = next(c for c in solido.caras if c.nombre == "superior")
    inferior = next(c for c in solido.caras if c.nombre == "inferior")
    assert superior.normal == pytest.approx([0, 0, 1])
    assert inferior.normal == pytest.approx([0, 0, -1])
    # La cara superior mide 4 m de largo por el ancho de 0.3 m.
    assert superior.area == pytest.approx(4.0 * 0.3)
    assert superior.ancho == pytest.approx(0.3)

    centro = (solido.puntos.min(axis=0) + solido.puntos.max(axis=0)) / 2
    for cara in solido.caras:
        assert np.dot(cara.normal, cara.centro - centro) > 0  # todas apuntan hacia fuera


def test_solido_barra_las_dimensiones_geometricas_son_las_de_la_seccion():
    p = _proyecto()
    solido = solido_barra(p, _barra(p, (0, 0, 0), (4, 0, 0)))

    extension = solido.puntos.max(axis=0) - solido.puntos.min(axis=0)
    assert extension == pytest.approx([4.0, 0.3, 0.5])


def test_solido_losa_horizontal():
    p = _proyecto()
    contorno = [p.agregar_nodo(x, y, 3.0).id for x, y in ((0, 0), (4, 0), (4, 6), (0, 6))]
    losa = p.agregar_losa(contorno, espesor=0.15, material="H")
    solido = solido_plano(p, losa)

    assert len(solido.caras) == 2 + 4  # inferior, superior y 4 bordes
    superior = solido.caras[1]
    inferior = solido.caras[0]
    assert superior.tipo == "SUPERIOR" and inferior.tipo == "INFERIOR"
    assert superior.normal == pytest.approx([0, 0, 1])
    assert inferior.normal == pytest.approx([0, 0, -1])
    assert superior.area == pytest.approx(24.0)
    assert solido.puntos[:, 2].min() == pytest.approx(3.0 - 0.075)
    assert solido.puntos[:, 2].max() == pytest.approx(3.0 + 0.075)

    bordes = [c for c in solido.caras if c.tipo == "BORDE"]
    assert sorted(round(c.area, 6) for c in bordes) == pytest.approx(sorted([4 * 0.15, 6 * 0.15, 4 * 0.15, 6 * 0.15]))
    assert [c.indice_borde for c in bordes] == [0, 1, 2, 3]


def test_solido_losa_se_orienta_hacia_arriba_aunque_el_contorno_este_en_sentido_horario():
    p = _proyecto()
    contorno = [p.agregar_nodo(x, y, 3.0).id for x, y in ((0, 0), (0, 6), (4, 6), (4, 0))]  # horario
    solido = solido_plano(p, p.agregar_losa(contorno, espesor=0.2, material="H"))

    assert solido.caras[1].tipo == "SUPERIOR"
    assert solido.caras[1].normal == pytest.approx([0, 0, 1])


def test_solido_muro_vertical_usa_caras_a_y_b():
    p = _proyecto()
    contorno = [p.agregar_nodo(x, 0.0, z).id for x, z in ((0, 0), (5, 0), (5, 3), (0, 3))]
    solido = solido_plano(p, p.agregar_muro(contorno, espesor=0.2, material="H"))

    tipos = {c.tipo for c in solido.caras}
    assert {"CARA_A", "CARA_B", "BORDE"} <= tipos
    cara_a = next(c for c in solido.caras if c.tipo == "CARA_A")
    assert abs(cara_a.normal[1]) == pytest.approx(1.0)
