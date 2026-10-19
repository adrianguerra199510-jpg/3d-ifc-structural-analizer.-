"""Pruebas de la detección automática de vigas de soporte de las losas."""

import pytest

from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.core.vigas_soporte import asignar_vigas_soporte


def _proyecto_con_losa_y_cuatro_vigas(z_viga: float = 2.75, desplazamiento_en_planta: float = 0.0) -> Proyecto:
    """Losa de 4 x 6 m en z=3 (espesor 0.2, plano medio en z=3) apoyada en
    cuatro vigas de 0.3 x 0.5 m cuyo eje está en ``z_viga``."""

    p = Proyecto("Soporte")
    p.agregar_material(Material(nombre="H", modulo_elasticidad=2.5e7))
    p.agregar_perfil(PerfilSeccion(nombre="V", area=0.15, inercia_y=1e-4, inercia_z=1e-3,
                                   inercia_torsional=1e-4, altura=0.5, ancho=0.3))
    esquinas = [(0, 0), (4, 0), (4, 6), (0, 6)]
    contorno = [p.agregar_nodo(x, y, 3.0).id for x, y in esquinas]
    p.agregar_losa(contorno, espesor=0.2, material="H", id_="L1")
    for i in range(4):
        (x0, y0), (x1, y1) = esquinas[i], esquinas[(i + 1) % 4]
        n0 = p.agregar_nodo(x0 + desplazamiento_en_planta, y0, z_viga)
        n1 = p.agregar_nodo(x1 + desplazamiento_en_planta, y1, z_viga)
        p.agregar_viga(n0.id, n1.id, "H", "V", id_=f"B{i + 1}")
    return p


def test_detecta_la_viga_de_cada_lado():
    p = _proyecto_con_losa_y_cuatro_vigas()

    asignadas = asignar_vigas_soporte(p)

    assert asignadas == 4
    assert p.losas["L1"].vigas_soporte == ["B1", "B2", "B3", "B4"]


def test_ignora_vigas_demasiado_lejos_en_vertical():
    p = _proyecto_con_losa_y_cuatro_vigas(z_viga=1.0)  # 2 m por debajo de la losa

    assert asignar_vigas_soporte(p) == 0
    assert p.losas["L1"].vigas_soporte == [None, None, None, None]


def test_ignora_vigas_perpendiculares_o_no_paralelas():
    p = _proyecto_con_losa_y_cuatro_vigas()
    # Se elimina B1 y se pone en su lugar una viga perpendicular al lado 1.
    del p.vigas["B1"]
    n0 = p.agregar_nodo(2, -0.5, 2.75)
    n1 = p.agregar_nodo(2, 0.5, 2.75)
    p.agregar_viga(n0.id, n1.id, "H", "V", id_="PERP")

    asignar_vigas_soporte(p)

    assert p.losas["L1"].vigas_soporte[0] is None
    assert p.losas["L1"].vigas_soporte[1:] == ["B2", "B3", "B4"]


def test_tolera_un_desfase_pequeno_en_planta_pero_no_uno_grande():
    p = _proyecto_con_losa_y_cuatro_vigas()
    # Se desplaza B1 0.2 m en Y (media viga 0.15 + tolerancia 0.15 = 0.30 permitido).
    p.nodos[p.vigas["B1"].nodo_inicio].y += 0.2
    p.nodos[p.vigas["B1"].nodo_fin].y += 0.2
    asignar_vigas_soporte(p)
    assert p.losas["L1"].vigas_soporte[0] == "B1"

    # Con 1 m de desfase ya no soporta ese lado.
    p.nodos[p.vigas["B1"].nodo_inicio].y += 0.8
    p.nodos[p.vigas["B1"].nodo_fin].y += 0.8
    asignar_vigas_soporte(p)
    assert p.losas["L1"].vigas_soporte[0] is None


def test_exige_que_la_viga_cubra_al_menos_la_mitad_del_lado():
    p = _proyecto_con_losa_y_cuatro_vigas()
    # B1 va de (0,0) a (4,0): se acorta a solo 1 m (25% del lado).
    p.nodos[p.vigas["B1"].nodo_fin].x = 1.0
    asignar_vigas_soporte(p)
    assert p.losas["L1"].vigas_soporte[0] is None


def test_elige_la_viga_que_mas_cubre_si_hay_varias():
    p = _proyecto_con_losa_y_cuatro_vigas()
    n0 = p.agregar_nodo(0.5, 0.0, 2.75)
    n1 = p.agregar_nodo(2.5, 0.0, 2.75)  # cubre 2 de 4 m: 50%
    p.agregar_viga(n0.id, n1.id, "H", "V", id_="CORTA")

    asignar_vigas_soporte(p)

    assert p.losas["L1"].vigas_soporte[0] == "B1"  # cubre el 100%
