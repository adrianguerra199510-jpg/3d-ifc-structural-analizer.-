"""Pruebas de la unión de nodos de barras que casi coinciden."""

import pytest

from ifc_structural.core.conectividad import conectar_nodos
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto


def _base() -> Proyecto:
    p = Proyecto("Conectividad")
    p.agregar_material(Material(nombre="H", modulo_elasticidad=2.5e7))
    p.agregar_perfil(PerfilSeccion(nombre="V", area=0.15, inercia_y=1e-4, inercia_z=1e-3, inercia_torsional=1e-4,
                                   altura=0.5, ancho=0.3))
    return p


def _nodo(p, x, y, z):
    return p.agregar_nodo(x, y, z).id


def test_la_viga_que_termina_en_la_cara_de_la_columna_se_une_al_eje_de_la_columna():
    p = _base()
    base, tope = _nodo(p, 0, 0, 0), _nodo(p, 0, 0, 3)
    p.agregar_columna(base, tope, "H", "V", id_="C1")
    # La viga termina 0.15 m a un lado y 0.1 m por debajo del tope de la columna.
    ext_viga, otro = _nodo(p, 0.15, 0, 2.9), _nodo(p, 4, 0, 2.9)
    p.agregar_viga(ext_viga, otro, "H", "V", id_="B1")

    unidos = conectar_nodos(p)

    assert unidos == 1
    assert p.vigas["B1"].nodo_inicio == tope  # se conserva el nodo de la columna
    assert ext_viga not in p.nodos  # el nodo huérfano se elimina


def test_no_une_nodos_lejanos():
    p = _base()
    a, b, c = _nodo(p, 0, 0, 0), _nodo(p, 4, 0, 0), _nodo(p, 4.8, 0, 0)
    p.agregar_viga(a, b, "H", "V", id_="B1")
    p.agregar_viga(c, _nodo(p, 9, 0, 0), "H", "V", id_="B2")

    assert conectar_nodos(p, tolerancia=0.4) == 0
    assert p.vigas["B2"].nodo_inicio == c


def test_sin_columnas_se_conserva_el_nodo_mas_cercano_al_centro_del_grupo():
    p = _base()
    a = _nodo(p, 4.0, 0, 0)
    b = _nodo(p, 4.1, 0, 0)
    c = _nodo(p, 4.2, 0, 0)
    p.agregar_viga(_nodo(p, 0, 0, 0), a, "H", "V", id_="B1")
    p.agregar_viga(b, _nodo(p, 4.1, 5, 0), "H", "V", id_="B2")
    p.agregar_viga(c, _nodo(p, 4.2, -5, 0), "H", "V", id_="B3")

    assert conectar_nodos(p) == 2
    assert p.vigas["B2"].nodo_inicio == b and p.vigas["B1"].nodo_fin == b and p.vigas["B3"].nodo_inicio == b


def test_una_barra_corta_no_se_colapsa_sobre_si_misma():
    p = _base()
    a, b = _nodo(p, 0, 0, 0), _nodo(p, 0.2, 0, 0)  # barra de 0.2 m < tolerancia
    p.agregar_viga(a, b, "H", "V", id_="CORTA")
    p.agregar_viga(b, _nodo(p, 5, 0, 0), "H", "V", id_="B2")

    conectar_nodos(p)

    corta = p.vigas["CORTA"]
    assert corta.nodo_inicio != corta.nodo_fin
    assert p.longitud_barra("CORTA") > 0.1


def test_actualiza_los_contornos_de_losas_que_usaban_un_nodo_unido():
    p = _base()
    tope = _nodo(p, 0, 0, 3)
    p.agregar_columna(_nodo(p, 0, 0, 0), tope, "H", "V", id_="C1")
    cercano = _nodo(p, 0.1, 0, 3)
    p.agregar_viga(cercano, _nodo(p, 4, 0, 3), "H", "V", id_="B1")
    losa = p.agregar_losa([cercano, _nodo(p, 4, 0, 3), _nodo(p, 4, 4, 3), _nodo(p, 0, 4, 3)], espesor=0.2, material="H")

    conectar_nodos(p)

    assert losa.nodos_contorno[0] == tope
    assert all(n in p.nodos for n in losa.nodos_contorno)
