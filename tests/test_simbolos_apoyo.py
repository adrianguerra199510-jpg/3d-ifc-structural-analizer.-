"""Símbolos de apoyo según los grados de libertad restringidos (sin OpenGL)."""

import numpy as np
import pytest

pytest.importorskip("pyvista")

from ifc_structural.gui.simbolos_apoyo import (
    COLOR_ARTICULADO, COLOR_EMPOTRADO, COLOR_GIRO, COLOR_RUEDA, nombre_apoyo, simbolo_apoyo,
)

TIPOS = {
    "empotrado": (1, 1, 1, 1, 1, 1),
    "articulado": (1, 1, 1, 0, 0, 0),
    "rodillo Uz": (0, 0, 1, 0, 0, 0),
    "rodillo Ux": (1, 0, 0, 0, 0, 0),
    "rodillo Ux Uz": (1, 0, 1, 0, 0, 0),
    "Rz": (0, 0, 0, 0, 0, 1),
    "Ux Uy Uz Rz": (1, 1, 1, 0, 0, 1),
    "Uz Rx": (0, 0, 1, 1, 0, 0),
}


@pytest.mark.parametrize("nombre,grados", TIPOS.items())
def test_nombre_del_apoyo(nombre, grados):
    assert nombre_apoyo(grados) == nombre


def test_cada_tipo_tiene_su_simbolo():
    colores = {n: set(simbolo_apoyo(g, (0, 0, 0), 1.0)) for n, g in TIPOS.items()}
    assert colores["empotrado"] == {COLOR_EMPOTRADO}
    assert colores["articulado"] == {COLOR_ARTICULADO}
    assert colores["rodillo Uz"] == {COLOR_ARTICULADO, COLOR_RUEDA}  # pirámide con ruedas
    assert colores["Rz"] == {COLOR_GIRO}
    assert colores["Ux Uy Uz Rz"] == {COLOR_ARTICULADO, COLOR_GIRO}
    assert colores["Uz Rx"] == {COLOR_ARTICULADO, COLOR_RUEDA, COLOR_GIRO}


def test_dos_traslaciones_dibujan_dos_rodillos():
    partes = simbolo_apoyo((1, 0, 1, 0, 0, 0), (0, 0, 0), 1.0)
    assert len(partes[COLOR_ARTICULADO]) == 2


def test_el_rodillo_se_orienta_segun_la_traslacion_restringida():
    for eje in range(3):
        g = [0] * 6
        g[eje] = 1
        piramide = simbolo_apoyo(g, (0, 0, 0), 1.0)[COLOR_ARTICULADO][0]
        # la punta está en el nodo y el cuerpo del símbolo en el lado negativo del eje
        centro = np.array(piramide.center)
        assert centro[eje] < -0.3
        assert np.allclose(np.delete(centro, eje), 0.0, atol=1e-6)


def test_el_simbolo_se_coloca_en_el_nodo():
    cubo = simbolo_apoyo((1,) * 6, (5.0, 2.0, 3.0), 1.0)[COLOR_EMPOTRADO][0]
    assert cubo.bounds[5] == pytest.approx(3.0)  # la cara superior toca el nodo
