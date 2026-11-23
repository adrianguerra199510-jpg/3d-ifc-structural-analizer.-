"""Uniones de láminas: empotramiento en un canto, uniones a tope y uniones en T (perfiles I).

CalculiX no transmite por ``*EQUATION`` los giros de los nodos de cáscaras: dos láminas que se tocan en una línea quedaban
articuladas; un perfil I (sus alas y su alma son trozos de la superficie media que no comparten nodos) se deshacía; y un
apoyo «empotrado» (Ux, Uy, Uz) en el canto de una lámina era una articulación.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.casos_carga import CasoCarga, TipoCaso
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido

from .test_solidos_libres import E, caja, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def _flecha_punta(proyecto, apoyado, largo, tamano=0.05, eje=0):
    s = proyecto.solidos[apoyado]
    # la cara extrema en el origen (por su posición: la tapa de una extrusión puede venir con la normal girada)
    extremos = [k for k, c in enumerate(s.caras) if abs(c.normal[eje]) > 0.99 and abs(c.centro[eje]) < 1e-6]
    apoyo = ApoyoSuperficie(id="AP", elemento_tipo="SOLIDO", elemento_id=apoyado, cara=extremos[0])
    r = ejecutar_analisis_solido(proyecto, {"D": CasoCarga("D", "D", TipoCaso.D)}, {}, [apoyo], {}, {}, {},
                                 Path(tempfile.mkdtemp()), tamano=tamano)
    assert not [a for a in r.advertencias if "unidas" in a or "mecanismo" in a]
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["D"]
    return -float(u[np.isclose(x[:, eje], largo, atol=1e-6), 2].mean())


@requiere_ccx
def test_chapa_en_voladizo_y_dos_chapas_a_tope():
    w, inercia = 78.5 * 0.2 * 0.01, 0.2 * 0.01 ** 3 / 12
    teoria = w * 2.0 ** 4 / (8 * E * inercia)
    una = _flecha_punta(proyecto_con(caja("A", 0, 0, 0, 2.0, 0.2, 0.01)), "A", 2.0)
    assert una == pytest.approx(teoria, rel=0.05)  # el canto empotrado no es una articulación
    dos = _flecha_punta(proyecto_con(caja("A", 0, 0, 0, 1.0, 0.2, 0.01), caja("B", 1.0, 0, 0, 1.0, 0.2, 0.01)), "A", 2.0)
    assert dos == pytest.approx(una, rel=0.05)  # la unión a tope transmite el momento
