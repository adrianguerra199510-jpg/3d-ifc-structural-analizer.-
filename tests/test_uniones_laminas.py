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


def _rejilla_s6(origen, eje_u, eje_v, nu, nv, coords):
    """Triángulos S6 de un rectángulo; reutiliza los nodos de ``coords`` que coinciden."""

    def nodo(p):
        p = tuple(np.round(p, 9))
        if p not in coords:
            coords[p] = len(coords)
        return coords[p]

    o, a, b = (np.asarray(v, float) for v in (origen, eje_u, eje_v))
    pt = lambda i, j: o + a * i / nu + b * j / nv  # noqa: E731
    elementos = []
    for i in range(nu):
        for j in range(nv):
            for tri in (((i, j), (i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i, j + 1))):
                p = [pt(*c) for c in tri]
                elementos.append([nodo(p[0]), nodo(p[1]), nodo(p[2]),
                                  nodo((p[0] + p[1]) / 2), nodo((p[1] + p[2]) / 2), nodo((p[2] + p[0]) / 2)])
    return elementos


def test_alma_y_ala_que_solo_comparten_un_nodo_se_acoplan_a_lo_largo():
    # en un perfil inclinado la superficie media dejaba el alma tocando el ala en un solo nodo: parecía una sola pieza y
    # el ala quedaba suelta del alma a lo largo de la viga
    from types import SimpleNamespace

    from ifc_structural.solido.contactos import _acople_interno

    coords = {}
    ala = _rejilla_s6((0, -0.04, 0), (1, 0, 0), (0, 0.1, 0), 8, 3, coords)  # sin nodos en y = 0
    alma = _rejilla_s6((0, 0, 0), (1, 0, 0), (0, 0, 0.2), 8, 2, coords)
    # el alma toca el ala en un solo nodo (el del origen pasa a ser un nodo del ala)
    comun, del_ala = coords[(0.0, 0.0, 0.0)], coords[tuple(np.round((0, -0.04 + 0.1 / 3, 0), 9))]
    alma = [[del_ala if n == comun else n for n in e] for e in alma]
    xyz = np.array(sorted(coords, key=coords.get))
    laminas = np.array(ala + alma)
    assert set(np.unique(ala).tolist()) & set(np.unique(alma).tolist()) == {del_ala}
    malla = SimpleNamespace(laminas=laminas, coordenadas=xyz, espesor_de_lamina=np.full(len(laminas), 0.01),
                            partes=[SimpleNamespace(primer_lamina=0, ultimo_lamina=len(laminas))])
    acople = _acople_interno(malla, 0, 0.125, set(), set(), 1)
    assert acople is not None
    unidos = xyz[np.asarray(acople.nodos_a)]
    assert np.ptp(unidos[:, 0]) > 0.8  # a lo largo de toda la viga, no solo en el nodo común
