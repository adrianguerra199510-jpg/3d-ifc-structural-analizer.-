"""Hexaedros cuadráticos C3D20R barridos en las piezas prismáticas."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.cargas import ApoyoSuperficie, CargaSuperficie
from ifc_structural.core.casos_carga import CasoCarga, TipoCaso
from ifc_structural.core.solidos import SolidoGeneral
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido
from ifc_structural.solido.barridos import (CARAS_HEX20, MEDIOS_CARA_HEX20, PESOS_CARA_HEX20, Prisma, caras_de_borde, extruir,
                                            prisma_de_solido, volumen)
from ifc_structural.solido.tetraedros import gmsh_disponible

from .test_solidos_libres import _carga, caja, cara_hacia, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")
requiere_gmsh = pytest.mark.skipif(not gmsh_disponible(), reason="Gmsh no está disponible")


def tubo(radio=0.1, interior=0.06, alto=1.0, lados=48, id_="T1") -> SolidoGeneral:
    """Tubo redondo facetado a lo largo de z (dos anillos de vértices en cada tapa)."""

    th = np.linspace(0, 2 * np.pi, lados, endpoint=False)
    puntos = np.array([[r * np.cos(x), r * np.sin(x), z] for z in (0.0, alto) for r in (radio, interior) for x in th])

    def k(nivel, anillo, i):
        return nivel * 2 * lados + anillo * lados + i % lados

    t = []
    for i in range(lados):
        j = i + 1
        t += [[k(0, 0, i), k(0, 0, j), k(1, 0, j)], [k(0, 0, i), k(1, 0, j), k(1, 0, i)]]
        t += [[k(0, 1, i), k(1, 1, j), k(0, 1, j)], [k(0, 1, i), k(1, 1, i), k(1, 1, j)]]
        t += [[k(0, 0, i), k(0, 1, j), k(0, 0, j)], [k(0, 0, i), k(0, 1, i), k(0, 1, j)]]
        t += [[k(1, 0, i), k(1, 0, j), k(1, 1, j)], [k(1, 0, i), k(1, 1, j), k(1, 1, i)]]
    return SolidoGeneral(id=id_, nombre=id_, tipo_ifc="IfcMember", material="ACERO", vertices=puntos, triangulos=np.array(t))


def test_detecta_prismas_y_su_eje():
    c = prisma_de_solido(caja("S1", 0, 0, 0, 1.0, 0.1, 0.2))
    assert c is not None and abs(abs(c.eje[0]) - 1) < 1e-9 and abs(c.largo - 1.0) < 1e-9  # el eje más largo
    assert abs(c.area_tapa - 0.02) < 1e-9
    t = prisma_de_solido(tubo())
    assert t is not None and abs(abs(t.eje[2]) - 1) < 1e-9 and abs(t.largo - 1.0) < 1e-9
    import pyvista as pv

    from ifc_structural.core.solidos import soldar_vertices

    esfera = pv.Sphere(radius=0.1).triangulate()
    v, tri = soldar_vertices(np.asarray(esfera.points), esfera.faces.reshape(-1, 4)[:, 1:])
    assert prisma_de_solido(SolidoGeneral(id="E", nombre="E", tipo_ifc="IfcPlate", material="ACERO", vertices=v,
                                          triangulos=tri)) is None  # una esfera no es un prisma


def test_extruir_da_hexaedros_de_20_nodos_validos():
    prisma = Prisma(eje=np.array([0.0, 0.0, 1.0]), z0=0.0, largo=0.3, vertices_tapa=np.zeros((0, 3)),
                    triangulos_tapa=np.zeros((0, 3), dtype=np.int64), area_tapa=0.02)
    seccion = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0], [0.2, 0.1, 0]], dtype=float)
    cuadros = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    x, h = extruir(prisma, seccion, cuadros, 0.1)
    assert h.shape == (2 * 3, 20) and abs(volumen(x, h) - 0.2 * 0.1 * 0.3) < 1e-12
    p = x[h]
    # vértices: abajo z0, arriba z0 + capa; nodos medios en el medio de su arista (orden de CalculiX)
    aristas = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for m, (a, b) in enumerate(aristas, start=8):
        assert np.allclose(p[:, m], 0.5 * (p[:, a] + p[:, b]))
    # volumen positivo: la cara inferior antihoraria vista desde arriba
    n = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
    assert np.all(np.einsum("ij,ij->i", n, p[:, 4] - p[:, 0]) > 0)
    # caras de borde: 2·(2 tapas) + laterales; los nodos medios de cada cara son los de sus aristas
    _, cara, nodos, _, normal, area = caras_de_borde(x, h)
    assert abs(area.sum() - 2 * (0.2 * 0.1 + 0.2 * 0.3 + 0.1 * 0.3)) < 1e-9 and np.all(np.isin(cara, [1, 2, 3, 4, 5, 6]))
    for k in range(4):
        assert np.allclose(x[nodos[:, 4 + k]], 0.5 * (x[nodos[:, k]] + x[nodos[:, (k + 1) % 4]]))
    assert abs(PESOS_CARA_HEX20.sum() - 1.0) < 1e-12 and len(CARAS_HEX20) == len(MEDIOS_CARA_HEX20) == 6


@requiere_ccx
@requiere_gmsh
def test_voladizo_con_hexaedros_barridos_coincide_con_la_teoria(tmp_path: Path):
    s = caja("S1", 0, 0, 0, 1.0, 0.1, 0.1)
    p = proyecto_con(s)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="S1", cara=cara_hacia(s, (-1, 0, 0)),
                            ux=True, uy=True, uz=True)
    carga = _carga(s, (1, 0, 0), "FUERZA", 10.0, (0, 0, -1))
    carga.caso_id = "L"
    r = ejecutar_analisis_solido(p, {"L": CasoCarga("L", "Viva", TipoCaso.L)}, {}, [apoyo], {}, {}, {}, tmp_path, tamano=0.05,
                                 cargas_superficie_por_caso={"L": [carga]})
    assert "C3D20R" in r.advertencias[0]
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    e = p.materiales[s.material].modulo_elasticidad
    teoria = 10.0 * 1.0**3 / (3 * e * 0.1**4 / 12)
    assert abs(-u[np.isclose(x[:, 0], 1.0), 2].mean() / teoria - 1) < 0.01
    medio = np.isclose(x[:, 0], 0.5, atol=0.02) & np.isclose(x[:, 2], 0.1, atol=1e-6)
    sx = np.abs(r.tensiones_por_caso["L"][medio, 0]).mean() / 1000.0  # MPa; M·c/I = 5 kN·m · 0,05 / I
    assert abs(sx / 30.0 - 1) < 0.02


@requiere_ccx
@requiere_gmsh
def test_tubo_barrido_con_presion_en_su_tapa_y_unido_a_una_placa(tmp_path: Path):
    """Tubo C3D20R sobre una placa (C3D20R también: es un prisma) unidos con *TIE; presión sobre la tapa del tubo
    (cargas consistentes de las caras de 8 nodos): la reacción total iguala a la carga."""

    t = tubo(alto=0.5, id_="T1")
    t.vertices = np.asarray(t.vertices) + [0.0, 0.0, 0.05]
    placa = caja("P1", -0.2, -0.2, 0.0, 0.4, 0.4, 0.05)
    p = proyecto_con(t, placa)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="P1", cara=cara_hacia(placa, (0, 0, -1)),
                            ux=True, uy=True, uz=True)
    tapa = cara_hacia(t, (0, 0, 1))
    carga = CargaSuperficie(id="C1", caso_id="L", elemento_tipo="SOLIDO", elemento_id="T1", cara=tapa, tipo="PRESION", valor=100.0)
    r = ejecutar_analisis_solido(p, {"L": CasoCarga("L", "Viva", TipoCaso.L)}, {}, [apoyo], {}, {}, {}, tmp_path, tamano=0.04,
                                 cargas_superficie_por_caso={"L": [carga]})
    assert "2 pieza(s) son prismáticas" in " ".join(r.advertencias)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    arriba = np.isclose(x[:, 2], 0.55)
    # acortamiento elástico del tubo: P·L / (E·A) con P = p·A → p·L/E
    e = p.materiales[t.material].modulo_elasticidad
    assert -u[arriba, 2].mean() > 0.9 * 100.0 * 0.5 / e
    assert np.isfinite(u).all()
