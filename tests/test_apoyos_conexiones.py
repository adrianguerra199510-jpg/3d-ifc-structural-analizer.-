"""Apoyos de área, conexiones (articulaciones y enlaces) y mecanismos, con
CalculiX real: los resultados se comparan con la teoría de vigas."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.combinaciones import esfuerzos
from ifc_structural.core.cargas import Apoyo, ApoyoArea, CargaLineal, CargaPuntual, Conexion
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.solver.analisis import ejecutar_analisis, validar_modelo
from ifc_structural.solver.mecanismos import detectar_mecanismos

pytestmark = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")

B, H = 0.3, 0.5
I_FUERTE = B * H**3 / 12.0  # 3.125e-3 m4


def _proyecto() -> Proyecto:
    p = Proyecto("Conexiones")
    p.agregar_material(Material(nombre="H", modulo_elasticidad=2.5e7, densidad=24.0, modulo_poisson=0.2))
    p.agregar_perfil(PerfilSeccion(nombre="V", area=B * H, inercia_y=3.4e-3, inercia_z=I_FUERTE,
                                   inercia_torsional=1e-3, altura=H, ancho=B))
    return p


def _casos():
    return {"L": CasoCarga("L", "Carga viva", TipoCaso.L)}, {"C": Combinacion("C", "L", {"L": 1.0})}


def _analizar(p, tmp_path, apoyos, conexiones=(), lineales=None, puntuales=None):
    casos, combinaciones = _casos()
    return ejecutar_analisis(
        p, casos, combinaciones, apoyos, {}, {"L": lineales or []}, {"L": puntuales or []}, tmp_path,
        conexiones=list(conexiones),
    )


def _portico_con_viga_articulada():
    """Columna empotrada (0,0,0)-(0,0,3) y viga de 4 m en su cabeza, articulada a la
    columna (traslaciones transmitidas, giros libres) y con un rodillo en el otro extremo."""
    p = _proyecto()
    p.agregar_nodo(0, 0, 0, "N0"); p.agregar_nodo(0, 0, 3, "N1"); p.agregar_nodo(4, 0, 3, "N2")
    p.agregar_columna("N0", "N1", "H", "V", id_="C1")
    p.agregar_viga("N1", "N2", "H", "V", id_="B1")
    apoyos = [
        Apoyo(id="A0", nodo_id="N0"),  # empotramiento
        Apoyo(id="A2", nodo_id="N2", ux=False, uy=True, uz=True, rx=True, ry=False, rz=False),  # rodillo
    ]
    return p, apoyos


def test_viga_articulada_a_la_columna_se_comporta_como_simplemente_apoyada(tmp_path: Path):
    p, apoyos = _portico_con_viga_articulada()
    w = 20.0
    cl = CargaLineal(id="W", caso_id="L", elemento_id="B1", w_inicio=w, w_fin=w)
    articulacion = Conexion(id="X1", nodo_id="N1", barra_id="B1")  # transmite ux,uy,uz; libera los 3 giros
    r = _analizar(p, tmp_path, apoyos, [articulacion], [cl])

    x = r.malla_tensiones.coordenadas
    sxx = r.tensiones_por_caso["L"][:, 0]
    enc = [r.malla.elementos[int(e)].barra_id for e in r.malla_tensiones.ids_elementos]
    nodos_viga = {int(i) for celda, b in zip(r.malla_tensiones.conectividad, enc) if b == "B1" for i in celda}
    nodos_viga = np.array(sorted(nodos_viga))
    centro = nodos_viga[np.isclose(x[nodos_viga, 0], 2.0, atol=0.02)]
    teorico = w * 4.0**2 / 8.0 * (H / 2) / I_FUERTE  # M = wL²/8 en el centro
    assert np.abs(sxx[centro]).max() == pytest.approx(teorico, rel=0.05)

    # La articulación no transmite momento a la columna: solo axial uniforme (N = wL/2 = 40 kN → 267 kPa)
    nodos_col = np.array(sorted({int(i) for celda, b in zip(r.malla_tensiones.conectividad, enc) if b == "C1" for i in celda}))
    szz = r.tensiones_por_caso["L"][nodos_col, 2]
    assert szz.mean() == pytest.approx(-w * 4.0 / 2.0 / (B * H), rel=0.05)
    assert np.ptp(szz) < 0.05 * teorico  # sin flexión: esfuerzo uniforme en la sección


def test_sin_articulacion_la_viga_transmite_momento_a_la_columna(tmp_path: Path):
    p, apoyos = _portico_con_viga_articulada()
    cl = CargaLineal(id="W", caso_id="L", elemento_id="B1", w_inicio=20.0, w_fin=20.0)
    r = _analizar(p, tmp_path, apoyos, [], [cl])
    enc = [r.malla.elementos[int(e)].barra_id for e in r.malla_tensiones.ids_elementos]
    nodos_col = np.array(sorted({int(i) for celda, b in zip(r.malla_tensiones.conectividad, enc) if b == "C1" for i in celda}))
    assert np.ptp(r.tensiones_por_caso["L"][nodos_col, 2]) > 1000.0  # kPa: hay flexión en la columna


def test_apoyo_de_area_en_una_viga_la_apoya_en_toda_su_longitud(tmp_path: Path):
    p = _proyecto()
    p.agregar_nodo(0, 0, 0, "N1"); p.agregar_nodo(4, 0, 0, "N2")
    p.agregar_viga("N1", "N2", "H", "V", id_="B1")
    # Apoyada a lo largo (ux, uy, uz) y con giro en X impedido: cargas hacia abajo no la flexionan
    area = ApoyoArea(id="A", elemento_tipo="VIGA", elemento_id="B1", rx=True)
    cl = CargaLineal(id="W", caso_id="L", elemento_id="B1", w_inicio=50.0, w_fin=50.0)
    r = _analizar(p, tmp_path, [area], [], [cl])
    vm = esfuerzos.von_mises(r.tensiones_por_caso["L"]).max()
    # Compresión pura por aplastamiento: la viga no se flexiona, el esfuerzo es muy pequeño
    assert vm < 0.05 * (50.0 * 4.0**2 / 8.0) * (H / 2) / I_FUERTE


def test_apoyo_de_area_solo_en_z_deja_libre_el_movimiento_horizontal(tmp_path: Path):
    p = _proyecto()
    p.agregar_nodo(0, 0, 0, "N1"); p.agregar_nodo(4, 0, 0, "N2")
    p.agregar_viga("N1", "N2", "H", "V", id_="B1")
    apoyos = [
        ApoyoArea(id="Z", elemento_tipo="VIGA", elemento_id="B1", ux=False, uy=False, uz=True),
        Apoyo(id="A1", nodo_id="N1", ux=True, uy=True, uz=False, rx=True, ry=True, rz=True),
    ]
    cl = CargaLineal(id="W", caso_id="L", elemento_id="B1", w_inicio=30.0, w_fin=30.0)
    errores, _ = validar_modelo(p, *_casos(), apoyos)
    assert not errores
    r = _analizar(p, tmp_path, apoyos, [], [cl])
    assert np.isfinite(esfuerzos.von_mises(r.tensiones_por_caso["L"]).max())


def test_carga_triangular_da_la_reaccion_y_momento_esperados(tmp_path: Path):
    """Voladizo de 4 m con carga triangular creciente 0 → w: M_empotramiento = w·L²/3."""
    p = _proyecto()
    p.agregar_nodo(0, 0, 0, "N1"); p.agregar_nodo(4, 0, 0, "N2")
    p.agregar_viga("N1", "N2", "H", "V", id_="B1")
    w = 30.0
    cl = CargaLineal(id="W", caso_id="L", elemento_id="B1", w_inicio=0.0, w_fin=w)
    r = _analizar(p, tmp_path, [Apoyo(id="A", nodo_id="N1")], [], [cl])
    x = r.malla_tensiones.coordenadas
    sxx = r.tensiones_por_caso["L"][:, 0]
    cerca = np.isclose(x[:, 0], 0.5, atol=0.02)  # cerca del empotramiento (evita el extremo extrapolado)
    # momento en x = a:  M(a) = w/(6L)·(L−a)²(2L+a)... integración de la carga triangular
    L, a = 4.0, 0.5
    momento = w / L * ((L**3 - a**3) / 3.0 - a * (L**2 - a**2) / 2.0)  # ∫ q(x)(x−a)dx desde a hasta L
    assert np.abs(sxx[cerca]).max() == pytest.approx(momento * (H / 2) / I_FUERTE, rel=0.05)


def test_portico_con_todas_las_uniones_articuladas_es_un_mecanismo():
    p = _proyecto()
    for id_, (x, y, z) in {"N0": (0, 0, 0), "N1": (0, 0, 3), "N2": (4, 0, 3), "N3": (4, 0, 0)}.items():
        p.agregar_nodo(x, y, z, id_)
    p.agregar_columna("N0", "N1", "H", "V", id_="C1")
    p.agregar_viga("N1", "N2", "H", "V", id_="B1")
    p.agregar_columna("N3", "N2", "H", "V", id_="C2")
    apoyos = [Apoyo(id="A0", nodo_id="N0", rx=True, ry=False, rz=False),
              Apoyo(id="A3", nodo_id="N3", rx=True, ry=False, rz=False)]
    articulaciones = [Conexion(id="X1", nodo_id="N1", barra_id="B1"), Conexion(id="X2", nodo_id="N2", barra_id="B1")]
    resultado = detectar_mecanismos(p, apoyos, articulaciones)
    assert resultado.cantidad >= 1
    errores, _ = validar_modelo(p, *_casos(), apoyos, articulaciones)
    assert any("mecanismo" in e for e in errores)


def test_portico_empotrado_con_una_articulacion_sigue_siendo_estable():
    p = _proyecto()
    for id_, (x, y, z) in {"N0": (0, 0, 0), "N1": (0, 0, 3), "N2": (4, 0, 3), "N3": (4, 0, 0)}.items():
        p.agregar_nodo(x, y, z, id_)
    p.agregar_columna("N0", "N1", "H", "V", id_="C1")
    p.agregar_viga("N1", "N2", "H", "V", id_="B1")
    p.agregar_columna("N3", "N2", "H", "V", id_="C2")
    apoyos = [Apoyo(id="A0", nodo_id="N0"), Apoyo(id="A3", nodo_id="N3")]
    assert detectar_mecanismos(p, apoyos, [Conexion(id="X1", nodo_id="N1", barra_id="B1")]).cantidad == 0
    assert detectar_mecanismos(p, apoyos, []).cantidad == 0


def test_enlace_rigido_entre_dos_nudos_separados(tmp_path: Path):
    """Voladizo B1 (0→2 m) unido por un enlace rígido a B2 (2.5→4.5 m): el empotramiento de B1 debe sostener a B2."""
    p = _proyecto()
    p.agregar_nodo(0, 0, 0, "N1"); p.agregar_nodo(2, 0, 0, "N2")
    p.agregar_nodo(2.5, 0, 0, "N3"); p.agregar_nodo(4.5, 0, 0, "N4")
    p.agregar_viga("N1", "N2", "H", "V", id_="B1")
    p.agregar_viga("N3", "N4", "H", "V", id_="B2")
    cp = CargaPuntual(id="F", caso_id="L", nodo_id="N4", fz=-10.0)
    enlace = Conexion(id="E1", nodo_id="N2", nodo_destino="N3", rx=True, ry=True, rz=True)  # enlace rígido
    r = _analizar(p, tmp_path, [Apoyo(id="A", nodo_id="N1")], [enlace], puntuales=[cp])
    x = r.malla_tensiones.coordenadas
    sxx = r.tensiones_por_caso["L"][:, 0]
    cerca = np.isclose(x[:, 0], 1.0, atol=0.02) & (x[:, 0] < 2.0)
    teorico = 10.0 * (4.5 - 1.0) * (H / 2) / I_FUERTE  # M = F·(distancia al punto de carga)
    assert np.abs(sxx[cerca]).max() == pytest.approx(teorico, rel=0.06)
