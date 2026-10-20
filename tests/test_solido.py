"""Análisis como modelo sólido (3D): mallado, uniones y esfuerzos con CalculiX
real comparados con la teoría de vigas y con el equilibrio."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.combinaciones import esfuerzos
from ifc_structural.core.cargas import Apoyo, CargaArea, CargaLineal, CargaPuntual
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso, casos_y_combinaciones_por_defecto
from ifc_structural.core.geometria import ejes_seccion
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.ifc.importador import importar_ifc
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido, estimar, tamano_sugerido
from ifc_structural.solido.contactos import detectar_uniones
from ifc_structural.solido.malla import construir_malla_solida
from ifc_structural.solido.miembros import construir_miembros, recortar_uniones
from ifc_structural.solido.seccion2d import malla_seccion, pesos_nodales
from ifc_structural.solver.analisis import ErrorAnalisis

from .ayuda_ifc import crear_ifc_conexion_metalica, crear_ifc_portico
from .test_perfiles_i import AREA, HW, H, I_FUERTE, TW, _voladizo

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def _casos():
    return {"L": CasoCarga("L", "Viva", TipoCaso.L)}, {"C": Combinacion("C", "L", {"L": 1.0})}


def _volumenes(malla) -> np.ndarray:
    c = malla.coordenadas[malla.hexaedros]
    total = np.zeros(len(c))
    for a, b, d, e in ((0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)):
        total += np.linalg.det(np.stack([c[:, b] - c[:, a], c[:, d] - c[:, a], c[:, e] - c[:, a]], axis=1)) / 6.0
    return total


@pytest.fixture(scope="module")
def conexion(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("c") / "conexion.ifc"
    crear_ifc_conexion_metalica(str(ruta))
    return importar_ifc(str(ruta))


# ---------------------------------------------------------------------------
# Mallado
# ---------------------------------------------------------------------------


def test_malla_de_la_seccion_en_i_cubre_exactamente_el_perfil():
    p = _voladizo()
    ejes = ejes_seccion(p, p.vigas["B1"])
    puntos, quads = malla_seccion(p.perfiles["IPE300"], ejes, 0.03)
    assert pesos_nodales(puntos, quads).sum() == pytest.approx(AREA)
    # sin nodos duplicados (alas y alma comparten los de la unión)
    assert len({(round(u, 8), round(v, 8)) for u, v in puntos}) == len(puntos)


def test_malla_de_barras_tiene_volumen_positivo_y_el_del_perfil(conexion):
    miembros = construir_miembros(conexion)
    recortar_uniones(conexion, miembros)
    malla = construir_malla_solida(conexion, miembros, 0.03)
    volumenes = _volumenes(malla)
    assert volumenes.min() > 0
    esperado = sum(conexion.perfiles[m.perfil].area * (m.longitud - m.recorte0 - m.recorte1) for m in miembros)
    assert volumenes.sum() == pytest.approx(esperado, rel=1e-6)


def test_miembros_de_la_conexion_columna_fundida_y_vigas_recortadas_hasta_el_ala(conexion):
    miembros = {m.id: m for m in construir_miembros(conexion)}
    assert set(miembros) == {"B1", "B2", "C1+C2"}
    avisos = recortar_uniones(conexion, list(miembros.values()))
    assert avisos == []
    assert miembros["C1+C2"].longitud == pytest.approx(4.0)
    # el peralte de la columna HEB va en X: las vigas se recortan hasta el ala (x = 0,1)
    assert miembros["B1"].recorte0 == pytest.approx(0.1)
    assert miembros["B2"].recorte1 == pytest.approx(0.1)
    assert miembros["C1+C2"].recorte0 == miembros["C1+C2"].recorte1 == 0.0


def test_las_vigas_se_unen_a_la_columna_por_la_cara_extrema(conexion):
    miembros = construir_miembros(conexion)
    recortar_uniones(conexion, miembros)
    malla = construir_malla_solida(conexion, miembros, 0.03)
    uniones, avisos = detectar_uniones(malla, 0.03)
    assert avisos == []
    pares = {frozenset((malla.partes[u.parte_esclava].id, malla.partes[u.parte_maestra].id)) for u in uniones}
    assert pares == {frozenset(("B1", "C1+C2")), frozenset(("B2", "C1+C2"))}


def test_la_losa_solapada_con_las_vigas_se_recorta_y_se_une(tmp_path: Path):
    ruta = tmp_path / "p.ifc"
    crear_ifc_portico(str(ruta))
    p = importar_ifc(str(ruta))
    miembros = construir_miembros(p)
    recortar_uniones(p, miembros)
    malla = construir_malla_solida(p, miembros, 0.1)
    assert _volumenes(malla).min() > 0
    uniones, avisos = detectar_uniones(malla, 0.1, {"L1": 0.2})
    ties_losa = {malla.partes[u.parte_maestra].id for u in uniones if malla.partes[u.parte_esclava].id == "L1"} | {
        malla.partes[u.parte_esclava].id for u in uniones if malla.partes[u.parte_maestra].id == "L1"}
    assert {"B1", "B2", "B3", "B4"} <= ties_losa
    assert any("L1" in a for a in avisos)


def test_estimacion_de_hexaedros_coincide_con_la_malla(conexion):
    miembros = construir_miembros(conexion)
    recortar_uniones(conexion, miembros)
    real = len(construir_malla_solida(conexion, miembros, 0.02).hexaedros)
    assert estimar(conexion, 0.02) == pytest.approx(real, rel=0.05)
    assert 0.01 <= tamano_sugerido(conexion) <= 0.25


# ---------------------------------------------------------------------------
# Errores de modelo
# ---------------------------------------------------------------------------


def test_sin_apoyos_no_se_analiza(tmp_path: Path):
    casos, comb = _casos()
    with pytest.raises(ErrorAnalisis, match="apoyos"):
        ejecutar_analisis_solido(_voladizo(), casos, comb, [], {}, {"L": []}, {"L": []}, tmp_path)


def test_pieza_flotante_se_detecta_antes_de_resolver(tmp_path: Path):
    p = _voladizo()
    p.agregar_nodo(0, 5, 0, "N8"); p.agregar_nodo(2, 5, 0, "N9")
    p.agregar_viga("N8", "N9", "S", "IPE300", id_="B2", vector_n1=(0, 0, 1))
    casos, comb = _casos()
    with pytest.raises(ErrorAnalisis, match="B2"):
        ejecutar_analisis_solido(p, casos, comb, [Apoyo(id="A", nodo_id="N0")], {}, {"L": []}, {"L": []}, tmp_path)


# ---------------------------------------------------------------------------
# Esfuerzos con ccx
# ---------------------------------------------------------------------------


@requiere_ccx
def test_voladizo_en_i_como_solido_coincide_con_la_teoria(tmp_path: Path):
    P, L = 25.0, 2.5
    p = _voladizo()
    casos, comb = _casos()
    cp = CargaPuntual(id="P", caso_id="L", nodo_id="N1", fz=-P)
    r = ejecutar_analisis_solido(p, casos, comb, [Apoyo(id="A", nodo_id="N0")], {}, {"L": []}, {"L": [cp]}, tmp_path, tamano=0.03)
    assert r.es_solido
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    for xs in (0.5, 1.0, 1.5, 2.0):
        en_estacion = np.isclose(x[:, 0], xs, atol=0.02)
        assert np.abs(t[en_estacion, 0]).max() == pytest.approx(P * (L - xs) * (H / 2) / I_FUERTE, rel=0.03), xs
    web = np.isclose(x[:, 0], 1.0, atol=0.02) & (np.abs(x[:, 1]) < TW)
    assert np.abs(t[web, 5]).max() == pytest.approx(P / (TW * HW), rel=0.1)
    assert set(r.barra_de_cada_celda()) == {"B1"}


@requiere_ccx
def test_viga_rectangular_con_carga_repartida_y_apoyos_articulados(tmp_path: Path):
    """Viga simplemente apoyada (pasador + rodillo en las caras extremas): M = wL²/8."""

    b, h, L, w = 0.3, 0.5, 4.0, 20.0
    p = Proyecto("R")
    p.agregar_material(Material("H", 2.5e7, 0.2, 24.0))
    p.agregar_perfil(PerfilSeccion("V", b * h, 1e-3, b * h**3 / 12, 1e-3, altura=h, ancho=b))
    p.agregar_nodo(0, 0, 0, "N0"); p.agregar_nodo(L, 0, 0, "N1")
    p.agregar_viga("N0", "N1", "H", "V", id_="B1", vector_n1=(0, 0, 1))
    casos, comb = _casos()
    apoyos = [Apoyo(id="A0", nodo_id="N0", ux=True, uy=True, uz=True, rx=True, ry=False, rz=False),
              Apoyo(id="A1", nodo_id="N1", ux=False, uy=True, uz=True, rx=True, ry=False, rz=False)]
    cl = CargaLineal(id="W", caso_id="L", elemento_id="B1", w_inicio=w, w_fin=w)
    r = ejecutar_analisis_solido(p, casos, comb, apoyos, {}, {"L": [cl]}, {"L": []}, tmp_path, tamano=0.05)
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    centro = np.isclose(x[:, 0], L / 2, atol=0.03)
    assert np.abs(t[centro, 0]).max() == pytest.approx(w * L**2 / 8 * (h / 2) / (b * h**3 / 12), rel=0.04)


@requiere_ccx
def test_conexion_metalica_como_solido_equilibrio_y_flexion_de_las_vigas(conexion, tmp_path: Path):
    casos, comb = _casos()
    cps = [CargaPuntual(id="1", caso_id="L", nodo_id=conexion.vigas["B1"].nodo_fin, fz=-25.0),
           CargaPuntual(id="2", caso_id="L", nodo_id=conexion.vigas["B2"].nodo_inicio, fz=-15.0),
           CargaPuntual(id="3", caso_id="L", nodo_id=conexion.columnas["C2"].nodo_fin, fz=-100.0)]
    apoyo = Apoyo(id="A", nodo_id=conexion.columnas["C1"].nodo_inicio)
    r = ejecutar_analisis_solido(conexion, casos, comb, [apoyo], {}, {"L": []}, {"L": cps}, tmp_path, tamano=0.03)
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    barras = np.array(r.barra_de_cada_celda())
    conectividad = r.malla_tensiones.conectividad
    ipe, heb = conexion.perfiles["IPE300"], conexion.perfiles["HEB200"]

    n = np.unique(conectividad[barras == "B1"])
    for xs in (0.3, 1.3):
        m = n[np.isclose(x[n, 0], xs, atol=0.02)]
        assert np.abs(t[m, 0]).max() == pytest.approx(25.0 * (2.6 - xs) * (ipe.altura / 2) / ipe.inercia_z, rel=0.03)

    columna = np.unique(conectividad[np.isin(barras, ["C1", "C2"])])
    arriba = columna[np.isclose(x[columna, 2], 3.5, atol=0.02)]
    assert t[arriba, 2].mean() == pytest.approx(-100.0 / heb.area, rel=0.03)
    # el tramo bajo la unión recibe 140 kN y un momento de 26 kN·m: la compresión media es la axial
    abajo = columna[np.isclose(x[columna, 2], 1.5, atol=0.02)]
    assert t[abajo, 2].mean() == pytest.approx(-140.0 / heb.area, rel=0.05)
    # el modelo sólido muestra concentraciones en la unión que el de barras no ve
    vm = esfuerzos.von_mises(t)
    assert vm.max() > 2.0 * 25.0 * 2.6 * (ipe.altura / 2) / ipe.inercia_z


@requiere_ccx
def test_portico_con_losa_equilibrio_de_la_carga_de_area(tmp_path: Path):
    ruta = tmp_path / "p.ifc"
    crear_ifc_portico(str(ruta))
    p = importar_ifc(str(ruta))
    casos, comb = casos_y_combinaciones_por_defecto()
    apoyos = [Apoyo(id=f"A{i}", nodo_id=c.nodo_inicio) for i, c in enumerate(p.columnas.values())]
    area = {"L": [CargaArea(id="Q", caso_id="L", presion=5.0, losa_id="L1")]}
    r = ejecutar_analisis_solido(p, casos, comb, apoyos, area, {}, {}, tmp_path / "w", tamano=0.1)
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    barras = np.array(r.barra_de_cada_celda())
    total = 0.0
    for c in p.columnas:
        n = np.unique(r.malla_tensiones.conectividad[barras == c])
        total += t[n[np.isclose(x[n, 2], 1.5, atol=0.03)], 2].mean() * 0.16
    assert total == pytest.approx(-5.0 * 4.0 * 6.0, rel=0.03)  # 120 kN repartidos en las 4 columnas
    assert set(barras) == {"B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "L1"}
    assert any("L1" in a for a in r.advertencias)


@requiere_ccx
def test_peso_propio_del_modelo_solido_en_el_caso_de_carga_muerta(tmp_path: Path):
    """Voladizo de hormigón: en el empotramiento M = γ·A·L²/2."""

    b, h, L = 0.3, 0.5, 3.0
    p = Proyecto("PP")
    p.agregar_material(Material("H", 2.5e7, 0.2, 24.0))
    p.agregar_perfil(PerfilSeccion("V", b * h, 1e-3, b * h**3 / 12, 1e-3, altura=h, ancho=b))
    p.agregar_nodo(0, 0, 0, "N0"); p.agregar_nodo(L, 0, 0, "N1")
    p.agregar_viga("N0", "N1", "H", "V", id_="B1", vector_n1=(0, 0, 1))
    casos, comb = casos_y_combinaciones_por_defecto()
    r = ejecutar_analisis_solido(p, casos, comb, [Apoyo(id="A", nodo_id="N0")], {}, {}, {}, tmp_path, tamano=0.05)
    t = r.tensiones_por_caso["D"]
    x = r.malla_tensiones.coordenadas
    en_estacion = np.isclose(x[:, 0], 1.5, atol=0.03)
    esperado = 24.0 * b * h * (L - 1.5) ** 2 / 2 * (h / 2) / (b * h**3 / 12)
    assert np.abs(t[en_estacion, 0]).max() == pytest.approx(esperado, rel=0.04)
    assert np.abs(r.tensiones_por_caso["L"]).max() < 1e-6  # sin cargas, el caso L no tiene esfuerzos
