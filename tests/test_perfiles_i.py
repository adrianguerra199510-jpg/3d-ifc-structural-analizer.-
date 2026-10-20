"""Perfiles en I (acero): geometría, importación de una conexión metálica y
esfuerzos con CalculiX real comparados con la teoría de vigas."""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.asignacion import Objetivo, ParametrosCarga, cargas_desde_parametros
from ifc_structural.core.cargas import Apoyo, CargaLineal, CargaPuntual
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ifc_structural.core.conectividad import conectar_uniones_en_t
from ifc_structural.core.geometria import partes_seccion, solido_barra
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.ifc.importador import importar_ifc
from ifc_structural.solver.analisis import ejecutar_analisis

from .ayuda_ifc import crear_ifc_conexion_metalica

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")

# IPE 300 sin radios de acuerdo (la forma con que se analiza)
H, B, TF, TW = 0.3, 0.15, 0.0107, 0.0071
HW = H - 2 * TF
AREA = 2 * B * TF + HW * TW
I_FUERTE = B * H**3 / 12.0 - (B - TW) * HW**3 / 12.0
I_DEBIL = 2 * TF * B**3 / 12.0 + HW * TW**3 / 12.0
E = 2.1e8  # kPa


def _ipe() -> PerfilSeccion:
    return PerfilSeccion(
        "IPE300", AREA, I_DEBIL, I_FUERTE, (2 * B * TF**3 + HW * TW**3) / 3.0,
        altura=H, ancho=B, forma="I", espesor_ala=TF, espesor_alma=TW,
    )


def _voladizo(destino=(2.5, 0.0, 0.0), n1=(0.0, 0.0, 1.0)) -> Proyecto:
    p = Proyecto("IPE")
    p.agregar_material(Material("S", E, 0.3, 77.0))
    p.agregar_perfil(_ipe())
    p.agregar_nodo(0, 0, 0, "N0")
    p.agregar_nodo(*destino, "N1")
    p.agregar_viga("N0", "N1", "S", "IPE300", id_="B1", vector_n1=n1)
    return p


def _analizar(p: Proyecto, tmp_path: Path, puntuales=(), lineales=(), apoyos=None):
    casos = {"L": CasoCarga("L", "Viva", TipoCaso.L)}
    combinaciones = {"C": Combinacion("C", "L", {"L": 1.0})}
    apoyos = apoyos if apoyos is not None else [Apoyo(id="A", nodo_id="N0")]
    return ejecutar_analisis(
        p, casos, combinaciones, apoyos, {}, {"L": list(lineales)}, {"L": list(puntuales)}, tmp_path,
    )


def _nodos_de(r, barra: str) -> np.ndarray:
    barras = np.array(r.barra_de_cada_celda())
    return np.unique(r.malla_tensiones.conectividad[barras == barra])


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------


def test_solido_del_perfil_en_i_tiene_doce_lados_y_caras_extremas_con_el_area_real():
    p = _voladizo()
    solido = solido_barra(p, p.vigas["B1"])
    assert len(solido.caras) == 14  # 2 extremos + 12 lados del contorno
    assert solido.caras[0].area == pytest.approx(AREA)
    assert solido.caras[1].area == pytest.approx(AREA)
    # todas las normales apuntan hacia fuera: la suma de área·normal de un sólido cerrado es nula
    total = sum(c.area * c.normal for c in solido.caras)
    assert np.linalg.norm(total) < 1e-6
    # la cara exterior del ala superior mira hacia arriba y mide ancho de ala × largo
    superior = max(solido.caras[2:], key=lambda c: c.centro[2])
    assert superior.normal == pytest.approx((0.0, 0.0, 1.0))
    assert superior.area == pytest.approx(B * 2.5)
    assert superior.ancho == pytest.approx(B)


def test_partes_del_perfil_en_i_suman_el_area_y_la_inercia_del_perfil():
    p = _voladizo()
    partes = partes_seccion(p, p.vigas["B1"])
    assert len(partes) == 3
    area = sum(x.dim1 * x.dim2 for x in partes)
    assert area == pytest.approx(AREA)
    # inercia fuerte por Steiner (el peralte va en e1 = vertical)
    inercia = sum(x.dim2 * x.dim1**3 / 12.0 + x.dim1 * x.dim2 * x.desplazamiento1**2 for x in partes)
    assert inercia == pytest.approx(I_FUERTE)


def test_presion_sobre_la_cara_extrema_de_un_perfil_en_i_usa_el_area_real():
    p = _voladizo()
    cargas = cargas_desde_parametros(
        p, Objetivo("VIGA", "B1", 1), "L", "C1", ParametrosCarga(tipo="PRESION", presion=1000.0)
    )
    assert len(cargas) == 1
    assert cargas[0].nodo_id == "N1"
    assert abs(cargas[0].fx) + abs(cargas[0].fy) + abs(cargas[0].fz) == pytest.approx(1000.0 * AREA)


# ---------------------------------------------------------------------------
# Esfuerzos con ccx comparados con la teoría
# ---------------------------------------------------------------------------


@requiere_ccx
def test_voladizo_ipe_flexion_fuerte_y_cortante_en_el_alma(tmp_path: Path):
    P, L = 25.0, 2.5
    p = _voladizo()
    r = _analizar(p, tmp_path, puntuales=[CargaPuntual(id="P", caso_id="L", nodo_id="N1", fz=-P)])
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    n = _nodos_de(r, "B1")
    for xs in (0.0, 0.5, 1.5, 2.25):
        en_estacion = n[np.isclose(x[n, 0], xs, atol=0.01)]
        teorico = P * (L - xs) * (H / 2) / I_FUERTE
        assert np.abs(t[en_estacion, 0]).max() == pytest.approx(teorico, rel=0.01), xs
    # El cortante lo lleva el alma: τ = V / (tw·hw); las alas no tienen cortante vertical.
    web = n[np.isclose(x[n, 0], 1.0, atol=0.01) & (np.abs(x[n, 1]) < TW)]
    assert np.abs(t[web, 5]).max() == pytest.approx(P / (TW * HW), rel=0.01)
    alas = n[np.isclose(x[n, 0], 1.0, atol=0.01) & (np.abs(x[n, 1]) > B / 2 - 1e-6)]
    assert np.abs(t[alas, 5]).max() < 1e-3 * P / (TW * HW)
    # no hay esfuerzos laterales espurios: Von Mises con σ y τ de la teoría
    sxx, tau = P * (L - 1.0) * (H / 2) / I_FUERTE, P / (TW * HW)
    from ifc_structural.combinaciones import esfuerzos

    vm = esfuerzos.von_mises(t[n[np.isclose(x[n, 0], 1.0, atol=0.01)]])
    assert vm.max() == pytest.approx(max(sxx, math.sqrt(sxx**2 * 0 + 3 * tau**2)), rel=0.02)


@requiere_ccx
def test_voladizo_ipe_flexion_debil(tmp_path: Path):
    P, L = 25.0, 2.5
    p = _voladizo()
    r = _analizar(p, tmp_path, puntuales=[CargaPuntual(id="P", caso_id="L", nodo_id="N1", fy=P)])
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    n = _nodos_de(r, "B1")
    en_raiz = n[np.isclose(x[n, 0], 0.0, atol=0.01)]
    assert np.abs(t[en_raiz, 0]).max() == pytest.approx(P * L * (B / 2) / I_DEBIL, rel=0.01)


@requiere_ccx
def test_perfil_en_i_en_diagonal_de_45_grados(tmp_path: Path):
    """Con e2 alineado a un eje global ccx falla; el intercambio de ejes de
    ``ejes_seccion`` no debe cambiar la sección física ni los esfuerzos."""

    P, L = 20.0, 2.0
    c = 1 / math.sqrt(2)
    p = _voladizo(destino=(L * c, 0.0, L * c), n1=(-c, 0.0, c))
    perpendicular = (c, 0.0, -c)  # en el plano de la flexión fuerte
    cp = CargaPuntual(id="P", caso_id="L", nodo_id="N1", fx=P * perpendicular[0], fz=P * perpendicular[2])
    r = _analizar(p, tmp_path, puntuales=[cp])
    t = r.tensiones_por_caso["L"]
    n = _nodos_de(r, "B1")
    x = r.malla_tensiones.coordenadas
    s = x[n] @ np.array([c, 0.0, c])
    en_raiz = n[np.isclose(s, 0.0, atol=0.01)]
    assert np.abs(t[en_raiz]).max(axis=0).max() / 1.0 > 0
    from ifc_structural.combinaciones import esfuerzos

    assert esfuerzos.von_mises(t[en_raiz]).max() == pytest.approx(P * L * (H / 2) / I_FUERTE, rel=0.02)


@requiere_ccx
def test_viga_ipe_simplemente_apoyada_con_carga_repartida(tmp_path: Path):
    w, L = 12.0, 4.0
    p = _voladizo(destino=(L, 0.0, 0.0))
    cl = CargaLineal(id="W", caso_id="L", elemento_id="B1", w_inicio=w, w_fin=w)
    apoyos = [Apoyo(id="A0", nodo_id="N0", ux=True, uy=True, uz=True, rx=True, ry=False, rz=False),
              Apoyo(id="A1", nodo_id="N1", ux=False, uy=True, uz=True, rx=True, ry=False, rz=False)]
    r = _analizar(p, tmp_path, lineales=[cl], apoyos=apoyos)
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    n = _nodos_de(r, "B1")
    centro = n[np.isclose(x[n, 0], L / 2, atol=0.01)]
    # La carga se aplica en el eje (sin excentricidad): w·L²/8 en el centro, tracción abajo
    assert np.abs(t[centro, 0]).max() == pytest.approx(w * L**2 / 8 * (H / 2) / I_FUERTE, rel=0.02)


@requiere_ccx
def test_columna_con_perfil_en_i_orientado_desde_el_ifc(tmp_path: Path):
    """La columna HEB de la conexión metálica tiene el peralte a lo largo de X:
    una fuerza horizontal en X flexiona su eje fuerte."""

    ruta = tmp_path / "c.ifc"
    crear_ifc_conexion_metalica(str(ruta))
    p = importar_ifc(str(ruta))
    heb = p.perfiles["HEB200"]
    r = _analizar(
        p, tmp_path / "w",
        puntuales=[CargaPuntual(id="P", caso_id="L", nodo_id=p.columnas["C2"].nodo_fin, fx=10.0)],
        apoyos=[Apoyo(id="A", nodo_id=p.columnas["C1"].nodo_inicio)],
    )
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    n = _nodos_de(r, "C1")
    base = n[np.isclose(x[n, 2], 0.0, atol=0.01)]
    # M = 10 kN · 4 m en la base, flexión fuerte del HEB (peralte 0,2 m en X)
    assert np.abs(t[base, 2]).max() == pytest.approx(10.0 * 4.0 * (heb.altura / 2) / heb.inercia_z, rel=0.02)


# ---------------------------------------------------------------------------
# Importación de la conexión metálica
# ---------------------------------------------------------------------------


def test_importa_la_conexion_metalica_con_perfiles_reales_y_union_en_t(tmp_path: Path):
    ruta = tmp_path / "conexion.ifc"
    crear_ifc_conexion_metalica(str(ruta))
    p = importar_ifc(str(ruta))

    assert len(p.vigas) == 2 and len(p.columnas) == 2  # la columna continua se parte donde llegan las vigas
    ipe, heb = p.perfiles["IPE300"], p.perfiles["HEB200"]
    assert ipe.forma == "I" and heb.forma == "I"
    assert ipe.altura == pytest.approx(0.3) and ipe.ancho == pytest.approx(0.15)
    assert ipe.area == pytest.approx(AREA)
    assert heb.altura == pytest.approx(0.2) and heb.espesor_alma == pytest.approx(0.009)

    nudos = {p.columnas["C1"].nodo_fin, p.columnas["C2"].nodo_inicio, p.vigas["B1"].nodo_inicio, p.vigas["B2"].nodo_fin}
    assert len(nudos) == 1  # los cuatro extremos comparten el mismo nodo
    coords = p.nodos[nudos.pop()].coordenadas
    assert coords == pytest.approx((0.0, 0.0, 3.0))

    # el peralte de las vigas es vertical y el de la columna va según X (leído de la colocación del perfil)
    assert abs(p.vigas["B1"].vector_n1[2]) == pytest.approx(1.0)
    assert abs(p.columnas["C1"].vector_n1[0]) == pytest.approx(1.0)

    # placas y pernos no son barras y se avisa de que no se modelan
    assert any("sólidos" in aviso and "Plate" in aviso for aviso in p.avisos)


def test_union_en_t_parte_la_barra_atravesada_y_traslada_el_extremo_al_eje():
    p = Proyecto("T")
    p.agregar_material(Material("S", E, 0.3, 77.0))
    p.agregar_perfil(_ipe())
    p.agregar_nodo(0, 0, 0, "A"); p.agregar_nodo(0, 0, 4, "B"); p.agregar_nodo(0.12, 0, 3, "C"); p.agregar_nodo(2.6, 0, 3, "D")
    p.agregar_columna("A", "B", "S", "IPE300", id_="C1", vector_n1=(1, 0, 0))
    p.agregar_viga("C", "D", "S", "IPE300", id_="B1", vector_n1=(0, 0, 1))

    assert conectar_uniones_en_t(p) == 1
    assert len(p.columnas) == 2, p.columnas
    assert p.nodos[p.vigas["B1"].nodo_inicio].coordenadas == pytest.approx((0.0, 0.0, 3.0))
    assert p.columnas["C1"].nodo_fin == p.vigas["B1"].nodo_inicio
    nueva = next(c for i, c in p.columnas.items() if i != "C1")
    assert p.columnas["C1"].nodo_inicio == "A" and nueva.nodo_fin == "B" and nueva.nodo_inicio == p.columnas["C1"].nodo_fin
    assert "C" not in p.nodos  # el nodo antiguo del extremo ya no se usa


def test_extremo_lejos_del_eje_no_se_conecta():
    p = Proyecto("T")
    p.agregar_material(Material("S", E, 0.3, 77.0))
    p.agregar_perfil(_ipe())
    p.agregar_nodo(0, 0, 0, "A"); p.agregar_nodo(0, 0, 4, "B"); p.agregar_nodo(0.6, 0, 3, "C"); p.agregar_nodo(3, 0, 3, "D")
    p.agregar_columna("A", "B", "S", "IPE300", id_="C1", vector_n1=(1, 0, 0))
    p.agregar_viga("C", "D", "S", "IPE300", id_="B1", vector_n1=(0, 0, 1))
    assert conectar_uniones_en_t(p) == 0
    assert len(p.columnas) == 1


@requiere_ccx
def test_conexion_metalica_con_cargas_en_los_extremos_coincide_con_el_calculo_manual(tmp_path: Path):
    """Columna empotrada en la base; 25 kN y 15 kN hacia abajo en los extremos de las
    vigas (en caras extremas) y 100 kN en la cabeza de la columna. Sin peso propio."""

    ruta = tmp_path / "c.ifc"
    crear_ifc_conexion_metalica(str(ruta))
    p = importar_ifc(str(ruta))
    apoyo = Apoyo(id="A", nodo_id=p.columnas["C1"].nodo_inicio)
    puntuales = [
        CargaPuntual(id="1", caso_id="L", nodo_id=p.vigas["B1"].nodo_fin, fz=-25.0),
        CargaPuntual(id="2", caso_id="L", nodo_id=p.vigas["B2"].nodo_inicio, fz=-15.0),
        CargaPuntual(id="3", caso_id="L", nodo_id=p.columnas["C2"].nodo_fin, fz=-100.0),
    ]
    r = _analizar(p, tmp_path / "w", puntuales=puntuales, apoyos=[apoyo])
    t = r.tensiones_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    ipe, heb = p.perfiles["IPE300"], p.perfiles["HEB200"]

    n = _nodos_de(r, "B1")
    raiz = n[np.isclose(x[n, 0], 0.0, atol=0.01)]
    assert np.abs(t[raiz, 0]).max() == pytest.approx(25.0 * 2.6 * 0.15 / ipe.inercia_z, rel=0.02)

    n = _nodos_de(r, "C1")
    base = n[np.isclose(x[n, 2], 0.0, atol=0.01)]
    axial = -140.0 / heb.area
    flexion = (25.0 - 15.0) * 2.6 * (heb.altura / 2) / heb.inercia_z
    assert t[base, 2].min() == pytest.approx(axial - flexion, rel=0.02)
    assert t[base, 2].max() == pytest.approx(axial + flexion, rel=0.03)

    n = _nodos_de(r, "C2")
    medio = n[np.isclose(x[n, 2], 3.5, atol=0.01)]
    assert t[medio, 2].mean() == pytest.approx(-100.0 / heb.area, rel=0.01)
    assert any("Perfiles en I" in a for a in r.advertencias)


def test_los_ifcmember_se_importan_como_vigas_o_columnas(tmp_path: Path):
    ruta = tmp_path / "c.ifc"
    crear_ifc_conexion_metalica(str(ruta))
    texto = ruta.read_text().replace("IFCBEAM(", "IFCMEMBER(").replace("IFCCOLUMN(", "IFCMEMBER(")
    ruta_miembros = tmp_path / "miembros.ifc"
    ruta_miembros.write_text(texto)
    p = importar_ifc(str(ruta_miembros))
    assert len(p.vigas) == 2 and len(p.columnas) == 2
    assert all(abs(c.vector_n1[0]) == pytest.approx(1.0) for c in p.columnas.values())
