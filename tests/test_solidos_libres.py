"""Sólidos de forma libre (geometría real de cualquier elemento): caras, mallado en hexaedros,
apoyos y cargas sobre cualquier superficie y análisis 3D con CalculiX comparado con la teoría."""

import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from ifc_structural.core.cargas import ApoyoSuperficie, CargaSuperficie, UnionSuperficie
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ifc_structural.core.modelo import Material, Proyecto
from ifc_structural.core.persistencia import cargar_cargas_superficie, cargar_proyecto, guardar_proyecto
from ifc_structural.core.solidos import SolidoGeneral, soldar_vertices
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido
from ifc_structural.solido.libres import mallar_solido
from ifc_structural.solido.malla import construir_malla_solida
from ifc_structural.solido.superficies import fuerzas_nodales, superficie_en_malla
from ifc_structural.solver.analisis import ErrorAnalisis

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")

E = 2.1e8  # kPa


def caja(id_, x, y, z, dx, dy, dz) -> SolidoGeneral:
    c = pv.Cube(center=(x + dx / 2, y + dy / 2, z + dz / 2), x_length=dx, y_length=dy, z_length=dz).triangulate()
    v, t = soldar_vertices(np.asarray(c.points), c.faces.reshape(-1, 4)[:, 1:])
    return SolidoGeneral(id=id_, nombre=id_, tipo_ifc="IfcPlate", material="ACERO", vertices=v, triangulos=t)


def cilindro(id_, radio, altura, resolucion=48) -> SolidoGeneral:
    c = pv.Cylinder(center=(0, 0, altura / 2), direction=(0, 0, 1), radius=radio, height=altura, resolution=resolucion).triangulate()
    v, t = soldar_vertices(np.asarray(c.points), c.faces.reshape(-1, 4)[:, 1:])
    return SolidoGeneral(id=id_, nombre=id_, tipo_ifc="IfcMechanicalFastener", material="ACERO", vertices=v, triangulos=t)


def proyecto_con(*solidos) -> Proyecto:
    p = Proyecto("Sólidos libres")
    p.agregar_material(Material(nombre="ACERO", modulo_elasticidad=E, densidad=78.5, modulo_poisson=0.3))
    for s in solidos:
        p.solidos[s.id] = s
    return p


def cara_hacia(s: SolidoGeneral, direccion) -> int:
    d = np.asarray(direccion, dtype=float)
    return max(range(len(s.caras)), key=lambda i: float(np.dot(s.caras[i].normal, d)) * 1e3 + float(s.caras[i].area))


def _casos():
    return {"L": CasoCarga("L", "Viva", TipoCaso.L)}, {"C": Combinacion("C", "L", {"L": 1.0})}


# ---------------------------------------------------------------------------
# Geometría y mallado
# ---------------------------------------------------------------------------


def test_una_caja_tiene_seis_superficies_con_sus_normales():
    s = caja("S1", 0, 0, 0, 0.3, 0.2, 0.02)
    assert s.cerrado and len(s.caras) == 6
    assert s.volumen == pytest.approx(0.3 * 0.2 * 0.02)
    assert s.caras[cara_hacia(s, (0, 0, 1))].area == pytest.approx(0.06)


def test_malla_de_una_caja_es_exacta_y_asocia_las_caras_de_hexaedro_a_cada_superficie():
    s = caja("S1", 0, 0, 0, 0.3, 0.2, 0.02)
    m = mallar_solido(s, 0.05)
    c = m.coordenadas[m.hexaedros]
    volumen = np.prod(c[:, 6] - c[:, 0], axis=1).sum()
    assert volumen == pytest.approx(s.volumen, rel=1e-5)
    # el área de las caras de borde asociadas a la superficie superior es la de la superficie
    sup = cara_hacia(s, (0, 0, 1))
    assert len(m.superficies[sup]) * m.tamano**2 == pytest.approx(0.06, rel=1e-6)
    assert set(m.superficies) == set(range(6))


def test_el_solido_menor_pierde_el_volumen_que_comparte_con_el_mayor():
    a, b = caja("A", 0, 0, 0, 0.2, 0.2, 0.2), caja("B", 0.1, 0.1, 0.1, 0.2, 0.2, 0.2)
    m = mallar_solido(b, 0.05, [a])
    c = m.coordenadas[m.hexaedros]
    assert np.prod(c[:, 6] - c[:, 0], axis=1).sum() == pytest.approx(0.2**3 - 0.1**3, rel=1e-5)
    assert m.descartados > 0


def test_un_solido_totalmente_dentro_de_otro_desaparece():
    assert mallar_solido(caja("B", 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), 0.02, [caja("A", 0, 0, 0, 0.2, 0.2, 0.2)]) is None


def test_el_cilindro_se_malla_con_su_volumen_aproximado_y_tres_superficies():
    s = cilindro("P", 0.01, 0.08)
    m = mallar_solido(s, 0.05)
    c = m.coordenadas[m.hexaedros]
    assert np.prod(c[:, 6] - c[:, 0], axis=1).sum() == pytest.approx(s.volumen, rel=0.1)
    assert len(s.caras) == 3 and set(m.superficies) == {0, 1, 2}


def test_superficie_en_malla_devuelve_las_caras_de_hexaedro_con_su_area():
    p = proyecto_con(caja("S1", 0, 0, 0, 0.3, 0.2, 0.02))
    malla = construir_malla_solida(p, [], 0.05)
    s = p.solidos["S1"]
    sup = superficie_en_malla(p, malla, "SOLIDO", "S1", cara_hacia(s, (1, 0, 0)), 0.05)
    assert sup.area_total == pytest.approx(0.2 * 0.02, rel=1e-6)
    assert np.allclose(sup.normal, (1, 0, 0))
    assert np.allclose(malla.coordenadas[sup.nodos][:, :, 0], 0.3)


# ---------------------------------------------------------------------------
# Reparto de cargas
# ---------------------------------------------------------------------------


def _carga(s, direccion_cara, tipo, valor, direccion):
    return CargaSuperficie(id="CS1", caso_id="L", elemento_tipo="SOLIDO", elemento_id=s.id, cara=cara_hacia(s, direccion_cara),
                           tipo=tipo, valor=valor, direccion=direccion)


def test_presion_fuerza_y_momento_se_reparten_con_la_resultante_correcta():
    p = proyecto_con(caja("S1", 0, 0, 0, 0.3, 0.2, 0.02))
    malla = construir_malla_solida(p, [], 0.05)
    s = p.solidos["S1"]
    for carga, fuerza, momento in (
        (_carga(s, (0, 0, 1), "PRESION", 50.0, None), (0, 0, -50.0 * 0.06), None),
        (_carga(s, (1, 0, 0), "FUERZA", 12.0, (0, 0, -1)), (0, 0, -12.0), None),
        (_carga(s, (0, 0, 1), "MOMENTO", 3.0, (0, 1, 0)), (0, 0, 0), (0, 3.0, 0)),
    ):
        sup = superficie_en_malla(p, malla, "SOLIDO", "S1", carga.cara, 0.05)
        nodos, f = fuerzas_nodales(malla, sup, carga)
        assert f.sum(axis=0) == pytest.approx(fuerza, abs=1e-5)
        if momento is not None:
            x = malla.coordenadas[nodos]
            centro = sup.centro.T @ sup.area / sup.area.sum()
            assert np.cross(x - centro, f).sum(axis=0) == pytest.approx(momento, abs=1e-5)


# ---------------------------------------------------------------------------
# Análisis con CalculiX
# ---------------------------------------------------------------------------


def _analizar(p, apoyos, cargas, tmp_path, tamano=0.05):
    casos, combinaciones = _casos()
    return ejecutar_analisis_solido(p, casos, combinaciones, apoyos, {}, {}, {}, tmp_path, tamano=tamano,
                                    cargas_superficie_por_caso={"L": cargas})


@requiere_ccx
def test_voladizo_de_un_bloque_libre_apoyado_y_cargado_en_sus_caras(tmp_path: Path):
    L, b, h, F = 1.0, 0.1, 0.1, 10.0
    s = caja("S1", 0, 0, 0, L, b, h)
    p = proyecto_con(s)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="S1", cara=cara_hacia(s, (-1, 0, 0)),
                            ux=True, uy=True, uz=True)
    carga = _carga(s, (1, 0, 0), "FUERZA", F, (0, 0, -1))
    r = _analizar(p, [apoyo], [carga], tmp_path, tamano=0.05)
    x = r.malla_tensiones.coordenadas
    u = r.desplazamientos_por_caso["L"]
    punta = np.isclose(x[:, 0], L)
    flecha = -u[punta, 2].mean()
    esperada = F * L**3 / (3 * E * b * h**3 / 12)
    assert flecha == pytest.approx(esperada, rel=0.08)
    # el empotramiento es total
    assert np.abs(u[np.isclose(x[:, 0], 0.0)]).max() < 1e-9


@requiere_ccx
def test_presion_sobre_una_cara_y_apoyo_deslizante_en_otra(tmp_path: Path):
    """Bloque comprimido entre una cara que solo restringe Uz y una presión: tensión uniforme σ = p."""

    s = caja("S1", 0, 0, 0, 0.2, 0.2, 0.4)
    p = proyecto_con(s)
    apoyos = [
        ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="S1", cara=cara_hacia(s, (0, 0, -1)), uz=True, ux=False, uy=False),
        ApoyoSuperficie(id="AP2", elemento_tipo="SOLIDO", elemento_id="S1", cara=cara_hacia(s, (-1, 0, 0)), ux=True, uy=False, uz=False),
        ApoyoSuperficie(id="AP3", elemento_tipo="SOLIDO", elemento_id="S1", cara=cara_hacia(s, (0, -1, 0)), ux=False, uy=True, uz=False),
    ]
    presion = 1000.0
    r = _analizar(p, apoyos, [_carga(s, (0, 0, 1), "PRESION", presion, None)], tmp_path, tamano=0.1)
    u = r.desplazamientos_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    esperado = -presion / E * 0.4
    assert u[np.isclose(x[:, 2], 0.4), 2].mean() == pytest.approx(esperado, rel=0.02)


@requiere_ccx
def test_perno_cilindrico_en_traccion_apoyado_en_su_tapa(tmp_path: Path):
    s = cilindro("P", 0.015, 0.2)
    p = proyecto_con(s)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="P", cara=cara_hacia(s, (0, 0, -1)),
                            ux=True, uy=True, uz=True)
    carga = _carga(s, (0, 0, 1), "FUERZA", 5.0, (0, 0, 1))
    r = _analizar(p, [apoyo], [carga], tmp_path, tamano=0.05)
    x = r.malla_tensiones.coordenadas
    u = r.desplazamientos_por_caso["L"]
    malla = construir_malla_solida(p, [], 0.05)
    c = malla.coordenadas[malla.hexaedros]
    area = (np.prod(c[:, 6] - c[:, 0], axis=1).sum()) / 0.2  # área de la sección escalonada
    assert u[np.isclose(x[:, 2], 0.2), 2].mean() == pytest.approx(5.0 * 0.2 / (E * area), rel=0.03)


@requiere_ccx
def test_dos_placas_apiladas_transmiten_la_carga_por_el_contacto(tmp_path: Path):
    abajo, arriba = caja("A", 0, 0, 0, 0.3, 0.3, 0.05), caja("B", 0, 0, 0.05, 0.3, 0.3, 0.05)
    p = proyecto_con(abajo, arriba)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(abajo, (0, 0, -1)),
                            ux=True, uy=True, uz=True)
    carga = _carga(arriba, (0, 0, 1), "PRESION", 2000.0, None)
    r = _analizar(p, [apoyo], [carga], tmp_path, tamano=0.05)
    x = r.malla_tensiones.coordenadas
    u = r.desplazamientos_por_caso["L"]
    # compresión uniaxial con Poisson restringido por el apoyo fijo en la base: u_z arriba ≈ -σ·0.1/E_edométrico
    assert u[np.isclose(x[:, 2], 0.1), 2].mean() < 0
    assert abs(u[np.isclose(x[:, 2], 0.05), 2].mean()) == pytest.approx(abs(u[np.isclose(x[:, 2], 0.1), 2].mean()) / 2, rel=0.15)


def test_sin_apoyos_ni_cargas_validos_el_analisis_avisa(tmp_path: Path):
    s = caja("S1", 0, 0, 0, 0.3, 0.2, 0.02)
    p = proyecto_con(s)
    casos, combinaciones = _casos()
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="NO_EXISTE", cara=0, uz=True)
    with pytest.raises(ErrorAnalisis):
        ejecutar_analisis_solido(p, casos, combinaciones, [apoyo], {}, {}, {}, tmp_path, tamano=0.05)


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def test_los_solidos_y_las_asignaciones_en_superficies_se_guardan_y_se_recuperan(tmp_path: Path):
    s = caja("S1", 0, 0, 0, 0.3, 0.2, 0.02)
    p = proyecto_con(s)
    casos, combinaciones = _casos()
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="S1", cara=2, ux=True, uy=True, uz=True, nombre="a")
    carga = _carga(s, (0, 0, 1), "PRESION", 10.0, None)
    ruta = tmp_path / "p.json"
    guardar_proyecto(ruta, p, [apoyo], casos, combinaciones, {}, {}, {}, cargas_superficie_por_caso={"L": [carga]})
    leido = cargar_cargas_superficie(ruta)
    assert leido["L"][0].valor == 10.0 and leido["L"][0].cara == carga.cara
    p2, apoyos2 = cargar_proyecto(ruta)[:2]
    assert isinstance(apoyos2[0], ApoyoSuperficie) and apoyos2[0].cara == 2
    s2 = p2.solidos["S1"]
    assert len(s2.caras) == 6 and s2.volumen == pytest.approx(s.volumen)
    assert np.allclose(s2.vertices, s.vertices)


# ---------------------------------------------------------------------------
# Constraints entre superficies
# ---------------------------------------------------------------------------


def _union(tipo, a, ca, b, cb, grados=(True,) * 6):
    return UnionSuperficie(id="UN1", tipo=tipo, a_tipo="SOLIDO", a_id=a.id, a_cara=cara_hacia(a, ca), b_tipo="SOLIDO",
                           b_id=b.id, b_cara=cara_hacia(b, cb), ux=grados[0], uy=grados[1], uz=grados[2],
                           rx=grados[3], ry=grados[4], rz=grados[5], nombre=tipo)


def _analizar_con_uniones(p, apoyos, cargas, uniones, tmp_path, tamano=0.05):
    casos, combinaciones = _casos()
    return ejecutar_analisis_solido(p, casos, combinaciones, apoyos, {}, {}, {}, tmp_path, tamano=tamano,
                                    cargas_superficie_por_caso={"L": cargas}, uniones_manuales=uniones)


@requiere_ccx
def test_soldadura_entre_dos_superficies_con_holgura_se_comporta_como_una_sola_pieza(tmp_path: Path):
    b, h, F = 0.1, 0.1, 10.0
    a, c = caja("A", 0, 0, 0, 0.5, b, h), caja("B", 0.53, 0, 0, 0.47, b, h)  # 3 cm de holgura
    p = proyecto_con(a, c)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (-1, 0, 0)), ux=True, uy=True, uz=True)
    carga = _carga(c, (1, 0, 0), "FUERZA", F, (0, 0, -1))
    with pytest.raises(ErrorAnalisis):  # sin constraint, B queda flotando
        _analizar_con_uniones(p, [apoyo], [carga], [], tmp_path)
    r = _analizar_con_uniones(p, [apoyo], [carga], [_union("SOLDADURA", a, (1, 0, 0), c, (-1, 0, 0))], tmp_path)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    flecha = -u[np.isclose(x[:, 0], 1.0), 2].mean()
    assert flecha == pytest.approx(F * 1.0**3 / (3 * E * b * h**3 / 12), rel=0.10)


@requiere_ccx
def test_igualar_movimientos_de_dos_superficies_equivale_a_soldarlas(tmp_path: Path):
    a, c = caja("A", 0, 0, 0, 0.2, 0.2, 0.1), caja("B", 0, 0, 0.101, 0.2, 0.2, 0.3)  # 1 mm de holgura
    base = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (0, 0, -1)), ux=True, uy=True, uz=True)
    carga = _carga(c, (1, 0, 0), "FUERZA", 20.0, (0, 1, 0))
    p = proyecto_con(a, c)
    r1 = _analizar_con_uniones(p, [base], [carga], [_union("IGUALAR", a, (0, 0, 1), c, (0, 0, -1))], tmp_path / "i")
    r2 = _analizar_con_uniones(p, [base], [carga], [_union("SOLDADURA", a, (0, 0, 1), c, (0, 0, -1))], tmp_path / "s")

    def punta(r):
        return r.desplazamientos_por_caso["L"][np.isclose(r.malla_tensiones.coordenadas[:, 2], 0.401), 1].mean()

    assert punta(r1) > 0 and punta(r1) == pytest.approx(punta(r2), rel=0.05)


@requiere_ccx
def test_cuerpo_rigido_entre_dos_superficies_las_mueve_como_una_sola_pieza(tmp_path: Path):
    a, c = caja("A", 0, 0, 0, 0.2, 0.2, 0.2), caja("B", 0.3, 0, 0, 0.2, 0.2, 0.2)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (-1, 0, 0)), ux=True, uy=True, uz=True)
    carga = _carga(c, (1, 0, 0), "FUERZA", 10.0, (0, 0, -1))
    r = _analizar_con_uniones(proyecto_con(a, c), [apoyo], [carga], [_union("CUERPO_RIGIDO", a, (1, 0, 0), c, (-1, 0, 0))], tmp_path)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    en_a, en_b = np.isclose(x[:, 0], 0.2), np.isclose(x[:, 0], 0.3)
    nodos = np.where(en_a | en_b)[0]
    # ajuste de un movimiento de cuerpo rígido u = t + θ × r: el residuo debe ser nulo
    r_ = x[nodos] - x[nodos].mean(axis=0)
    filas = np.vstack([np.hstack([np.eye(3), -_antisimetrica(ri)]) for ri in r_])
    solucion, *_ = np.linalg.lstsq(filas, u[nodos].ravel(), rcond=None)
    assert np.abs(filas @ solucion - u[nodos].ravel()).max() < 1e-9
    assert np.abs(u[nodos, 2]).max() > 1e-6  # y sí se mueven


def _antisimetrica(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


@requiere_ccx
def test_articulacion_entre_superficies_iguala_traslaciones_y_los_giros_marcados(tmp_path: Path):
    a, c = caja("A", 0, 0, 0, 0.2, 0.2, 0.2), caja("B", 0, 0, 0.2, 0.2, 0.2, 0.4)
    base = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (0, 0, -1)), ux=True, uy=True, uz=True)
    carga = _carga(c, (0, 0, 1), "MOMENTO", 2.0, (1, 0, 0))  # momento alrededor de X: Rx está igualado, la rótula lo resiste
    # solo el giro alrededor de Y queda libre (sin carga que lo excite)
    union = _union("ARTICULACION", a, (0, 0, 1), c, (0, 0, -1), (True, True, True, True, False, True))
    r = _analizar_con_uniones(proyecto_con(a, c), [base], [carga], [union], tmp_path)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    assert np.abs(u[:, 1]).max() > 0  # el momento flexiona la parte superior


def test_constraint_entre_elementos_enteros_toma_las_caras_mas_cercanas_y_se_guarda(tmp_path: Path):
    from ifc_structural.core.asignacion import ErrorAsignacion, Objetivo
    from ifc_structural.core.asignacion_superficie import union_para

    a, c = caja("A", 0, 0, 0, 0.2, 0.2, 0.1), caja("B", 0.5, 0, 0, 0.2, 0.2, 0.1)
    p = proyecto_con(a, c)
    u = union_para(p, Objetivo("SOLIDO", "A"), Objetivo("SOLIDO", "B"), "CT1", "SOLDADURA", (True,) * 6)
    assert (u.a_cara, u.b_cara) == (cara_hacia(a, (1, 0, 0)), cara_hacia(c, (-1, 0, 0)))
    with pytest.raises(ErrorAsignacion):
        union_para(p, Objetivo("SOLIDO", "A"), Objetivo("SOLIDO", "A"), "CT2", "SOLDADURA", (True,) * 6)
    with pytest.raises(ErrorAsignacion):  # igualar sin ninguna traslación
        union_para(p, Objetivo("SOLIDO", "A"), Objetivo("SOLIDO", "B"), "CT3", "IGUALAR", (False,) * 6)
    p.uniones.append(union_para(p, Objetivo("SOLIDO", "A"), Objetivo("SOLIDO", "B"), "CT4", "IGUALAR", (True, False, True, True, True, True)))
    casos, combinaciones = _casos()
    ruta = tmp_path / "p.json"
    guardar_proyecto(ruta, p, [], casos, combinaciones, {}, {}, {})
    leida = cargar_proyecto(ruta)[0].uniones
    assert len(leida) == 1 and leida[0].tipo == "IGUALAR" and leida[0].grados() == [1, 3]


def test_articulacion_escribe_cuerpos_rigidos_y_ecuaciones(tmp_path: Path):
    from ifc_structural.solido.generador import Enlace3D, ModeloSolido, escribir_comun

    p = proyecto_con(caja("A", 0, 0, 0, 0.2, 0.2, 0.2), caja("B", 0, 0, 0.2, 0.2, 0.2, 0.2))
    malla = construir_malla_solida(p, [], 0.1)
    a = np.unique(malla.hexaedros[malla.partes[0].primer_hexaedro:malla.partes[0].ultimo_hexaedro])
    b = np.unique(malla.hexaedros[malla.partes[1].primer_hexaedro:malla.partes[1].ultimo_hexaedro])
    modelo = ModeloSolido(malla=malla, uniones=[], apoyos=[], materiales={"ACERO": "ACERO"},
                          enlaces=[Enlace3D("ARTICULACION", a[:4], b[:4], {1, 2, 3, 4}, "rótula")])
    texto = "\n".join(escribir_comun(modelo, p))
    assert texto.count("*RIGID BODY") == 2 and texto.count("\n2\n") == 4  # 3 traslaciones + 1 giro igualados
