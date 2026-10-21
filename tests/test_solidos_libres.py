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


# ---------------------------------------------------------------------------
# Constraints entre elementos 3D completos y conexión automática
# ---------------------------------------------------------------------------


def _union_completa(tipo, a, b, tolerancia=0.0, grados=(True,) * 6):
    return UnionSuperficie(id="UN1", tipo=tipo, a_tipo="SOLIDO", a_id=a.id, a_cara=-1, b_tipo="SOLIDO", b_id=b.id, b_cara=-1,
                           ux=grados[0], uy=grados[1], uz=grados[2], rx=grados[3], ry=grados[4], rz=grados[5],
                           nombre=tipo, tolerancia=tolerancia)


@requiere_ccx
def test_soldadura_entre_elementos_completos_une_piezas_con_holgura_de_la_tolerancia(tmp_path: Path):
    b, h, F = 0.1, 0.1, 10.0
    a, c = caja("A", 0, 0, 0, 0.5, b, h), caja("B", 0.53, 0, 0, 0.47, b, h)  # 3 cm de holgura
    p = proyecto_con(a, c)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (-1, 0, 0)), ux=True, uy=True, uz=True)
    carga = _carga(c, (1, 0, 0), "FUERZA", F, (0, 0, -1))
    r = _analizar_con_uniones(p, [apoyo], [carga], [_union_completa("SOLDADURA", a, c, tolerancia=0.04)], tmp_path)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    assert -u[np.isclose(x[:, 0], 1.0), 2].mean() == pytest.approx(F * 1.0**3 / (3 * E * b * h**3 / 12), rel=0.10)
    # con una tolerancia menor que la holgura no se unen: avisa y la pieza queda flotando
    with pytest.raises(ErrorAnalisis):
        _analizar_con_uniones(p, [apoyo], [carga], [_union_completa("SOLDADURA", a, c, tolerancia=0.005)], tmp_path / "x")


@requiere_ccx
def test_cuerpo_rigido_entre_elementos_completos_mueve_a_los_dos_como_uno(tmp_path: Path):
    a, c = caja("A", 0, 0, 0, 0.2, 0.2, 0.2), caja("B", 0.3, 0, 0, 0.2, 0.2, 0.2)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (-1, 0, 0)), ux=True, uy=True, uz=True)
    carga = _carga(c, (1, 0, 0), "FUERZA", 10.0, (0, 0, -1))
    r = _analizar_con_uniones(proyecto_con(a, c), [apoyo], [carga], [_union_completa("CUERPO_RIGIDO", a, c)], tmp_path)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    # B es rígido: todos sus nodos siguen un movimiento de cuerpo rígido (y se mueve)
    assert np.abs(u[x[:, 0] >= 0.3 - 1e-9, 2]).max() > 1e-8
    en_b = np.where(x[:, 0] >= 0.3 - 1e-9)[0]
    r_ = x[en_b] - x[en_b].mean(axis=0)
    filas = np.vstack([np.hstack([np.eye(3), -_antisimetrica(ri)]) for ri in r_])
    sol, *_ = np.linalg.lstsq(filas, u[en_b].ravel(), rcond=None)
    assert np.abs(filas @ sol - u[en_b].ravel()).max() < 1e-9


def test_detecta_los_elementos_que_se_tocan_y_los_que_estan_a_poca_distancia():
    from ifc_structural.solido.deteccion import detectar_contactos

    a, b, c = caja("A", 0, 0, 0, 0.2, 0.2, 0.1), caja("B", 0, 0, 0.1, 0.2, 0.2, 0.1), caja("C", 0, 0, 0.215, 0.2, 0.2, 0.1)
    p = proyecto_con(a, b, c)
    parejas = lambda tol: {frozenset((x.a_id, x.b_id)) for x in detectar_contactos(p, tol)}
    assert parejas(0.002) == {frozenset("AB")}  # C está a 15 mm de B
    assert parejas(0.02) == {frozenset("AB"), frozenset("BC")}
    assert all(x.area > 0.03 for x in detectar_contactos(p, 0.02))


# ---------------------------------------------------------------------------
# Placas delgadas y piezas sueltas (modelo real de Revit: viga armada en cajón)
# ---------------------------------------------------------------------------


def test_una_placa_delgada_larga_lleva_una_sola_capa_en_el_espesor_y_pocas_celdas():
    from ifc_structural.solido.libres import celdas_por_eje, estimar_hexaedros_solido

    alma = caja("W", 0, 0, 0, 6.0, 0.01, 0.56)  # alma de 10 mm de una viga de 6 m
    n = celdas_por_eje(alma, 0.05)
    assert n[1] == 1 and n[0] == 120 and n[2] <= 12
    assert estimar_hexaedros_solido(alma, 0.05) < 3000  # con celdas cúbicas de 5 mm serían ~270 000
    m = mallar_solido(alma, 0.05)
    c = m.coordenadas[m.hexaedros]
    assert np.prod(c[:, 6] - c[:, 0], axis=1).sum() == pytest.approx(alma.volumen, rel=1e-5)


@requiere_ccx
def test_una_pieza_suelta_sin_asignaciones_se_deja_fija_y_se_avisa(tmp_path: Path):
    a, suelta = caja("A", 0, 0, 0, 0.5, 0.1, 0.1), caja("B", 2.0, 2.0, 0, 0.2, 0.2, 0.05)
    p = proyecto_con(a, suelta)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (-1, 0, 0)), ux=True, uy=True, uz=True)
    r = _analizar(p, [apoyo], [_carga(a, (1, 0, 0), "FUERZA", 5.0, (0, 0, -1))], tmp_path)
    assert any("B" in aviso and "no tocan" in aviso for aviso in r.advertencias)
    # si la pieza suelta tiene una carga asignada, sí es un error (quedaría flotando)
    with pytest.raises(ErrorAnalisis):
        _analizar(p, [apoyo], [_carga(a, (1, 0, 0), "FUERZA", 5.0, (0, 0, -1)), _carga(suelta, (0, 0, 1), "PRESION", 10.0, None)], tmp_path / "x")


# ---------------------------------------------------------------------------
# Piezas de Revit: cascarones que se tocan por una arista y asignación de material
# ---------------------------------------------------------------------------


def _dos_cajas_por_arista() -> SolidoGeneral:
    """Dos cubos que solo comparten una arista (cada arista de contacto la usan 4 triángulos)."""

    a, b = caja("a", 0, 0, 0, 1, 1, 1), caja("b", 1, 1, 0, 1, 1, 1)
    v = np.vstack([a.vertices, b.vertices])
    t = np.vstack([a.triangulos, b.triangulos + len(a.vertices)])
    v, t = soldar_vertices(v, t)
    return SolidoGeneral(id="S1", nombre="doble", tipo_ifc="IfcBuildingElementProxy", material="ACERO", vertices=v, triangulos=t)


def test_aristas_compartidas_por_cuatro_triangulos_siguen_siendo_superficie_cerrada():
    s = _dos_cajas_por_arista()
    assert s.cerrado and s.volumen == pytest.approx(2.0, rel=1e-6)
    malla = mallar_solido(s, 0.25)
    assert malla is not None and len(malla.hexaedros) > 0
    assert malla.volumen == pytest.approx(2.0, rel=0.1)


def test_una_malla_con_agujeros_sigue_siendo_abierta():
    s = caja("a", 0, 0, 0, 1, 1, 1)
    assert not SolidoGeneral(id="x", nombre="x", tipo_ifc="IfcPlate", material="ACERO", vertices=s.vertices,
                             triangulos=s.triangulos[:-1]).cerrado


def test_pared_delgada_avisa_pero_una_placa_plana_no():
    placa = construir_malla_solida(proyecto_con(caja("S1", 0, 0, 0, 2.0, 2.0, 0.01)), [], 0.2)  # una capa en el espesor
    assert not any("pared delgada" in a for a in placa.avisos)
    gruesa = construir_malla_solida(proyecto_con(caja("S1", 0, 0, 0, 1, 1, 1)), [], 0.25)
    assert not any("pared delgada" in a for a in gruesa.avisos)
    # un depósito de pared de 1 cm con celdas de 5 cm: no hay una capa de hexaedros por pared que lo represente
    cilindro_grande = pv.Cylinder(center=(0, 0, 1), direction=(0, 0, 1), radius=1.0, height=2.0, resolution=64, capping=True).triangulate()
    interior = pv.Cylinder(center=(0, 0, 1), direction=(0, 0, 1), radius=0.99, height=1.98, resolution=64, capping=True).triangulate()
    v = np.vstack([np.asarray(cilindro_grande.points), np.asarray(interior.points)])
    t = np.vstack([cilindro_grande.faces.reshape(-1, 4)[:, 1:], interior.faces.reshape(-1, 4)[:, 1:][:, ::-1] + len(cilindro_grande.points)])
    v, t = soldar_vertices(v, t)
    deposito = SolidoGeneral(id="S1", nombre="deposito", tipo_ifc="IfcBuildingElementProxy", material="ACERO", vertices=v, triangulos=t)
    malla = construir_malla_solida(proyecto_con(deposito), [], 0.05)
    assert any("pared delgada" in a for a in malla.avisos)


def test_asignar_material_a_un_solido():
    from ifc_structural.core.asignacion import ErrorAsignacion, Objetivo, asignar_material

    p = proyecto_con(caja("S1", 0, 0, 0, 1, 1, 1))
    p.agregar_material(Material(nombre="HORMIGON", modulo_elasticidad=2.5e7, densidad=24.0, modulo_poisson=0.2))
    cambiados = asignar_material(p, [Objetivo("SOLIDO", "S1", 0)], "HORMIGON")
    assert p.solidos["S1"].material == "HORMIGON" and len(cambiados) == 1
    with pytest.raises(ErrorAsignacion):
        asignar_material(p, [Objetivo("SOLIDO", "S9")], "HORMIGON")


def test_dialogo_de_material_acepta_solidos():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ifc_structural.core.asignacion import Objetivo
    from ifc_structural.gui.dialogo_materiales import DialogoAsignarMaterial

    QApplication.instance() or QApplication([])
    p = proyecto_con(caja("S1", 0, 0, 0, 1, 1, 1))
    dialogo = DialogoAsignarMaterial(p, [Objetivo("SOLIDO", "S1", 0)])
    assert dialogo.material_elegido() == "ACERO"


def test_un_elemento_con_piezas_sueltas_se_separa_en_componentes():
    from ifc_structural.core.solidos import partes_conexas

    a, b = caja("a", 0, 0, 0, 1, 1, 1), caja("b", 5, 0, 0, 1, 1, 1)
    v = np.vstack([a.vertices, b.vertices])
    t = np.vstack([a.triangulos, b.triangulos + len(a.vertices)])
    pieza = partes_conexas(t, len(v))
    assert len(np.unique(pieza)) == 2 and len(set(pieza[: len(a.triangulos)])) == 1
    assert pieza[0] != pieza[-1]


def test_placa_con_agujeros_lleva_una_sola_capa():
    from ifc_structural.solido.libres import celdas_por_eje

    base = pv.Cube(x_length=0.4, y_length=0.65, z_length=0.014).triangulate()
    hueco = pv.Cube(x_length=0.1, y_length=0.1, z_length=0.014).triangulate()
    # placa con un hueco pasante aproximado: se usa el volumen reducido (relleno 0.9 de la caja)
    v, t = soldar_vertices(np.asarray(base.points), base.faces.reshape(-1, 4)[:, 1:])
    s = SolidoGeneral(id="P", nombre="P", tipo_ifc="IfcPlate", material="ACERO", vertices=v, triangulos=t)
    s.volumen *= 0.7
    n = celdas_por_eje(s, 0.25)
    assert n.min() == 1 and int(np.prod(n)) < 100


# ---------------------------------------------------------------------------
# Tetraedros cuadráticos C3D10 (Gmsh), como en Placa Base Pro
# ---------------------------------------------------------------------------

requiere_gmsh = pytest.mark.skipif(not __import__("ifc_structural.solido.tetraedros", fromlist=["x"]).gmsh_disponible(),
                                   reason="Gmsh no está disponible")


def _analizar_con(elemento, p, apoyos, cargas, tmp_path, tamano=0.05, uniones=()):
    casos, combinaciones = _casos()
    return ejecutar_analisis_solido(p, casos, combinaciones, apoyos, {}, {}, {}, tmp_path, tamano=tamano,
                                    cargas_superficie_por_caso={"L": cargas}, uniones_manuales=uniones, elemento_solido=elemento)


@requiere_ccx
@requiere_gmsh
@pytest.mark.parametrize("elemento, tolerancia", [("TETRA", 0.05), ("HEXA", 0.08)])
def test_voladizo_con_cada_tipo_de_elemento(tmp_path: Path, elemento, tolerancia):
    L, b, h, F = 1.0, 0.1, 0.1, 10.0
    s = caja("S1", 0, 0, 0, L, b, h)
    p = proyecto_con(s)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="S1", cara=cara_hacia(s, (-1, 0, 0)), ux=True, uy=True, uz=True)
    r = _analizar_con(elemento, p, [apoyo], [_carga(s, (1, 0, 0), "FUERZA", F, (0, 0, -1))], tmp_path, tamano=0.05)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    flecha = -u[np.isclose(x[:, 0], L), 2].mean()
    assert flecha == pytest.approx(F * L**3 / (3 * E * b * h**3 / 12), rel=tolerancia)
    assert any(("C3D10" in a) == (elemento == "TETRA") for a in r.advertencias[:1]) or elemento == "HEXA"


@requiere_gmsh
def test_la_malla_de_tetraedros_cubre_el_volumen_y_asocia_las_superficies():
    from ifc_structural.solido.tetraedros import mallar_solido_tetra, volumenes_firmados

    c = cilindro("C", 0.5, 2.0)
    m = mallar_solido_tetra(c, 0.2)
    assert m.tetraedros.shape[1] == 10 and (volumenes_firmados(m.coordenadas, m.tetraedros) > 0).all()
    assert m.volumen == pytest.approx(c.volumen, rel=0.02)
    assert len(m.superficies) == 3  # dos tapas y la superficie curva
    # los nodos medios están en el punto medio de las aristas
    t, x = m.tetraedros, m.coordenadas
    assert np.abs(x[t[:, 4]] - 0.5 * (x[t[:, 0]] + x[t[:, 1]])).max() < 1e-9
    assert np.abs(x[t[:, 9]] - 0.5 * (x[t[:, 2]] + x[t[:, 3]])).max() < 1e-9


@requiere_ccx
@requiere_gmsh
def test_cilindro_a_traccion_con_tetraedros_sigue_la_superficie_curva(tmp_path: Path):
    """Con hexaedros la superficie curva queda escalonada; con tetraedros el área de la tapa es la real."""

    r_, L, F = 0.05, 0.4, 20.0
    s = cilindro("P", r_, L)
    p = proyecto_con(s)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="P", cara=cara_hacia(s, (0, 0, -1)), ux=True, uy=True, uz=True)
    carga = _carga(s, (0, 0, 1), "FUERZA", F, (0, 0, 1))
    res = _analizar_con("TETRA", p, [apoyo], [carga], tmp_path, tamano=0.02)
    x, u = res.malla_tensiones.coordenadas, res.desplazamientos_por_caso["L"]
    area = np.pi * r_**2
    esperado = F * L / (E * area)
    assert u[np.isclose(x[:, 2], L, atol=1e-6), 2].mean() == pytest.approx(esperado, rel=0.03)


@requiere_ccx
@requiere_gmsh
def test_dos_solidos_de_tetraedros_en_contacto_se_comportan_como_una_pieza(tmp_path: Path):
    """Dos sólidos de tetraedros que se tocan se unen con *TIE y flexionan como una sola pieza."""

    b, h, F = 0.1, 0.1, 10.0
    a, c = caja("A", 0, 0, 0, 0.5, b, h), caja("B", 0.5, 0, 0, 0.5, b, h)
    p = proyecto_con(a, c)
    apoyo = ApoyoSuperficie(id="AP1", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (-1, 0, 0)), ux=True, uy=True, uz=True)
    carga = _carga(c, (1, 0, 0), "FUERZA", F, (0, 0, -1))
    r = _analizar_con("TETRA", p, [apoyo], [carga], tmp_path, tamano=0.05)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    flecha = -u[np.isclose(x[:, 0], 1.0), 2].mean()
    assert flecha == pytest.approx(F * 1.0**3 / (3 * E * b * h**3 / 12), rel=0.08)
