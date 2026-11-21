"""Validación del modelo antes de analizar: ¿va a correr el análisis y va a converger?"""

import shutil
from pathlib import Path

import pytest

from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.casos_carga import CasoCarga, Combinacion, TipoCaso
from ifc_structural.core.materiales import material_desde_valores
from ifc_structural.solido.validacion_modelo import AVISO, ERROR, validar_modelo_solido

from .test_solidos_libres import _carga, caja, cara_hacia, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def _casos():
    return {"L": CasoCarga("L", "Viva", TipoCaso.L)}, {"C": Combinacion("C", "1.6L", {"L": 1.6})}


def _apoyo(s, direccion, id_="AP", **grados):
    return ApoyoSuperficie(id=id_, elemento_tipo="SOLIDO", elemento_id=s.id, cara=cara_hacia(s, direccion), **grados)


def _textos(informe, nivel=None, seccion=None):
    return [h.texto for h in informe.hallazgos if (nivel is None or h.nivel == nivel) and (seccion is None or h.seccion == seccion)]


def test_un_modelo_bien_apoyado_es_valido():
    s = caja("V", 0, 0, 0, 1.0, 0.1, 0.1)
    casos, comb = _casos()
    informe = validar_modelo_solido(proyecto_con(s), casos, comb, [_apoyo(s, (-1, 0, 0))],
                                    {"L": [_carga(s, (0, 0, 1), "FUERZA", 10.0, (0, 0, -1))]}, tamano=0.05)
    assert informe.correcto, _textos(informe, ERROR)
    assert any("los 6 movimientos" in t for t in _textos(informe, seccion="Estabilidad"))
    assert any("(0.00; 0.00; -10.00) kN" in t for t in _textos(informe, seccion="Cargas"))


def test_apoyos_que_dejan_movimientos_libres_son_un_error():
    s = caja("Z", 0, 0, 0, 2.0, 2.0, 0.5)
    casos, comb = _casos()
    # zapata apoyada solo en Uz: puede deslizar en X e Y y girar alrededor de Z
    informe = validar_modelo_solido(proyecto_con(s), casos, comb, [_apoyo(s, (0, 0, -1), ux=False, uy=False, uz=True)],
                                    {"L": [_carga(s, (0, 0, 1), "PRESION", 10.0, None)]}, tamano=0.25)
    errores = _textos(informe, ERROR, "Estabilidad")
    assert len(errores) == 1
    assert all(m in errores[0] for m in ("traslación en X", "traslación en Y", "giro alrededor de Z"))


def test_dos_pasadores_alineados_dejan_libre_el_giro_alrededor_de_su_recta():
    s = caja("V", 0, 0, 0, 2.0, 0.1, 0.1)
    casos, comb = _casos()
    apoyos = [_apoyo(s, (-1, 0, 0), "A", pasador=True), _apoyo(s, (1, 0, 0), "B", pasador=True)]
    informe = validar_modelo_solido(proyecto_con(s), casos, comb, apoyos,
                                    {"L": [_carga(s, (0, 0, 1), "FUERZA", 10.0, (0, 0, -1))]}, tamano=0.05)
    errores = _textos(informe, ERROR, "Estabilidad")
    assert len(errores) == 1 and "giro alrededor de X" in errores[0]


def test_una_pieza_cargada_que_no_toca_nada_flota_y_una_suelta_se_avisa():
    s, flota, suelta = caja("S", 0, 0, 0, 1, 1, 0.1), caja("F", 3, 0, 0, 0.2, 0.2, 0.2), caja("X", 6, 0, 0, 0.2, 0.2, 0.2)
    casos, comb = _casos()
    informe = validar_modelo_solido(proyecto_con(s, flota, suelta), casos, comb, [_apoyo(s, (0, 0, -1))],
                                    {"L": [_carga(flota, (0, 0, 1), "FUERZA", 1.0, (0, 0, -1))]}, tamano=0.05)
    errores = [h for h in informe.errores if h.seccion == "Conectividad"]
    assert len(errores) == 1 and "F" in errores[0].texto and [o.id for o in errores[0].objetivos] == ["F"]
    assert any("X" in h.texto and h.nivel == AVISO for h in informe.hallazgos if h.seccion == "Conectividad")


def test_datos_y_materiales_incorrectos():
    s = caja("V", 0, 0, 0, 1.0, 0.1, 0.1)
    p = proyecto_con(s)
    s.material = "NO_EXISTE"
    casos, comb = _casos()
    malo = ApoyoSuperficie(id="AP", elemento_tipo="SOLIDO", elemento_id="BORRADO", cara=0)
    informe = validar_modelo_solido(p, casos, comb, [_apoyo(s, (-1, 0, 0)), malo], {}, tamano=0.05)
    assert any("NO_EXISTE" in t for t in _textos(informe, ERROR, "Materiales"))
    assert any("ya no existe" in t for t in _textos(informe, ERROR, "Datos"))
    assert any("no tiene cargas ni peso propio" in t for t in _textos(informe, AVISO, "Cargas"))


@requiere_ccx
def test_la_prueba_con_calculix_estima_si_el_analisis_elastoplastico_convergera(tmp_path: Path):
    s = caja("B", 0, 0, 0, 1.0, 0.1, 0.1)
    p = proyecto_con(s)
    p.materiales["ACERO"] = material_desde_valores("ACERO", 200000, 0.3, 0.0, 250.0)
    casos = {"L": CasoCarga("L", "Viva", TipoCaso.L)}
    apoyos = [_apoyo(s, (-1, 0, 0))]
    for fuerza, esperado in ((1000.0, "no llega a fy"), (4000.0, "plastificación amplia")):  # 100 MPa y 400 MPa
        informe = validar_modelo_solido(p, casos, {}, apoyos, {"L": [_carga(s, (1, 0, 0), "FUERZA", fuerza, (1, 0, 0))]},
                                        tamano=0.05, no_lineal=True, prueba_ccx=True, directorio_trabajo=tmp_path / str(fuerza))
        textos = " ".join(_textos(informe, seccion="CalculiX"))
        assert "resolvió" in textos and esperado in textos
    assert not informe.correcto  # con el análisis no lineal pedido, la plastificación amplia es un error


def test_el_informe_muestra_los_hallazgos_y_selecciona_las_piezas():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ifc_structural.core.asignacion import Objetivo
    from ifc_structural.gui.dialogo_validacion import DialogoValidacion
    from ifc_structural.solido.validacion_modelo import CORRECTO, InformeValidacion

    QApplication.instance() or QApplication([])
    informe = InformeValidacion()
    informe.agregar(ERROR, "Estabilidad", "El modelo es inestable.", [Objetivo("SOLIDO", "Z")])
    informe.agregar(CORRECTO, "Cargas", "Caso D: resultante.")
    d = DialogoValidacion(informe)
    assert d.boton_analizar is None and d.boton_prueba is None  # con errores no se ofrece analizar ni la prueba
    antes = DialogoValidacion(informe, antes_de_analizar=True)
    assert antes.boton_analizar.text() == "Analizar de todos modos"
    pedidos = []
    d.seleccion_pedida.connect(pedidos.append)
    estabilidad = d.arbol.topLevelItem(0)
    assert estabilidad.text(0) == "Estabilidad"
    d._al_doble_clic(estabilidad.child(0), 0)
    assert pedidos == [[Objetivo("SOLIDO", "Z")]]
    bien = InformeValidacion()
    bien.agregar(CORRECTO, "Cargas", "ok")
    d = DialogoValidacion(bien)
    assert d.boton_analizar.text() == "Analizar" and d.boton_prueba is not None


def test_la_validacion_malla_y_avisa_si_calculix_no_tendra_memoria(monkeypatch):
    import ifc_structural.solido.validacion_modelo as vm

    s = caja("V", 0, 0, 0, 1.0, 0.1, 0.1)
    casos, comb = _casos()
    cargas = {"L": [_carga(s, (0, 0, 1), "FUERZA", 10.0, (0, 0, -1))]}
    informe = validar_modelo_solido(proyecto_con(s), casos, comb, [_apoyo(s, (-1, 0, 0))], cargas, tamano=0.05, mallar=True)
    assert informe.correcto and informe.malla_previa is not None  # ANALIZAR la vuelve a usar
    assert any("GB de memoria" in t for t in _textos(informe, seccion="Malla"))
    monkeypatch.setattr(vm, "memoria_disponible_mb", lambda: 100.0)  # un equipo con 100 MB
    informe = validar_modelo_solido(proyecto_con(s), casos, comb, [_apoyo(s, (-1, 0, 0))], cargas, tamano=0.05, mallar=True)
    assert any("sin memoria" in t for t in _textos(informe, ERROR, "Malla"))


def test_memoria_estimada_de_calculix():
    from ifc_structural.solido.validacion_modelo import memoria_calculix_mb, memoria_disponible_mb

    # puntos medidos (sólido macizo de 34 651 nodos; lámina de 51 253 nodos): orden de magnitud
    assert memoria_calculix_mb(34651, 0) == pytest.approx(857, rel=0.1)
    assert memoria_calculix_mb(0, 51253) == pytest.approx(1060, rel=0.1)
    # la plataforma (unos 470 000 nodos de lámina y 115 000 de sólido) no cabía en 16 GB
    assert memoria_calculix_mb(115000, 470000) > 16 * 1024
    assert (memoria_disponible_mb() or 1.0) > 0


def test_calculix_detenido_por_el_sistema_se_explica_como_falta_de_memoria():
    from ifc_structural.solver.ejecutor_ccx import ResultadoEjecucion

    r = ResultadoEjecucion(ruta_inp=Path("x.inp"), codigo_salida=-9, salida="", exitoso=False)
    assert "falta de memoria" in r.diagnostico()
