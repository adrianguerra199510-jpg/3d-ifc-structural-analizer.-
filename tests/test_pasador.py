"""Apoyo tipo pasador: se fija el centro de la superficie (acople distribuido), no cada nodo.

Con todos los nodos fijos, la superficie del agujero queda pegada a un perno infinitamente rígido: la pieza no puede
girar alrededor del perno ni el agujero deformarse, y aparecen picos de esfuerzo que no existen.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.asignacion import Objetivo
from ifc_structural.core.asignacion_superficie import apoyo_superficie_para
from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.persistencia import cargar_proyecto, guardar_proyecto
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido

from .test_solidos_libres import E, _carga, _casos, caja, cara_hacia, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")

L, B, H, W = 2.0, 0.1, 0.1, 10.0  # m, m, m, kN


def _apoyo(s, direccion, pasador, ux=True, uy=True, uz=True, id_="AP"):
    return ApoyoSuperficie(id=id_, elemento_tipo="SOLIDO", elemento_id=s.id, cara=cara_hacia(s, direccion),
                           ux=ux, uy=uy, uz=uz, pasador=pasador)


def _flecha(tmp_path, apoyos, s):
    p = proyecto_con(s)
    r = ejecutar_analisis_solido(p, *_casos(), apoyos, {}, {}, {}, tmp_path, tamano=0.05,
                                 cargas_superficie_por_caso={"L": [_carga(s, (0, 0, 1), "FUERZA", W, (0, 0, -1))]},
                                 barridos=False)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    centro = np.argsort(np.linalg.norm(x - [L / 2, B / 2, H / 2], axis=1))[:4]
    return -float(u[centro, 2].mean()), r


@requiere_ccx
def test_viga_sobre_dos_pasadores_es_simplemente_apoyada(tmp_path: Path):
    s = caja("V", 0, 0, 0, L, B, H)
    inercia = B * H ** 3 / 12
    # pasador fijo en un extremo, deslizante en x en el otro; el fondo no se mueve de lado (impide el giro de toda la
    # viga alrededor de la recta de los dos pasadores, que estos dejan libre)
    apoyos = [_apoyo(s, (-1, 0, 0), True, id_="A"), _apoyo(s, (1, 0, 0), True, ux=False, id_="B"),
              _apoyo(s, (0, 0, -1), False, ux=False, uy=True, uz=False, id_="T")]
    flecha, r = _flecha(tmp_path / "pasador", apoyos, s)
    teoria = 5 * W * L ** 3 / (384 * E * inercia)  # carga repartida, simplemente apoyada
    cortante = W * L / (8 * (5 / 6) * E / 2.6 * B * H)
    assert flecha == pytest.approx(teoria + cortante, rel=0.08)
    # con los extremos «empotrados» (todos los nodos fijos) la viga es biempotrada: unas cinco veces más rígida
    fijos = [_apoyo(s, (-1, 0, 0), False, id_="A"), _apoyo(s, (1, 0, 0), False, id_="B")]
    flecha_fija, _ = _flecha(tmp_path / "fija", fijos, s)
    assert flecha / flecha_fija > 3.0


@requiere_ccx
def test_los_pasadores_alineados_avisan_que_el_modelo_puede_girar(tmp_path: Path):
    s = caja("V", 0, 0, 0, L, B, H)
    apoyos = [_apoyo(s, (-1, 0, 0), True, id_="A"), _apoyo(s, (1, 0, 0), True, ux=False, id_="B"),
              _apoyo(s, (0, 0, -1), False, ux=False, uy=True, uz=False, id_="T")]
    # con el apoyo lateral del fondo hay otro apoyo además de los pasadores: no hay aviso
    _, r = _flecha(tmp_path / "a", apoyos, s)
    assert not any("misma recta" in a for a in r.advertencias)
    from ifc_structural.solido.generador import Apoyo3D, ModeloSolido, escribir_comun
    from ifc_structural.solido.malla import construir_malla_solida

    malla = construir_malla_solida(proyecto_con(s), [], 0.05)
    x = malla.coordenadas
    extremo = [np.flatnonzero(np.isclose(x[:, 0], v)) for v in (0.0, L)]
    modelo = ModeloSolido(malla=malla, uniones=[], apoyos=[Apoyo3D(n, {1, 2, 3}, pasador=True, nombre=f"P{k}")
                                                           for k, n in enumerate(extremo)],
                          materiales={"ACERO": "ACERO"}, enlaces=[], acoples=[])
    texto = "\n".join(escribir_comun(modelo, proyecto_con(s)))
    assert any("misma recta" in a for a in modelo.avisos)
    # 3 ecuaciones por pasador, cada una con todos los nodos de su cara y pesos que suman 1
    bloque = texto.split("*EQUATION\n")[1].split("\n*")[0].splitlines()
    assert sum(1 for linea in bloque if "," not in linea) == 6 and "*BOUNDARY" not in texto


def test_el_pasador_se_crea_sin_giros_y_se_guarda(tmp_path: Path):
    s = caja("V", 0, 0, 0, L, B, H)
    p = proyecto_con(s)
    a = apoyo_superficie_para(p, Objetivo("SOLIDO", "V", cara_hacia(s, (-1, 0, 0))), "AP1", (True,) * 6, pasador=True)
    assert a.pasador and a.grados_restringidos() == [1, 2, 3] and "pasador" in a.nombre
    ruta = tmp_path / "p.json"
    guardar_proyecto(ruta, p, [a], *_casos(), {}, {}, {}, {})
    assert cargar_proyecto(ruta)[1][0].pasador


def test_el_dialogo_ofrece_el_pasador_con_los_giros_libres():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from ifc_structural.gui.dialogo_superficie import PRESET_PASADOR, DialogoApoyoSuperficie

    QApplication.instance() or QApplication([])
    s = caja("V", 0, 0, 0, L, B, H)
    d = DialogoApoyoSuperficie(proyecto_con(s), [Objetivo("SOLIDO", "V", cara_hacia(s, (-1, 0, 0)))])
    d.combo_preset.setCurrentText(PRESET_PASADOR)
    assert d.pasador and d.restricciones() == (True, True, True, False, False, False)
    assert not d.checks[3].isEnabled() and "RBE3" in d.nota.text()
    d.checks[0].setChecked(False)  # sigue siendo un pasador, deslizante en X
    assert d.pasador and d.restricciones() == (False, True, True, False, False, False)
    creados, errores = d.apoyos(lambda: "AP1")
    assert not errores and creados[0].pasador and not creados[0].ux
