"""Sin límite de elementos (solver directo de CalculiX), estimación de memoria y rutas largas."""

import shutil
from pathlib import Path

import pytest

from ifc_structural.solido.analisis_solido import memoria_disponible, memoria_solver
from ifc_structural.solver.ejecutor_ccx import ejecutar_ccx

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def test_memoria_del_solver_directo():
    assert memoria_solver(96_500) == pytest.approx(2.95e9)  # el balde: 193 000 nodos, medido
    assert memoria_solver(400_000) > 2 * memoria_solver(200_000)  # crece más que los elementos
    libre = memoria_disponible()
    assert libre is None or libre > 0


def test_el_dialogo_no_impide_analizar_mallas_grandes():
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from ifc_structural.gui.dialogo_analisis_3d import DialogoAnalisis3D

    from .test_solidos_libres import caja, proyecto_con

    QApplication.instance() or QApplication([])
    dialogo = DialogoAnalisis3D(proyecto_con(caja("S1", 0, 0, 0, 3.0, 3.0, 3.0)))
    dialogo.spin_tamano.setValue(dialogo.spin_tamano.minimum())  # millones de elementos
    assert dialogo.botones.button(QDialogButtonBox.Ok).isEnabled()
    assert "GB" in dialogo.estimacion.text()


@requiere_ccx
def test_calculix_funciona_con_rutas_largas(tmp_path: Path):
    """CalculiX rechaza nombres de más de 127 caracteres: se le pasa solo el nombre, desde la carpeta del .inp."""

    carpeta = tmp_path
    while len(str(carpeta)) < 160:
        carpeta = carpeta / "carpeta_de_usuario_con_un_nombre_largo"
    carpeta.mkdir(parents=True)
    inp = carpeta / "modelo.inp"
    inp.write_text("*NODE\n1,0,0,0\n2,1,0,0\n3,0,1,0\n4,0,0,1\n*ELEMENT,TYPE=C3D4,ELSET=E\n1,1,2,3,4\n*MATERIAL,NAME=M\n"
                   "*ELASTIC\n210000,0.3\n*SOLID SECTION,ELSET=E,MATERIAL=M\n*BOUNDARY\n1,1,3\n2,2,3\n3,3,3\n*STEP\n*STATIC\n"
                   "*CLOAD\n4,3,1.\n*NODE FILE\nU\n*END STEP\n", encoding="utf-8")
    assert ejecutar_ccx(inp).exitoso


@requiere_ccx
@pytest.mark.parametrize("tipo", ["IGUALAR", "CUERPO_RIGIDO"])
def test_una_lamina_acoplada_y_con_un_constraint_no_repite_dependientes(tmp_path: Path, tipo):
    """Una chapa (lámina) sobre un bloque se acopla sola por *EQUATION; si además hay un constraint entre las dos, CalculiX
    abortaba con «the DOF ... is detected on the dependent side of two different MPC's»."""

    import numpy as np

    from ifc_structural.core.cargas import ApoyoSuperficie
    from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido

    from .test_solidos_libres import _carga, _casos, _union, _union_completa, caja, cara_hacia, proyecto_con

    bloque, chapa = caja("B", 0, 0, 0, 0.3, 0.3, 0.2), caja("C", 0, 0, 0.2, 0.3, 0.3, 0.005)
    apoyo = ApoyoSuperficie(id="A", elemento_tipo="SOLIDO", elemento_id="B", cara=cara_hacia(bloque, (0, 0, -1)),
                            ux=True, uy=True, uz=True)
    union = (_union("IGUALAR", chapa, (0, 0, -1), bloque, (0, 0, 1)) if tipo == "IGUALAR"
             else _union_completa("CUERPO_RIGIDO", chapa, bloque))
    r = ejecutar_analisis_solido(proyecto_con(bloque, chapa), *_casos(), [apoyo], {}, {}, {}, tmp_path, tamano=0.05,
                                 cargas_superficie_por_caso={"L": [_carga(chapa, (0, 0, 1), "PRESION", 10.0, None)]},
                                 uniones_manuales=[union])
    assert np.isfinite(r.desplazamientos_por_caso["L"]).all()
