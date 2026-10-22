"""Constraints (Conectar) y uniones entre piezas: casos que fallaban.

* La superficie media de un cajón (el anillo inferior del balde) quedaba con grietas en las esquinas.
* «Conectar automáticamente» medía sobre vóxeles y no veía los contactos inclinados.
* Con dos elementos elegidos, las caras «más cercanas» se medían entre vértices.
* Cuerpos rígidos encadenados (A–B y B–C) o junto a una soldadura automática dejaban nodos dependientes dos veces.
* La tolerancia de una soldadura con holgura variable salía de la distancia mínima.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.core.asignacion import Objetivo
from ifc_structural.core.asignacion_superficie import caras_mas_cercanas, union_para
from ifc_structural.core.cargas import ApoyoSuperficie
from ifc_structural.core.solidos import SolidoGeneral
from ifc_structural.solido.analisis_solido import ejecutar_analisis_solido
from ifc_structural.solido.deteccion import detectar_contactos
from ifc_structural.solido.laminas import _aristas_libres, _cerrar_grietas

from .test_solidos_libres import _carga, _casos, caja, cara_hacia, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def _tira(filas, columnas):
    """Triángulos de una rejilla de ``filas`` × ``columnas`` vértices (índices por filas)."""

    t = []
    for i in range(filas - 1):
        for j in range(columnas - 1):
            a, b, c, d = i * columnas + j, i * columnas + j + 1, (i + 1) * columnas + j + 1, (i + 1) * columnas + j
            t += [[a, b, c], [a, c, d]]
    return np.array(t)


def test_se_cosen_las_grietas_de_esquina_de_la_superficie_media():
    # fondo (z = 0) y pared (y = -5 mm) de un cajón de 10 mm: sus superficies medias terminan a 7 mm una de la otra y con
    # los vértices a distinta distancia a lo largo del canto (11 en el fondo, 9 en la pared)
    xa, xb = np.linspace(0, 1, 11), np.linspace(0, 1, 9)
    fondo = np.array([[x, y, 0.0] for y in (0.0, 0.2) for x in xa])
    pared = np.array([[x, -0.005, z] for z in (0.005, 0.2) for x in xb])
    v = np.vstack([fondo, pared])
    t = np.vstack([_tira(2, 11), _tira(2, 9) + len(fondo)])
    antes = _aristas_libres(t)
    vm, tm, _, _ = _cerrar_grietas(v, t, np.arange(len(t)), np.full(len(v), 0.0105))
    libres = _aristas_libres(tm)
    centro = vm[libres].mean(axis=1)
    en_el_canto = (np.abs(centro[:, 1]) < 0.01) & (np.abs(centro[:, 2]) < 0.01)
    assert len(antes) > len(libres) and not en_el_canto.any()  # el canto quedó cosido: solo quedan los bordes de verdad
    # el área no cambia (los triángulos solo se parten)
    def area(vv, tt):
        p = vv[tt]
        return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum()
    assert area(vm, tm) == pytest.approx(area(v, t), rel=0.02)


def _girar(s: SolidoGeneral, grados: float) -> SolidoGeneral:
    a = np.radians(grados)
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return SolidoGeneral(id=s.id, nombre=s.nombre, tipo_ifc=s.tipo_ifc, material=s.material,
                         vertices=np.asarray(s.vertices) @ r.T, triangulos=s.triangulos)


def test_conectar_automaticamente_ve_contactos_inclinados_y_solapes():
    # dos bloques girados 30° que se tocan cara con cara, y un perno metido en el segundo
    a = _girar(caja("A", 0, 0, 0, 0.3, 0.2, 0.1), 30)
    b = _girar(caja("B", 0.3, 0, 0, 0.3, 0.2, 0.1), 30)
    ang = np.radians(30)
    centro_b = np.array([0.45 * np.cos(ang) - 0.1 * np.sin(ang), 0.45 * np.sin(ang) + 0.1 * np.cos(ang), 0.0])
    perno = caja("P", centro_b[0] - 0.01, centro_b[1] - 0.01, -0.1, 0.02, 0.02, 0.3)  # atraviesa B por el medio
    contactos = {frozenset((c.a_id, c.b_id)): c for c in detectar_contactos(proyecto_con(a, b, perno), 0.002)}
    # área aproximada: cuentan también las franjas de las caras vecinas a menos de la separación del borde del contacto
    assert frozenset("AB") in contactos and contactos[frozenset("AB")].area == pytest.approx(0.2 * 0.1, rel=0.25)
    assert frozenset("BP") in contactos and contactos[frozenset("BP")].separacion == 0.0  # solapan
    assert frozenset("AP") not in contactos


def test_las_caras_mas_cercanas_se_miden_entre_superficies_no_entre_vertices():
    # placa pequeña sobre una losa grande, junto a su borde: el centro de la cara lateral de la losa queda más cerca de los
    # vértices de la placa que cualquier vértice de la cara superior
    losa, placa = caja("L", 0, 0, 0, 4.0, 4.0, 0.2), caja("P", 1.8, 0.1, 0.2, 0.3, 0.3, 0.02)
    a, b = caras_mas_cercanas(proyecto_con(losa, placa), Objetivo("SOLIDO", "L"), Objetivo("SOLIDO", "P"))
    assert a.cara == cara_hacia(losa, (0, 0, 1)) and b.cara == cara_hacia(placa, (0, 0, -1))


def _completa(p, tipo, x, y):
    return union_para(p, Objetivo("SOLIDO", x), Objetivo("SOLIDO", y), f"CT{x}{y}", tipo, (True,) * 6, completo=True)


@requiere_ccx
def test_cuerpos_rigidos_encadenados_se_funden_en_uno(tmp_path: Path):
    a, b, c = caja("A", 0, 0, 0, 0.2, 0.2, 0.2), caja("B", 0.2, 0, 0, 0.2, 0.2, 0.2), caja("C", 0.4, 0, 0, 0.2, 0.2, 0.2)
    p = proyecto_con(a, b, c)
    apoyo = ApoyoSuperficie(id="AP", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (-1, 0, 0)), ux=True, uy=True, uz=True)
    uniones = [_completa(p, "CUERPO_RIGIDO", "A", "B"), _completa(p, "CUERPO_RIGIDO", "B", "C")]
    r = ejecutar_analisis_solido(p, *_casos(), [apoyo], {}, {}, {}, tmp_path, tamano=0.1,
                                 cargas_superficie_por_caso={"L": [_carga(c, (1, 0, 0), "FUERZA", 10.0, (0, 0, -1))]},
                                 uniones_manuales=uniones, barridos=False)
    assert any("se fundieron en uno solo" in x for x in r.advertencias)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    assert np.isfinite(u).all()
    # B y C forman con A un único cuerpo rígido (sin los nodos con apoyo, que no pueden depender de él: la primera capa de
    # A se deforma un poco): todo lo que está más allá del apoyo se mueve como un sólido rígido
    lejos = np.where(x[:, 0] > 0.1 + 1e-6)[0]
    r_ = x[lejos] - x[lejos].mean(axis=0)
    filas = np.vstack([np.hstack([np.eye(3), -np.array([[0, -ri[2], ri[1]], [ri[2], 0, -ri[0]], [-ri[1], ri[0], 0]])]) for ri in r_])
    sol, *_ = np.linalg.lstsq(filas, u[lejos].ravel(), rcond=None)
    assert np.abs(filas @ sol - u[lejos].ravel()).max() < 1e-3 * np.abs(u[lejos]).max()


@requiere_ccx
def test_un_cuerpo_rigido_no_toma_los_nodos_esclavos_de_una_soldadura(tmp_path: Path):
    # B se apoya en la placa A (unión automática: la cara de B es la esclava) y se une a C con un cuerpo rígido completo
    a, b, c = caja("A", -0.2, -0.2, 0, 0.6, 0.6, 0.05), caja("B", 0, 0, 0.05, 0.2, 0.2, 0.2), caja("C", 0, 0, 0.26, 0.2, 0.2, 0.1)
    p = proyecto_con(a, b, c)
    apoyo = ApoyoSuperficie(id="AP", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (0, 0, -1)), ux=True, uy=True, uz=True)
    r = ejecutar_analisis_solido(p, *_casos(), [apoyo], {}, {}, {}, tmp_path, tamano=0.05,
                                 cargas_superficie_por_caso={"L": [_carga(c, (0, 0, 1), "FUERZA", 10.0, (1, 0, 0))]},
                                 uniones_manuales=[_completa(p, "CUERPO_RIGIDO", "B", "C")], barridos=False)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    assert np.isfinite(u).all() and np.abs(u).max() > 0
    # la placa trabaja: su cara superior se mueve bajo B (el cuerpo rígido B+C está soldado a la placa)
    assert np.abs(u[np.isclose(x[:, 2], 0.05) & (np.abs(x[:, 0] - 0.1) < 0.1), 0]).max() > 0


@requiere_ccx
def test_soldadura_con_holgura_variable_une_toda_la_superficie(tmp_path: Path):
    # B inclinado sobre A: toca en un borde y se abre 10 mm en el otro
    a = caja("A", 0, 0, 0, 0.3, 0.3, 0.05)
    b = caja("B", 0, 0, 0.05, 0.3, 0.3, 0.05)
    v = np.asarray(b.vertices, dtype=float).copy()
    v[:, 2] += np.where(v[:, 2] < 0.075, 0.01 * v[:, 0] / 0.3, 0.0)  # la cara inferior se levanta hacia +x
    b = SolidoGeneral(id="B", nombre="B", tipo_ifc="IfcPlate", material="ACERO", vertices=v, triangulos=b.triangulos)
    p = proyecto_con(a, b)
    apoyo = ApoyoSuperficie(id="AP", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (0, 0, -1)), ux=True, uy=True, uz=True)
    union = union_para(p, Objetivo("SOLIDO", "A", cara_hacia(a, (0, 0, 1))), Objetivo("SOLIDO", "B", cara_hacia(b, (0, 0, -1))),
                       "CT1", "SOLDADURA", (True,) * 6)
    r = ejecutar_analisis_solido(p, *_casos(), [apoyo], {}, {}, {}, tmp_path, tamano=0.05,
                                 cargas_superficie_por_caso={"L": [_carga(b, (0, 0, 1), "FUERZA", 10.0, (1, 0, 0))]},
                                 uniones_manuales=[union], barridos=False)
    assert not any("no se une" in x for x in r.advertencias)
    x, u = r.malla_tensiones.coordenadas, r.desplazamientos_por_caso["L"]
    assert np.isfinite(u).all() and np.abs(u).max() < 1e-3  # B no queda suelto


def test_el_reporte_sale_en_el_formato_elegido_aunque_el_nombre_diga_pdf():
    pytest.importorskip("PySide6")
    from ifc_structural.gui.ventana_principal import ruta_con_formato

    assert ruta_con_formato("C:/x/balde_reporte.pdf", "Word (*.docx)") == "C:/x/balde_reporte.docx"
    assert ruta_con_formato("C:/x/balde_reporte.docx", "PDF (*.pdf)") == "C:/x/balde_reporte.pdf"
    assert ruta_con_formato("C:/x/balde", "Word (*.docx)") == "C:/x/balde.docx"
    assert ruta_con_formato("C:/x/balde.PDF", "") == "C:/x/balde.pdf"
