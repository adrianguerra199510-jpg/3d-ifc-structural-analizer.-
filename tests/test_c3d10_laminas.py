"""Análisis con solo tetraedros cuadráticos C3D10 y láminas S6: criterio de lámina, superficie media vuelta a mallar
con Gmsh, un solo proceso de Gmsh para todas las piezas (lanzar uno por pieza hace saltar a los antivirus), conversión
de barras, losas y muros a C3D10, piezas solapadas unidas por proximidad y tamaño de elemento sugerido."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ifc_structural.ifc.importador import importar_ifc
from ifc_structural.solido import tetraedros as T
from ifc_structural.solido.analisis_solido import TAMANOS, ejecutar_analisis_solido, estimar, tamano_sugerido
from ifc_structural.solido.laminas import NoEsLamina, es_lamina, mallar_lamina, superficie_media
from ifc_structural.solido.malla import construir_malla_solida
from ifc_structural.solido.miembros import construir_miembros, recortar_uniones

from .ayuda_ifc import crear_ifc_portico
from .test_solidos_libres import E, _carga, _casos, caja, cara_hacia, cilindro, proyecto_con

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")
requiere_gmsh = pytest.mark.skipif(not T.gmsh_disponible(), reason="Gmsh no está disponible")


def _calidad(coordenadas, triangulos):
    p = coordenadas[triangulos]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    l2 = sum(((p[:, (k + 1) % 3] - p[:, k]) ** 2).sum(axis=1) for k in range(3))
    return 4 * np.sqrt(3) * area / l2


def _cono_en_abanico(radio=1.0, altura=1.5, espesor=0.01, n=48):
    """Pared cónica (abierta arriba y abajo) triangulada en abanico desde la generatriz, como las exporta Revit: cada
    piel son tiras de dos triángulos larguísimos de arriba abajo."""

    from ifc_structural.core.solidos import SolidoGeneral

    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r_arriba = radio + 0.6
    capas = []
    for dr in (0.0, espesor):  # piel exterior e interior
        abajo = np.column_stack([(radio - dr) * np.cos(a), (radio - dr) * np.sin(a), np.zeros(n)])
        arriba = np.column_stack([(r_arriba - dr) * np.cos(a), (r_arriba - dr) * np.sin(a), np.full(n, altura)])
        capas.append((abajo, arriba))
    v = np.vstack([capas[0][0], capas[0][1], capas[1][0], capas[1][1]])
    ea, eb, ia, ib = 0, n, 2 * n, 3 * n
    t = []
    for k in range(n):
        j = (k + 1) % n
        t += [(ea + k, ea + j, eb + j), (ea + k, eb + j, eb + k)]  # exterior
        t += [(ia + k, ib + j, ia + j), (ia + k, ib + k, ib + j)]  # interior
        t += [(ea + k, ia + j, ea + j), (ea + k, ia + k, ia + j)]  # canto inferior
        t += [(eb + k, eb + j, ib + j), (eb + k, ib + j, ib + k)]  # canto superior
    s = SolidoGeneral(id="CONO", nombre="Cono", tipo_ifc="IfcPlate", material="ACERO", vertices=v, triangulos=np.array(t))
    if s.volumen < 0:
        s = SolidoGeneral(id="CONO", nombre="Cono", tipo_ifc="IfcPlate", material="ACERO", vertices=v,
                          triangulos=np.array(t)[:, ::-1].copy())
    return s


# ---------------------------------------------------------------------------
# Criterio de lámina
# ---------------------------------------------------------------------------


def test_una_chapa_es_lamina_y_un_bloque_no():
    assert es_lamina(caja("C", 0, 0, 0, 1.0, 0.4, 0.01))  # 10 mm frente a ~0,63 m de lado equivalente
    assert not es_lamina(caja("B", 0, 0, 0, 0.3, 0.3, 0.3))
    assert not es_lamina(caja("P", 0, 0, 0, 0.3, 0.3, 0.05))  # placa gruesa: 50 mm frente a 0,33 m


def test_una_barra_redonda_esbelta_pasa_el_criterio_pero_se_reconoce_maciza():
    barra = cilindro("R", 0.01, 3.0, resolucion=24)  # 2V/A ≈ radio: parece delgada frente a su área
    assert es_lamina(barra)
    with pytest.raises(NoEsLamina):
        superficie_media(barra, 0.05)


# ---------------------------------------------------------------------------
# Un solo proceso de Gmsh para todas las piezas
# ---------------------------------------------------------------------------


@requiere_gmsh
def test_todas_las_piezas_se_mallan_con_un_solo_proceso(monkeypatch):
    lanzados = []
    original = subprocess.Popen

    def contar(*args, **kwargs):
        lanzados.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(T.subprocess, "Popen", contar)
    chapa = caja("C", 0, 0, 0, 1.0, 0.4, 0.01)
    media = superficie_media(chapa, 0.1)
    piezas = [(f"B{i}", caja("x", i, 0, 0, 0.2, 0.2, 0.2).vertices, caja("x", i, 0, 0, 0.2, 0.2, 0.2).triangulos) for i in range(4)]
    piezas.append(("C", media.vertices, media.triangulos, 2))
    resultado = T.mallar_lote_gmsh(piezas, 0.1)
    assert len(lanzados) == 1
    for i in range(4):
        coordenadas, tetra = resultado[f"B{i}"]
        assert tetra.shape[1] == 10 and (T.volumenes_firmados(coordenadas, tetra) > 0).all()
    coordenadas, triangulos = resultado["C"]
    assert triangulos.shape[1] == 3 and _calidad(coordenadas, triangulos).min() > 0.3


_HIJO_FALSO = '''
import sys, time
from pathlib import Path
import numpy as np
args = sys.argv[1:]
entrada, carpeta = Path(args[1]), Path(args[2])
desde = int(args[4]) if len(args) > 4 else 0
datos = np.load(entrada)
n = int(datos["n"])
for i in range(desde, n):
    (carpeta / "actual.txt").write_text(str(i))
    if i == 1:
        time.sleep(600)  # se cuelga con la pieza 1
    if i == 2:
        (carpeta / f"err_{i}.txt").write_text("Gmsh falló a propósito")
        continue
    v = datos[f"v{i}"]
    np.savez(carpeta / f"res_{i}.npz", coordenadas=v, tetraedros=np.zeros((0, 10), dtype=np.int64))
(carpeta / "actual.txt").write_text(str(n))
'''


@requiere_gmsh
def test_una_pieza_que_cuelga_gmsh_no_detiene_las_demas_y_se_relanza_una_sola_vez(monkeypatch, tmp_path):
    guion = tmp_path / "hijo.py"
    guion.write_text(_HIJO_FALSO, encoding="utf-8")
    lanzados = []

    def comando(argumentos):
        lanzados.append(list(argumentos))
        return [sys.executable, str(guion), *argumentos], None

    monkeypatch.setattr(T, "_comando_hijo", comando)
    c = caja("x", 0, 0, 0, 0.2, 0.2, 0.2)
    resultado = T.mallar_lote_gmsh([(f"P{i}", c.vertices, c.triangulos) for i in range(4)], 0.1, tiempo_por_pieza=3.0)
    assert len(lanzados) == 2 and lanzados[1][-2:] == ["--desde", "2"]  # se relanza desde la pieza siguiente
    assert "no terminó" in resultado["P1"]
    assert resultado["P2"] == "Gmsh falló a propósito"
    assert not isinstance(resultado["P0"], str) and not isinstance(resultado["P3"], str)


# ---------------------------------------------------------------------------
# Láminas: superficie media vuelta a mallar
# ---------------------------------------------------------------------------


@requiere_gmsh
def test_la_pared_en_abanico_se_vuelve_a_mallar_con_triangulos_casi_equilateros():
    cono = _cono_en_abanico()
    assert es_lamina(cono)
    media = superficie_media(cono, 0.15)
    assert _calidad(media.vertices, media.triangulos).max() < 0.3  # de partida, tiras larguísimas

    def remallar(v, t, h):
        return T.mallar_lote_gmsh([("m", v, t, 2)], h)["m"]

    m = mallar_lamina(cono, 0.15, remallar)
    q = _calidad(m.coordenadas, m.elementos[:, :3])
    assert np.median(q) > 0.85 and q.min() > 0.3
    assert np.allclose(m.espesores, 0.01 * 1.5 / np.hypot(1.5, 0.6), rtol=0.03)  # espesor normal a la pared inclinada
    assert m.volumen == pytest.approx(cono.volumen, rel=0.03)
    # normales coherentes: todas hacia fuera del cono (o todas hacia dentro)
    centro = m.coordenadas[m.elementos[:, :3]].mean(axis=1)
    radial = np.einsum("ij,ij->i", m.normales[:, :2], centro[:, :2])
    assert (radial > 0).all() or (radial < 0).all()


def test_sin_gmsh_la_lamina_se_refina_por_biseccion():
    m = mallar_lamina(caja("C", 0, 0, 0.5, 1.0, 0.4, 0.01), 0.1)  # remallar=None
    assert m.elementos.shape[1] == 6 and np.allclose(m.espesores, 0.01, atol=1e-6)


# ---------------------------------------------------------------------------
# Todo en C3D10
# ---------------------------------------------------------------------------


@requiere_ccx
def test_el_portico_se_analiza_solo_con_tetraedros_cuadraticos(tmp_path: Path):
    from ifc_structural.core.cargas import Apoyo, CargaPuntual

    crear_ifc_portico(str(tmp_path / "portico.ifc"))
    p = importar_ifc(str(tmp_path / "portico.ifc"))
    miembros = construir_miembros(p)
    recortar_uniones(p, miembros)
    malla = construir_malla_solida(p, miembros, 0.1, None, "TETRA", usar_laminas=True)
    assert len(malla.hexaedros) == 0 and len(malla.tetraedros) > 0
    for parte in malla.partes:
        if parte.tipo == "BARRA":
            # cada anillo lleva los nodos medios de las aristas de la sección; los pesos (solo en los medios) suman el área
            assert all(len(a) == len(parte.pesos) for a in parte.anillos)
            area = p.perfiles[parte.miembro.perfil].area
            assert parte.pesos.sum() == pytest.approx(area, rel=0.02)
            assert 0 < np.count_nonzero(parte.pesos) < len(parte.pesos)  # los vértices de la sección no llevan carga
    # análisis completo: equilibrio de una carga puntual en una viga
    casos, combinaciones = _casos()
    base = [n for n, nodo in p.nodos.items() if abs(nodo.z - min(m.z for m in p.nodos.values())) < 1e-6]
    apoyos = [Apoyo(id=f"A{n}", nodo_id=n) for n in base]
    viga = next(iter(p.vigas))
    carga = CargaPuntual(id="P1", caso_id="L", elemento_id=viga, posicion=0.5, fz=-10.0)
    r = ejecutar_analisis_solido(p, casos, combinaciones, apoyos, {}, {}, {"L": [carga]}, tmp_path / "ccx", tamano=0.1)
    assert "C3D10" in r.advertencias[0]
    inp = (tmp_path / "ccx" / "solido_L.inp").read_text()
    assert "TYPE=C3D10" in inp and "C3D8" not in inp
    u = r.desplazamientos_por_caso["L"]
    assert np.isfinite(u).all() and u[:, 2].min() < 0


def test_una_losa_con_un_vertice_alineado_no_da_tetraedros_planos():
    from ifc_structural.core.modelo import Losa, Nodo
    from ifc_structural.solido.tetraedros import volumenes_firmados

    p = proyecto_con()
    for i, (x, y) in enumerate([(0, 0), (3, 0), (6, 0), (6, 4), (0, 4)]):  # (3, 0) está alineado: ángulo de 180°
        p.nodos[f"N{i}"] = Nodo(id=f"N{i}", x=float(x), y=float(y), z=0.0)
    p.losas["L1"] = Losa(id="L1", nodos_contorno=[f"N{i}" for i in range(5)], espesor=0.25, material="ACERO")
    malla = construir_malla_solida(p, [], 1.0, None, "TETRA")
    v = volumenes_firmados(malla.coordenadas, malla.tetraedros)
    assert len(malla.hexaedros) == 0 and v.min() > 1e-3 * v.mean()
    assert v.sum() == pytest.approx(6 * 4 * 0.25, rel=1e-9)


# ---------------------------------------------------------------------------
# Piezas solapadas (una tuerca metida en la placa)
# ---------------------------------------------------------------------------


@requiere_ccx
@requiere_gmsh
def test_una_pieza_metida_en_otra_se_une_por_proximidad(tmp_path: Path):
    """El taco B entra 1 cm en el bloque A: cede el volumen común y su borde allí es un corte sin caras planas
    enfrentadas. Se une igual y la carga en B llega al apoyo de A."""

    a = caja("A", 0, 0, 0, 0.4, 0.4, 0.2)
    b = caja("B", 0.12, 0.12, 0.19, 0.16, 0.16, 0.16)
    p = proyecto_con(a, b)
    apoyo = __import__("ifc_structural.core.cargas", fromlist=["x"]).ApoyoSuperficie(
        id="AP", elemento_tipo="SOLIDO", elemento_id="A", cara=cara_hacia(a, (0, 0, -1)), ux=True, uy=True, uz=True)
    carga = _carga(b, (0, 0, 1), "FUERZA", 5.0, (1, 0, 0))
    casos, combinaciones = _casos()
    r = ejecutar_analisis_solido(p, casos, combinaciones, [apoyo], {}, {}, {}, tmp_path, tamano=0.05,
                                 cargas_superficie_por_caso={"L": [carga]})
    assert not any("se dejaron fijos" in x for x in r.advertencias)
    u = r.desplazamientos_por_caso["L"]
    x = r.malla_tensiones.coordenadas
    arriba = x[:, 2] > 0.34
    assert u[arriba, 0].mean() > 0  # el taco se mueve con la carga, unido al bloque
    assert np.abs(u).max() < 1e-3


# ---------------------------------------------------------------------------
# Estimación y tamaño sugerido
# ---------------------------------------------------------------------------


@requiere_gmsh
def test_la_estimacion_se_parece_a_la_malla_real():
    p = proyecto_con(caja("B", 0, 0, 0, 0.6, 0.4, 0.3), cilindro("P", 0.03, 0.4), caja("C", 0, 0, 1.0, 1.0, 0.5, 0.01))
    for h in (0.03, 0.06):
        malla = construir_malla_solida(p, [], h, None, "TETRA", usar_laminas=True)
        real = len(malla.tetraedros) + len(malla.laminas)
        assert 0.6 * real <= estimar(p, h) <= 1.6 * real


def test_el_tamano_sugerido_no_engorda_los_elementos_sin_ganar_nada(monkeypatch):
    from ifc_structural.solido import analisis_solido as A

    # muchas piezas pequeñas: el número de elementos casi no baja a partir de cierto tamaño
    cuenta = {h: int(50_000 + 1e3 / h) for h in TAMANOS}
    monkeypatch.setattr(A, "estimar_hexaedros", lambda proyecto, miembros, h, tetra: cuenta[h])
    h = tamano_sugerido(proyecto_con(), objetivo=40_000)
    assert h < TAMANOS[-1] and cuenta[h] <= 1.15 * cuenta[TAMANOS[-1]]
    assert tamano_sugerido(proyecto_con(), objetivo=10**9) == TAMANOS[0]
