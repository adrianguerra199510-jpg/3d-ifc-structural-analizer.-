"""Conversión de hexaedros ``C3D8`` en tetraedros ``C3D10``: tabla de divisiones, conformidad, volúmenes, nodos medios,
caras y un voladizo calculado con CalculiX de verdad."""

import itertools
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pytest

from ifc_structural.solido.hex_a_tet import (CARAS_HEXAEDRO, CARAS_TETRAEDRO, MEDIOS_CARA, _convertir, _tabla,
                                             divisiones_de_patron, hexaedros_a_tetraedros, patrones_diagonales)

requiere_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")

CUBO = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _rejilla(nx, ny, nz, dx=1.0, dy=1.0, dz=1.0):
    """Cuadrícula de vóxeles: (coordenadas, hexaedros) con el orden de nodos de ``C3D8``."""

    x, y, z = np.meshgrid(np.arange(nx + 1) * dx, np.arange(ny + 1) * dy, np.arange(nz + 1) * dz, indexing="ij")
    coordenadas = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    def nodo(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    i, j, k = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
    hexaedros = np.stack([nodo(i, j, k), nodo(i + 1, j, k), nodo(i + 1, j + 1, k), nodo(i, j + 1, k),
                          nodo(i, j, k + 1), nodo(i + 1, j, k + 1), nodo(i + 1, j + 1, k + 1), nodo(i, j + 1, k + 1)], axis=1)
    return coordenadas, hexaedros


def _volumenes(coordenadas, tetraedros):
    p = coordenadas[tetraedros[:, :4]]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def _area_triangulos(p):
    return 0.5 * np.linalg.norm(np.cross(p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 0, :]), axis=-1)


def _quads_de_borde(hexaedros):
    """(hexaedro, cara 0..5) de las caras de hexaedro que no comparte ningún otro."""

    todas = hexaedros[:, np.array(CARAS_HEXAEDRO)]  # (M, 6, 4)
    _, primero, cuenta = np.unique(np.sort(todas.reshape(-1, 4), axis=1), axis=0, return_index=True, return_counts=True)
    return np.divmod(primero[cuenta == 1], 6)


def _volumen_encerrado(coordenadas, hexaedros):
    """Volumen encerrado por el borde de la malla de hexaedros con cada cuadrilátero partido por la diagonal de su
    nodo menor (teorema de la divergencia; las caras de ``CARAS_HEXAEDRO`` miran hacia dentro)."""

    h, c = _quads_de_borde(hexaedros)
    q = hexaedros[h][np.arange(len(h))[:, None], np.array(CARAS_HEXAEDRO)[c]]  # (n, 4)
    pos = np.argmin(q, axis=1) % 2
    tri = np.where(pos[:, None, None] == 0, q[:, [[0, 1, 2], [0, 2, 3]]], q[:, [[0, 1, 3], [1, 2, 3]]]).reshape(-1, 3)
    centro = coordenadas.mean(axis=0)
    p = coordenadas[tri] - centro
    return -np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0


def _comprobar(coordenadas, hexaedros, conv, volumen=None, calidad_minima=0.02):
    """Comprobaciones comunes de una conversión: nodos originales, volúmenes, conformidad y nodos medios."""

    n, m = len(coordenadas), len(hexaedros)
    tet = conv.tetraedros
    assert tet.shape[1] == 10 and tet.dtype.kind == "i"
    # los nodos originales conservan su índice y sus coordenadas
    assert conv.n_originales == n
    np.testing.assert_array_equal(conv.coordenadas[:n], coordenadas)
    # volúmenes positivos y no degenerados, y volumen total conservado
    v = _volumenes(conv.coordenadas, tet)
    assert v.min() > 0.0
    assert v.min() > 1e-3 * v.mean(), "tetraedro casi plano"
    assert conv.calidad.min() > calidad_minima
    if volumen is not None:
        assert v.sum() == pytest.approx(volumen, rel=1e-10, abs=1e-12)
    # cada hexaedro tiene sus tetraedros seguidos y solo usa sus nodos (y su centro, si lo lleva)
    np.testing.assert_array_equal(conv.hexaedro_de_tetraedro, np.repeat(np.arange(m), np.diff(conv.primer_tetraedro)))
    assert set(np.diff(conv.primer_tetraedro).tolist()) <= {5, 6, 12}
    propios = np.concatenate([hexaedros, conv.centro_de_hexaedro[:, None]], axis=1)[conv.hexaedro_de_tetraedro]
    assert (tet[:, :4, None] == propios[:, None, :]).any(axis=2).all()
    assert (conv.centro_de_hexaedro >= 0).sum() == conv.n_centros
    assert np.array_equal(np.diff(conv.primer_tetraedro) == 12, conv.centro_de_hexaedro >= 0)

    # conformidad: cada triángulo está en 1 tetraedro (borde) o en 2 (interior)
    caras = np.sort(tet[:, np.array(CARAS_TETRAEDRO)].reshape(-1, 3), axis=1)
    unicas, cuenta = np.unique(caras, axis=0, return_counts=True)
    assert set(cuenta.tolist()) <= {1, 2}
    borde = unicas[cuenta == 1]
    # ... y los triángulos de borde cubren exactamente el borde de la malla de hexaedros: 2 por cuadrilátero
    h, c = _quads_de_borde(hexaedros)
    quads = hexaedros[h][np.arange(len(h))[:, None], np.array(CARAS_HEXAEDRO)[c]]
    assert len(borde) == 2 * len(quads)
    de_quad = {}
    for k, q in enumerate(quads.tolist()):
        for tri in itertools.combinations(sorted(q), 3):
            de_quad[tri] = k
    por_quad = {}
    for tri in map(tuple, borde.tolist()):
        assert tri in de_quad, f"triángulo de borde {tri} fuera del borde de los hexaedros"
        por_quad.setdefault(de_quad[tri], []).append(set(tri))
    assert len(por_quad) == len(quads)
    for k, (t1, t2) in por_quad.items():
        assert t1 | t2 == set(quads[k].tolist())
        assert min(quads[k]) in t1 & t2, "la diagonal no pasa por el nodo menor de la cara"
    p = conv.coordenadas[borde]
    area_quads = 0.5 * np.linalg.norm(np.cross(coordenadas[quads[:, 2]] - coordenadas[quads[:, 0]],
                                               coordenadas[quads[:, 3]] - coordenadas[quads[:, 1]]), axis=1)
    if np.allclose(np.einsum("ij,ij->i", np.cross(coordenadas[quads[:, 1]] - coordenadas[quads[:, 0]],
                                                   coordenadas[quads[:, 2]] - coordenadas[quads[:, 0]]),
                             coordenadas[quads[:, 3]] - coordenadas[quads[:, 0]]), 0.0, atol=1e-12):  # cuadriláteros planos
        assert _area_triangulos(p).sum() == pytest.approx(area_quads.sum(), rel=1e-10)

    # nodos medios: uno por arista, en su punto medio exacto, detrás de los vértices
    esquinas = tet[:, :4]
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))):
        np.testing.assert_allclose(conv.coordenadas[tet[:, 4 + k]],
                                   0.5 * (conv.coordenadas[esquinas[:, i]] + conv.coordenadas[esquinas[:, j]]), atol=1e-12)
    a = esquinas[:, [0, 1, 2, 0, 1, 2]].ravel()
    b = esquinas[:, [1, 2, 0, 3, 3, 3]].ravel()
    arista = np.minimum(a, b) * len(conv.coordenadas) + np.maximum(a, b)
    pares = np.unique(np.stack([arista, tet[:, 4:].ravel()], axis=1), axis=0)
    assert len(pares) == len(np.unique(arista)) == len(np.unique(tet[:, 4:]))
    assert tet[:, :4].max() < conv.primer_medio <= tet[:, 4:].min()
    assert len(conv.coordenadas) == conv.primer_medio + len(conv.aristas)
    np.testing.assert_array_equal(conv.medio_de_arista(a, b), tet[:, 4:].ravel())
    return v


def _bloque_deformado(nx, ny, nz, amplitud, semilla):
    """Cuadrícula con la numeración de nodos permutada al azar y los nodos movidos un poco: los interiores en las
    tres direcciones y los del borde solo dentro de su cara (el volumen total sigue siendo nx·ny·nz)."""

    rng = np.random.default_rng(semilla)
    coordenadas, hexaedros = _rejilla(nx, ny, nz)
    libre = (coordenadas > 1e-9) & (coordenadas < np.array([nx, ny, nz]) - 1e-9)
    coordenadas = coordenadas + libre * rng.uniform(-amplitud, amplitud, coordenadas.shape)
    permutacion = rng.permutation(len(coordenadas))  # nodo nuevo -> nodo viejo
    nuevo = np.argsort(permutacion)
    return coordenadas[permutacion], nuevo[hexaedros][rng.permutation(len(hexaedros))]


# ---------------------------------------------------------------------------
# Tabla de divisiones del cubo
# ---------------------------------------------------------------------------


def test_tabla_de_divisiones_del_cubo():
    divisiones, sin_centro, con_centro = _tabla()
    assert sum(len(d) for d in sin_centro) == 74  # triangulaciones del cubo sin vértices nuevos
    assert sum(1 for d in sin_centro if d) == 46
    sin_division = [p for p in range(64) if not sin_centro[p]]
    assert len(sin_division) == 18
    for patron in range(64):
        for t in divisiones_de_patron(patron):
            assert len(t) in (5, 6)
            v = _volumenes(CUBO, t)
            assert v.min() > 0 and v.sum() == pytest.approx(1.0)
        assert len(divisiones[con_centro[patron]].tetraedros) == 12


def test_regla_del_nodo_menor_siempre_admite_division_sin_centro():
    """Con la diagonal del nodo menor, el nodo menor del hexaedro tiene las diagonales de sus tres caras y siempre hay
    división en 5 o 6 tetraedros: el centro solo hace falta en hexaedros muy deformados."""

    _, sin_centro, _ = _tabla()
    numeraciones = np.array(list(itertools.permutations(range(8))))
    alcanzables = np.unique(patrones_diagonales(numeraciones))
    assert len(alcanzables) == 42
    assert all(sin_centro[p] for p in alcanzables.tolist())


@pytest.mark.parametrize("patron", range(64))
def test_cada_patron_en_el_cubo(patron):
    """Con un patrón de diagonales impuesto: volumen 1, tetraedros positivos y caras partidas según el patrón."""

    conv = _convertir(CUBO, np.arange(8)[None], np.array([patron]))
    v = _volumenes(conv.coordenadas, conv.tetraedros)
    assert v.min() > 0 and v.sum() == pytest.approx(1.0)
    _, sin_centro, _ = _tabla()
    if sin_centro[patron]:
        assert len(v) in (5, 6) and conv.n_centros == 0
    else:
        assert len(v) == 12 and conv.n_centros == 1
        np.testing.assert_allclose(conv.coordenadas[8], [0.5, 0.5, 0.5])
    for f, cara in enumerate(CARAS_HEXAEDRO):
        a, b, c, d = cara
        diagonal = {a, c} if not (patron >> f) & 1 else {b, d}
        triangulos = [set(conv.tetraedros[t, list(CARAS_TETRAEDRO[k - 1])].tolist())
                      for t, k in conv.cara_hexaedro_a_tetraedros(0, f + 1)]
        assert triangulos[0] | triangulos[1] == set(cara) and triangulos[0] & triangulos[1] == diagonal


# ---------------------------------------------------------------------------
# Mallas
# ---------------------------------------------------------------------------


def test_cubo():
    conv = hexaedros_a_tetraedros(CUBO, np.arange(8)[None])
    _comprobar(CUBO, np.arange(8)[None], conv, volumen=1.0)
    assert len(conv.tetraedros) in (5, 6) and conv.n_centros == 0


def test_rejilla_de_voxeles_3x3x3():
    coordenadas, hexaedros = _rejilla(3, 3, 3, 0.5, 0.4, 0.3)
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    _comprobar(coordenadas, hexaedros, conv, volumen=1.5 * 1.2 * 0.9)
    assert conv.n_centros == 0


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_bloque_con_numeracion_aleatoria_y_deformado(semilla):
    coordenadas, hexaedros = _bloque_deformado(5, 4, 3, 0.2, semilla)
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    _comprobar(coordenadas, hexaedros, conv, volumen=60.0)


def test_bloque_deformado_sin_caras_planas():
    """Con todos los nodos movidos (también los del borde) las caras dejan de ser planas: el volumen es el que
    encierran los triángulos del borde."""

    rng = np.random.default_rng(7)
    coordenadas, hexaedros = _rejilla(4, 3, 3)
    coordenadas = coordenadas + rng.uniform(-0.15, 0.15, coordenadas.shape)
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    _comprobar(coordenadas, hexaedros, conv, volumen=_volumen_encerrado(coordenadas, hexaedros))


def test_barrido_curvo_de_una_seccion():
    """Hexaedros de una barra curva (como los de ``malla._malla_barra``: anillos de la sección seguidos)."""

    u, v = np.meshgrid(np.linspace(-0.05, 0.05, 4), np.linspace(-0.1, 0.1, 5), indexing="ij")
    seccion = np.stack([u.ravel(), v.ravel()], axis=1)
    nu, nv = 4, 5
    # sentido antihorario visto desde el avance (u × v = -tangente, por eso se recorren al revés)
    quads = np.array([[i * nv + j + 1, (i + 1) * nv + j + 1, (i + 1) * nv + j, i * nv + j]
                      for i in range(nu - 1) for j in range(nv - 1)])
    radio, estaciones = 2.0, np.linspace(0.0, np.pi / 2, 25)
    anillos = [np.stack([(radio + seccion[:, 0]) * np.cos(a), (radio + seccion[:, 0]) * np.sin(a), seccion[:, 1]], axis=1)
               for a in estaciones]
    coordenadas = np.vstack(anillos)
    n2 = len(seccion)
    hexaedros = np.concatenate([np.concatenate([quads + k * n2, quads + (k + 1) * n2], axis=1)
                                for k in range(len(estaciones) - 1)])
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    v = _comprobar(coordenadas, hexaedros, conv, volumen=_volumen_encerrado(coordenadas, hexaedros), calidad_minima=0.005)
    assert v.sum() == pytest.approx(0.1 * 0.2 * radio * np.pi / 2, rel=2e-3)  # Pappus
    # los nodos de un anillo más sus nodos medios son los de esa sección
    anillo = conv.ampliar_con_medios(np.arange(5 * n2, 6 * n2))
    angulo = np.arctan2(conv.coordenadas[:, 1], conv.coordenadas[:, 0])
    np.testing.assert_array_equal(anillo, np.flatnonzero(np.isclose(angulo, estaciones[5], atol=1e-12)))


def test_capas_finas_de_losa():
    """Losa de 1 × 1 m y 2 cm de espesor en 2 capas, con los cuadriláteros de un polígono no rectangular."""

    puntos2d = np.array([[0, 0], [1, 0], [1.2, 0.9], [0.1, 1.1], [0.55, 0.5], [0.5, 0.0], [1.1, 0.45], [0.65, 1.0],
                         [0.05, 0.55]], dtype=float)
    quads = np.array([[0, 5, 4, 8], [5, 1, 6, 4], [4, 6, 2, 7], [8, 4, 7, 3]])
    capas = [np.c_[puntos2d, np.full(len(puntos2d), z)] for z in (0.0, 0.01, 0.02)]
    coordenadas = np.vstack(capas)
    n2 = len(puntos2d)
    hexaedros = np.concatenate([np.concatenate([quads + k * n2, quads + (k + 1) * n2], axis=1) for k in range(2)])
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    x, y = puntos2d[:4, 0], puntos2d[:4, 1]  # contorno: los 4 primeros puntos
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    _comprobar(coordenadas, hexaedros, conv, volumen=0.02 * area, calidad_minima=1e-3)


def test_numeracion_invertida():
    """Hexaedros numerados al revés (cara inferior en sentido horario): se da la vuelta a sus tetraedros."""

    coordenadas, hexaedros = _rejilla(2, 2, 2)
    invertidos = hexaedros[:, [0, 3, 2, 1, 4, 7, 6, 5]]
    assert _volumen_encerrado(coordenadas, invertidos) == pytest.approx(-8.0)
    conv = hexaedros_a_tetraedros(coordenadas, invertidos)
    _comprobar(coordenadas, invertidos, conv, volumen=8.0, calidad_minima=0.3)
    # la cara 1 de cada hexaedro (ahora 1-4-3-2) sigue saliendo en sus triángulos
    for h in range(len(invertidos)):
        tri = [set(conv.tetraedros[t, list(CARAS_TETRAEDRO[c - 1])].tolist()) for t, c in conv.cara_hexaedro_a_tetraedros(h, 1)]
        assert tri[0] | tri[1] == set(invertidos[h, :4].tolist())


def test_hexaedro_torcido_usa_el_centro():
    """Cara superior girada 60°: la única división en tetraedros sin punto interior compatible con sus diagonales
    tiene un tetraedro invertido; se añade el centro y se une con los 12 triángulos de las caras."""

    b = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    giro = np.radians(60.0)
    arriba = b @ np.array([[np.cos(giro), np.sin(giro)], [-np.sin(giro), np.cos(giro)]])
    coordenadas = np.vstack([np.c_[b, np.zeros(4)], np.c_[arriba, np.ones(4)]])
    hexaedros = np.arange(8)[None]
    patron = int(patrones_diagonales(hexaedros)[0])
    assert all(_volumenes(coordenadas, t).min() < 0 for t in divisiones_de_patron(patron))
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    assert conv.n_centros == 1 and len(conv.tetraedros) == 12
    np.testing.assert_allclose(conv.coordenadas[8], coordenadas.mean(axis=0))
    _comprobar(coordenadas, hexaedros, conv, volumen=_volumen_encerrado(coordenadas, hexaedros))


def test_detecta_tetraedros_degenerados():
    """Un hexaedro aplastado (cara superior sobre la inferior) da tetraedros planos: las comprobaciones lo detectan."""

    coordenadas = CUBO.copy()
    coordenadas[4:, 2] = 1e-9
    conv = hexaedros_a_tetraedros(coordenadas, np.arange(8)[None])
    assert conv.calidad.min() < 1e-3
    with pytest.raises(AssertionError):
        _comprobar(coordenadas, np.arange(8)[None], conv)


def test_malla_vacia():
    coordenadas = np.random.default_rng(0).random((5, 3))
    conv = hexaedros_a_tetraedros(coordenadas, np.zeros((0, 8), dtype=int))
    assert conv.tetraedros.shape == (0, 10) and conv.caras_de_hexaedro.shape == (0, 6, 2, 2)
    np.testing.assert_array_equal(conv.coordenadas, coordenadas)


def test_nodos_inexistentes():
    with pytest.raises(ValueError):
        hexaedros_a_tetraedros(CUBO, np.arange(1, 9)[None])


def test_constantes_iguales_a_las_de_la_malla():
    from ifc_structural.solido import malla, tetraedros

    assert CARAS_HEXAEDRO == malla.CARAS_HEXAEDRO
    assert CARAS_TETRAEDRO == tetraedros.CARAS_TETRAEDRO and MEDIOS_CARA == tetraedros.MEDIOS_CARA


# ---------------------------------------------------------------------------
# Caras y nodos medios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("semilla", [3, 4])
def test_caras_de_hexaedro_a_tetraedros(semilla):
    coordenadas, hexaedros = _bloque_deformado(4, 3, 3, 0.2, semilla)
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    caras = np.sort(conv.tetraedros[:, np.array(CARAS_TETRAEDRO)].reshape(-1, 3), axis=1)
    unicas, cuenta = np.unique(caras, axis=0, return_counts=True)
    de_borde = {tuple(t) for t in unicas[cuenta == 1].tolist()}
    h_borde, c_borde = _quads_de_borde(hexaedros)
    for h, c in zip(h_borde.tolist(), c_borde.tolist()):
        cuadrilatero = hexaedros[h, list(CARAS_HEXAEDRO[c])]
        p = conv.coordenadas[cuadrilatero]
        area = 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[1]))  # caras del borde: planas
        pares = conv.cara_hexaedro_a_tetraedros(h, c + 1)
        assert len(pares) == 2
        total = 0.0
        for t, k in pares:
            assert conv.hexaedro_de_tetraedro[t] == h and 1 <= k <= 4
            vertices = conv.tetraedros[t, list(CARAS_TETRAEDRO[k - 1])]
            medios = conv.tetraedros[t, list(MEDIOS_CARA[k - 1])]
            assert set(vertices.tolist()) <= set(cuadrilatero.tolist())
            assert tuple(sorted(vertices.tolist())) in de_borde
            q = conv.coordenadas[vertices]
            total += _area_triangulos(q)
            # la normal del triángulo (orden de CalculiX) mira hacia dentro del tetraedro, como la del cuadrilátero
            normal = np.cross(q[1] - q[0], q[2] - q[0])
            assert np.dot(normal, np.cross(p[2] - p[0], p[3] - p[1])) > 0
            np.testing.assert_allclose(conv.coordenadas[medios], 0.5 * (q + q[[1, 2, 0]]), atol=1e-12)
        assert total == pytest.approx(area, rel=1e-10)
    # versión vectorizada
    t, k = conv.caras_a_tetraedros(h_borde, c_borde + 1)
    esperado = [par for h, c in zip(h_borde.tolist(), c_borde.tolist()) for par in conv.cara_hexaedro_a_tetraedros(h, c + 1)]
    assert list(zip(t.tolist(), k.tolist())) == esperado
    with pytest.raises(ValueError):
        conv.cara_hexaedro_a_tetraedros(0, 7)


def test_caras_interiores_compartidas():
    """La cara común de dos hexaedros vecinos da los mismos dos triángulos desde los dos lados."""

    coordenadas, hexaedros = _bloque_deformado(3, 3, 2, 0.15, 5)
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    claves = {}
    for h in range(len(hexaedros)):
        for c in range(6):
            clave = tuple(sorted(hexaedros[h, list(CARAS_HEXAEDRO[c])].tolist()))
            tri = sorted(tuple(sorted(conv.tetraedros[t, list(CARAS_TETRAEDRO[k - 1])].tolist()))
                         for t, k in conv.cara_hexaedro_a_tetraedros(h, c + 1))
            claves.setdefault(clave, []).append(tri)
    compartidas = [v for v in claves.values() if len(v) == 2]
    assert len(compartidas) == 2 * 3 * 2 + 3 * 2 * 2 + 3 * 3 * 1
    assert all(a == b for a, b in compartidas)


def test_medio_de_arista_y_nodos_de_una_cara():
    coordenadas, hexaedros = _rejilla(3, 3, 3)
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    t = conv.tetraedros[17]
    assert conv.medio_de_arista(t[0], t[1]) == t[4] == conv.medio_de_arista(t[1], t[0])
    assert isinstance(conv.medio_de_arista(int(t[2]), int(t[3])), int)
    np.testing.assert_array_equal(conv.medio_de_arista(t[[0, 1, 2]], t[[3, 3, 3]]), t[[7, 8, 9]])
    with pytest.raises(KeyError):
        conv.medio_de_arista(0, len(coordenadas) - 1)  # esquinas opuestas de la rejilla
    with pytest.raises(KeyError):
        conv.medio_de_arista(-1, 0)
    # nodos de la cara x = 0 con sus nodos medios: todos los nodos de la malla en ese plano
    cara = np.flatnonzero(coordenadas[:, 0] == 0.0)
    np.testing.assert_array_equal(conv.ampliar_con_medios(cara), np.flatnonzero(conv.coordenadas[:, 0] == 0.0))
    assert conv.tetraedros_de_hexaedros(0, len(hexaedros)) == range(len(conv.tetraedros))
    assert list(conv.tetraedros_de_hexaedros(2, 3)) == np.flatnonzero(conv.hexaedro_de_tetraedro == 2).tolist()


def test_rapidez():
    coordenadas, hexaedros = _rejilla(40, 40, 40, 0.1, 0.1, 0.1)
    rng = np.random.default_rng(0)
    permutacion = rng.permutation(len(coordenadas))
    coordenadas, hexaedros = coordenadas[permutacion], np.argsort(permutacion)[hexaedros]
    inicio = time.perf_counter()
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    duracion = time.perf_counter() - inicio
    assert len(hexaedros) == 64_000
    assert duracion < 30.0  # unos 1-2 s; el margen es para máquinas cargadas
    v = _volumenes(conv.coordenadas, conv.tetraedros)
    assert v.min() > 0 and v.sum() == pytest.approx(64.0)


# ---------------------------------------------------------------------------
# Análisis con CalculiX
# ---------------------------------------------------------------------------


def _escribir_inp(ruta, conv, empotrados, cargas, E, nu):
    lineas = ["*HEADING", "Voladizo de tetraedros C3D10 convertidos de hexaedros", "*NODE"]
    lineas += [f"{i + 1}, {x:.15g}, {y:.15g}, {z:.15g}" for i, (x, y, z) in enumerate(conv.coordenadas)]
    lineas.append("*ELEMENT, TYPE=C3D10, ELSET=EALL")
    lineas += [f"{e + 1}, " + ", ".join(str(n + 1) for n in nodos) for e, nodos in enumerate(conv.tetraedros.tolist())]
    lineas.append("*NSET, NSET=EMPOTRADO")
    lineas += [", ".join(str(n + 1) for n in empotrados[i: i + 12]) for i in range(0, len(empotrados), 12)]
    lineas.append("*NSET, NSET=PUNTA")
    punta = sorted(cargas)
    lineas += [", ".join(str(n + 1) for n in punta[i: i + 12]) for i in range(0, len(punta), 12)]
    lineas += ["*MATERIAL, NAME=ACERO", "*ELASTIC", f"{E:.6g}, {nu}", "*SOLID SECTION, ELSET=EALL, MATERIAL=ACERO",
               "*BOUNDARY", "EMPOTRADO, 1, 3", "*STEP", "*STATIC", "*CLOAD"]
    lineas += [f"{n + 1}, 3, {f:.15g}" for n, f in sorted(cargas.items())]
    lineas += ["*NODE PRINT, NSET=PUNTA", "U", "*END STEP", ""]
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def _desplazamientos_dat(ruta):
    """Desplazamientos (nodo -> (ux, uy, uz)) del bloque ``displacements`` del .dat."""

    resultado, dentro = {}, False
    numero = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
        if "displacements" in linea:
            dentro = True
            continue
        if dentro:
            m = re.match(rf"^\s*(\d+)\s+({numero})\s+({numero})\s+({numero})\s*$", linea)
            if m:
                resultado[int(m.group(1)) - 1] = tuple(float(m.group(k)) for k in (2, 3, 4))
            elif resultado:
                dentro = False
    return resultado


@requiere_ccx
@pytest.mark.parametrize("divisiones", [(10, 2, 2), (20, 4, 4)])
def test_voladizo_c3d10_con_calculix(tmp_path, divisiones):
    """Voladizo de 1 m × 0.1 × 0.1 empotrado en x = 0 con una carga en la punta: flecha de Timoshenko a menos del 5 %."""

    L, b, h, E, nu, F = 1.0, 0.1, 0.1, 210e9, 0.3, -1000.0
    nx, ny, nz = divisiones
    coordenadas, hexaedros = _rejilla(nx, ny, nz, L / nx, b / ny, h / nz)
    conv = hexaedros_a_tetraedros(coordenadas, hexaedros)
    _comprobar(coordenadas, hexaedros, conv, volumen=L * b * h)

    empotrados = conv.ampliar_con_medios(np.flatnonzero(np.isclose(coordenadas[:, 0], 0.0))).tolist()
    # carga de la punta: en los nodos medios de los triángulos de la cara x = L, en proporción al área (consistente)
    en_punta = np.isclose(coordenadas[:, 0], L)
    h_punta, c_punta = [], []
    for c, cara in enumerate(CARAS_HEXAEDRO):
        todos = en_punta[hexaedros[:, list(cara)]].all(axis=1)
        h_punta += np.flatnonzero(todos).tolist()
        c_punta += [c + 1] * int(todos.sum())
    t, k = conv.caras_a_tetraedros(np.array(h_punta), np.array(c_punta))
    vertices = conv.tetraedros[t[:, None], np.array(CARAS_TETRAEDRO)[k - 1]]
    medios = conv.tetraedros[t[:, None], np.array(MEDIOS_CARA)[k - 1]]
    area = _area_triangulos(conv.coordenadas[vertices])
    assert area.sum() == pytest.approx(b * h)
    cargas = {}
    for fila, a in zip(medios.tolist(), area.tolist()):
        for n in fila:
            cargas[n] = cargas.get(n, 0.0) + F * a / (b * h) / 3.0
    nodos_punta = conv.ampliar_con_medios(np.flatnonzero(en_punta))
    for n in nodos_punta.tolist():
        cargas.setdefault(n, 0.0)
    assert sum(cargas.values()) == pytest.approx(F)

    ruta = tmp_path / "voladizo.inp"
    _escribir_inp(ruta, conv, empotrados, cargas, E, nu)
    entorno = dict(os.environ, OMP_NUM_THREADS="1", CCX_NPROC_EQUATION_SOLVER="1")
    proceso = subprocess.run(["ccx", "-i", "voladizo"], cwd=tmp_path, capture_output=True, text=True, timeout=600, env=entorno)
    salida = proceso.stdout + proceso.stderr
    assert "Job finished" in salida and "*ERROR" not in salida, salida[-2000:]
    u = _desplazamientos_dat(tmp_path / "voladizo.dat")
    assert set(u) == set(nodos_punta.tolist())
    flecha = np.mean([u[n][2] for n in nodos_punta.tolist()])
    inercia = b * h**3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    teoria = F * L**3 / (3.0 * E * inercia) + F * L / (5.0 / 6.0 * G * b * h)
    assert flecha == pytest.approx(teoria, rel=0.05), f"flecha {flecha:.6e} frente a {teoria:.6e}"
