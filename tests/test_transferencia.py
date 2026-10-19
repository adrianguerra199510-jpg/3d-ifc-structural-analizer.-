"""Pruebas del algoritmo de transferencia de cargas de área a vigas."""

import math

import pytest

from ifc_structural.core.cargas import CargaArea
from ifc_structural.core.modelo import DireccionLosa, Material, PerfilSeccion, Proyecto
from ifc_structural.core.transferencia import (
    ErrorTransferencia,
    cargas_area_peso_propio_losas,
    peso_propio_barras,
    transferir_carga_area,
)


def _proyecto_basico() -> Proyecto:
    p = Proyecto("Pruebas")
    p.agregar_material(Material(nombre="HORMIGON", modulo_elasticidad=25e6, densidad=24.0))
    p.agregar_perfil(
        PerfilSeccion(
            nombre="VIGA_30X50",
            area=0.15,
            inercia_y=3.125e-4,
            inercia_z=1.125e-3,
            inercia_torsional=1e-4,
        )
    )
    return p


def _integral_segmento(seg) -> float:
    """Integral de un CargaLineal (trapecio) sobre su longitud real, en kN."""

    longitud = seg.s_fin - seg.s_inicio  # fracción; se multiplica por longitud real afuera
    return 0.5 * (seg.w_inicio + seg.w_fin) * longitud


def test_losa_rectangular_dos_direcciones_conserva_carga_total():
    p = _proyecto_basico()
    lx, ly = 4.0, 6.0
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(lx, 0, 0, "N2")
    n3 = p.agregar_nodo(lx, ly, 0, "N3")
    n4 = p.agregar_nodo(0, ly, 0, "N4")

    b1 = p.agregar_viga(n1.id, n2.id, "HORMIGON", "VIGA_30X50", id_="B1")
    b2 = p.agregar_viga(n2.id, n3.id, "HORMIGON", "VIGA_30X50", id_="B2")
    b3 = p.agregar_viga(n3.id, n4.id, "HORMIGON", "VIGA_30X50", id_="B3")
    b4 = p.agregar_viga(n4.id, n1.id, "HORMIGON", "VIGA_30X50", id_="B4")

    losa = p.agregar_losa(
        nodos_contorno=[n1.id, n2.id, n3.id, n4.id],
        espesor=0.15,
        material="HORMIGON",
        direccion=DireccionLosa.DOS_DIRECCIONES,
        vigas_soporte=[b1.id, b2.id, b3.id, b4.id],
    )

    presion = 10.0  # kPa
    carga = CargaArea(id="CA1", caso_id="L", presion=presion, losa_id=losa.id)
    segmentos = transferir_carga_area(p, carga)

    # Reconstruir la carga total integrando cada segmento sobre su longitud real.
    longitudes = {"B1": lx, "B2": ly, "B3": lx, "B4": ly}
    total = 0.0
    for seg in segmentos:
        longitud_real = longitudes[seg.elemento_id]
        total += _integral_segmento(seg) * longitud_real

    area_losa = lx * ly
    assert total == pytest.approx(presion * area_losa, rel=1e-6)

    # El pico de intensidad en cualquier viga debe ser presion * luz_corta / 2.
    pico_esperado = presion * (lx / 2.0)
    pico_obtenido = max(max(s.w_inicio, s.w_fin) for s in segmentos)
    assert pico_obtenido == pytest.approx(pico_esperado, rel=1e-6)

    # Las vigas cortas (B1, B3) reciben un único segmento triangular.
    segmentos_b1 = [s for s in segmentos if s.elemento_id == "B1"]
    assert len(segmentos_b1) == 2  # rampa ascendente + descendente
    assert min(s.w_inicio for s in segmentos_b1) == pytest.approx(0.0, abs=1e-9)

    # Las vigas largas (B2, B4) reciben un perfil trapezoidal (3 segmentos).
    segmentos_b2 = [s for s in segmentos if s.elemento_id == "B2"]
    assert len(segmentos_b2) == 3


def test_losa_una_direccion_reparte_mitad_de_luz():
    p = _proyecto_basico()
    luz = 5.0
    ancho = 3.0
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(luz, 0, 0, "N2")
    n3 = p.agregar_nodo(luz, ancho, 0, "N3")
    n4 = p.agregar_nodo(0, ancho, 0, "N4")

    b1 = p.agregar_viga(n1.id, n2.id, "HORMIGON", "VIGA_30X50", id_="B1")
    b3 = p.agregar_viga(n3.id, n4.id, "HORMIGON", "VIGA_30X50", id_="B3")

    losa = p.agregar_losa(
        nodos_contorno=[n1.id, n2.id, n3.id, n4.id],
        espesor=0.15,
        material="HORMIGON",
        direccion=DireccionLosa.UNA_DIRECCION,
        # Solo las vigas que delimitan la luz (lados "verticales" del rectángulo:
        # N2-N3 y N4-N1) tienen viga asignada aquí es al revés: delimitan la luz
        # las vigas B1 (N1-N2) y B3 (N3-N4), paralelas, separadas por "ancho".
        vigas_soporte=[b1.id, None, b3.id, None],
    )

    presion = 8.0
    carga = CargaArea(id="CA2", caso_id="L", presion=presion, losa_id=losa.id)
    segmentos = transferir_carga_area(p, carga)

    assert len(segmentos) == 2
    for seg in segmentos:
        assert seg.es_uniforme
        assert seg.w_inicio == pytest.approx(presion * (ancho / 2.0), rel=1e-6)


def test_losa_no_rectangular_usa_reparto_proporcional_y_avisa():
    p = _proyecto_basico()
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(4, 0, 0, "N2")
    n3 = p.agregar_nodo(4, 4, 0, "N3")
    n4 = p.agregar_nodo(2, 6, 0, "N4")
    n5 = p.agregar_nodo(0, 4, 0, "N5")

    ids_nodos = [n1.id, n2.id, n3.id, n4.id, n5.id]
    vigas = []
    for i in range(5):
        a, b = ids_nodos[i], ids_nodos[(i + 1) % 5]
        vigas.append(p.agregar_viga(a, b, "HORMIGON", "VIGA_30X50", id_=f"B{i+1}").id)

    losa = p.agregar_losa(
        nodos_contorno=ids_nodos,
        espesor=0.15,
        material="HORMIGON",
        direccion=DireccionLosa.DOS_DIRECCIONES,
        vigas_soporte=vigas,
    )

    carga = CargaArea(id="CA3", caso_id="L", presion=5.0, losa_id=losa.id)
    with pytest.warns(UserWarning):
        segmentos = transferir_carga_area(p, carga)

    assert len(segmentos) == 5
    for seg in segmentos:
        assert seg.es_uniforme


def test_losa_sin_vigas_de_soporte_lanza_error():
    p = _proyecto_basico()
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(4, 0, 0, "N2")
    n3 = p.agregar_nodo(4, 4, 0, "N3")
    n4 = p.agregar_nodo(0, 4, 0, "N4")
    losa = p.agregar_losa(
        nodos_contorno=[n1.id, n2.id, n3.id, n4.id],
        espesor=0.15,
        material="HORMIGON",
        vigas_soporte=[None, None, None, None],
    )
    carga = CargaArea(id="CA4", caso_id="L", presion=5.0, losa_id=losa.id)
    with pytest.raises(ErrorTransferencia):
        transferir_carga_area(p, carga)


def test_peso_propio_barras():
    p = _proyecto_basico()
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(5, 0, 0, "N2")
    p.agregar_viga(n1.id, n2.id, "HORMIGON", "VIGA_30X50", id_="B1")

    cargas = peso_propio_barras(p, caso_id="D")
    assert len(cargas) == 1
    esperado = 24.0 * 0.15  # densidad * area
    assert cargas[0].w_inicio == pytest.approx(esperado)
    assert cargas[0].direccion == (0.0, 0.0, -1.0)


def test_peso_propio_losas_genera_carga_area_y_se_transfiere():
    p = _proyecto_basico()
    lx, ly = 4.0, 4.0
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(lx, 0, 0, "N2")
    n3 = p.agregar_nodo(lx, ly, 0, "N3")
    n4 = p.agregar_nodo(0, ly, 0, "N4")
    b1 = p.agregar_viga(n1.id, n2.id, "HORMIGON", "VIGA_30X50", id_="B1")
    b2 = p.agregar_viga(n2.id, n3.id, "HORMIGON", "VIGA_30X50", id_="B2")
    b3 = p.agregar_viga(n3.id, n4.id, "HORMIGON", "VIGA_30X50", id_="B3")
    b4 = p.agregar_viga(n4.id, n1.id, "HORMIGON", "VIGA_30X50", id_="B4")
    losa = p.agregar_losa(
        nodos_contorno=[n1.id, n2.id, n3.id, n4.id],
        espesor=0.2,
        material="HORMIGON",
        vigas_soporte=[b1.id, b2.id, b3.id, b4.id],
    )

    cargas_area = cargas_area_peso_propio_losas(p, caso_id="D")
    assert len(cargas_area) == 1
    assert cargas_area[0].presion == pytest.approx(24.0 * 0.2)

    segmentos = transferir_carga_area(p, cargas_area[0])
    assert len(segmentos) > 0
