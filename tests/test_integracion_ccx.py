"""Prueba de integración: genera un .inp, ejecuta ccx de verdad y lee los
resultados. Se omite si no hay un ejecutable `ccx` disponible en el sistema
(p. ej. en un entorno de CI sin CalculiX instalado).
"""

import shutil
from pathlib import Path

import pytest

from ifc_structural.combinaciones.motor_superposicion import superponer_combinacion
from ifc_structural.core.cargas import Apoyo, CargaPuntual
from ifc_structural.core.casos_carga import Combinacion
from ifc_structural.core.modelo import Material, PerfilSeccion, Proyecto
from ifc_structural.core.transferencia import peso_propio_barras
from ifc_structural.solver.ejecutor_ccx import ejecutar_ccx_todos_los_casos
from ifc_structural.solver.generador_inp import construir_malla, generar_inp_todos_los_casos
from ifc_structural.solver.lector_resultados import leer_resultados_caso

pytestmark = pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema")


def test_voladizo_peso_propio_y_carga_puntual(tmp_path: Path):
    p = Proyecto("Integración ccx")
    p.agregar_material(Material(nombre="HORMIGON", modulo_elasticidad=2.5e7, densidad=24.0, modulo_poisson=0.2))
    p.agregar_perfil(
        PerfilSeccion(
            nombre="V30X50", area=0.15, inercia_y=3.125e-4, inercia_z=1.125e-3,
            inercia_torsional=1e-4, altura=0.5, ancho=0.3,
        )
    )
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(4, 0, 0, "N2")
    p.agregar_viga(n1.id, n2.id, "HORMIGON", "V30X50", id_="B1")
    apoyo = Apoyo(id="AP1", nodo_id="N1")  # empotramiento (los 6 gdl, por defecto)

    cargas_lineales = {"D": peso_propio_barras(p, "D"), "L": []}
    cargas_puntuales = {"D": [], "L": [CargaPuntual(id="CP1", caso_id="L", nodo_id="N2", fz=-10.0)]}

    malla = construir_malla(p, cargas_lineales, cargas_puntuales)
    rutas_inp = generar_inp_todos_los_casos(p, malla, cargas_lineales, cargas_puntuales, [apoyo], tmp_path)
    resultados_ejecucion = ejecutar_ccx_todos_los_casos(rutas_inp)

    for caso_id, resultado in resultados_ejecucion.items():
        assert resultado.exitoso, f"ccx falló para el caso {caso_id}:\n{resultado.salida}"

    resultados_por_caso = {
        caso_id: leer_resultados_caso(r.ruta_dat, r.ruta_frd) for caso_id, r in resultados_ejecucion.items()
    }

    nodo_libre = malla.nodo_ccx_de_nodo_original["N2"]
    nodo_apoyo = malla.orden_nodos_por_barra["B1"][0]

    # Bajo peso propio, el extremo libre debe desplazarse hacia abajo (Z<0).
    assert resultados_por_caso["D"].desplazamientos[nodo_libre][2] < 0

    # El primer elemento (cerca del apoyo) debe tener el mayor momento en
    # valor absoluto (el momento flector de un voladizo es máximo en el
    # empotramiento).
    m2_apoyo_D = abs(resultados_por_caso["D"].fuerzas_seccion[nodo_apoyo][5])
    m2_libre_D = abs(resultados_por_caso["D"].fuerzas_seccion[nodo_libre][5])
    assert m2_apoyo_D > m2_libre_D

    # Combinación 1.2D + 1.6L: el momento en el apoyo debe superar al de
    # cualquiera de los dos casos por separado (ambos suman en el mismo
    # sentido para esta geometría).
    combinacion = Combinacion(id="C1", nombre="1.2D+1.6L", factores={"D": 1.2, "L": 1.6})
    resultado_combinado = superponer_combinacion(resultados_por_caso, combinacion)
    m2_apoyo_comb = abs(resultado_combinado.fuerzas_seccion[nodo_apoyo][5])
    m2_apoyo_L = abs(resultados_por_caso["L"].fuerzas_seccion[nodo_apoyo][5])
    assert m2_apoyo_comb > max(m2_apoyo_D, m2_apoyo_L)
