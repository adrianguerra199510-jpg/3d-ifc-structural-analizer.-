"""Prueba de guardado/carga del proyecto completo en JSON."""

from pathlib import Path

from ifc_structural.core.cargas import Apoyo, CargaArea, CargaLineal, CargaPuntual
from ifc_structural.core.casos_carga import Combinacion, CasoCarga, TipoCaso
from ifc_structural.core.modelo import DireccionLosa, Material, PerfilSeccion, Proyecto
from ifc_structural.core.persistencia import cargar_proyecto, guardar_proyecto


def _proyecto_completo():
    p = Proyecto("Prueba persistencia")
    p.agregar_material(Material(nombre="HORMIGON", modulo_elasticidad=2.5e7, densidad=24.0))
    p.agregar_perfil(PerfilSeccion(nombre="V30X50", area=0.15, inercia_y=3e-4, inercia_z=1e-3, inercia_torsional=1e-4))
    n1 = p.agregar_nodo(0, 0, 0, "N1")
    n2 = p.agregar_nodo(4, 0, 0, "N2")
    n3 = p.agregar_nodo(4, 4, 0, "N3")
    n4 = p.agregar_nodo(0, 4, 0, "N4")
    b1 = p.agregar_viga(n1.id, n2.id, "HORMIGON", "V30X50", id_="B1")
    p.agregar_columna(n1.id, n1.id, "HORMIGON", "V30X50", id_="C1")
    p.agregar_losa(
        nodos_contorno=[n1.id, n2.id, n3.id, n4.id], espesor=0.15, material="HORMIGON",
        direccion=DireccionLosa.DOS_DIRECCIONES, vigas_soporte=[b1.id, None, None, None], id_="L1",
    )

    apoyos = [Apoyo(id="AP1", nodo_id="N1")]
    casos = {"D": CasoCarga(id="D", nombre="Carga muerta", tipo=TipoCaso.D)}
    combinaciones = {"C1": Combinacion(id="C1", nombre="1.4D", factores={"D": 1.4}, norma="ACI 318")}
    cargas_area = {"D": [CargaArea(id="CA1", caso_id="D", presion=5.0, losa_id="L1")]}
    cargas_lineales = {"D": [CargaLineal(id="CL1", caso_id="D", elemento_id="B1", w_inicio=3.6, w_fin=3.6)]}
    cargas_puntuales = {"D": [CargaPuntual(id="CP1", caso_id="D", nodo_id="N2", fz=-10.0)]}

    return p, apoyos, casos, combinaciones, cargas_area, cargas_lineales, cargas_puntuales


def test_guardar_y_cargar_proyecto_round_trip(tmp_path: Path):
    datos_originales = _proyecto_completo()
    ruta = tmp_path / "proyecto.json"
    guardar_proyecto(ruta, *datos_originales)

    (proyecto, apoyos, casos, combinaciones, cargas_area, cargas_lineales, cargas_puntuales) = cargar_proyecto(ruta)

    assert proyecto.nombre == "Prueba persistencia"
    assert set(proyecto.vigas) == {"B1"}
    assert set(proyecto.columnas) == {"C1"}
    assert set(proyecto.losas) == {"L1"}
    assert proyecto.losas["L1"].vigas_soporte[0] == "B1"
    assert proyecto.losas["L1"].vigas_soporte[1] is None

    assert apoyos[0].nodo_id == "N1"
    assert casos["D"].tipo == TipoCaso.D
    assert combinaciones["C1"].factores["D"] == 1.4
    assert cargas_area["D"][0].presion == 5.0
    assert cargas_lineales["D"][0].w_inicio == 3.6
    assert cargas_puntuales["D"][0].fz == -10.0
