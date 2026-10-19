"""Autoprueba del solver: resuelve un voladizo con CalculiX y compara con la
teoría de vigas. Sirve para comprobar, sobre la versión portable ya
compilada, que ``ccx.exe`` y sus bibliotecas funcionan en este equipo.

    AnalizadorEstructuralIFC.exe --autoprueba

Escribe ``autoprueba.txt`` junto al ejecutable y devuelve 0 si todo salió bien.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import numpy as np

from .config import carpeta_aplicacion, cargar_configuracion
from .core.cargas import Apoyo, CargaPuntual
from .core.casos_carga import CasoCarga, Combinacion, TipoCaso
from .core.modelo import Material, PerfilSeccion, Proyecto
from .solver.analisis import ejecutar_analisis


def _voladizo() -> Proyecto:
    p = Proyecto("Autoprueba")
    p.agregar_material(Material(nombre="HORMIGON", modulo_elasticidad=2.5e7, densidad=24.0, modulo_poisson=0.2))
    p.agregar_perfil(
        PerfilSeccion(nombre="V30X50", area=0.15, inercia_y=3.125e-4, inercia_z=1.125e-3,
                      inercia_torsional=1e-4, altura=0.5, ancho=0.3)
    )
    p.agregar_nodo(0, 0, 0, id_="N1")
    p.agregar_nodo(4, 0, 0, id_="N2")
    p.agregar_viga("N1", "N2", "HORMIGON", "V30X50", id_="B1")
    return p


def ejecutar_autoprueba() -> int:
    configuracion = cargar_configuracion()
    lineas = [f"CalculiX: {configuracion.ruta_ccx_efectiva}"]
    try:
        with tempfile.TemporaryDirectory(prefix="autoprueba_") as carpeta:
            resultado = ejecutar_analisis(
                _voladizo(),
                {"L": CasoCarga("L", "Carga viva", TipoCaso.L)},
                {"C1": Combinacion("C1", "1.0L", {"L": 1.0})},
                [Apoyo(id="AP1", nodo_id="N1")], {}, {},
                {"L": [CargaPuntual(id="CP1", caso_id="L", nodo_id="N2", fz=-10.0)]},
                Path(carpeta), configuracion.ruta_ccx_efectiva,
            )
        x = resultado.malla_tensiones.coordenadas[:, 0]
        sxx = np.abs(resultado.tensiones_por_caso["L"][:, 0])
        obtenido = float(sxx[np.isclose(x, 2.0, atol=0.02)].max())
        esperado = 10.0 * (4.0 - 2.0) * 0.25 / (0.3 * 0.5**3 / 12.0)  # M·c/I en kPa
        error = abs(obtenido - esperado) / esperado
        lineas.append(f"Esfuerzo en x = 2 m: {obtenido:.1f} kPa (teoria de vigas: {esperado:.1f} kPa, error {error:.2%})")
        ok = error < 0.05
        lineas.append("RESULTADO: CORRECTO" if ok else "RESULTADO: FALLO (esfuerzo fuera de tolerancia)")
    except Exception:  # noqa: BLE001 - se informa cualquier fallo
        lineas.append(traceback.format_exc())
        lineas.append("RESULTADO: FALLO")
        ok = False
    texto = "\n".join(lineas)
    try:
        (carpeta_aplicacion() / "autoprueba.txt").write_text(texto, encoding="utf-8")
    except OSError:
        pass
    print(texto)
    return 0 if ok else 1
