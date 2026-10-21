"""Punto de entrada de la aplicación de escritorio.

Ejecutar con:

    python -m ifc_structural.app
"""

from __future__ import annotations

import logging
import sys


def main() -> int:
    if "--malla-gmsh" in sys.argv:  # proceso hijo: malla un sólido con Gmsh (ver solido/proceso_gmsh.py)
        from .solido.proceso_gmsh import main as malla_gmsh

        return malla_gmsh(sys.argv[sys.argv.index("--malla-gmsh") + 1:])
    from PySide6.QtWidgets import QApplication

    from .gui.estilo import aplicar_estilo
    from .gui.iconos import icono_aplicacion
    from .gui.ventana_principal import VentanaPrincipal

    if "--autoprueba" in sys.argv:
        from .autoprueba import ejecutar_autoprueba

        return ejecutar_autoprueba()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if sys.platform == "win32":  # que Windows use el icono propio en la barra de tareas
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AnalizadorEstructuralIFC.App")
        except Exception:  # noqa: BLE001 - solo estético
            pass
    app = QApplication(sys.argv)
    app.setApplicationName("Analizador Estructural 3D")
    app.setWindowIcon(icono_aplicacion())
    aplicar_estilo(app)
    ventana = VentanaPrincipal()
    ventana.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
