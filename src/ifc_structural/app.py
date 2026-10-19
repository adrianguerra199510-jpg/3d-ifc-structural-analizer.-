"""Punto de entrada de la aplicación de escritorio.

Ejecutar con:

    python -m ifc_structural.app
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from .gui.ventana_principal import VentanaPrincipal


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
