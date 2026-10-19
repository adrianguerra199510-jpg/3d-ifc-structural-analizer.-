"""Configuración común de pytest.

Fuerza la plataforma Qt "offscreen" para que las pruebas de la GUI puedan
construir widgets (QApplication, QMainWindow, etc.) en un entorno sin
servidor de ventanas, como este contenedor. Debe fijarse ANTES de
importar cualquier módulo de PySide6/Qt, por eso se hace aquí, en la
carga del propio conftest.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
