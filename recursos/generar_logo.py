"""Genera los archivos del logo a partir del SVG definido en la aplicación.

    python recursos/generar_logo.py

Escribe ``recursos/logo.svg``, ``recursos/logo.png`` (512 px) y
``recursos/logo.ico`` (varios tamaños; lo usa PyInstaller como icono del .exe).
Necesita PySide6 y Pillow.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from PySide6.QtGui import QGuiApplication  # noqa: E402

app = QGuiApplication([])

from PIL import Image  # noqa: E402

from ifc_structural.gui.iconos import LOGO_SVG, guardar_logo  # noqa: E402

destino = Path(__file__).resolve().parent
(destino / "logo.svg").write_text(LOGO_SVG, encoding="utf-8")
guardar_logo(str(destino / "logo.png"), 512)
Image.open(destino / "logo.png").save(
    destino / "logo.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
)
print("Logo generado en", destino)
