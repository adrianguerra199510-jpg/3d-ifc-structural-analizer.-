"""Punto de entrada alternativo para empaquetar con PyInstaller.

``python -m ifc_structural.app`` funciona bien para ejecutar la app desde
el código fuente, pero PyInstaller necesita un script de nivel superior
(no un submódulo de un paquete) para que las importaciones relativas
dentro de ``ifc_structural`` sigan funcionando una vez compilado. Este
archivo es ese punto de entrada; no contiene lógica propia.
"""

from ifc_structural.app import main

if __name__ == "__main__":
    raise SystemExit(main())
