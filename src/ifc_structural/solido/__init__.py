"""Análisis como modelo sólido (3D).

Cada viga, columna, losa y muro se malla con su geometría real en
hexaedros (CalculiX ``C3D8I``), las piezas que se tocan se unen con
``*TIE`` y se resuelve con el mismo CalculiX que el análisis de barras. El
mallador es propio (barrido de una sección de cuadriláteros), de modo que no
hace falta instalar nada más: la versión portable sigue siendo «ejecutar y
listo».
"""
