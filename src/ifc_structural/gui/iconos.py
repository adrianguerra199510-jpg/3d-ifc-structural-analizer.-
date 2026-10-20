"""Logo e iconos de la aplicación, dibujados como SVG en código.

Todos los iconos comparten una rejilla de 24×24, trazo redondeado de
1.7 px y dos tonos: ``@T@`` (trazo base, gris azulado) y ``@A@`` (acento).
Así se ven coherentes y se pueden recolorear (p. ej. blancos sobre el botón
verde de ANALIZAR) sin archivos de imagen. Se renderizan con ``QSvgRenderer``
a varios tamaños y con soporte de pantallas de alta densidad.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

COLOR_TRAZO = "#3b4a5c"
COLOR_ACENTO = "#1f6feb"
COLOR_DESACTIVADO = "#aab4c0"

_PLANTILLA = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="@T@" '
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{cuerpo}</svg>'
)

# Caras del cubo isométrico (compartidas por los iconos de vista)
_CUBO_SUP = "M12 3l8 4.5-8 4.5-8-4.5z"
_CUBO_IZQ = "M4 7.5L12 12v9l-8-4.5z"
_CUBO_DER = "M12 12l8-4.5v9L12 21z"


def _cubo(resaltada: str = "") -> str:
    caras = {"sup": _CUBO_SUP, "izq": _CUBO_IZQ, "der": _CUBO_DER}
    base = "".join(f'<path d="{d}"/>' for d in caras.values())
    if resaltada:
        base += f'<path d="{caras[resaltada]}" fill="@A@" fill-opacity=".38" stroke="@A@"/>'
    return base


CUERPOS: Dict[str, str] = {
    # -- archivo --------------------------------------------------------------
    "importar": (
        '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<path d="M12 10v6M9.5 13.5L12 16l2.5-2.5" stroke="@A@"/>'
    ),
    "abrir": (
        '<path d="M3 18V7a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v2"/>'
        '<path d="M3 18l2.4-7H21l-2.4 7z" stroke="@A@"/>'
    ),
    "guardar": (
        '<path d="M5 3h11l4 4v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>'
        '<path d="M7 3v5h8V3" stroke="@A@"/><path d="M7 21v-6.5h10V21"/>'
    ),
    # -- selección ---------------------------------------------------------------
    "sel_elemento": (
        '<path d="M10.5 3l6.5 3.6v7L10.5 17 4 13.6v-7z"/><path d="M10.5 10.3l6.5-3.7M10.5 10.3V17M10.5 10.3L4 6.6" stroke-opacity=".45"/>'
        '<path d="M14 13l6.5 2.2-2.8 1.5 1.9 3.5-1.5.8-1.9-3.6-2.2 2z" fill="@A@" stroke="@A@" stroke-width="1.2"/>'
    ),
    "sel_cara": (
        '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" stroke-opacity=".45"/>'
        '<path d="M12 12l8-4.5v9L12 21z" fill="@A@" fill-opacity=".38" stroke="@A@"/>'
    ),
    # -- asignaciones --------------------------------------------------------------
    "apoyo": (
        '<circle cx="12" cy="6" r="1.9"/><path d="M12 8l6 8H6z"/>'
        '<path d="M4 19h16" stroke="@A@"/><path d="M6.5 21.5l2-2.5M11 21.5l2-2.5M15.5 21.5l2-2.5" stroke="@A@"/>'
    ),
    "apoyo_area": (
        '<path d="M3 7.5h18" stroke-width="2.6"/>'
        '<path d="M6 7.5l2.2 4.6H3.8zM12 7.5l2.2 4.6H9.8zM18 7.5l2.2 4.6h-4.4z"/>'
        '<path d="M3 16h18" stroke="@A@"/><path d="M5.5 20.5l1.8-3.2M10.5 20.5l1.8-3.2M15.5 20.5l1.8-3.2M20 20l1-1.8" stroke="@A@"/>'
    ),
    "carga": (
        '<path d="M12 3v11"/><path d="M7.5 10l4.5 4.5 4.5-4.5" stroke="@A@"/><path d="M4 19h16" stroke-width="2.6"/>'
    ),
    "carga_dist": (
        '<path d="M4 4.5h16" stroke="@A@"/><path d="M6 4.5v8M12 4.5v8M18 4.5v8" stroke="@A@"/>'
        '<path d="M4.3 10.2L6 12.5l1.7-2.3M10.3 10.2L12 12.5l1.7-2.3M16.3 10.2L18 12.5l1.7-2.3" stroke="@A@"/>'
        '<path d="M3 18.5h18" stroke-width="2.6"/>'
    ),
    "material": (
        '<path d="M12 3l9 4.5-9 4.5-9-4.5z"/><path d="M3 12l9 4.5 9-4.5" stroke="@A@"/>'
        '<path d="M3 16.5l9 4.5 9-4.5" stroke="@A@" stroke-opacity=".55"/>'
    ),
    "materiales": (
        '<path d="M9 3h6M10 3v6L5 18a2 2 0 0 0 1.8 3h10.4A2 2 0 0 0 19 18l-5-9V3"/>'
        '<path d="M7.6 15h8.8" stroke="@A@"/><circle cx="10.5" cy="18" r=".7" fill="@A@" stroke="@A@"/>'
        '<circle cx="13.5" cy="17.2" r=".7" fill="@A@" stroke="@A@"/>'
    ),
    "conectar": (
        '<circle cx="6" cy="17.5" r="2.7"/><circle cx="18" cy="6.5" r="2.7"/>'
        '<path d="M8.1 15.6l7.8-7.2" stroke="@A@"/><circle cx="12" cy="12" r="1.5" fill="@A@" stroke="@A@"/>'
    ),
    "casos": (
        '<path d="M9.5 6h11M9.5 12h11M9.5 18h11"/>'
        '<path d="M3.4 6l1.3 1.3L7 4.8M3.4 12l1.3 1.3L7 10.8M3.4 18l1.3 1.3L7 16.8" stroke="@A@"/>'
    ),
    # -- análisis y vistas -----------------------------------------------------------
    "analizar": '<path d="M7.5 4.8v14.4a.6.6 0 0 0 .9.5l11.6-7.2a.6.6 0 0 0 0-1L8.4 4.3a.6.6 0 0 0-.9.5z" fill="@A@" stroke="@A@"/>',
    "solido": (
        '<path d="M12 3l8.5 4.8v8.4L12 21l-8.5-4.8V7.8z"/><path d="M12 12l8.5-4.2M12 12v9M12 12L3.5 7.8"/>'
        '<path d="M8.2 5.4l8.5 4.8M3.5 12l8.5 4.8 8.5-4.8M16.1 5.4L7.6 10.2" stroke="@A@" stroke-width="1.1"/>'
    ),
    "zona": (
        '<rect x="4" y="5" width="12" height="10" rx="1" stroke-dasharray="3 2.4"/>'
        '<path d="M13 12l7 2.5-3 1.2 2 3.3-1.6.9-2-3.3-2.4 2z" fill="@A@" fill-opacity=".35" stroke="@A@"/>'
    ),
    "mover": (
        '<path d="M12 3v18M3 12h18"/><path d="M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3" stroke="@A@"/>'
    ),
    "rotar": (
        '<path d="M20 12a8 8 0 1 1-2.6-5.9"/><path d="M20 4.5v4.6h-4.6" stroke="@A@"/>'
    ),
    "deshacer": (
        '<path d="M9 7L4.5 11.5 9 16"/><path d="M5 11.5h9a5 5 0 0 1 0 10h-3" stroke="@A@"/>'
    ),
    "rehacer": (
        '<path d="M15 7l4.5 4.5L15 16"/><path d="M19 11.5h-9a5 5 0 0 0 0 10h3" stroke="@A@"/>'
    ),
    "eliminar": (
        '<path d="M5 7h14M10 7V4.5h4V7M7 7l1 13h8l1-13"/><path d="M10.5 11v5.5M13.5 11v5.5" stroke="@A@"/>'
    ),
    "modelo": (
        '<path d="M12 3l8.5 4.8v8.4L12 21l-8.5-4.8V7.8z"/><path d="M12 12l8.5-4.2M12 12v9M12 12L3.5 7.8" stroke="@A@"/>'
    ),
    "esfuerzos": (
        '<rect x="3" y="7" width="18" height="10" rx="2.2"/>'
        '<path d="M3 9.2a2.2 2.2 0 0 1 2.2-2.2H9v10H5.2A2.2 2.2 0 0 1 3 14.8z" fill="@A@" fill-opacity=".14" stroke="none"/>'
        '<path d="M9 7h6v10H9z" fill="@A@" fill-opacity=".34" stroke="none"/>'
        '<path d="M15 7h3.8A2.2 2.2 0 0 1 21 9.2v5.6a2.2 2.2 0 0 1-2.2 2.2H15z" fill="@A@" fill-opacity=".62" stroke="none"/>'
        '<path d="M9 7v10M15 7v10" stroke="@A@"/>'
    ),
    "vistas": (
        '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3" stroke="@A@"/>'
    ),
    "vista_iso": _cubo(),
    "vista_planta": _cubo("sup"),
    "vista_frontal": _cubo("izq"),
    "vista_lateral": _cubo("der"),
    "vista_inferior": (
        '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" stroke-opacity=".6"/><path d="M4 16.5l8 4.5 8-4.5-8-4.5z" fill="@A@" fill-opacity=".38" stroke="@A@"/>'
    ),
    "ajustar": (
        '<path d="M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15"/>'
        '<rect x="8.5" y="8.5" width="7" height="7" rx="1.2" stroke="@A@"/>'
    ),
    # -- elementos -------------------------------------------------------------------------
    "viga": (
        '<rect x="2.5" y="8.5" width="19" height="7" rx="1.4"/><path d="M7.5 8.5v7M12 8.5v7M16.5 8.5v7" stroke="@A@" stroke-opacity=".6"/>'
    ),
    "columna": (
        '<rect x="8.5" y="2.5" width="7" height="19" rx="1.4"/><path d="M8.5 7.5h7M8.5 12h7M8.5 16.5h7" stroke="@A@" stroke-opacity=".6"/>'
    ),
    "losa": (
        '<path d="M3 14l5-6h13l-5 6z" fill="@A@" fill-opacity=".18"/><path d="M3 14v3.5h13V14M16 14l5-6v3.5l-5 6"/>'
    ),
    "muro": (
        '<rect x="3" y="4" width="18" height="16" rx="1.4"/><path d="M3 9.3h18M3 14.7h18M9 4v5.3M15 9.3v5.4M9 14.7V20" stroke="@A@" stroke-opacity=".65"/>'
    ),
    # -- utilidades ---------------------------------------------------------------------------
    "ayuda": (
        '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.6a2.5 2.5 0 1 1 3.4 2.3c-.8.4-1 1-1 1.7" stroke="@A@"/>'
        '<circle cx="12" cy="17" r=".7" fill="@A@" stroke="@A@"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5" stroke="@A@"/><circle cx="12" cy="7.8" r=".8" fill="@A@" stroke="@A@"/>'
    ),
    "quitar": '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6" stroke="@A@"/>',
    "eliminar": (
        '<path d="M4 7h16M10 7V4.5h4V7M6 7l1 13h10l1-13"/><path d="M10 11v6M14 11v6" stroke="@A@"/>'
    ),
    "advertencia": (
        '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.5" stroke="@A@"/><circle cx="12" cy="17.3" r=".8" fill="@A@" stroke="@A@"/>'
    ),
    "expandir": '<path d="M7 5l5 5 5-5M7 12l5 5 5-5"/>',
    "contraer": '<path d="M7 11l5-5 5 5M7 18l5-5 5 5"/>',
    "buscar": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21" stroke="@A@"/>',
    "correcto": '<circle cx="12" cy="12" r="9"/><path d="M7.8 12.4l2.9 2.9 5.6-6" stroke="@A@"/>',
}

#: Para qué tipo de elemento/asignación se usa cada icono (paneles y tablas).
ICONO_DE_TIPO = {
    "VIGA": "viga", "COLUMNA": "columna", "LOSA": "losa", "MURO": "muro", "SOLIDO": "solido",
    "Apoyo en superficie": "apoyo", "Constraint": "conectar", "Carga en superficie": "carga_dist",
    "Apoyo": "apoyo", "Apoyo de área": "apoyo_area", "Conexión": "conectar",
    "Carga de área": "carga", "Carga lineal": "carga_dist", "Carga puntual": "carga",
}


def svg_icono(nombre: str, trazo: str = COLOR_TRAZO, acento: str = COLOR_ACENTO) -> str:
    cuerpo = CUERPOS[nombre]
    return _PLANTILLA.format(cuerpo=cuerpo).replace("@T@", trazo).replace("@A@", acento)


def _renderizar(svg: str, lado: int, ratio: float = 1.0) -> QPixmap:
    px = QPixmap(int(lado * ratio), int(lado * ratio))
    px.setDevicePixelRatio(ratio)
    px.fill(Qt.transparent)
    pintor = QPainter(px)
    pintor.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(pintor, QRectF(0, 0, lado, lado))
    pintor.end()
    return px


_CACHE: Dict[Tuple[str, str, str], QIcon] = {}


def icono(nombre: str, trazo: str = COLOR_TRAZO, acento: str = COLOR_ACENTO) -> QIcon:
    """Icono del conjunto, con variantes normal y desactivada en varios
    tamaños (nítido en pantallas normales y de alta densidad)."""

    clave = (nombre, trazo, acento)
    if clave not in _CACHE:
        resultado = QIcon()
        svg_normal = svg_icono(nombre, trazo, acento)
        svg_gris = svg_icono(nombre, COLOR_DESACTIVADO, COLOR_DESACTIVADO)
        for lado in (16, 20, 24, 32, 48):
            for ratio in (1.0, 2.0):
                resultado.addPixmap(_renderizar(svg_normal, lado, ratio), QIcon.Normal, QIcon.Off)
                resultado.addPixmap(_renderizar(svg_gris, lado, ratio), QIcon.Disabled, QIcon.Off)
        _CACHE[clave] = resultado
    return _CACHE[clave]


def icono_blanco(nombre: str) -> QIcon:
    """Versión blanca (para botones de color, como ANALIZAR)."""

    return icono(nombre, "#ffffff", "#ffffff")


# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs>
    <linearGradient id="fondo" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0d2a50"/><stop offset="1" stop-color="#1f6feb"/>
    </linearGradient>
    <linearGradient id="esfuerzo" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#2f6bff"/><stop offset=".38" stop-color="#19c3a4"/>
      <stop offset=".7" stop-color="#f7d548"/><stop offset="1" stop-color="#ff6a3d"/>
    </linearGradient>
    <linearGradient id="brillo" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity=".18"/><stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="240" height="240" rx="58" fill="url(#fondo)"/>
  <rect x="8" y="8" width="240" height="120" rx="58" fill="url(#brillo)"/>
  <!-- cargas -->
  <g stroke="#ffffff" stroke-opacity=".85" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="M88 40h80"/>
    <path d="M96 40v26M128 40v26M160 40v26"/>
    <path d="M88 58l8 10 8-10M120 58l8 10 8-10M152 58l8 10 8-10"/>
  </g>
  <!-- columnas -->
  <rect x="66" y="112" width="20" height="92" rx="6" fill="#ffffff" fill-opacity=".95"/>
  <rect x="170" y="112" width="20" height="92" rx="6" fill="#ffffff" fill-opacity=".95"/>
  <!-- viga con mapa de esfuerzos -->
  <rect x="50" y="84" width="156" height="32" rx="11" fill="url(#esfuerzo)" stroke="#ffffff" stroke-width="4"/>
  <!-- apoyos -->
  <path d="M76 204l-19 20h38z" fill="#ffffff" fill-opacity=".95"/>
  <path d="M180 204l-19 20h38z" fill="#ffffff" fill-opacity=".95"/>
  <rect x="42" y="226" width="172" height="6" rx="3" fill="#ffffff" fill-opacity=".6"/>
  <!-- nudos -->
  <circle cx="76" cy="100" r="8" fill="#ffffff" stroke="#0d2a50" stroke-width="4"/>
  <circle cx="180" cy="100" r="8" fill="#ffffff" stroke="#0d2a50" stroke-width="4"/>
</svg>"""


def pixmap_logo(lado: int = 128, ratio: float = 1.0) -> QPixmap:
    return _renderizar(LOGO_SVG, lado, ratio)


def icono_aplicacion() -> QIcon:
    resultado = QIcon()
    for lado in (16, 24, 32, 48, 64, 128, 256):
        resultado.addPixmap(_renderizar(LOGO_SVG, lado, 1.0))
    return resultado


def guardar_logo(ruta_png: str, lado: int = 512) -> None:
    """Escribe el logo en un PNG (lo usa ``recursos/generar_logo.py``)."""

    imagen = QImage(lado, lado, QImage.Format_ARGB32)
    imagen.fill(Qt.transparent)
    pintor = QPainter(imagen)
    pintor.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(QByteArray(LOGO_SVG.encode("utf-8"))).render(pintor, QRectF(0, 0, lado, lado))
    pintor.end()
    imagen.save(ruta_png)
