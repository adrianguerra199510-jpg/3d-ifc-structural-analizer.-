"""El logo y todos los iconos se dibujan (sin necesidad de OpenGL)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QGuiApplication, QIcon


@pytest.fixture(scope="module")
def app():
    return QGuiApplication.instance() or QGuiApplication([])


def test_todos_los_iconos_se_dibujan(app):
    from ifc_structural.gui import iconos

    for nombre in iconos.CUERPOS:
        pixmap = iconos.icono(nombre).pixmap(24, 24)
        imagen = pixmap.toImage()
        assert not imagen.isNull()
        assert any(imagen.pixelColor(x, y).alpha() > 0 for x in range(24) for y in range(24)), nombre
        # La variante desactivada también existe y es distinta del color normal
        assert not iconos.icono(nombre).pixmap(24, 24, QIcon.Disabled).isNull()


def test_los_iconos_de_tipo_existen(app):
    from ifc_structural.gui import iconos

    assert set(iconos.ICONO_DE_TIPO.values()) <= set(iconos.CUERPOS)


def test_logo_e_icono_de_la_aplicacion(app):
    from ifc_structural.gui import iconos

    assert not iconos.pixmap_logo(128).isNull()
    assert not iconos.icono_aplicacion().isNull()
    assert iconos.LOGO_SVG.startswith("<svg")
