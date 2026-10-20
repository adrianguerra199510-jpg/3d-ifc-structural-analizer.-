"""Prueba de la GUI con clics de mouse reales.

El visor 3D necesita OpenGL, que la plataforma Qt «offscreen» de pytest no
tiene; por eso el guion ``guion_gui_xvfb.py`` se ejecuta en un subproceso
bajo ``xvfb-run``. Se omite si faltan xvfb-run, ccx o PySide6/pyvista.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

pytestmark = [
    pytest.mark.skipif(shutil.which("ccx") is None, reason="ccx no está instalado en este sistema"),
    pytest.mark.skipif(shutil.which("xvfb-run") is None, reason="xvfb-run no está instalado en este sistema"),
]


def _ejecutar(guion: str, tmp_path: Path):
    raiz = Path(__file__).resolve().parents[1]
    entorno = dict(os.environ)
    entorno.pop("QT_QPA_PLATFORM", None)
    entorno["PYTHONPATH"] = os.pathsep.join([str(raiz / "src"), str(raiz)])
    proceso = subprocess.run(
        ["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24", sys.executable, "-m", f"tests.{guion}", str(tmp_path)],
        cwd=raiz, env=entorno, capture_output=True, text=True, timeout=600,
    )
    assert proceso.returncode == 0 and "OK" in proceso.stdout, proceso.stdout + proceso.stderr


def test_flujo_completo_con_mouse(tmp_path: Path):
    _ejecutar("guion_gui_xvfb", tmp_path)


def test_conexion_metalica_con_cargas_en_caras_extremas(tmp_path: Path):
    _ejecutar("guion_gui_conexion_xvfb", tmp_path)


def test_geometria_libre_apoyos_y_cargas_en_superficies(tmp_path: Path):
    _ejecutar("guion_gui_libre_xvfb", tmp_path)
