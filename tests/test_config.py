"""Configuración: localización del CalculiX incluido con la aplicación."""

from ifc_structural import config
from ifc_structural.config import Configuracion, resolver_ruta_ccx


def test_ruta_explicita_se_respeta(tmp_path, monkeypatch):
    assert resolver_ruta_ccx(r"C:\herramientas\ccx.exe") == r"C:\herramientas\ccx.exe"


def test_usa_ccx_incluido_junto_a_la_aplicacion(tmp_path, monkeypatch):
    (tmp_path / "ccx").mkdir()
    monkeypatch.setattr(config.sys, "platform", "win32")
    exe = tmp_path / "ccx" / "ccx.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(config, "carpeta_aplicacion", lambda: tmp_path)
    assert resolver_ruta_ccx("auto") == str(exe)
    assert resolver_ruta_ccx("ccx") == str(exe)
    assert Configuracion().ruta_ccx_efectiva == str(exe)


def test_sin_ccx_incluido_cae_al_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "carpeta_aplicacion", lambda: tmp_path)
    monkeypatch.setattr(config.shutil, "which", lambda nombre: "/usr/bin/ccx")
    assert resolver_ruta_ccx("auto") == "/usr/bin/ccx"
    monkeypatch.setattr(config.shutil, "which", lambda nombre: None)
    assert resolver_ruta_ccx("auto") == "ccx"


def test_portable_trabaja_junto_al_ejecutable(tmp_path, monkeypatch):
    monkeypatch.setattr(config.sys, "frozen", True, raising=False)
    monkeypatch.setattr(config, "carpeta_aplicacion", lambda: tmp_path)
    assert Configuracion().directorio_trabajo_absoluto == (tmp_path / "trabajo_ccx").resolve()
