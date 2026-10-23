"""Smoke test: the mibici package can be imported in the project environment."""

import mibici


def test_package_imports():
    # Si falla, el entorno no tiene el paquete instalado (revisar `make setup`)
    assert mibici is not None
