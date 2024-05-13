from pathlib import Path

import pytest


@pytest.fixture
def data_path():
    data_path = Path("tests/data")
    return data_path


@pytest.fixture
def default_config_path(data_path):
    config_path = data_path / "default.yml"
    return config_path
