import yaml
import pytest
from src.utils import load_config

def test_load_config_successfully(tmp_path):
    """
    Tests that load_config can successfully load a valid YAML file.
    """
    config_content = {
        "paths": {
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv"
        },
        "debug_sample_size": 100
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    # Pass the path of the temporary config file to the function
    config = load_config(str(config_file))
    
    assert config is not None
    assert isinstance(config, dict)
    assert config["debug_sample_size"] == 100
    assert config["paths"]["train_csv"] == "data/train.csv"

def test_load_config_file_not_found():
    """
    Tests that load_config raises FileNotFoundError for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")

def test_load_config_default_path(monkeypatch):
    """
    Tests that load_config uses the default path if none is provided.
    This test assumes that 'configs/config.yaml' exists.
    """
    # This is more of an integration test, but useful.
    # We can use monkeypatch to ensure we are in the project root.
    monkeypatch.chdir('d:\\AIE_Project\\jigsaw')
    try:
        config = load_config()
        assert config is not None
        assert isinstance(config, dict)
    except FileNotFoundError:
        pytest.fail("Default config file 'configs/config.yaml' not found from project root.")
