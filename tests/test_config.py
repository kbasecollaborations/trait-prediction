from trait_prediction.pipeline import Config


def test_default_config(default_config_path):
    Config.load_config(default_config_path)
