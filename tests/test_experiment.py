from trait_prediction.pipeline import Experiment


def test_experiment_initialization(tmp_path):
    experiment = Experiment.initialize(tmp_path, sep="_")
    assert experiment.experiment_dir.is_dir()
    assert experiment.experiment_dir in list(tmp_path.iterdir())
    assert experiment.metadata == {}


def test_experimentresult_initialization(tmp_path):
    experiment = Experiment.initialize(tmp_path, sep="_")
    experiment_result = experiment.create_result()
    assert experiment_result.run_dir.is_dir()
    assert experiment_result.run_dir in list(experiment.experiment_dir.iterdir())
    assert experiment_result.metadata == {}
