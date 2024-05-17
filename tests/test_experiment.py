from trait_prediction.pipeline import Experiment, ExperimentSet


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


def test_experimentset_initialization(tmp_path):
    experimentset = ExperimentSet.initialize(tmp_path, sep="_")
    assert experimentset.experimentset_dir.is_dir()
    assert experimentset.experimentset_dir in list(tmp_path.iterdir())
    assert experimentset.metadata == {}


def test_experimentset_create_experiments(tmp_path, default_configset):
    experimentset = ExperimentSet.initialize(tmp_path, sep="_")
    configset = default_configset
    common_metadata = {
        "dataset": "leaf",
    }
    experimentset.create_experiments(configset, common_metadata=common_metadata)
    assert len(experimentset) == 4
    assert all([experiment.experiment_dir.is_dir() for experiment in experimentset])
    assert all(
        [
            experiment.experiment_dir in list(experimentset.experimentset_dir.iterdir())
            for experiment in experimentset
        ]
    )
    assert all(
        [
            (experiment.experiment_dir / "metadata.yaml").is_file()
            for experiment in experimentset
        ]
    )
