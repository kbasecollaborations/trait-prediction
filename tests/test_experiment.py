from trait_prediction.pipeline import Experiment, ExperimentSet


def test_experiment_initialization(tmp_path):
    experiment = Experiment.initialize(tmp_path, seed=42, sep="_")
    assert experiment.experiment_dir.is_dir()
    assert experiment.experiment_dir in list(tmp_path.iterdir())
    assert experiment.metadata == {}


def test_experimentresult_initialization(tmp_path):
    experiment = Experiment.initialize(tmp_path, seed=42, sep="_")
    experiment_result = experiment.create_result(metadata={"dataset": "leaf"})
    assert experiment_result.run_dir.is_dir()
    assert experiment_result.run_dir in list(experiment.experiment_dir.iterdir())
    assert experiment_result.metadata == {}


def test_experimentresult_seed(tmp_path):
    experiment_1 = Experiment.initialize(tmp_path, seed=42, sep="_")
    experiment_result_1 = experiment_1.create_result(metadata={"dataset": "leaf"})
    experiment_2 = Experiment.initialize(tmp_path, seed=42, sep="_", resume=True)
    experiment_result_2 = experiment_2.create_result(metadata={"dataset": "leaf"})
    assert experiment_result_1.run_dir == experiment_result_2.run_dir


def test_experimentset_initialization(tmp_path):
    experimentset = ExperimentSet.initialize(tmp_path, sep="_")
    assert experimentset.experimentset_dir.is_dir()
    assert experimentset.experimentset_dir == tmp_path
    assert experimentset.metadata == {}


def test_experimentset_seed(tmp_path):
    experimentset_1 = ExperimentSet.initialize(tmp_path, sep="_")
    experimentset_2 = ExperimentSet.initialize(tmp_path, sep="_", resume=True)
    assert experimentset_1.experimentset_dir == experimentset_2.experimentset_dir


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
