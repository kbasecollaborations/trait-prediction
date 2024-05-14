from trait_prediction.pipeline import PredictionPipeline

from .conftest import make_classifier


def test_prediction_pipeline_init(
    default_config_path,
    leaf_phenotype_pinputs,
    leaf_feature_finputs,
    tmp_path,
    random_state,
):
    config_path = default_config_path / "default.yaml"
    pinputs = leaf_phenotype_pinputs
    finputs = leaf_feature_finputs
    output_dir = tmp_path
    n_cpus = 2
    pipeline = PredictionPipeline(
        config_path, pinputs, finputs, make_classifier, output_dir, n_cpus, random_state
    )
    assert pipeline.experiment.experiment_dir.is_dir()
    assert pipeline.experiment.experiment_dir in list(tmp_path.iterdir())
    assert (pipeline.experiment.experiment_dir / "experiment.log").is_file()
    assert (pipeline.experiment.experiment_dir / "metadata.yaml").is_file()
    assert pipeline.dataset is not None


def test_prediction_pipeline_run(leaf_pipeline):
    pipeline = leaf_pipeline
    pipeline.run()
