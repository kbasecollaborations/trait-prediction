from trait_prediction.pipeline import TrainingPipeline


def test_training_pipeline_init(
    default_configset,
    leaf_phenotype_pinputs,
    leaf_feature_finputs,
    classifier_factory,
    tmp_path,
):
    configset = default_configset
    pinputs = leaf_phenotype_pinputs
    finputs = leaf_feature_finputs
    output_dir = tmp_path
    n_cpus = 2
    pipeline = TrainingPipeline(
        configset,
        pinputs,
        finputs,
        classifier_factory,
        output_dir,
        n_cpus,
    )
    assert pipeline.experimentset.experimentset_dir.is_dir()
    assert pipeline.experimentset.experimentset_dir == tmp_path
    assert (pipeline.experimentset.experimentset_dir / "experimentset.log").is_file()
    assert (pipeline.experimentset.experimentset_dir / "metadata.json").is_file()
    assert pipeline.dataset is not None


def test_training_pipeline_run(leaf_pipeline):
    pipeline = leaf_pipeline
    pipeline.run()
