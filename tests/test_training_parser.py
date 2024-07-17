from trait_prediction.pipeline import TrainingParser


def test_training_parser(leaf_pipeline):
    pipeline = leaf_pipeline
    pipeline.run()
    experimentset_dir = pipeline.experimentset.experimentset_dir
    parser = TrainingParser(experimentset_dir)
    assert len(parser.metadata) == 4
    assert len(list(parser.metadata.values())[0]) == 6
    assert parser.scores.shape == (120, 40)
    assert parser.importances.shape[1] == 29
