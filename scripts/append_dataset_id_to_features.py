#!/usr/bin/env python3

from pathlib import Path

from trait_prediction.main import Feature, FeatureIndex, FeatureInput


def index_format_func(x):
    return (
        x.strip()
        .split("?")[-1]
        .removesuffix(".RAST")
        .removesuffix(".fna")
        .removeprefix("g")
    )


def main(
    data_folder: Path, dataset_id_map: dict[str, int], output_folder: Path
) -> None:
    for feature_file in data_folder.glob("**/*.tsv"):
        dataset_name = feature_file.parent.stem
        file_name = feature_file.name
        findex = FeatureIndex(name=dataset_name, ftype="binary", dtype="uint8")
        finput = FeatureInput(
            path=feature_file, findex=findex, index_format_func=index_format_func
        )
        feature = Feature.read_data(finput)
        feature_df = feature.feature_data
        feature_df["dataset"] = dataset_id_map[dataset_name]
        output_file = output_folder / dataset_name / file_name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_file, sep="\t", index=True)
        print(f"Processed {output_file}")


if __name__ == "__main__":
    data_folder = Path("../data/interim/features")
    output_folder = Path("../data/interim/features_mod/")
    dataset_id_map = {"atleaf": 0, "lit": 1, "pmi": 2}
    main(data_folder, dataset_id_map, output_folder)
