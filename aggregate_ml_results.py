#!/usr/bin/env python3

import pathlib

import pandas as pd


def get_scores(folder: pathlib.Path, feature_name: str) -> pd.DataFrame | None:
    scores_file = folder / "scores.csv"
    if not scores_file.is_file():
        return None
    scores_df = pd.read_csv(scores_file, index_col=0)
    scores_df["feature"] = [feature_name]
    return scores_df


def main(input_dirs: list[pathlib.Path], datasets: list[str], output_dir: pathlib.Path):
    scores_data_list = []
    untrained_data_list = []
    for input_dir in input_dirs:
        feature_name = input_dir.stem
        for category in input_dir.iterdir():
            if not category.is_dir():
                continue
            if category.stem not in datasets:
                continue
            for name in category.iterdir():
                if not name.is_dir():
                    continue
                scores_df = get_scores(name, feature_name)
                if scores_df is None:
                    untrained_data_list.append(
                        {
                            "category": category,
                            "name": name,
                        }
                    )
                else:
                    scores_data_list.append(scores_df)
    untrained_df = pd.DataFrame(untrained_data_list)
    untrained_df.to_csv(output_dir / "untrained_data.csv", sep=",", index=True)
    all_scores_df = pd.concat(scores_data_list)
    all_scores_df.to_csv(output_dir / "all_scores_data.csv", sep=",", index=True)


if __name__ == "__main__":
    output_dir = pathlib.Path("data/outputs/biolog")
    input_dirs = [
        output_dir / "cluster30",
        output_dir / "cluster70",
        output_dir / "eggnog_kegg",
        output_dir / "kofam",
        output_dir / "kofam_modules",
        output_dir / "rast",
        output_dir / "uniprot_trembl",
        output_dir / "uniref30",
        output_dir / "uniref90",
    ]
    # change this to include other datasets
    datasets = ["ch_biolog"]
    main(input_dirs, datasets, output_dir)
