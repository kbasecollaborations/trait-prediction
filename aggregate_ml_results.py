#!/usr/bin/env python3

import argparse
import pathlib
from calendar import c

import pandas as pd


def get_scores(
    folder: pathlib.Path, feature_name: str, run_id: str
) -> pd.DataFrame | None:
    scores_file = folder / "scores.csv"
    if not scores_file.is_file():
        return None
    scores_df = pd.read_csv(scores_file, index_col=0)
    scores_df["feature"] = feature_name
    scores_df["run_id"] = run_id
    return scores_df


def get_cv_scores(
    folder: pathlib.Path, feature_name: str, run_id: str
) -> pd.DataFrame | None:
    cv_scores_file = folder / "cv_scores.csv"
    if not cv_scores_file.is_file():
        return None
    cv_scores_df = pd.read_csv(cv_scores_file, index_col=0)
    cv_scores_df["fold_num"] = cv_scores_df.index + 1
    cv_scores_df["feature"] = feature_name
    cv_scores_df["run_id"] = run_id
    return cv_scores_df


def get_cv_features(
    folder: pathlib.Path,
    feature_name: str,
    run_id: str,
    n: int = 20,
) -> pd.DataFrame | None:
    cv_feature_files = list(folder.glob("cv_shap_features_*.csv"))
    if not cv_feature_files:
        return None
    features_list = []
    for cv_features_file in cv_feature_files:
        if not cv_features_file.is_file():
            continue
        cv_feature_df = pd.read_csv(cv_features_file, index_col=0)
        top_n_features = list(
            cv_feature_df.sort_values(by="0", ascending=False).index[:n]
        )
        # Create a new dataframe with the top n features along the rows
        features_list.append({f"feat_{k+1}": v for k, v in enumerate(top_n_features)})
    cv_features_df = pd.DataFrame(features_list)
    cv_features_df["fold_num"] = cv_features_df.index + 1
    cv_features_df["feature"] = feature_name
    cv_features_df["run_id"] = run_id
    cv_features_df["name"] = folder.stem
    cv_features_df["category"] = folder.parent.stem
    return cv_features_df


def main(
    prediction_runs: list[pathlib.Path],
    feature_list: list[str],
    datasets: list[str],
):
    for prediction_run in prediction_runs:
        run_id = prediction_run.stem
        print(f"Processing {prediction_run}")
        scores_all_feat_list = []
        cv_scores_all_feat_list = []
        cv_features_all_feat_list = []
        untrained_all_feat_list = []
        for feature in feature_list:
            feature_folder = prediction_run / feature
            if not feature_folder.is_dir():
                continue
            feature_name = feature
            for category_folder in feature_folder.iterdir():
                scores_data_list = []
                cv_scores_data_list = []
                cv_features_data_list = []
                untrained_data_list = []
                if not category_folder.is_dir():
                    continue
                if category_folder.stem not in datasets:
                    continue
                for phenotype_folder in category_folder.iterdir():
                    if not phenotype_folder.is_dir():
                        continue
                    scores_df = get_scores(phenotype_folder, feature_name, run_id)
                    cv_scores_df = get_cv_scores(phenotype_folder, feature_name, run_id)
                    cv_features_df = get_cv_features(
                        phenotype_folder, feature_name, run_id, n=20
                    )
                    if (
                        scores_df is None
                        or cv_scores_df is None
                        or cv_features_df is None
                    ):
                        untrained_data_list.append(
                            {
                                "category": category_folder,
                                "feature": feature_name,
                                "run_id": run_id,
                            }
                        )
                    else:
                        scores_data_list.append(scores_df)
                        cv_scores_data_list.append(cv_scores_df)
                        cv_features_data_list.append(cv_features_df)
                all_untrained_df = pd.DataFrame(untrained_data_list)
                untrained_all_feat_list.append(all_untrained_df)
                all_untrained_df.to_csv(
                    category_folder / "all_untrained_data.csv", sep=",", index=True
                )
                all_scores_df = pd.concat(scores_data_list)
                scores_all_feat_list.append(all_scores_df)
                all_scores_df.to_csv(
                    category_folder / "all_scores_data.csv", sep=",", index=True
                )
                all_cv_scores_df = pd.concat(cv_scores_data_list)
                cv_scores_all_feat_list.append(all_cv_scores_df)
                all_cv_scores_df.to_csv(
                    category_folder / "all_cv_scores_data.csv", sep=",", index=True
                )
                all_cv_features_df = pd.concat(cv_features_data_list)
                cv_features_all_feat_list.append(all_cv_features_df)
                all_cv_features_df.to_csv(
                    category_folder / "all_cv_features_data.csv", sep=",", index=True
                )

        all_untrained_df = pd.concat(untrained_all_feat_list)
        all_untrained_df.to_csv(
            prediction_run / "all_feat_untrained_data.csv", sep=",", index=True
        )
        all_scores_df = pd.concat(scores_all_feat_list)
        all_scores_df.to_csv(
            prediction_run / "all_feat_scores_data.csv", sep=",", index=True
        )
        all_cv_scores_df = pd.concat(cv_scores_all_feat_list)
        all_cv_scores_df.to_csv(
            prediction_run / "all_feat_cv_scores_data.csv", sep=",", index=True
        )
        all_cv_features_df = pd.concat(cv_features_all_feat_list)
        all_cv_features_df.to_csv(
            prediction_run / "all_feat_cv_features_data.csv", sep=",", index=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate ML results")
    parser.add_argument(
        "output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "dataset",
        type=str,
        required=True,
        help="Name of the dataset",
    )
    args = parser.parse_args()
    output_dir = pathlib.Path(args.output_dir)
    datasets = [args.dataset]
    prediction_runs = [fol for fol in output_dir.iterdir() if fol.is_dir()]
    feature_list = [
        "cluster30",
        "cluster50",
        "cluster70",
        "cluster90",
        "eggnog_kegg",
        "eggnog_seed",
        "kofam",
        "kofam_modules",
        "rast",
        "uniprot_trembl",
        "uniref30",
        "uniref50",
        "uniref70",
        "uniref90",
    ]
    main(prediction_runs, feature_list, datasets)
