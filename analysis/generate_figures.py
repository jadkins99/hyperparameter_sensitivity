import pandas as pd
from utils import compute_scores
import os
import argparse
from plots import generate_sensitivity_plane, generate_dimensionality_plot

parser = argparse.ArgumentParser()
parser.add_argument("--plot_type", choices=["sensitivity", "dimensionality"], type=str)
args = parser.parse_args()

dataset_path = f"analysis/data_expected_over_seeds.csv"

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
else:
    raise Exception("Cannot find dataset")


scores = compute_scores(df)


if args.plot_type == "sensitivity":
    generate_sensitivity_plane(df, scores)
elif args.plot_type == "dimensionality":
    generate_dimensionality_plot(df, scores)
else:
    raise NotImplementedError("Plot type not supported")
