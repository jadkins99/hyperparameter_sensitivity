import pandas as pd


def compute_scores(expected_performance):

    env_num = 5
    # only take hypers, algs which have been run on all envs
    a = (
        expected_performance.groupby(
            by=["alg_type", "gae_lambda", "ent_coef", "actor_lr", "critic_lr"]
        ).env_name.nunique()
        == env_num
    ).reset_index()
    #
    b = a[a.iloc[:, -1]].iloc[:, :-1]
    expected_performance_hypers_no_nan_all_envs = expected_performance.merge(
        b,
        on=["alg_type", "gae_lambda", "ent_coef", "actor_lr", "critic_lr"],
        how="inner",
    )

    # Identify Per-Environment Tuned Performance
    # Find the indices of the maximum percentile_normalized_return within each alg_type and env_name
    idx = expected_performance.groupby(["alg_type", "env_name"])[
        "percentile_normalized_return"
    ].idxmax()
    per_env_tuned_perf = expected_performance.loc[idx].reset_index(drop=True)

    #  Average Performance Across Environments
    per_env_avg_perf = (
        per_env_tuned_perf.groupby("alg_type")["percentile_normalized_return"]
        .mean()
        .reset_index()
    )
    per_env_avg_perf.rename(
        columns={"percentile_normalized_return": "per_env_tuned_performance"},
        inplace=True,
    )

    #  Compute Cross-Environment Tuned Performance
    avg_across_envs = (
        expected_performance_hypers_no_nan_all_envs.groupby(
            ["alg_type", "gae_lambda", "ent_coef", "actor_lr", "critic_lr"]
        )["percentile_normalized_return"]
        .mean()
        .reset_index()
    )

    # Find the indices of the maximum percentile_normalized_return within each alg_type
    idx_cross = avg_across_envs.groupby("alg_type")[
        "percentile_normalized_return"
    ].idxmax()
    cross_env_tuned_perf = avg_across_envs.loc[idx_cross].reset_index(drop=True)
    cross_env_tuned_perf.rename(
        columns={"percentile_normalized_return": "cross_env_tuned_performance"},
        inplace=True,
    )
    #  Compute Sensitivity
    sensitivity = per_env_avg_perf.merge(
        cross_env_tuned_perf[["alg_type", "cross_env_tuned_performance"]],
        on="alg_type",
        suffixes=("_per_env", "_cross_env"),
    )

    sensitivity["sensitivity_difference"] = (
        sensitivity["per_env_tuned_performance"]
        - sensitivity["cross_env_tuned_performance"]
    )
    print("Sensitivity Results:")
    print(
        sensitivity[["alg_type", "per_env_tuned_performance", "sensitivity_difference"]]
    )
    return sensitivity
