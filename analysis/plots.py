import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools


def generate_sensitivity_plane(df, scores_old_lab):

    change_labels = {
        "lambda_ac": "PPO",
        "symlog_obs": "Symlog observation",
        "norm_obs": "Observation $\mu -\sigma$ normalization",
        "advn_norm_ema": "Percentile scaling",
        "advn_norm_max_ema": "Lower bounded percentile scaling",
        "advn_norm_mean": "Per-minibatch $\mu -\sigma$ normalization",
        "symlog_critic_targets": "Symlog value target",
    }

    alg_list = df.alg_type.unique()

    scores = {}
    plt.rcParams.update({"font.size": 18})
    plt.rcParams["font.family"] = "Times"
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    for alg in alg_list:
        scores[change_labels[alg]] = {
            "sensitivity": scores_old_lab.loc[scores_old_lab["alg_type"] == alg][
                "sensitivity_difference"
            ].item(),
            "performance": scores_old_lab.loc[scores_old_lab["alg_type"] == alg][
                "per_env_tuned_performance"
            ].item(),
        }

    CB_color_cycle = {
        "PPO": "#377eb8",
        "Symlog observation": "#ff7f00",
        "Observation $\mu -\sigma$ normalization": "#4daf4a",
        "Percentile scaling": "#f781bf",
        "Lower bounded percentile scaling": "#a65628",
        "Per-minibatch $\mu -\sigma$ normalization": "#984ea3",
        "Symlog value target": "#999999",
    }
    ref_sens = scores["PPO"]["sensitivity"]
    ref_perf = scores["PPO"]["performance"]

    unit_line = lambda x: ref_perf + 1.0 * (x - ref_sens)
    xs = np.linspace(0, 0.6, 1000)
    alpha = 0.1

    fig, ax = plt.subplots(layout="constrained")
    fig.supxlabel(
        "Sensitivity (x-axis)",
    )
    fig.suptitle(
        "Environment Tuned Performance (y-axis)",
        rotation="horizontal",
    )
    y_l_lim = ref_perf - 0.2
    y_u_lim = 1.5  # ref_perf + 0.12
    x_u_lim = 0.32  # ref_sens + 0.22
    x_l_Lim = max(0, ref_sens - 0.08)
    ax.set_ylim(y_l_lim, y_u_lim)
    ax.set_xlim(x_l_Lim, x_u_lim)
    ax.set_xticks([0.0, 0.15, 0.3])
    ax.set_yticks([1.0, 1.2, 1.5])

    ax.fill_between(
        xs[xs >= ref_sens],
        unit_line(xs[xs >= ref_sens]),
        y_u_lim,
        color="yellow",
        alpha=alpha,
    )
    ax.fill_between(
        xs[xs <= ref_sens],
        unit_line(xs[xs <= ref_sens]),
        ref_perf,
        color="blue",
        alpha=alpha,
    )

    ax.fill_between(
        xs[xs <= ref_sens],
        ref_perf,
        y_u_lim,
        color="green",
        alpha=alpha,
    )
    ax.fill_between(
        xs,
        0.0,
        unit_line(xs),
        color="red",
        alpha=alpha,
    )
    ax.fill_between(
        xs,
        unit_line(xs),
        ref_perf,
        color="red",
        alpha=alpha,
    )

    for key, score_dict in scores.items():

        plt.errorbar(
            score_dict["sensitivity"],
            score_dict["performance"],
            label=key,
            fmt="o",
            linewidth=2,
            capsize=6,
            c=CB_color_cycle[key],
        )

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # # Put a legend below current axis
    # ax.legend(
    #     loc="upper left",
    #     bbox_to_anchor=(0.9, 1),
    #     fancybox=True,
    #     shadow=True,
    #     ncol=5,
    # )
    plt.show()


def generate_dimensionality_plot(df_expected, scores):

    # only take hypers, algs which have been run on all envs
    a = (
        df_expected.groupby(
            by=["alg_type", "gae_lambda", "ent_coef", "actor_lr", "critic_lr"]
        ).env_name.nunique()
        == 5
    ).reset_index()
    #
    b = a[a.iloc[:, -1]].iloc[:, :-1]
    df_expected = df_expected.merge(
        b,
        on=["alg_type", "gae_lambda", "ent_coef", "actor_lr", "critic_lr"],
        how="inner",
    )

    # find default hypers
    cross_envs_hypers_algs = (
        df_expected.groupby(
            ["alg_type", "actor_lr", "critic_lr", "ent_coef", "gae_lambda"],
            dropna=True,
        )
        .mean(numeric_only=True)["percentile_normalized_return"]
        .reset_index()
    )

    idx = cross_envs_hypers_algs.groupby("alg_type")[
        "percentile_normalized_return"
    ].idxmax()

    cross_envs_alg_max_hypers = cross_envs_hypers_algs.loc[idx]

    hyper_list = ["ent_coef", "gae_lambda", "actor_lr", "critic_lr"]

    best_tune_n_performance = {}

    cross_envs_scores = (
        cross_envs_hypers_algs.groupby(["alg_type"])
        .max()["percentile_normalized_return"]
        .reset_index()
    )

    best_tune_n_performance = pd.DataFrame(
        0, index=np.arange(5), columns=df_expected.alg_type.unique()
    )

    # consider all n choose k ways to fix hypers
    most_important_hypers = {}
    for alg in df_expected.alg_type.unique():
        most_important_hypers[alg] = {}
        alg_df_subset = df_expected.loc[df_expected["alg_type"] == alg]
        alg_best_default_hypers = cross_envs_alg_max_hypers.loc[
            cross_envs_alg_max_hypers["alg_type"] == alg
        ]

        best_tune_n_performance[alg][4] = cross_envs_scores.loc[
            cross_envs_scores["alg_type"] == alg
        ]["percentile_normalized_return"].item()
        best_tune_n_performance[alg][0] = scores.loc[scores["alg_type"] == alg][
            "per_env_tuned_performance"
        ].item()  # scores[f"{alg}_performance"]

        for k in range(1, len(hyper_list)):
            combinations = itertools.combinations(hyper_list, k)
            for hyper_settings in combinations:
                alg_hyper_subset = alg_df_subset.copy()
                for hyper_setting in hyper_settings:
                    alg_hyper_subset = alg_hyper_subset.loc[
                        alg_hyper_subset[hyper_setting]
                        == alg_best_default_hypers[hyper_setting].item()
                    ]
                alg_hyper_subset_score = (
                    alg_hyper_subset.groupby(["env_name"])
                    .max()["percentile_normalized_return"]
                    .reset_index()["percentile_normalized_return"]
                    .mean()
                )
                if (
                    alg_hyper_subset_score.item()
                    > best_tune_n_performance[alg][
                        k
                    ]  # > best_tune_n_performance[alg][f"best_choose_{k}_performance"]
                ):
                    most_important_hypers[alg][k] = [
                        hyper for hyper in hyper_list if hyper not in hyper_settings
                    ]
                    best_tune_n_performance[alg][k] = alg_hyper_subset_score.item()

    # flip so that order goes from tuning zero hypers to 4 along axis 0
    best_tune_n_performance = best_tune_n_performance.iloc[::-1].reset_index()

    target_performances = 0.95 * best_tune_n_performance.iloc[4]

    # for key, val in most_important_hypers.items():
    #     print(key, ":", val)

    CB_color_cycle = {
        "lambda_ac": "#377eb8",
        "symlog_obs": "#ff7f00",
        "norm_obs": "#4daf4a",
        "advn_norm_ema": "#f781bf",
        "advn_norm_max_ema": "#a65628",
        "advn_norm_mean": "#984ea3",
        "symlog_critic_targets": "#999999",
    }

    fig, axs = plt.subplots(1, 6)
    plt.rcParams["font.family"] = "Times"
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7

    alg_list = df_expected.alg_type.unique()
    alg_list = list(alg_list)
    alg_list.remove("lambda_ac")
    for alg_idx in range(0, len(alg_list)):

        axs[alg_idx].tick_params(axis="both", which="major", labelsize=7)
        axs[alg_idx].tick_params(axis="both", which="minor", labelsize=8)

        color = CB_color_cycle[alg_list[alg_idx]]
        axs[alg_idx].plot(
            best_tune_n_performance[alg_list[alg_idx]],
            color=color,
            marker="o",
            markersize=4,
            label="line with marker",
        )
        axs[alg_idx].set_xticks([0, 1, 2, 3, 4])
        axs[alg_idx].set_yticks([1.0, 1.1, 1.2])

        axs[alg_idx].plot(
            best_tune_n_performance["lambda_ac"],
            color=CB_color_cycle["lambda_ac"],
            marker="o",
            markersize=4,
            label="line with marker",
        )

        for i in range(4):
            if (
                target_performances[alg_list[alg_idx]]
                < best_tune_n_performance[alg_list[alg_idx]][i + 1]
                and target_performances[alg_list[alg_idx]]
                > best_tune_n_performance[alg_list[alg_idx]][i]
            ):
                point = i
                break

        x = (
            target_performances[alg_list[alg_idx]]
            - best_tune_n_performance[alg_list[alg_idx]][point]
        ) / (
            best_tune_n_performance[alg_list[alg_idx]][point + 1]
            - best_tune_n_performance[alg_list[alg_idx]][point]
        ) + point
        axs[alg_idx].plot(
            x,
            target_performances[alg_list[alg_idx]],
            label="marker only",
            markersize=4,
            color=CB_color_cycle[alg_list[alg_idx]],
            marker="x",
        )
        if alg_list[alg_idx] != "symlog_critic_targets":

            axs[alg_idx].vlines(
                x,
                1.1,
                target_performances[alg_list[alg_idx]],
                linestyles="dashed",
                color=CB_color_cycle[alg_list[alg_idx]],
            )
        else:
            axs[alg_idx].vlines(
                x,
                0.98,
                target_performances[alg_list[alg_idx]],
                linestyles="dashed",
                color=CB_color_cycle[alg_list[alg_idx]],
            )

        for i in range(4):
            if (
                target_performances["lambda_ac"]
                < best_tune_n_performance["lambda_ac"][i + 1]
                and target_performances["lambda_ac"]
                > best_tune_n_performance["lambda_ac"][i]
            ):
                ppo_point = i
                break

        ppo_x = (
            target_performances["lambda_ac"]
            - best_tune_n_performance["lambda_ac"][ppo_point]
        ) / (
            best_tune_n_performance["lambda_ac"][ppo_point + 1]
            - best_tune_n_performance["lambda_ac"][ppo_point]
        ) + ppo_point
        axs[alg_idx].plot(
            ppo_x,
            target_performances["lambda_ac"],
            "go",
            label="marker only",
            markersize=4,
            marker="x",
            color=CB_color_cycle["lambda_ac"],
        )
        axs[alg_idx].vlines(
            ppo_x,
            1.1,
            target_performances["lambda_ac"],
            linestyles="dashed",
            color=CB_color_cycle["lambda_ac"],
        )
        axs[alg_idx].set_box_aspect(1)
        axs[alg_idx].set_xticks([0, 1, 2, 3, 4])
        if alg_list[alg_idx] == "symlog_critic_targets":
            axs[alg_idx].set_yticks([1.0, 1.1, 1.2])
            # axs[alg_idx].set_yticklabels([])
            # axs[alg_idx].set_xticklabels([])

        elif alg_list[alg_idx] == "advn_norm_ema" or "advn_norm_max_ema":
            axs[alg_idx].set_yticks([1.1, 1.2, 1.3])
            # axs[alg_idx].set_yticklabels([])
            # axs[alg_idx].set_xticklabels([])

        else:
            axs[alg_idx].set_yticks([1.1, 1.15, 1.25])
            # axs[alg_idx].set_yticklabels([])
            # axs[alg_idx].set_xticklabels([])

    fig.tight_layout()

    plt.show()
