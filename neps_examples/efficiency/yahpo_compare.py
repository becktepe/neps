import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import random

from neps.optimizers.multi_objective.parego_promotion_policy import ParEGOPromotionPolicy
from neps.optimizers.multi_objective.parego import ParEGO
from neps.optimizers.multi_objective.epsnet import EpsNet
from neps.optimizers.multi_objective.mo_sample_policy import MOEnsemblePolicy

from neps_examples.efficiency.yahpo_pb_parego import evaluate_mopb
from neps_examples.efficiency.yahpo_smac import evaluate_smac
from neps_examples.efficiency.yahpo_smac_mf import evaluate_smac_mf

OPENML_TASK_ID = "3945"
N_EVALUATIONS = [5, 10, 20, 35]
ETA = 2

sns.set(style="whitegrid")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


for n_evaluations in N_EVALUATIONS:
    result_parego_epsnet = evaluate_mopb(
        OPENML_TASK_ID,
        n_evaluations,
        eta=ETA,
    )
    result_smac = evaluate_smac(OPENML_TASK_ID, n_evaluations)
    result_smac_mf = evaluate_smac_mf(OPENML_TASK_ID, n_evaluations, eta=ETA)

    result_parego_epsnet["Approach"] = "MOPriorBand"
    result_smac["Approach"] = "SMAC"
    result_smac_mf["Approach"] = "SMAC MF"

    result = pd.concat([result_parego_epsnet, result_smac, result_smac_mf])

    plt.figure(figsize=(10,6))
    plt.title(f"OpenML task {OPENML_TASK_ID}, {n_evaluations} evaluations")
    sns.scatterplot(
        data=result, 
        x="loss",
        y="time", 
        hue="Approach", 
        size="budget",
        alpha=0.5
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f"./scatterplot_{n_evaluations}.png", dpi=500)

    pareto_fronts = {}

    for approach in result["Approach"].unique():
        approach_result = result[result["Approach"] == approach]
        pareto_front = approach_result.sort_values(by="time")
        pareto_front = pareto_front[pareto_front['loss'] == pareto_front['loss'].cummin()]
        pareto_fronts[approach] = pareto_front

        # Plotting the points and the Pareto front using Seaborn
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=approach_result, 
            x="time", 
            y="loss", 
            size="budget",
            color="blue"
        )
        sns.lineplot(data=pareto_front, x="time", y="loss", label="Pareto Front", color="red", marker="o")

        # Labels and title
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.title("Pareto Front Minimizing Time and Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./pareto_front_{n_evaluations}_{approach}.png", dpi=500)

    # Plot all pareto fronts in one plot

    plt.figure(figsize=(10,6))
    plt.title(f"OpenML task {OPENML_TASK_ID}, {n_evaluations} evaluations")

    for approach in pareto_fronts.keys():
        pareto_front = pareto_fronts[approach]
        sns.lineplot(data=pareto_front, x="time", y="loss", label=approach, marker="o")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f"./pareto_fronts_{n_evaluations}.png", dpi=500)

