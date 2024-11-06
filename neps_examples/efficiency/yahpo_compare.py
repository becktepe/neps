import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import random

from neps.optimizers.multi_objective.parego_promotion_policy import ParEGOPromotionPolicy

from neps_examples.efficiency.yahpo_pb_parego import evaluate_mopb
from neps_examples.efficiency.yahpo_smac import evaluate_smac
from neps_examples.efficiency.yahpo_smac_mf import evaluate_smac_mf

OPENML_TASK_ID = "3945"
N_EVALUATIONS = [10, 20, 30]
ETA = 2

sns.set(style="whitegrid")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


for n_evaluations in N_EVALUATIONS:
    result_parego_hb = evaluate_mopb(OPENML_TASK_ID, n_evaluations, eta=ETA)
    parego_band_kwargs = dict(
        promotion_policy=ParEGOPromotionPolicy,
    )
    result_parego_band = evaluate_mopb(OPENML_TASK_ID, n_evaluations, eta=ETA)
    result_smac = evaluate_smac(OPENML_TASK_ID, n_evaluations)
    result_smac_mf = evaluate_smac_mf(OPENML_TASK_ID, n_evaluations, eta=ETA)

    result_parego_hb["Approach"] = "ParEGO + PB"
    result_parego_band["Approach"] = "ParEGOBand"
    result_smac["Approach"] = "SMAC"
    result_smac_mf["Approach"] = "SMAC MF"

    result = pd.concat([result_parego_hb, result_parego_band, result_smac, result_smac_mf])

    plt.figure(figsize=(10,6))
    plt.title(f"OpenML task {OPENML_TASK_ID}, {n_evaluations} evaluations")
    sns.scatterplot(data=result, x="loss", y="time", hue="Approach", size="budget", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()

    plt.savefig(f"./scatterplot_{n_evaluations}.png", dpi=500)

    for approach in result["Approach"].unique():
        approach_result = result[result["Approach"] == approach]
        pareto_front = approach_result.sort_values(by="time")
        pareto_front = pareto_front[pareto_front['loss'] == pareto_front['loss'].cummin()]

        # Plotting the points and the Pareto front using Seaborn
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=approach_result, x="time", y="loss", label="Points", s=50, color="blue")
        sns.lineplot(data=pareto_front, x="time", y="loss", label="Pareto Front", color="red", marker="o")

        # Labels and title
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.title("Pareto Front Minimizing Time and Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./pareto_front_{n_evaluations}_{approach}.png", dpi=500)

