import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from neps_examples.efficiency.yahpo_pb_parego import evaluate_mopb
from neps_examples.efficiency.yahpo_smac import evaluate_smac
from neps_examples.efficiency.yahpo_smac_mf import evaluate_smac_mf

OPENML_TASK_ID = "3945"
N_EVALUATIONS = [10, 20, 35, 50]
ETA = [1, 2, 3]
TCHEBYCHEFF = [True, False]
SAMPLING = ["uniform", "dirichlet"]

for n_evaluations, tchebycheff, sampling, eta in itertools.product(N_EVALUATIONS, TCHEBYCHEFF, SAMPLING, ETA):
    par_ego_kwargs = {
        "tchebycheff": tchebycheff,
        "weight_distribution": sampling,
        "eta": eta
    }
    result_parego = evaluate_mopb(OPENML_TASK_ID, n_evaluations, par_ego_kwargs)
    result_smac = evaluate_smac(OPENML_TASK_ID, n_evaluations)
    # result_smac_mf = evaluate_smac_mf(OPENML_TASK_ID, n_evaluations)

    result_parego["Approach"] = "ParEGO + PB"
    result_smac["Approach"] = "SMAC"
    # result_smac_mf["Approach"] = "SMAC MF"

    result = pd.concat([result_parego, result_smac])
    # result = pd.concat([result_parego, result_smac, result_smac_mf])

    plt.figure()
    plt.title(f"OpenML task {OPENML_TASK_ID}, {n_evaluations} evaluations")
    sns.scatterplot(data=result, x="loss", y="time", hue="Approach", size="budget", alpha=0.5)

    filename = f"./scatterplot_{n_evaluations}_{sampling}_{eta}"
    if tchebycheff:
        filename += "_tchebycheff"

    plt.savefig(f"{filename}.png")

