import logging

import numpy as np

import neps
import neps.optimizers
import neps.optimizers.multi_objective
import neps.optimizers.multi_objective.parego


def run_pipeline(float1, float2, integer1, fidelity):
    loss = -float(float1 + integer1) / fidelity
    loss2 = -float(-float2 + integer1) / fidelity
    return {"loss": loss, "loss2": loss2}


pipeline_space = dict(
    float1=neps.FloatParameter(
        lower=1, upper=1000, log=False, default=600, default_confidence="medium"
    ),
    float2=neps.FloatParameter(
        lower=-10, upper=10, default=0, default_confidence="medium"
    ),
    integer1=neps.IntegerParameter(
        lower=0, upper=50, default=35, default_confidence="low"
    ),
    fidelity=neps.IntegerParameter(lower=1, upper=10, is_fidelity=True),
)

import datetime as datetime

# Get datetime string
datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

mo_optimizer = neps.optimizers.multi_objective.parego.ParEGO(objectives=["loss", "loss2"])

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/multifidelity_priors/" + datetime_str,
    max_evaluations_total=25,  # For an alternate stopping method see multi_fidelity.py,
    searcher="priorband",
    mo_optimizer=mo_optimizer
)

import pandas as pd

result = pd.read_csv(f"results/multifidelity_priors/{datetime_str}/summary_csv/config_data.csv")
result["id"] = np.arange(len(result))

# Dataframe has columns "id", "result.loss1", "result.loss2"
# Create scatterplot for two objectives and visualize id as color hue

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.scatterplot(data=result, x="result.loss", y="result.loss2", hue="id")
# Invert axis to show Pareto front
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.savefig("./results/multifidelity_priors/" + datetime_str + "/scatterplot.png")
