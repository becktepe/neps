import logging

import numpy as np

import neps
import neps.optimizers
import neps.optimizers.multi_objective
from neps.optimizers.multi_objective.mo_priorband import MOPriorBand
import pandas as pd

from neps.search_spaces.search_space import pipeline_space_from_configspace, SearchSpace


from yahpo_gym import benchmark_set
from yahpo_gym import local_config


def evaluate_mopb(
        OpenML_task_id: str,
        n_evaluations: int = 100,
        eta: int = 3,
        **searcher_kwargs
    ) -> pd.DataFrame:
    local_config.init_config()

    bench = benchmark_set.BenchmarkSet("lcbench")
    bench.set_instance(OpenML_task_id)

    pipeline_space = pipeline_space_from_configspace(bench.get_opt_space())
    pipeline_space.pop("OpenML_task_id")
    pipeline_space["epoch"].is_fidelity = True
    pipeline_space = SearchSpace(**pipeline_space)

    def run_pipeline(**config):
        config["OpenML_task_id"] = OpenML_task_id
        result = bench.objective_function(config)[0]

        time = result["time"] / config["epoch"]
        
        return {
            "loss": 1 - result["val_accuracy"],
            "time": time,
            "cost": config["epoch"]
        }

    import datetime as datetime

    # Get datetime string
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    searcher = MOPriorBand(
        pipeline_space=pipeline_space,
        budget=pipeline_space["epoch"].upper * n_evaluations,
        objectives=["loss", "time"],
        eta=eta,
        **searcher_kwargs
    )

    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/multifidelity_priors/" + datetime_str,
        searcher=searcher,
        budget=pipeline_space["epoch"].upper * n_evaluations,
    )

    result = pd.read_csv(f"results/multifidelity_priors/{datetime_str}/summary_csv/config_data.csv")
    result["id"] = np.arange(len(result))

    result = result.rename(columns={
        "result.loss": "loss",
        "result.time": "time",
        "config.epoch": "budget"
    })

    return result
