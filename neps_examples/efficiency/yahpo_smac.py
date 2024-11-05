from yahpo_gym import benchmark_set
from yahpo_gym import local_config

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.multi_objective.parego import ParEGO
import ConfigSpace as CS

import pandas as pd


def evaluate_smac(OpenML_task_id: str, n_evaluations: int = 100) -> pd.DataFrame:
    local_config.init_config()

    bench = benchmark_set.BenchmarkSet("lcbench")
    bench.set_instance(OpenML_task_id) 

    def run_pipeline(config, seed: int = 0, budget: int = 52):
        config = dict(config)
        config["OpenML_task_id"] = OpenML_task_id
        config = CS.Configuration(bench.get_opt_space(), values={**config, "epoch": budget})

        result = bench.objective_function(config)[0]

        time = result["time"] / config["epoch"]

        return {
            "loss": 1 - result["val_accuracy"],
            "time": time,
        }

    objectives = ["loss", "time"]

    # Define our environment variables
    config_space = bench.get_opt_space()
    # remove epoch hyperparameter
    config_space = CS.ConfigurationSpace(
        {k: v for k, v in config_space._hyperparameters.items() if k != "epoch"})
    scenario = Scenario(
        config_space,
        objectives=objectives,
        n_trials=n_evaluations,  # Evaluate max 200 different trials
        n_workers=1,
        deterministic=True,
        min_budget=bench.get_opt_space()["epoch"].lower,
        max_budget=bench.get_opt_space()["epoch"].upper,
    )

    # We want to run five random configurations before starting the optimization.
    multi_objective_algorithm = ParEGO(scenario)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        run_pipeline,
        multi_objective_algorithm=multi_objective_algorithm,
        overwrite=True,
    )

    # Let's optimize
    incumbents = smac.optimize()

    loss, time, id = [], [], []

    i = 0
    for k, v in smac.runhistory.items():
        config = smac.runhistory.get_config(k.config_id)
        loss.append(v.cost[0])
        time.append(v.cost[1])
        print(k)
        print(v)
        print(config)
        id.append(i)
        i += 1

    budget = [bench.get_opt_space()["epoch"].upper] * len(loss)

    result = pd.DataFrame({"loss": loss, "time": time, "id": id, "budget": budget})

    return result

