import warnings
warnings.filterwarnings("ignore")

from yahpo_gym import benchmark_set
from yahpo_gym import local_config

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from smac.multi_objective.parego import ParEGO
import ConfigSpace as CS

import pandas as pd


def evaluate_smac_mf(OpenML_task_id: str, n_trials: int = 23) -> pd.DataFrame:
    local_config.init_config()

    bench = benchmark_set.BenchmarkSet("lcbench")
    bench.set_instance(OpenML_task_id) 

    def run_pipeline(config, seed: int = 0, budget: int = 50):
        config = dict(config)
        config["OpenML_task_id"] = OpenML_task_id
        config["epoch"] = round(budget)
        config = CS.Configuration(bench.get_opt_space(), values=config)

        result = bench.objective_function(config)[0]

        time = result["time"] / budget

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
        n_trials=500, 
        walltime_limit=1000,
        n_workers=1,
        min_budget=1,
        max_budget=52
    )

    # We want to run five random configurations before starting the optimization.
    multi_objective_algorithm = ParEGO(scenario)

    # Create our SMAC object and pass the scenario and the train method
    intensifier = Hyperband(scenario)
    smac = MFFacade(
        scenario,
        run_pipeline,
        multi_objective_algorithm=multi_objective_algorithm,
        overwrite=True,
        intensifier=intensifier,
        initial_design = MFFacade.get_initial_design(scenario, n_configs=5)
    )

    # Let's optimize
    incumbents = smac.optimize()

    loss, time, id, budget = [], [], [], []

    i = 0
    for k, v in smac.runhistory.items():
        config = smac.runhistory.get_config(k.config_id)
        loss.append(v.cost[0])
        time.append(v.cost[1])
        id.append(i)
        print(v)
        print()
        exit()
        i += 1

    result = pd.DataFrame({"loss": loss, "time": time, "id": id, "budget": budget})

    return result

