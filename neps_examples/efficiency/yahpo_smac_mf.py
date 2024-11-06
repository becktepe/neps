import warnings
warnings.filterwarnings("ignore")

from yahpo_gym import benchmark_set
from yahpo_gym import local_config

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from smac.multi_objective.parego import ParEGO

import pandas as pd
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
)

from smac.multi_objective.parego import ParEGO


from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband


def evaluate_smac_mf(OpenML_task_id: str, n_evaluations: int = 10, eta: int = 3) -> pd.DataFrame:
    local_config.init_config()

    bench = benchmark_set.BenchmarkSet("lcbench")
    bench.set_instance(OpenML_task_id) 

    def run_pipeline(config: Configuration, seed: int = 0, budget: int = 25) -> dict:
        config = dict(config)
        config["OpenML_task_id"] = OpenML_task_id
        config["epoch"] = round(budget)
        config = Configuration(bench.get_opt_space(), values=config)

        result = bench.objective_function(config)[0]

        time = result["time"] / budget

        return {
            "loss": 1 - result["val_accuracy"],
            "time": time,
        }

    # Define our environment variables
    config_space = bench.get_opt_space()
    # remove epoch hyperparameter
    config_space = ConfigurationSpace(
        {k: v for k, v in config_space._hyperparameters.items() if k != "epoch"})

    # Define our environment variables
    scenario = Scenario(
        config_space,
        objectives=["loss", "time"],
        walltime_limit=60,  
        n_trials=200,  
        min_budget=1,  
        max_budget=52, 
        n_workers=1,
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    multi_objective_algorithm = ParEGO(scenario)

    intensifier = Hyperband(scenario, incumbent_selection="highest_budget", eta=eta)

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        run_pipeline,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
        multi_objective_algorithm=multi_objective_algorithm,
    )

    # Let's optimize
    incumbent = smac.optimize()

    loss, time, id, budget = [], [], [], []

    i = 0
    for k, v in smac.runhistory.items():
        loss.append(v.cost[0])
        time.append(v.cost[1])
        id.append(i)
        budget.append(k.budget)
        i += 1

    result = pd.DataFrame({"loss": loss, "time": time, "id": id, "budget": budget})
    
    result.loc[:, "budget_used"] = result["budget"].cumsum()
    allowed_budget = bench.get_opt_space()["epoch"].upper * n_evaluations
    result = result[result["budget_used"] <= allowed_budget]

    return result
