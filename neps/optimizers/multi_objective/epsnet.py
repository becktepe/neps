import numpy as np
from typing import Literal
from collections import defaultdict
from neps.utils.types import ConfigResult
from neps.optimizers.multi_objective.multi_objective_optimizer import MultiObjectiveOptimizer

import logging
logger = logging.getLogger("EpsNet")

class EpsNet(MultiObjectiveOptimizer):
    __name__ = "EpsNet"

    """Class for the EpsNet multi-objective optimization algorithm."""
    def __init__(
            self,
            objectives: list[str],
        ) -> None:
        """
        Initialize the EpsNet optimizer.
        
        Parameters
        ----------
        objectives : list[str]
            List of objectives to optimize.
        eta : float, optional
            Number of configurations to sample before resampling weights, by default 3.
        tchebycheff : bool, optional
            Whether to use the Tchebycheff scalarization function, by default True.
        objective_bounds : list | None, optional
            List of (min, max) tuples for each objective, by default None.
        weight_distribution : Literal["uniform", "dirichlet"], optional
            Distribution to sample weights from, by default "uniform".
        """
        super().__init__(objectives)

        logger.info("Initializing EpsNet.")
        self._results = defaultdict(dict)
        self._rung = 0
    
    def add_config(self, config_id: str, is_default_config: bool = False) -> None:
        """Add a configuration to the optimizer."""

    def add_config_result(self, config_result: ConfigResult) -> None:
        """Register the result of a config to update the objective bounds."""
        if not isinstance(config_result.result, dict):
            raise ValueError("ConfigResult.result should be a dictionary.")
        
        config_id, rung = config_result.id.split("_")
        config_id, rung = int(config_id), int(rung)

        result = tuple(config_result.result.values())

        self._results[rung][config_id] = result
        self._rung = rung

    def _non_dominated_sorting(self, configs: dict[int, tuple]) -> list[list[int]]:
        """Get the non-dominated configurations."""
        population = configs.copy()
        fronts = []  # config IDs for each front

        while population:
            current_front = []  # config IDs of current non-dominated front

            for config_id, performance in population.items():
                dominated = False
                for other_id, other_performance in population.items():
                    if other_id != config_id:
                        if (other_performance[0] <= performance[0] and other_performance[1] <= performance[1]) and (
                                other_performance[0] < performance[0] or other_performance[1] < performance[1]):
                            dominated = True
                            break

                if not dominated:
                    current_front.append(config_id)

            fronts.append(current_front)

            for config_id in current_front:
                population.pop(config_id)

        return fronts
    
    def _compute_distance(self, result1: tuple, result2: tuple) -> float:
        """Compute the distance between two results."""
        return float(np.linalg.norm(np.array(result1) - np.array(result2)))

    def get_result(self, config_result: ConfigResult, rung: int | None = None) -> float:
        """Get the result of a configuration."""
        if not isinstance(config_result.result, dict):
            raise ValueError("ConfigResult.result should be a dictionary.")
        
        rung = int(config_result.id.split("_")[1])
        fronts = self._non_dominated_sorting(self._results[rung])

        sorted_configs = [] 

        # We start by selecting the first element of the first front
        sorted_configs.append(fronts[0].pop(0))  

        # Iterate over each front
        for front in fronts:
            while front:
                # Find the configuration in the current that maximizes the minimum distance to the current set C
                max_min_distance = -np.inf
                best_config = None

                for config_id in front:
                    # Compute the minimum distance between the current config and each config in C
                    distances = [self._compute_distance(self._results[rung][config_id], self._results[rung][c]) for c in sorted_configs]

                    min_distance = min(distances)
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_config = config_id
                
                # Add the selected configuration to C and remove it from F
                if best_config is not None:
                    sorted_configs.append(best_config)
                    front.remove(best_config)


        # Now we have a list of config IDs sorted by their distance to the non-dominated front
        # The rank is the position of the configuration in the sorted list
        # We want to have a list of len(sorted_configs) with the rank of each configuration normalized to [0, 1]
        ranks = np.linspace(0, 1, len(sorted_configs))
        ranks = dict(zip(sorted_configs, ranks))

        assert len(sorted_configs) == len(self._results[rung])

        return ranks[int(config_result.id.split("_")[0])]
       