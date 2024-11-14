import numpy as np
from collections import defaultdict
from neps.utils.types import ConfigResult
from neps.optimizers.multi_objective.multi_objective_optimizer import MultiObjectiveOptimizer
from neps.search_spaces.search_space import SearchSpace

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
        self._results_per_rung = defaultdict(dict)
        self._rung = 0
    
    def add_config(self, config_id: str, is_default_config: bool = False) -> None:
        """Add a configuration to the optimizer."""
        pass

    def add_config_result(self, config_result: ConfigResult) -> None:
        """Register the result of a config to update the objective bounds."""
        super().add_config_result(config_result)

        config_id, rung = config_result.id.split("_")
        config_id, rung = int(config_id), int(rung)

        self._results_per_rung[rung][config_id] = config_result
        self._rung = rung
    
    def _compute_distance(self, result1: ConfigResult, result2: ConfigResult) -> float:
        """Compute the distance between two results."""
        assert isinstance(result1.result, dict)
        assert isinstance(result2.result, dict)
        
        r1 = np.array(list(result1.result.values()))
        r2 = np.array(list(result2.result.values()))

        return float(np.linalg.norm(r1 - r2))
    
    def get_pareto_front(self) -> list[SearchSpace]:
        """Get the Pareto front."""
        fronts = self._non_dominated_sorting(self._all_results)

        pareto_front = [self._all_results[config_id].config for config_id in fronts[0]]
        return pareto_front

    def get_result(self, config_result: ConfigResult) -> float:
        """Get the result of a configuration."""
        if not isinstance(config_result.result, dict):
            raise ValueError("ConfigResult.result should be a dictionary.")
        
        rung = int(config_result.id.split("_")[1])
        fronts = self._non_dominated_sorting(self._results_per_rung[rung])

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
                    distances = [self._compute_distance(self._results_per_rung[rung][config_id], self._results_per_rung[rung][c]) for c in sorted_configs]

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

        assert len(sorted_configs) == len(self._results_per_rung[rung])

        config_rank = ranks[int(config_result.id.split("_")[0])]

        return config_rank
       