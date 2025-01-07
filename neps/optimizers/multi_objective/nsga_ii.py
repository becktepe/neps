import numpy as np
from collections import defaultdict
from neps.utils.types import ConfigResult
from neps.optimizers.multi_objective.multi_objective_optimizer import MultiObjectiveOptimizer
from neps.search_spaces.search_space import SearchSpace

import logging
logger = logging.getLogger("EpsNet")


class NSGAII(MultiObjectiveOptimizer):
    __name__ = "NSGAII"

    """Class for the NSGA-II-based multi-objective optimization algorithm."""
    def __init__(
            self,
            objectives: list[str],
            reference_point: list[float] | None = None,
        ) -> None:
        """
        Initialize the NSGA-II-based multi-objective optimizer.
        
        Parameters
        ----------
        objectives : list[str]
            List of objectives to optimize.
        reference_point : list[float] | None
            Reference point for the hypervolume calculation. Defaults to [0.0] * len(objectives).
        """
        super().__init__(objectives=objectives)

        logger.info("Initializing NSGA-II based MO-optimizer.")
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

    def crowding_distance_assignment(self, configs: dict[str, ConfigResult]) -> dict[str, float]:
        """Assign crowding distances to the configurations."""
        distances = {c: 0. for c in configs}

        for o in self._objectives: 
            o_min, o_max = self._objective_bounds[o]
            sorted_configs = sorted(configs.keys(), key=lambda c: configs[c].result[o])

            distances[sorted_configs[0]] = np.inf
            distances[sorted_configs[-1]] = np.inf

            for i in range(2, len(sorted_configs) - 1):
                config_key = sorted_configs[i]
                distances[config_key] += (configs[sorted_configs[i + 1]].result[o] - configs[sorted_configs[i + 1]].result[o]) / (o_max - o_min)

        return distances    

    def get_result(self, config_result: ConfigResult) -> float:
        """Get the result of a configuration."""
        if not isinstance(config_result.result, dict):
            raise ValueError("ConfigResult.result should be a dictionary.")
        
        rung = int(config_result.id.split("_")[1])
        rung_results = self._results_per_rung[rung]
        fronts = self._non_dominated_sorting(rung_results)
        sorted_configs = []

        for front in fronts:
            front_results = {c: rung_results[c] for c in front}
            crowding_distance = self.crowding_distance_assignment(front_results)

            # Since we want to sort ascendingly by crowding distance, we need to negate it
            # As second sorting criterion, we use the first objective since we consider it 
            # to be most important
            _sorted_configs = sorted(
                crowding_distance.keys(),
                key=lambda c: (-crowding_distance[c], rung_results[c].result[self._objectives[0]]),
            )
            sorted_configs.extend(_sorted_configs)

        # Now we have a list of config IDs sorted by their distance to the non-dominated front
        # The rank is the position of the configuration in the sorted list
        # We want to have a list of len(sorted_configs) with the rank of each configuration normalized to [0, 1]
        ranks = np.linspace(0, 1, len(sorted_configs))
        ranks = dict(zip(sorted_configs, ranks))

        assert len(sorted_configs) == len(rung_results)

        config_rank = ranks[int(config_result.id.split("_")[0])]

        return config_rank
       