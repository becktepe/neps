from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from neps.utils.types import ConfigResult
from neps.search_spaces.search_space import SearchSpace

class MultiObjectiveOptimizer(ABC):
    __name__ = "MultiObjectiveOptimizer"

    """Base class for an multi-objective optimization algorithm."""
    def __init__(
            self,
            objectives: list[str],
        ) -> None:
        """
        Initialize an MultiObjective optimizer.
        
        Parameters
        ----------
        objectives : list[str]
            List of objectives to optimize.
        """
        self._objectives = objectives
        self._all_results: dict[str, ConfigResult] = {}
        self._objective_bounds = {objective: (np.inf, -np.inf) for objective in objectives}

        # Through normalization the values are transformed to the range [0, 1],
        # so we set the reference point to be slightly above 1.
        self._reference_point = [1.1] * len(objectives)

    def _normalize_objective(self, objective: str, value: float) -> float:
        """Normalize the objective value."""
        lower, upper = self._objective_bounds[objective]
        if np.isinf(lower) or np.isinf(upper) or lower == upper:
            return value

        return (value - lower) / (upper - lower)

    def _non_dominated_sorting(self, configs: dict[Any, ConfigResult]) -> list[list[Any]]:
        """Get the non-dominated configurations."""
        population = configs.copy()
        fronts = []  

        while population:
            current_front = [] 

            for config_id, config_result in population.items():
                dominated = False
                for other_id, other_config_result in population.items():
                    if config_id == other_id:
                        continue
                    
                    assert isinstance(config_result.result, dict)
                    assert isinstance(other_config_result.result, dict)

                    performance = list(config_result.result.values())
                    other_performance = list(other_config_result.result.values())

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
    
    def _get_pareto_front(self) -> list[ConfigResult]:
        """Get the Pareto front containing ConfigResults."""
        fronts = self._non_dominated_sorting(self._all_results)

        pareto_front = [self._all_results[config_id] for config_id in fronts[0]]
        return pareto_front
    
    def get_pareto_front(self) -> list[SearchSpace]:
        """Get the Pareto front containing actual configurations."""
        pareto_front = self._get_pareto_front()

        return [c.config for c in pareto_front]
    
    def _compute_area(self, config_result: ConfigResult) -> float:
        if config_result.result == "error":
            return 0

        objective_values = [
            self._normalize_objective(
                objective,
                config_result.result[objective]
            ) for objective in self._objectives
        ]

        return float(np.prod([np.abs(self._reference_point[i] - objective_values[i]) for i in range(len(objective_values))]))

    def get_incumbent(self) -> SearchSpace:
        """Compute the incumbent configuration based on the area spanned by configuration and reference point."""
        pareto_front = self._get_pareto_front()

        incumbent_config = None
        incumbent_area = -np.inf

        for config_result in pareto_front:
            area = self._compute_area(config_result)

            if area > incumbent_area:
                incumbent_config = config_result.config
                incumbent_area = area
        
        if incumbent_config is None:
            raise ValueError("No incumbent found.")
        
        return incumbent_config

    @abstractmethod
    def add_config(self, config_id: str, is_default_config: bool = False) -> None:
        """Add a configuration to the optimizer."""
        raise NotImplementedError

    def add_config_result(self, config_result: ConfigResult) -> None:
        """Register the result of a configuration."""
        if not isinstance(config_result.result, dict):
            raise ValueError("ConfigResult.result should be a dictionary.")
        self._all_results[config_result.id] = config_result
        self._objective_bounds = {
            objective: (
                min(self._objective_bounds[objective][0], config_result.result[objective]),
                max(self._objective_bounds[objective][1], config_result.result[objective]),
            )
            for objective in self._objectives
        }

    @abstractmethod
    def get_result(self, config_result: ConfigResult) -> float:
        """Scalarize the result of a configuration."""
        raise NotImplementedError