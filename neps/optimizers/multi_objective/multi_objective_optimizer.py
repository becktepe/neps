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
            reference_point: list[float] | None = None,
        ) -> None:
        """
        Initialize an MultiObjective optimizer.
        
        Parameters
        ----------
        objectives : list[str]
            List of objectives to optimize.
        """
        self._objectives = objectives
        self._reference_point = reference_point or [0.0] * len(objectives)
        self._all_results: dict[str, ConfigResult] = {}

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
    
    def get_incumbent(self) -> SearchSpace:
        """Compute the incumbent configuration based on the hypervolume."""
        pareto_front = self._get_pareto_front()
        
        incumbent_config = None
        incumbent_hypervolume = -np.inf

        hypervolumes = []

        for config_result in pareto_front:
            if config_result.result == "error":
                continue

            objectives = [config_result.result[objective] for objective in self._objectives]
            hypervolume = np.prod([self._reference_point[i] - objectives[i] for i in range(len(objectives))])
            hypervolumes.append(hypervolume)

            if hypervolume > incumbent_hypervolume:
                incumbent_config = config_result.config
                incumbent_hypervolume = hypervolume
        
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

    @abstractmethod
    def get_result(self, config_result: ConfigResult) -> float:
        """Scalarize the result of a configuration."""
        raise NotImplementedError