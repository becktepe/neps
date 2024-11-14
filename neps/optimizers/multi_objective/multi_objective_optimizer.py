from abc import ABC, abstractmethod
from typing import Any

from neps.utils.types import ConfigResult


class MultiObjectiveOptimizer(ABC):
    __name__ = "MultiObjectiveOptimizer"

    """Base class for an multi-objective optimization algorithm."""
    def __init__(
            self,
            objectives: list[str]
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
    
    def get_pareto_front(self) -> list[ConfigResult]:
        """Get the Pareto front."""
        fronts = self._non_dominated_sorting(self._all_results)

        pareto_front = [self._all_results[config_id] for config_id in fronts[0]]
        return pareto_front

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