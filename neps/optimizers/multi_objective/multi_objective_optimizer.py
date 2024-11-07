from abc import ABC, abstractmethod

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

    @abstractmethod
    def add_config(self, config_id: str, is_default_config: bool = False) -> None:
        """Add a configuration to the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def add_config_result(self, config_result: ConfigResult) -> None:
        """Register the result of a configuration."""
        raise NotImplementedError

    @abstractmethod
    def get_result(self, config_result: ConfigResult, rung: int | None = None) -> float:
        """Scalarize the result of a configuration."""
        raise NotImplementedError