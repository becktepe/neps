import numpy as np
from typing import Literal

from neps.utils.types import ConfigResult
from neps.optimizers.multi_objective.multi_objective_optimizer import MultiObjectiveOptimizer

import logging
logger = logging.getLogger("ParEGO")

class ParEGO(MultiObjectiveOptimizer):
    __name__ = "ParEGO"

    """Class for the ParEGO multi-objective optimization algorithm.
    
    Based on https://meta-learn.github.io/2020/papers/24_paper.pdf
    """
    def __init__(
            self,
            objectives: list[str], 
            eta: int = 1,
            k: int = 3,
            tchebycheff: bool = True,
            objective_bounds: list | None = None,
            weight_distribution: Literal["uniform", "dirichlet"] = "uniform"
        ) -> None:
        """
        Initialize the ParEGO optimizer.
        
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

        logger.info("Initializing ParEGO.")
        self._eta = eta
        self._k = k
        self._tchebycheff = tchebycheff
        self._weight_distribution = weight_distribution
        self._config_weights = {}
        self._weights = self._sample_weights()

        # We assign configurations with the same weights to the same group
        # to allow for hyperband promotion based on groups
        self._config_groups = {}
        self._cur_config_group = 0

        if objective_bounds is None:
            # List of (min, max) tuples for each objective
            self._objective_bounds = [(None, None)] * len(objectives)

    def _update_objective_bounds(self, objectives: np.ndarray) -> None:
        """Update the bounds of the objectives."""
        for i, objective in enumerate(objectives):
            min_val, max_val = self._objective_bounds[i]

            min_val = min(min_val, objective) if min_val is not None else objective
            max_val = max(max_val, objective) if max_val is not None else objective

            self._objective_bounds[i] = (min_val, max_val)

    def _normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Normalize the objectives."""
        normalized_objectives = objectives.copy()
        
        for i, (min_val, max_val) in enumerate(self._objective_bounds):
            if min_val is not None and max_val is not None and max_val - min_val != 0:
                normalized_objectives[i] = (objectives[i] - min_val) / (max_val - min_val)

        return normalized_objectives

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize the weights."""
        return weights / np.sum(weights, axis=0)

    def _sample_weights(self) -> np.ndarray:
        """Sample weights for the objectives."""
        if self._weight_distribution == "uniform":
            # Sample len(self._objectives) x k weights
            weights = np.random.uniform(size=(len(self._objectives), self._k))
        else:
            weights = np.random.dirichlet(alpha=np.ones(shape=(len(self._objectives), self._k)))
            
        normalized_weights = self._normalize_weights(weights)

        return normalized_weights
    
    def _extract_config_id(self, config_id: str) -> str:
        """Extract the configuration ID from a full ID."""
        return config_id.split("_")[0]

    def _next_group(self) -> None:
        """Move to the next group of configurations."""
        logger.info(f"Moving to next group of configurations.")

        self._cur_config_group += 1
        self._weights = self._sample_weights()

    def get_group_id(self, config_id: str) -> int:
        """Get the group ID of a configuration."""
        config_id = self._extract_config_id(config_id)
        return self._config_groups[config_id]
    
    def add_config(self, config_id: str, is_default_config: bool = False) -> None:
        """Add a configuration to the optimizer."""
        config_id = self._extract_config_id(config_id)

        # This means that we are in the first stage of the current
        # hyperband bracket. Here we want so sample new weights
        if config_id not in self._config_weights:
            self._config_weights[config_id] = self._weights
            self._config_groups[config_id] = self._cur_config_group
            logger.info(f"Added configuration {config_id} to ParEGO.")

            # This ensure that we have groups of <eta> configurations
            if len(self._config_weights) % self._eta == 1 or is_default_config:
                self._next_group()
        
        # Configurations that we have already seen keep their weights 
        # until the bitter end :-)
        else:
            logger.info(f"Configuration {config_id} already exists in ParEGO.")

    def add_config_result(self, config_result: ConfigResult) -> None:
        """Register the result of a config to update the objective bounds."""
        if not isinstance(config_result.result, dict):
            raise ValueError("ConfigResult.result should be a dictionary.")
        
        objectives = np.array([config_result.result[objective] for objective in self._objectives])
        self._update_objective_bounds(objectives)

    def get_result(self, config_result: ConfigResult, rung: int | None = None) -> float:
        """Scalarize the result of a configuration."""
        if not isinstance(config_result.result, dict):
            raise ValueError("ConfigResult.result should be a dictionary.")
        
        objectives = np.array([config_result.result[objective] for objective in self._objectives])
        
        # Just in case add_config_result() was not called
        self._update_objective_bounds(objectives)

        normalized_objectives = self._normalize_objectives(objectives)

        config_id = self._extract_config_id(config_result.id)
        weights = self._config_weights[config_id]

        weighted_objectives = normalized_objectives * weights.T

        if self._tchebycheff:
            scalarized_objectives = np.max(weighted_objectives, axis=1) + 0.05 * np.sum(weighted_objectives, axis=1)
        else:
            scalarized_objectives = np.sum(weighted_objectives)

        return float(np.min(scalarized_objectives))
