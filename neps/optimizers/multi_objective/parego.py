import numpy as np
from typing import Literal

from neps.utils.types import ConfigResult

import logging
logger = logging.getLogger("ParEGO")

class ParEGO:
    __name__ = "ParEGO"

    """Class for the ParEGO multi-objective optimization algorithm."""
    def __init__(
            self,
            objectives: list[str], 
            eta: float = 3,
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
        logger.info("Initializing ParEGO.")

        self._objectives = objectives
        self._eta = eta
        self._tchebycheff = tchebycheff
        self._weight_distribution = weight_distribution
        self._config_weights = {}
        self._weights = self._sample_weights()

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
            if min_val is not None and max_val is not None:
                normalized_objectives[i] = (objectives[i] - min_val) / (max_val - min_val)

        return normalized_objectives

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize the weights."""
        return weights / (np.sum(weights) + 1e-10)

    def _sample_weights(self) -> np.ndarray:
        """Sample weights for the objectives."""
        if self._weight_distribution == "uniform":
            weights = np.random.rand(len(self._objectives))
        else:
            weights = np.random.dirichlet(np.ones(len(self._objectives)))

        normalized_weights = self._normalize_weights(weights)

        return normalized_weights
    
    def _extract_config_id(self, config_id: str) -> str:
        """Extract the configuration ID from a full ID."""
        return config_id.split("_")[0]
        
    def add_config(self, config_id: str) -> None:
        """Add a configuration to the optimizer."""
        config_id = self._extract_config_id(config_id)
        if config_id not in self._config_weights:
            self._config_weights[config_id] = self._weights
            logger.info(f"Added configuration {config_id} to ParEGO.")

            if len(self._config_weights) % self._eta == 0:
                logger.info(f"Resampling Weights")
                self._weights = self._sample_weights()
        else:
            logger.info(f"Configuration {config_id} already exists in ParEGO.")

    def scalarize_result(self, config_result: ConfigResult) -> float:
        """Scalarize the result of a configuration."""
        if not isinstance(config_result.result, dict):
            raise ValueError("ConfigResult.result should be a dictionary.")
        
        objectives = np.array([config_result.result[objective] for objective in self._objectives])
        self._update_objective_bounds(objectives)

        normalized_objectives = self._normalize_objectives(objectives)

        config_id = self._extract_config_id(config_result.id)
        weights = self._config_weights[config_id]

        weighted_objectives = normalized_objectives * weights

        # SMAC ParEGO
        if self._tchebycheff:
            return float(np.max(weighted_objectives, axis=0) + 0.05 * np.sum(weighted_objectives, axis=0))
        else:
            return float(np.sum(weighted_objectives))
