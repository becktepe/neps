from __future__ import annotations

import random

from ...search_spaces.search_space import SearchSpace
from ...utils.result_utils import get_loss
from ..base_optimizer import BaseOptimizer


class RegularizedEvolution(BaseOptimizer):
    def __init__(
        self,
        pipeline_space: SearchSpace,
        population_size: int = 30,
        sample_size: int = 10,
        patience: int = 100,
        budget: None | int | float = None,
        logger=None,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            patience=patience,
            logger=logger,
            budget=budget,
        )

        if population_size < 1:
            raise ValueError("RegularizedEvolution needs a population size >= 1")
        self.population_size = population_size
        self.sample_size = sample_size
        self.population: list = []
        self.num_train_x: int = 0

    def load_results(self, previous_results: dict, pending_evaluations: dict) -> None:
        train_x = [el.config for el in previous_results.values()]
        train_y = [get_loss(el.result) for el in previous_results.values()]
        self.num_train_x = len(train_x)
        self.population = [
            (x, y)
            for x, y in zip(
                train_x[-self.population_size :], train_y[-self.population_size :]
            )
        ]
        if bool(pending_evaluations):
            raise NotImplementedError(
                "Pending evaluations are currently not supported for RegularizedEvolution"
            )

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        if len(self.population) < self.population_size:
            config = self.pipeline_space.sample(patience=self.patience, user_priors=True)
        else:
            candidates = [random.choice(self.population) for _ in range(self.sample_size)]
            parent = min(candidates, key=lambda c: c[1])[0]
            config = self._mutate(parent)
            if config is False:
                config = self.pipeline_space.sample(
                    patience=self.patience, user_priors=True
                )
        config_id = str(self.num_train_x + 1)
        return config, config_id, None

    def _mutate(self, parent):
        for _ in range(self.patience):
            try:
                # needs to throw an Exception if config is not valid, e.g., empty graph etc.!
                return parent.mutate()
            except Exception:
                continue
        return False
