# mypy: disable-error-code = assignment
from __future__ import annotations

import numpy as np

from ...search_spaces.search_space import SearchSpace

from ..multi_fidelity_prior.utils import (
    compute_config_dist,
    custom_crossover,
    local_mutation,
)

from neps.optimizers.multi_fidelity.sampling_policy import EnsemblePolicy

TOLERANCE = 1e-2  # 1%
SAMPLE_THRESHOLD = 1000  # num samples to be rejected for increasing hypersphere radius
DELTA_THRESHOLD = 1e-2  # 1%
TOP_EI_SAMPLE_COUNT = 10


class MOEnsemblePolicy(EnsemblePolicy):
    """Multi-objective ensemble of sampling policies including sampling randomly, from prior & incumbent.

    Args:
        SamplingPolicy ([type]): [description]
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        inc_type: str = "mutation",
        logger=None,
    ):
        """Samples a policy as per its weights and performs the selected sampling.

        Args:
            pipeline_space: Space in which to search
            inc_type: str
                if "hypersphere", uniformly samples from around the incumbent within its
                    distance from the nearest neighbour in history
                if "gaussian", samples from a gaussian around the incumbent
                if "crossover", generates a config by crossover between a random sample
                    and the incumbent
                if "mutation", generates a config by perturbing each hyperparameter with
                    50% (mutation_rate=0.5) probability of selecting each hyperparmeter
                    for perturbation, sampling a deviation N(value, mutation_std=0.5))
        """
        super().__init__(pipeline_space=pipeline_space, logger=logger)
        assert inc_type == "mutation"

    def sample(
        self, inc: list[SearchSpace] | SearchSpace | None = None, weights: dict[str, float] = None, *args, **kwargs
    ) -> SearchSpace:
        """Samples from the prior with a certain probability

        Returns:
            SearchSpace: [description]
        """
        if isinstance(inc, SearchSpace) or inc is None:
            return super().sample(incumbent=inc, weights=weights, *args, **kwargs)
        else:
            assert isinstance(inc, list)
            # We select on incumbent randomly
            inc_idx = np.random.choice(len(inc))
            incumbent = inc[inc_idx]
        
            return super().sample(incumbent=incumbent, weights=weights, *args, **kwargs)