from __future__ import annotations

import numpy as np
from neps.optimizers.multi_fidelity.promotion_policy import SyncPromotionPolicy


class EpsNetPromotionPolicy(SyncPromotionPolicy):
    __name__ = "EpsNetPromotionPolicy"

    """Implements a synchronous promotion from lower to higher fidelity.

    Promotes only when all predefined number of config slots are full.
    Based on https://arxiv.org/pdf/2106.12639.
    """

    def _non_dominated_sorting(self, population: np.ndarray) -> np.ndarray:
        pass

    def retrieve_promotions(self) -> dict:
        """Returns the top 1/eta configurations per rung if enough configurations seen"""
        assert self.config_map is not None

        self.rung_promotions = {rung: [] for rung in self.config_map.keys()}
        total_rung_evals = 0
        for rung in reversed(sorted(self.config_map.keys())):
            print(self.rung_members_performance[rung])
            total_rung_evals += len(self.rung_members[rung])
            if (
                total_rung_evals >= self.config_map[rung]
                and np.isnan(self.rung_members_performance[rung]).sum()
            ):
                # if rung is full but incomplete evaluations, pause on promotions, wait
                return self.rung_promotions
            if rung == self.max_rung:
                # cease promotions for the highest rung (configs at max budget)
                continue
            if (
                total_rung_evals >= self.config_map[rung]
                and np.isnan(self.rung_members_performance[rung]).sum() == 0
            ):
                # if rung is full and no incomplete evaluations, find promotions
                top_k = (self.config_map[rung] // self.eta) - (
                    self.config_map[rung] - len(self.rung_members[rung])
                )
                selected_idx = np.argsort(self.rung_members_performance[rung])[:top_k]
                self.rung_promotions[rung] = self.rung_members[rung][selected_idx]

        return self.rung_promotions


