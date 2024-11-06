from __future__ import annotations

import numpy as np
from neps.optimizers.multi_fidelity.promotion_policy import SyncPromotionPolicy


class ParEGOPromotionPolicy(SyncPromotionPolicy):
    __name__ = "ParEGOPromotionPolicy"

    """Implements a synchronous promotion from lower to higher fidelity.

    Promotes only when all predefined number of config slots are full.
    """
    def set_state(
        self,
        *,  # allows only keyword args
        max_rung: int,
        members: dict,
        performances: dict,
        config_map: dict,
        group_ids: dict,
        **kwargs,
    ) -> None:
        super().set_state(
            max_rung=max_rung,
            members=members,
            performances=performances,
            config_map=config_map
        )
        self.rung_members_group_ids = group_ids

    def retrieve_promotions(self) -> dict:
        """Returns the top 1/eta configurations per rung if enough configurations seen"""
        assert self.config_map is not None

        self.rung_promotions = {rung: [] for rung in self.config_map.keys()}
        self.dummy_rung_promotions = {rung: [] for rung in self.config_map.keys()}
        total_rung_evals = 0
        for rung in reversed(sorted(self.config_map.keys())):
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

                # if rung is full and no incomplete evaluations, find promotions
                configs_to_promote = (self.config_map[rung] // self.eta) - (
                    self.config_map[rung] - len(self.rung_members[rung])
                )

                # Now we can compute how many configurations we need to promote for each group
                group_ids_to_promote = dict(
                    zip(
                        *np.unique(
                            self.rung_members_group_ids[rung], return_counts=True
                        )
                    )
                )

                # We only need to promote configurations from groups that are not already in the next rung
                group_ids_to_promote = {
                    group_id: count // self.eta for group_id, count in group_ids_to_promote.items()}

                promoted_members = []  
                if configs_to_promote == 0:
                    continue
                elif configs_to_promote == 1 and sum(group_ids_to_promote.values()) == 0:
                    # For the final rung, we only promote the best performing configuration
                    selected_idx = np.argmin(self.rung_members_performance[rung])
                    promoted_members = [self.rung_members[rung][selected_idx]]
                elif len(self.rung_members[rung]) > 0:
                    import pandas as pd
                    groups_and_performances = pd.DataFrame({
                        "id": self.rung_members[rung],
                        "group": self.rung_members_group_ids[rung],
                        "performance": self.rung_members_performance[rung]}
                    )

                    # Now we select the best performing members within each group
                    for group_id, top_k in group_ids_to_promote.items():
                        group_members = groups_and_performances[groups_and_performances["group"] == group_id]
                        group_members = group_members.sort_values("performance", ascending=True)
                        group_members = group_members.head(top_k)
                        promoted_members.append(group_members["id"].values)

                    if len(promoted_members) > 0:
                        promoted_members = np.concatenate(promoted_members)

                self.dummy_rung_promotions[rung] = list(promoted_members)

        for rung in self.dummy_rung_promotions.keys():
            assert len(self.rung_promotions[rung]) == len(self.dummy_rung_promotions[rung])

        return self.dummy_rung_promotions


