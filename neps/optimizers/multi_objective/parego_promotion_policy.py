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
                group_ids_to_promote = {
                    group_id: 1 for group_id in np.unique(self.rung_members_group_ids[rung])

                }

                if configs_to_promote != len(group_ids_to_promote):
                    print(f"Rung {rung} configs_to_promote: {configs_to_promote}")
                    print(f"Rung {rung} group_ids_to_promote: {group_ids_to_promote}")
                    exit()

                # First, we check how many configurations of each group are already in the next rung
                next_rung = rung + 1
                if next_rung in self.config_map.keys():
                    for group_id, next_rung_count in zip(*np.unique(
                                self.rung_members_group_ids[next_rung], return_counts=True)):
                        group_ids_to_promote[group_id] -= next_rung_count

                print(f"Rung {rung} configs_to_promote: {configs_to_promote}")
                print(f"Rung {rung} group_ids_to_promote: {group_ids_to_promote}")

                promoted_members = []  
                if configs_to_promote == 0:
                    continue
                elif configs_to_promote == 1:
                    # For the final rung, we only promote the best performing configuration
                    selected_idx = np.argmin(self.rung_members_performance[rung])
                    promoted_members = [self.rung_members[rung][selected_idx]]
                # elif len(self.rung_members[rung]) > 0:
                else:
                    import pandas as pd
                    groups_and_performances = pd.DataFrame({
                        "id": self.rung_members[rung],
                        "group": self.rung_members_group_ids[rung],
                        "performance": self.rung_members_performance[rung]}
                    )

                    # Now we select the best performing member within each group
                    selected_ids_and_performances = []
                    for group_id, members_to_promote in group_ids_to_promote.items():
                        if members_to_promote == 0:
                            continue

                        group_members = groups_and_performances[groups_and_performances["group"] == group_id]
                        group_members = group_members.sort_values("performance", ascending=True)

                        selected_ids_and_performances += [{
                            "id": group_members["id"].values[0],
                            "performance": group_members["performance"].values[0]
                        }]

                    selected_ids_and_performances = pd.DataFrame(selected_ids_and_performances)
                    selected_ids_and_performances = selected_ids_and_performances.sort_values("performance", ascending=True)
                    promoted_members = selected_ids_and_performances["id"].values

                self.dummy_rung_promotions[rung] = list(promoted_members)

        for rung in self.dummy_rung_promotions.keys():
            if not len(self.rung_promotions[rung]) == len(self.dummy_rung_promotions[rung]):
                print(f"Rung promotions {len(self.rung_promotions[rung])}: {self.rung_promotions[rung]}")
                print(f"Dummy rung promotions {len(self.dummy_rung_promotions[rung])}: {self.dummy_rung_promotions[rung]}")
                print(f"Rung members: {self.rung_members}")
                print(f"Rung performances: {self.rung_members_performance}")
                raise ValueError("Promotions and dummy promotions do not match")

        return self.dummy_rung_promotions


