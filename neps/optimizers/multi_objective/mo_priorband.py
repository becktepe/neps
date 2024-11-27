from __future__ import annotations

import typing

import numpy as np
from typing_extensions import Literal

from neps.search_spaces.search_space import SearchSpace
from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import (
    BaseAcquisition,
)
from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from neps.optimizers.multi_fidelity_prior.priorband import PriorBand
from neps.optimizers.multi_fidelity.promotion_policy import SyncPromotionPolicy
from neps.optimizers.multi_fidelity.sampling_policy import ModelPolicy
from neps.optimizers.multi_objective.mo_sample_policy import MOEnsemblePolicy
from neps.optimizers.multi_objective.multi_objective_optimizer import MultiObjectiveOptimizer
from neps.optimizers.multi_objective.epsnet import EpsNet
from neps.optimizers.multi_objective.parego import ParEGO


class MOPriorBand(PriorBand):
    def __init__(
        self,
            pipeline_space: SearchSpace,
            budget: int,
            objectives: list[str],
            mo_optimizer: type[MultiObjectiveOptimizer] = EpsNet,
            incumbent_selection: Literal["hypervolume", "pareto_front"] = "hypervolume",
            reference_point: list[float] | None = None,
            eta: int = 3,
            initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
            sampling_policy: typing.Any = MOEnsemblePolicy,
            promotion_policy: typing.Any = SyncPromotionPolicy,
            loss_value_on_error: None | float = None,
            cost_value_on_error: None | float = None,
            ignore_errors: bool = False,
            logger=None,
            prior_confidence: Literal["low", "medium", "high"] = "medium",
            random_interleave_prob: float = 0.0,
            sample_default_first: bool = True,
            sample_default_at_target: bool = True,
            prior_weight_type: str = "geometric",  # could also be {"linear", "50-50"}
            inc_sample_type: str = "mutation",  # or {"crossover", "gaussian", "hypersphere"}
            inc_mutation_rate: float = 0.5,
            inc_mutation_std: float = 0.25,
            inc_style: str = "dynamic",  # could also be {"decay", "constant"}
            # arguments for model
            model_based: bool = False,  # crucial argument to set to allow model-search
            modelling_type: str = "joint",  # could also be {"rung"}
            initial_design_size: int = None,
            model_policy: typing.Any = ModelPolicy,
            surrogate_model: str | typing.Any = "gp",
            domain_se_kernel: str = None,
            hp_kernels: list = None,
            surrogate_model_args: dict = None,
            acquisition: str | BaseAcquisition = "EI",
            log_prior_weighted: bool = False,
            acquisition_sampler: str | AcquisitionSampler = "random",
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            logger=logger,
            prior_confidence=prior_confidence,
            random_interleave_prob=random_interleave_prob,
            sample_default_first=sample_default_first,
            sample_default_at_target=sample_default_at_target,
            prior_weight_type=prior_weight_type,
            inc_sample_type=inc_sample_type,
            inc_mutation_rate=inc_mutation_rate,
            inc_mutation_std=inc_mutation_std,
            inc_style=inc_style,
            model_based=model_based,
            modelling_type=modelling_type,
            initial_design_size=initial_design_size,
            model_policy=model_policy,
            surrogate_model=surrogate_model,
            domain_se_kernel=domain_se_kernel,
            hp_kernels=hp_kernels,
            surrogate_model_args=surrogate_model_args,
            acquisition=acquisition,
            log_prior_weighted=log_prior_weighted,
            acquisition_sampler=acquisition_sampler,
        )
        self.mo_optimizer = mo_optimizer(objectives=objectives, reference_point=reference_point)
        self.incumbent_selection = incumbent_selection

    def set_sampling_weights_and_inc(self, rung: int):
        sampling_args = self.calc_sampling_args(rung)
        if not self.is_activate_inc():
            sampling_args["prior"] += sampling_args["inc"]
            sampling_args["inc"] = 0
            inc = None

            self.sampling_args = {"inc": inc, "weights": sampling_args}
        else:
            # For multi-objective optimization, the incumbent should be the
            # Pareto front, not the best performing configuration.
            if self.mo_optimizer is not None:
                if self.inc_sample_type != "mutation":
                    raise ValueError(
                        "Multi-objective optimization only supports incumbent sampling "
                        "via mutation. Please set `inc_sample_type='mutation'`."
                    )
                if not isinstance(self.sampling_policy, MOEnsemblePolicy):
                    raise ValueError(
                        "Multi-objective optimization only supports MOEnsemblePolicy. "
                        "Please set `sampling_policy=MOEnsemblePolicy`."
                    )
                
                if self.incumbent_selection == "hypervolume":
                    inc = self.mo_optimizer.get_incumbent()
                    self.logger.info("Using hypervolume as incumbent.")
                    self.logger.info(f"Incumbent: {inc}")
                elif self.incumbent_selection == "pareto_front":
                    inc = self.mo_optimizer.get_pareto_front()
                    self.logger.info("Using Pareto front as incumbent.")
                    self.logger.info(f"Incumbent: {inc}")
                else:
                    raise ValueError(
                        f"Invalid incumbent selection method: {self.incumbent_selection}"
                    )
            else:
                inc = self.find_incumbent()

            self.sampling_args = {"inc": inc, "weights": sampling_args}
            if self.inc_sample_type == "hypersphere":
                min_dist = self.find_1nn_distance_from_incumbent(inc)
                self.sampling_args.update({"distance": min_dist})
            elif self.inc_sample_type == "mutation":
                self.sampling_args.update(
                    {
                        "inc_mutation_rate": self.inc_mutation_rate,
                        "inc_mutation_std": self.inc_mutation_std,
                    }
                )
        return self.sampling_args