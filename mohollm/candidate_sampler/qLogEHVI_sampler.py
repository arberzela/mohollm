import logging
from typing import List, Dict

import torch
from torch import tensor

from botorch.utils.transforms import normalize, unnormalize
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.pareto import is_non_dominated
from mohollm.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER

logger = logging.getLogger("qLogEHVISampler")


class qLogEHVISampler(CANDIDATE_SAMPLER):
    def __init__(self):
        super().__init__()
        self.fitted_model = None
        self.global_bounds = None
        self.config_keys = None
        self.train_Y_neg = None
        self.ref_point_tensor = None
        self.pareto_Y = None

    def fit_model(self):
        """Fit GP models once on the current observed data."""
        logger.debug("Fitting GP models on observed data")

        # Get global bounds from config_space
        if self.config_space is None:
            logger.warning("No config_space available. Cannot fit model.")
            return

        self.config_keys = list(self.config_space.keys())

        # Build global bounds tensor from config_space
        global_lowers = []
        global_uppers = []
        for k in self.config_keys:
            b = self.config_space[k]
            # expect range-like = [low, high]
            if (
                isinstance(b, list)
                and len(b) == 2
                and all(isinstance(x, (int, float)) for x in b)
            ):
                global_lowers.append(float(b[0]))
                global_uppers.append(float(b[1]))
            else:
                # fallback: if categorical, map to integer index range
                global_lowers.append(0.0)
                global_uppers.append(float(len(b) - 1))

        self.global_bounds = tensor([global_lowers, global_uppers], dtype=torch.double)

        # Get observed data
        observed_configs = self.statistics.observed_configs
        observed_fvals = self.statistics.observed_fvals

        if not observed_configs or len(observed_configs) < 4:
            logger.warning("Not enough observed data to fit GP models")
            return

        # Build train_X and train_Y tensors
        train_X_list = [
            [float(cfg.get(k)) for k in self.config_keys] for cfg in observed_configs
        ]

        # Determine objective keys / order
        if self.metrics:
            obj_keys = list(self.metrics)
        else:
            # infer from observed_fvals dict
            obj_keys = list(observed_fvals[0].keys())

        train_Y_list = [
            [float(fval.get(k)) for k in obj_keys] for fval in observed_fvals
        ]

        dtype = torch.double
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_X = tensor(train_X_list, dtype=dtype, device=device)
        train_Y = tensor(train_Y_list, dtype=dtype, device=device)

        # Normalize X to [0,1] using global bounds
        train_X_normalized = normalize(
            train_X, self.global_bounds.to(device=device, dtype=dtype)
        )

        # BoTorch expects to maximize; if our fvals are minimization objectives we negate
        # Assume observed_fvals are minimization (common in this codebase) -> negate
        self.train_Y_neg = -train_Y

        # Fit independent GPs for each objective
        models = []
        for i in range(self.train_Y_neg.shape[-1]):
            train_y = self.train_Y_neg[..., i : i + 1]
            gp = SingleTaskGP(train_X_normalized, train_y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        self.fitted_model = ModelListGP(*models)

        # Compute reference point and Pareto front
        self.ref_point_tensor = self.train_Y_neg.min(0).values

        with torch.no_grad():
            pareto_mask = is_non_dominated(self.train_Y_neg)
            self.pareto_Y = self.train_Y_neg[pareto_mask]

        logger.debug("GP models fitted successfully")

    def generate_points(
        self, target_number_of_candidates: int, optionals: dict = {}
    ) -> List[Dict]:
        """Generate candidates using qLogExpectedHypervolumeImprovement.

        This builds GP models from the observed data in `self.statistics` and
        optimizes the qLogEHVI acquisition function to propose `target_number_of_candidates`.

        If there are too few observed points or optimization fails, falls back to
        Sobol sampling inside the provided region or the full config space.
        """
        logger.debug(f"qLogEHVI: generating {target_number_of_candidates} candidates")

        region_constraints = optionals.get("region_constraints", None)

        # If model is not fitted yet, fall back to Sobol sampling
        if self.fitted_model is None or self.global_bounds is None:
            logger.info("Model not fitted yet; using Sobol fallback")

            # Get config_keys and global bounds for fallback
            if self.config_space is None:
                logger.warning("No config_space available. Returning empty list.")
                return []

            config_keys = list(self.config_space.keys())

            # Build global bounds for fallback
            global_lowers = []
            global_uppers = []
            for k in config_keys:
                b = self.config_space[k]
                if (
                    isinstance(b, list)
                    and len(b) == 2
                    and all(isinstance(x, (int, float)) for x in b)
                ):
                    global_lowers.append(float(b[0]))
                    global_uppers.append(float(b[1]))
                else:
                    global_lowers.append(0.0)
                    global_uppers.append(float(len(b) - 1))

            # Build region bounds for fallback
            region_lowers = []
            region_uppers = []
            for k in config_keys:
                if region_constraints and k in region_constraints.boundaries:
                    b = region_constraints.boundaries[k]
                    region_lowers.append(float(b[0]))
                    region_uppers.append(float(b[1]))
                else:
                    region_lowers.append(global_lowers[config_keys.index(k)])
                    region_uppers.append(global_uppers[config_keys.index(k)])

            region_bounds = tensor([region_lowers, region_uppers], dtype=torch.double)

            samples = draw_sobol_samples(
                bounds=region_bounds, n=target_number_of_candidates, q=1
            ).squeeze(1)
            candidates = []
            for s in samples:
                cand = {k: float(v) for k, v in zip(config_keys, s)}
                candidates.append(cand)
            logger.debug(f"qLogEHVI fallback candidates: {candidates}")
            return candidates

        # Use pre-fitted model
        dtype = torch.double
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use pre-fitted model
        dtype = torch.double
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build region bounds (use region constraints if present, otherwise use global bounds)
        region_lowers = []
        region_uppers = []

        # Extract global bounds values
        global_lowers = self.global_bounds[0].tolist()
        global_uppers = self.global_bounds[1].tolist()

        for i, k in enumerate(self.config_keys):
            if region_constraints and k in region_constraints.boundaries:
                b = region_constraints.boundaries[k]
                region_lowers.append(float(b[0]))
                region_uppers.append(float(b[1]))
            else:
                # Use global bounds
                region_lowers.append(global_lowers[i])
                region_uppers.append(global_uppers[i])

        region_bounds = tensor([region_lowers, region_uppers], dtype=dtype)

        # Create partitioning and acquisition function using pre-fitted model
        ref_point_list = self.ref_point_tensor.tolist()

        with torch.no_grad():
            partitioning = FastNondominatedPartitioning(
                ref_point=self.ref_point_tensor,
                Y=self.pareto_Y,
            )

        acqf = qLogExpectedHypervolumeImprovement(
            model=self.fitted_model,
            ref_point=ref_point_list,
            partitioning=partitioning,
            sampler=None,
        )

        # Normalize region bounds to [0,1] using global bounds for acquisition optimization
        region_bounds_normalized_lower = (region_bounds[0] - self.global_bounds[0]) / (
            self.global_bounds[1] - self.global_bounds[0]
        )
        region_bounds_normalized_upper = (region_bounds[1] - self.global_bounds[0]) / (
            self.global_bounds[1] - self.global_bounds[0]
        )
        region_bounds_normalized = torch.stack(
            [region_bounds_normalized_lower, region_bounds_normalized_upper]
        ).to(device=device, dtype=dtype)

        candidates_normalized, _ = optimize_acqf(
            acq_function=acqf,
            bounds=region_bounds_normalized,
            q=target_number_of_candidates,
            num_restarts=10,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
        )

        with torch.no_grad():
            # Unnormalize using global bounds
            new_x = unnormalize(
                candidates_normalized.detach(),
                bounds=self.global_bounds.to(device=device, dtype=dtype),
            )

            candidates = []
            for pt in new_x:
                cand = {
                    k: float(v) for k, v in zip(self.config_keys, pt.cpu().tolist())
                }
                candidates.append(cand)

        logger.debug(f"qLogEHVI candidates: {candidates}")
        return candidates

    def evaluate_desired_values(self, args=None, **kwargs):
        # qLogEHVI does not use desired values in this implementation
        return {}
