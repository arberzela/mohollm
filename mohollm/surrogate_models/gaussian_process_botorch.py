import logging
from typing import List, Dict

import torch
from torch import tensor
import numpy as np

from botorch.utils.transforms import normalize
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from mohollm.surrogate_models.surrogate_model import SURROGATE_MODEL

logger = logging.getLogger("GaussianProcessBoTorchSurrogate")


class GaussianProcessBoTorchSurrogate(SURROGATE_MODEL):
    """
    Gaussian Process surrogate model using BoTorch.

    This implementation uses BoTorch's SingleTaskGP with independent models
    for each objective, similar to the approach in qLogEHVISampler.
    """

    def __init__(self):
        super().__init__()
        self.surrogate_models: List[SingleTaskGP] = []
        self.config_keys = None
        self.bounds_tensor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

    def _build_bounds_tensor(self):
        """Build bounds tensor from benchmark.bounds."""
        if self.benchmark is None or not hasattr(self.benchmark, "bounds"):
            logger.warning("No benchmark.bounds available.")
            return None

        # Get bounds from benchmark (dict with tuples)
        bounds_dict = self.benchmark.bounds

        # Ensure config_keys are set
        if self.config_keys is None:
            self.config_keys = list(bounds_dict.keys())

        lowers = []
        uppers = []

        for key in self.config_keys:
            if key in bounds_dict:
                lower, upper = bounds_dict[key]
                lowers.append(float(lower))
                uppers.append(float(upper))
            else:
                logger.warning(f"Key {key} not found in benchmark.bounds")
                return None

        return tensor([lowers, uppers], dtype=self.dtype)

    def evaluate_candidates(self, candidate_points: List[Dict], optionals={}):
        """
        Evaluate a list of candidate points using independent Gaussian Process models
        for each objective (BoTorch implementation).

        Args:
            candidate_points (List[Dict]): A list of candidate points to evaluate.
            optionals (dict): Optional parameters.

        Returns:
            List[Dict]: Predicted values for each candidate point in a list of dictionaries format.
        """
        logger.debug("Evaluating candidates with BoTorch Gaussian Process models")

        # Build bounds tensor if not already built
        if self.bounds_tensor is None:
            self.bounds_tensor = self._build_bounds_tensor()
            if self.bounds_tensor is None:
                logger.error("Failed to build bounds tensor from benchmark.bounds")
                return [], 0, 0

        # --- 1. Prepare Training Data ---
        observed_configs = self.statistics.observed_configs
        observed_fvals = self.statistics.observed_fvals

        if not observed_configs or len(observed_configs) < 1:
            logger.warning("No training data available. Cannot evaluate candidates.")
            return [], 0, 0

        # Set config_keys if not already set
        if self.config_keys is None:
            self.config_keys = list(self.benchmark.bounds.keys())

        # Determine objective keys / order
        if self.metrics_names:
            obj_keys = list(self.metrics_names)
        else:
            # infer from observed_fvals dict
            obj_keys = list(observed_fvals[0].keys())

        # Build train_X and train_Y tensors
        train_X_list = [
            [float(cfg.get(k)) for k in self.config_keys] for cfg in observed_configs
        ]
        train_Y_list = [
            [float(fval.get(k)) for k in obj_keys] for fval in observed_fvals
        ]

        train_X = tensor(train_X_list, dtype=self.dtype, device=self.device)
        train_Y = tensor(train_Y_list, dtype=self.dtype, device=self.device)

        # Normalize X to [0,1] using benchmark bounds
        train_X_normalized = normalize(
            train_X, self.bounds_tensor.to(device=self.device, dtype=self.dtype)
        )

        num_objectives = len(obj_keys)
        logger.debug(f"Training {num_objectives} independent GP models")

        # --- 2. Train One Model Per Objective ---
        self.surrogate_models = []

        try:
            for i in range(num_objectives):
                logger.debug(f"Training GP model for objective {obj_keys[i]}")

                # Get the training data for this objective (single column)
                train_y = train_Y[..., i : i + 1]

                # Create and fit GP model
                gp = SingleTaskGP(train_X_normalized, train_y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

                self.surrogate_models.append(gp)

            # --- 3. Make Predictions ---
            if not candidate_points:
                return [], 0, 0

            # Prepare candidate points for prediction
            X_predict_list = [
                [float(candidate.get(k)) for k in self.config_keys]
                for candidate in candidate_points
            ]
            X_predict = tensor(X_predict_list, dtype=self.dtype, device=self.device)

            # Normalize candidate points using benchmark bounds
            X_predict_normalized = normalize(
                X_predict, self.bounds_tensor.to(device=self.device, dtype=self.dtype)
            )

            # Collect predictions from each model
            all_predictions = []
            for i, model in enumerate(self.surrogate_models):
                with torch.no_grad():
                    posterior = model.posterior(X_predict_normalized)
                    predictions = posterior.mean.squeeze(-1).cpu().numpy()
                    all_predictions.append(predictions)

            predictions_by_candidate = np.array(all_predictions).T

            # --- 4. Format Output ---
            prediction_dicts = [
                {name: float(pred) for name, pred in zip(obj_keys, prediction_row)}
                for prediction_row in predictions_by_candidate
            ]

            logger.debug(
                f"Predictions for {len(candidate_points)} candidates completed"
            )
            return prediction_dicts, 0, 0

        except Exception as e:
            logger.error(f"Failed to train/predict with BoTorch GP: {e}")
            return [], 0, 0

    def evaluate_candidate(
        self, candidate: Dict, target_number_of_evaluations: int, optionals={}
    ):
        pass
