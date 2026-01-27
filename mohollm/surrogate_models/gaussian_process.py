import logging
from typing import List, Dict
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from mohollm.surrogate_models.surrogate_model import SURROGATE_MODEL

logger = logging.getLogger("GaussianProcessSurrogate")


class GaussianProcessSurrogate(SURROGATE_MODEL):
    def __init__(self):
        self.surrogate_models: List[GaussianProcessRegressor] = []

    def evaluate_candidates(self, candidate_points: List[Dict], optionals={}):
        """
        Evaluate a list of candidate points using independent Gaussian Process models
        for each objective.

        Args:
            candidate_points (List[Dict]): A list of candidate points to evaluate.

        Returns:
            List[Dict]: Predicted values for each candidate point in a list of dictionaries format.
        """
        logger.debug("Evaluating candidates with independent Gaussian Process models")

        # --- 1. Prepare Training Data ---
        X_train = [list(config.values()) for config in self.statistics.observed_configs]
        y_train_raw = [list(fval.values()) for fval in self.statistics.observed_fvals]

        # Convert y_train to a numpy array for easier slicing
        y_train = np.array(y_train_raw)

        if not X_train:
            logger.warning("No training data available. Cannot evaluate candidates.")
            return [], 0, 0

        num_objectives = len(self.metrics_names)
        logger.debug(f"Identified {num_objectives} objectives to model.")

        # --- 2. Train One Model Per Objective ---
        self.surrogate_models = []
        all_objective_predictions = []

        for i in range(num_objectives):
            logger.debug(f"Training Gaussian Process model for Objective F{i + 1}")

            # Create a new GP model instance for this specific objective
            kernel = Matern(nu=2.5)
            # Use a new model for each objective to avoid state conflicts
            gp_model = GaussianProcessRegressor(
                kernel=kernel, normalize_y=False, n_restarts_optimizer=10
            )

            # Get the training data for this objective only (a single column)
            y_train_objective = y_train[:, i]

            gp_model.fit(X_train, y_train_objective)
            self.surrogate_models.append(gp_model)

        # --- 3. Predict Using the Correct Model for Each Objective ---
        X_predict = [list(candidate.values()) for candidate in candidate_points]

        if not X_predict:
            return [], 0, 0

        # Collect predictions from each independent model
        for i, model in enumerate(self.surrogate_models):
            predictions_for_objective = model.predict(X_predict, return_std=False)
            all_objective_predictions.append(predictions_for_objective)

        # Transpose the results to group predictions by candidate point
        # Before: [[f1_cand1, f1_cand2], [f2_cand1, f2_cand2]]
        # After:  [[f1_cand1, f2_cand1], [f1_cand2, f2_cand2]]
        predictions_by_candidate = np.array(all_objective_predictions).T

        # --- 4. Format Output ---
        prediction_dicts = [
            {name: pred for name, pred in zip(self.metrics_names, prediction_row)}
            for prediction_row in predictions_by_candidate
        ]

        logger.debug(f"Predictions for candidates: {prediction_dicts}")
        return prediction_dicts, 0, 0

    def evaluate_candidate(
        self, candidate: Dict, target_number_of_evaluations: int, optionals={}
    ):
        """
        Evaluate a single candidate point using the trained independent models.
        Note: This requires the models to be trained first via `evaluate_candidates`.
        """
        if not self.surrogate_models:
            raise RuntimeError(
                "Surrogate models have not been trained yet. Call `evaluate_candidates` first."
            )

        logger.debug(f"Evaluating single candidate: {candidate}")
        X = [list(candidate.values())]

        prediction = {}

        for i, model in enumerate(self.surrogate_models):
            pred_val = model.predict(X, return_std=False)[0]
            prediction[self.metrics_names[i]] = pred_val

        return prediction
