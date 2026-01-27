import logging
import numpy as np
from mohollm.surrogate_models.surrogate_model import SURROGATE_MODEL
from tabpfn import TabPFNRegressor

logger = logging.getLogger("TabPFNSurrogateModel")


class TabPFNSurrogate(SURROGATE_MODEL):
    def __init__(self):
        self.surrogate_models: list[TabPFNRegressor] = []

    # TODO: Implement this. https://github.com/PriorLabs/TabPFN
    def evaluate_candidates(self, candidate_points, optionals={}):
        logger.debug("Evaluating candidates with TabPFN surrogate model")
        X_train = [list(config.values()) for config in self.statistics.observed_configs]

        y_train_raw = [list(fval.values()) for fval in self.statistics.observed_fvals]
        y_train = np.array(y_train_raw)

        logger.debug(
            f"Training TabPFN model with {len(X_train)} training points and {len(self.metrics_names)} objectives"
        )

        self.surrogate_models = []
        num_objectives = len(self.metrics_names)
        logger.debug(f"Identified {num_objectives} objectives to model.")
        for i in range(num_objectives):
            model = TabPFNRegressor()
            model.fit(X_train, y_train[:, i])
            self.surrogate_models.append(model)

        X_predict = [list(candidate.values()) for candidate in candidate_points]
        all_objective_predictions = []

        for i, model in enumerate(self.surrogate_models):
            prediction_for_objective = model.predict(X_predict)
            all_objective_predictions.append(prediction_for_objective)

        predictions_by_candidate = np.array(all_objective_predictions).T

        prediction_dicts = [
            {name: pred for name, pred in zip(self.metrics_names, prediction_row)}
            for prediction_row in predictions_by_candidate
        ]
        logger.debug(f"Predictions for candidates: {prediction_dicts}")
        return prediction_dicts, 0, 0

    def evaluate_candidate(self, candidate_point, optionals={}):
        pass
