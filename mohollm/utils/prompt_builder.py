import random
import logging
from string import Template
from typing import Dict, Tuple, List
from mohollm.statistics.statistics import Statistics

logger = logging.getLogger("PromptBuilder")


class PromptBuilder:
    def __init__(
        self,
        template_dir: str = None,
    ):
        """
        Initializes the PromptBuilder with a given prompt template directory.

        Args:
            template_dir : str
                The path to the directory containing the prompt template.

        Attributes:
            prompt_template : Template
                The prompt template to be used for generating prompts.
            statistics : Statistics
                The statistics module to be used for generating prompts.
            prompt : Dict
                The prompt information to be used for generating prompts.
            initial_samples : int
                The initial samples to be used for generating prompts.
        """
        self.prompt_template: Template = self._load_prompt_template(template_dir)
        self.prompt: Dict = None
        self.statistics: Statistics = None
        self.shuffle_icl_columns = None
        self.shuffle_icl_rows = None
        self.use_few_shot_examples = False
        self.few_shot_examples: Tuple = None

        # This dict represents the global bounds for each feature
        self.parameter_constraints: Dict = None

        self.feature_names: List[str] = None
        self.metrics_names: List[str] = None
        self.metrics_targets: list = None

    def initialize_templates(self):
        self.icl_example_template: Template = Template(self._create_response_format())
        self.icl_examples_template: Template = Template(
            self._create_icl_examples_template()
        )

    def _load_prompt_template(self, template_dir: str = None) -> Template:
        """
        Loads a prompt template from a given directory and returns it as a Template object.

        Args
            template_dir : str
                The path to the directory containing the prompt template.

        Returns
            Template
                The loaded prompt template.

        Raises
            ValueError
                If no template directory is provided.
            FileNotFoundError
                If the template file is not found.
            OSError
                If there is an issue with reading the template file.
        """
        if template_dir is None:
            logger.warning("No template directory provided")
            raise ValueError("No template directory provided")

        template = None
        try:
            with open(template_dir, "r") as file:
                template = file.read()
        except OSError as e:
            logger.warning(f"Loading of prompt template failed. Error: {e}")

        if template is None:
            logger.warning(
                f"Loading of prompt template failed. Check if path {template_dir} is correct."
            )
            raise FileNotFoundError(
                f"Loading of prompt template failed. Check if path {template_dir} is correct."
            )

        return Template(template)

    def build_prompt(self, **kwargs) -> str:
        """
        Builds a prompt based on the given task context and optional additional keyword arguments.

        Args:
            **kwargs
                Additional keyword arguments will be used to substitute placeholders in the prompt template.

        Returns
            str
                The built prompt as a string.
        """

        substitution_values = {
            **self.prompt,
            **kwargs,
            **{
                "ICL_examples": self._build_icl_examples(),
                "few_shot_examples": self._build_few_shot_examples(),
                "metrics_desired_values_template": self._build_desired_values_template(
                    **kwargs
                ),
                "Region_ICL_examples": self._build_region_icl_examples(
                    kwargs.get("region_icl_examples", {})
                ),
                "warmstarting_response_format": self._create_response_format(),
                "candidate_sampler_response_format": self._create_response_format(),
                "surrogate_model_response_format": self._create_surrogate_model_response_format(),
                "metrics": self._create_metrics_description(),
                "constraints": self.parameter_constraints,
                "description": self.prompt.get("description", ""),
                "instruction": self.prompt.get("instruction", ""),
            },
        }

        prompt = self.prompt_template.safe_substitute(**substitution_values)
        return prompt

    def _create_response_format(self) -> str:
        """
        Create a response format template for mohollm based on the configuration space.

        :param config_space: The configuration space
        :return: A JSON template string
        """
        template_parts = ["{"]
        logger.debug(f"Building response format for features: {self.feature_names}")
        for i, name in enumerate(self.feature_names):
            template_parts.append(f'"{name}": ${name}')
            if i < len(self.feature_names) - 1:
                template_parts.append(", ")
        template_parts.append("}")

        return "".join(template_parts)

    def _create_surrogate_model_response_format(self) -> str:
        """
        Create a response format template for the surrogate model based on the configuration space.

        :return: A JSON template string
        """
        template_parts = ["{"]
        for i, name in enumerate(self.metrics_names):
            template_parts.append(f'"{name}": ${name}')
            if i < len(self.metrics_names) - 1:
                template_parts.append(", ")
        template_parts.append("}")

        return "".join(template_parts)

    def _create_icl_examples_template(self) -> str:
        """
        Create an ICL examples template string using the metrics_names list.
        The format will be:

        Configuration: $configuration \nF1: $F1, F2: $F2\n

        Returns:
            str: The template string for ICL examples.
        """
        metrics_str = ", ".join([f"{name}: ${name}" for name in self.metrics_names])
        template = f"Configuration: $configuration \n{metrics_str}\n"
        return template

    def _create_metrics_description(self) -> str:
        """
        Create a metrics description string for mohollm.

        :param metrics: List of metrics
        :return: A string describing the metrics
        """
        m_strings = []
        for metric, target in zip(self.metrics_names, self.metrics_targets):
            if target == "min":
                m_strings.append(f"{metric} (lower is better)")
            elif target == "max":
                m_strings.append(f"{metric} (higher is better)")
        return ", ".join([m_string for m_string in m_strings])

    def _build_region_icl_examples(
        self, region_icl_examples: Tuple[List[Dict], List[Dict]]
    ) -> str:
        if not region_icl_examples:
            return ""

        configs, fvals = region_icl_examples

        icl_examples = []
        for config, fval in zip(configs, fvals):
            fval = {key: round(value, 6) for key, value in fval.items()}

            configuration = self.icl_example_template.safe_substitute(
                **{
                    **config,
                }
            )
            icl_example = self.icl_examples_template.safe_substitute(
                **{
                    "configuration": configuration,
                    **fval,
                }
            )
            icl_examples.append(icl_example)

        if self.shuffle_icl_rows:
            icl_examples = self._shuffle_config_rows(icl_examples)

        return "\n".join(icl_examples)

    def _build_icl_examples(self) -> str:
        """
        Builds a string of few-shot examples of hyperparameter configurations and corresponding function values.

        Returns
            str
                The string of ICL examples.
        """
        configs, fvals = self.statistics.get_statistics_for_icl()

        icl_examples = []
        for config, fval in zip(configs, fvals):
            fval = {key: round(value, 6) for key, value in fval.items()}
            if self.shuffle_icl_columns:
                config = self._shuffle_config_columns(config)
            configuration = self.icl_example_template.safe_substitute(
                **{
                    **config,
                }
            )
            icl_example = self.icl_examples_template.safe_substitute(
                **{
                    "configuration": configuration,
                    **fval,
                }
            )
            icl_examples.append(icl_example)

        if self.shuffle_icl_rows:
            icl_examples = self._shuffle_config_rows(icl_examples)
        return "\n".join(icl_examples)

    def _build_few_shot_examples(self) -> str:
        """
        Builds a string of few-shot examples using the provided models and evaluations.

        This method generates few-shot examples by iterating over the models and their
        corresponding evaluations, substituting these into a template, and appending
        the resulting examples to a list. If few-shot examples are not being used,
        it returns an empty string.

        Returns:
            str: A concatenated string of the few-shot examples.
        """
        if not self.use_few_shot_examples:
            return ""
        icl_examples = []

        for model, evaluation in self.few_shot_examples:
            configuration = self.icl_example_template.safe_substitute(
                **{
                    **model,
                }
            )
            icl_example = self.icl_examples_template.safe_substitute(
                **{
                    "configuration": configuration,
                    **evaluation,
                }
            )
            icl_examples.append(icl_example)

        return "\n".join(icl_examples)

    def _build_desired_values_template(self, **kwargs) -> str:
        metrics_desired_template_string = self.prompt.get(
            "metrics_desired_values_template", None
        )
        if metrics_desired_template_string is None:
            return ""
        metrics_desired_template = Template(metrics_desired_template_string)
        desired_metrics_values = metrics_desired_template.safe_substitute(**kwargs)
        return desired_metrics_values

    def _shuffle_config_columns(self, config: Dict) -> Dict:
        """
        Shuffles the columns of a given configuration dictionary.


        Args:
            config (Dict)
                The configuration dictionary to shuffle.

        Returns:
            Dict
                The shuffled configuration dictionary.
        """
        list_config = list(config.items())
        random.shuffle(list_config)
        return dict(list_config)

    def _shuffle_config_rows(self, config: list) -> list:
        """
        Shuffles the rows of a given list of configuration strings.

        Args:
            config (list)
                The list of configuration strings to shuffle.

        Returns:
            list
                The shuffled list of configuration strings.
        """
        random.shuffle(config)
        return config
