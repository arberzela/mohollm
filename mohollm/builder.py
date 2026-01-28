import logging
from typing import Dict, Optional, Type, TypeVar
from mohollm.llm.llm import LLMInterface
from mohollm.llm.models.local import LOCAL
from mohollm.llm.models.gpt import GPT
from mohollm.llm.models.gemini import GEMINI
from mohollm.llm.models.groq import GROQ
from mohollm.llm.models.huggingface import HUGGINGFACE
from mohollm.llm.models.nebius import NEBIUS
from mohollm.llm.models.openrouter import OPENROUTER
from mohollm.settings import (
    MODELS,
    ACQUISITION_FUNCTIONS,
    SURROGATE_MODELS,
    CANDIDATE_SAMPLERS,
    WARMSTARTERS,
    CONTEXT_LIMIT_STRATEGIES,
    SCHEDULERS,
    SPACE_PARTITIONING_STRATEGIES,
    REGION_ACQUISITION_FUNCTIONS,
)
from mohollm.optimization_strategy.optimization_strategy import OptimizationStrategy
from mohollm.optimization_strategy.mohollm import mohollm
from mohollm.optimization_strategy.space_partitioning_mohollm import (
    SpacePartitioningmohollm,
)
from mohollm.benchmarks.benchmark import BENCHMARK
from mohollm.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION
from mohollm.region_acquisition_functions.region_acq import RegionACQ
from mohollm.surrogate_models.surrogate_model import (
    SURROGATE_MODEL,
)
from mohollm.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER
from mohollm.warmstarter.warmstarter import WARMSTARTER
from mohollm.statistics.context_limit_strategy.context_limit_strategy import (
    ContextLimitStrategy,
)

from mohollm.space_partitioning.space_partitioning_strategy import (
    SPACE_PARTITIONING_STRATEGY,
)
from mohollm.space_partitioning.utils import BoundingBox

from mohollm.statistics.statistics import Statistics
from mohollm.utils.rate_limiter import RateLimiter
from mohollm.utils.prompt_builder import PromptBuilder

logger = logging.getLogger("Builder")

T = TypeVar("T")


class Builder:
    def __init__(
        self,
        config: Dict,
        benchmark: BENCHMARK,
        custom_model: LLMInterface = None,
        custom_acquisition_function: ACQUISITION_FUNCTION = None,
        custom_candidate_sampler: CANDIDATE_SAMPLER = None,
        custom_region_acquisition_function: RegionACQ = None,
        custom_space_partitioning_strategy: SPACE_PARTITIONING_STRATEGY = None,
        custom_warmstarter: WARMSTARTER = None,
        custom_surrogate_model: SURROGATE_MODEL = None,
    ) -> None:
        logger.debug(f"Setting up builder with configuration: {config}")

        # Store custom implementations
        self._custom_model = custom_model
        self._custom_acquisition_function = custom_acquisition_function
        self._custom_candidate_sampler = custom_candidate_sampler
        self._custom_space_partitioning_strategy = custom_space_partitioning_strategy
        self._custom_warmstarter = custom_warmstarter
        self._custom_surrogate_model = custom_surrogate_model
        self._custom_region_acquisition_function = custom_region_acquisition_function

        # Model related attributes
        self.llm_settings: dict = config.get("llm_settings", {})

        # Configuration parameters
        self.optimization_method = config.get("optimization_method", "mohollm")
        self.alpha = config.get("alpha", -0.2)
        self.n_trials = config.get("n_trials", 5)
        self.total_trials = config.get("total_trials", 5)
        self.max_candidates_per_trial = config.get("max_candidates_per_trial", 5)
        self.candidates_per_request = config.get("candidates_per_request", 5)
        self.evaluations_per_request = config.get("evaluations_per_request", 5)
        self.max_evaluations_per_trial = config.get("max_evaluations_per_trial", 5)
        self.space_partitioning_settings = config.get("space_partitioning_settings", {})
        self.warmstarter_settings = config.get("warmstarter_settings", {})
        self.initial_samples = config.get("initial_samples", 5)
        self.seed = config.get("seed", 42)
        self.top_k = config.get("top_k", 4)
        self.max_context_configs = config.get("max_context_configs", 100)
        self.max_requests_per_minute = config.get("max_requests_per_minute", 700)
        self.context_limit_strategy = self.get_context_limit_strategy(
            config.get("context_limit_strategy", None)
        )
        self.shuffle_icl_columns = config.get("shuffle_icl_columns", False)
        self.shuffle_icl_rows = config.get("shuffle_icl_rows", False)
        self.metrics = config.get("metrics", [])
        self.metrics_targets = config.get("metrics_targets", [])
        self.range_parameter_keys = config.get("range_parameter_keys", [])
        self.integer_parameter_keys = config.get("integer_parameter_keys", [])
        self.float_parameter_keys = config.get("float_parameter_keys", [])
        self.parameter_constraints: dict = config.get("parameter_constraints", {})

        self.benchmark_name: str = config.get("benchmark", "")
        self.method_name: str = config.get("method_name", " ")
        self.use_few_shot_examples = config.get("use_few_shot_examples", False)

        self.prompt = config.get("prompt", {})

        self.model_name: str = self.llm_settings.get("model", None)
        self.benchmark: BENCHMARK = benchmark
        self.benchmark.method_name = self.method_name

        self.statistics: Statistics = self.get_statistics()
        self.model: LLMInterface = self.get_model(self.model_name)
        self.warmstarting_prompt_template_dir: str = config.get(
            "warmstarting_prompt_template", None
        )
        self.candidate_sampler_prompt_template_dir: str = config.get(
            "candidate_sampler_prompt_template", None
        )
        self.surrogate_model_prompt_template_dir: str = config.get(
            "surrogate_model_prompt_template", None
        )

        self.acquisition_function: ACQUISITION_FUNCTION = self.get_acquisition_function(
            config.get("acquisition_function", "")
        )
        self.surrogate_model: SURROGATE_MODEL = self.get_surrogate_model(
            config.get("surrogate_model", "")
        )
        self.candidate_sampler: CANDIDATE_SAMPLER = self.get_candidate_sampler(
            config.get("candidate_sampler", "")
        )
        self.warmstarter: WARMSTARTER = self.get_warmstarter(
            config.get("warmstarter", "")
        )

        logger.debug("Setting up builder complete")

    def _create_component(
        self,
        name: str,
        registry: Dict[str, Type[T]],
        custom: Optional[T],
        label: str,
    ) -> Optional[T]:
        if custom is not None:
            logger.debug(f"Using custom {label}")
            return custom
        selected = registry.get(name, None)
        if selected is None:
            logger.warning(f"No {label} specified: {name}")
            return None
        return selected()

    def get_candidate_sampler(self, candidate_sampler: str) -> Optional[CANDIDATE_SAMPLER]:
        """
        Retrieve and configure a candidate sampler based on the provided name.

        This method first checks if a custom candidate sampler was provided at initialization.
        If not, it looks up the candidate sampler specified by the 'candidate_sampler'
        argument in the CANDIDATE_SAMPLERS dictionary. If found, it configures the sampler
        with the necessary attributes from the current object's context.

        Args:
            candidate_sampler (str): The name of the candidate sampler to retrieve.

        Returns:
            CANDIDATE_SAMPLER: An instance of the configured candidate sampler.

        """
        selected_candidate_sampler = self._create_component(
            candidate_sampler,
            CANDIDATE_SAMPLERS,
            self._custom_candidate_sampler,
            "candidate sampler",
        )
        if selected_candidate_sampler is None:
            return None

        selected_candidate_sampler.max_candidates_per_trial = (
            self.max_candidates_per_trial
        )
        selected_candidate_sampler.candidates_per_request = self.candidates_per_request
        selected_candidate_sampler.model = self.model
        selected_candidate_sampler.statistics = self.statistics
        selected_candidate_sampler.alpha = self.alpha
        selected_candidate_sampler.metrics = self.metrics
        selected_candidate_sampler.benchmark = self.benchmark
        selected_candidate_sampler.prompt_builder = self.get_prompt_builder(
            template_dir=self.candidate_sampler_prompt_template_dir,
        )
        selected_candidate_sampler.range_parameter_keys = self.range_parameter_keys
        selected_candidate_sampler.config_space = self.parameter_constraints
        return selected_candidate_sampler

    def get_warmstarter(self, warmstarter: str) -> Optional[WARMSTARTER]:
        """
        Retrieves and configures a warmstarter based on the provided name.

        This method first checks if a custom warmstarter was provided at initialization.
        If not, it looks up the warmstarter specified by the 'warmstarter'
        argument in the WARMSTARTERS dictionary. If found, it configures the
        warmstarter with the necessary attributes from the current object's context.

        Args:
            warmstarter (str): The name of the warmstarter to retrieve.

        Returns:
            WARMSTARTER: An instance of the configured warmstarter.

        """
        selected_warmstarter = self._create_component(
            warmstarter,
            WARMSTARTERS,
            self._custom_warmstarter,
            "warmstarter",
        )
        if selected_warmstarter is None:
            return None

        selected_warmstarter.model = self.model
        selected_warmstarter.initial_samples = self.initial_samples
        selected_warmstarter.benchmark = self.benchmark
        selected_warmstarter.prompt_builder = self.get_prompt_builder(
            template_dir=self.warmstarting_prompt_template_dir,
        )
        selected_warmstarter.warmstarter_settings = self.warmstarter_settings

        return selected_warmstarter

    def get_acquisition_function(
        self, acquisition_function: str
    ) -> Optional[ACQUISITION_FUNCTION]:
        """
        Retrieves and configures an acquisition function based on the provided name.

        This method first checks if a custom acquisition function was provided at initialization.
        If not, it looks up the acquisition function specified by the
        'acquisition_function' argument in the ACQUISITION_FUNCTIONS dictionary. If
        found, it configures the acquisition function with the necessary attributes
        from the current object's context.

        Args:
            acquisition_function (str): The name of the acquisition function to
                retrieve.

        Returns:
            ACQUISITION_FUNCTION: An instance of the configured acquisition
                function.

        """
        selected_acquisition_function = self._create_component(
            acquisition_function,
            ACQUISITION_FUNCTIONS,
            self._custom_acquisition_function,
            "acquisition function",
        )
        if selected_acquisition_function is None:
            return None

        selected_acquisition_function.statistics = self.statistics
        selected_acquisition_function.metrics_targets = self.metrics_targets
        return selected_acquisition_function

    def get_surrogate_model(self, surrogate_model: str) -> Optional[SURROGATE_MODEL]:
        """
        Retrieves and configures a surrogate model based on the provided name.

        This method first checks if a custom surrogate model was provided at initialization.
        If not, it looks up the surrogate model specified by the
        'surrogate_model' argument in the SURROGATE_MODELS dictionary. If
        found, it configures the surrogate model with the necessary attributes
        from the current object's context.

        Args:
            surrogate_model (str): The name of the surrogate model to
                retrieve.

        Returns:
            SURROGATE_MODEL: An instance of the configured surrogate
                model.

        """
        selected_surrogate_model = self._create_component(
            surrogate_model,
            SURROGATE_MODELS,
            self._custom_surrogate_model,
            "surrogate model",
        )
        if selected_surrogate_model is None:
            return None

        selected_surrogate_model.model = self.model
        selected_surrogate_model.statistics = self.statistics
        selected_surrogate_model.benchmark = self.benchmark
        selected_surrogate_model.max_evaluations_per_trial = (
            self.max_evaluations_per_trial
        )
        selected_surrogate_model.evaluations_per_request = self.evaluations_per_request
        logger.debug(f"Template dir: {self.surrogate_model_prompt_template_dir}")
        selected_surrogate_model.metrics_names = self.metrics

        selected_surrogate_model.prompt_builder = self.get_prompt_builder(
            template_dir=self.surrogate_model_prompt_template_dir,
        )

        return selected_surrogate_model

    def get_prompt_builder(self, template_dir: str) -> PromptBuilder:
        """
        Retrieves and configures a PromptBuilder based on the provided template directory.

        Args:
            template_dir (str): The directory containing the prompt template.

        Returns:
            PromptBuilder: An instance of the configured PromptBuilder.
        """
        prompt_builder: PromptBuilder = PromptBuilder(template_dir=template_dir)
        prompt_builder.initial_samples = self.initial_samples
        prompt_builder.statistics = self.statistics
        prompt_builder.shuffle_icl_columns = self.shuffle_icl_columns
        prompt_builder.shuffle_icl_rows = self.shuffle_icl_rows
        prompt_builder.use_few_shot_examples = self.use_few_shot_examples
        prompt_builder.few_shot_examples = self.benchmark.get_few_shot_samples()
        prompt_builder.metrics_ranges = self.benchmark.get_metrics_ranges()
        prompt_builder.prompt = self.prompt
        prompt_builder.parameter_constraints = self.parameter_constraints

        prompt_builder.feature_names = self.parameter_constraints.keys()
        prompt_builder.metrics_names = self.metrics
        prompt_builder.metrics_targets = self.metrics_targets
        prompt_builder.initialize_templates()
        return prompt_builder

    def get_context_limit_strategy(
        self, context_limit_strategy: str
    ) -> Optional[ContextLimitStrategy]:
        selected_context_limit_strategy = self._create_component(
            context_limit_strategy,
            CONTEXT_LIMIT_STRATEGIES,
            None,
            "context limit strategy",
        )
        if selected_context_limit_strategy is None:
            return None

        selected_context_limit_strategy.max_context_configs = self.max_context_configs
        return selected_context_limit_strategy

    def get_statistics(self) -> Statistics:
        """
        Retrieves and configures a Statistics object.

        Returns:
            Statistics: An instance of the configured Statistics object.
        """
        statistics = Statistics()
        statistics.model_name = self.model_name
        statistics.seed = self.seed
        statistics.initial_samples = self.initial_samples
        statistics.max_context_configs = self.max_context_configs
        statistics.context_limit_strategy = self.context_limit_strategy
        statistics.metrics = self.metrics
        statistics.metrics_targets = self.metrics_targets
        statistics.benchmark_name = self.benchmark_name
        return statistics

    def get_model(self, model: str) -> LLMInterface:
        """
        Retrieves and configures a model based on the provided model name.
        If a custom model was provided at initialization, it will be used instead.

        Args:
            model (str): The name of the model to retrieve.

        Returns:
            LLMInterface: An instance of the configured model.

        """
        # Check if a custom model was provided
        if self._custom_model is not None:
            logger.debug("Using custom model")
            selected_model = self._custom_model
        else:
            selected_model: LLMInterface = MODELS.get(model, None)
            if selected_model is None:
                logger.warning(f"No model specified: {model}")
                return
            else:
                selected_model: LLMInterface = selected_model(self.model_name)
        # Load the model to memory if it's a LOCAL model.
        # This prevents the model from being reloaded after each prompt.
        if isinstance(selected_model, LOCAL):
            selected_model._load_model_to_memory()
        # If the model is of type GPT, GEMINI, GROQ or HUGGINGFACE, set the rate limiter
        if isinstance(
            selected_model, (GPT, GEMINI, GROQ, HUGGINGFACE, NEBIUS, OPENROUTER)
        ):
            selected_model.rate_limiter = RateLimiter(
                max_tokens=self.llm_settings.get("max_tokens_per_minute", 10000),
                time_frame=60,
                max_requests=self.llm_settings.get("max_requests_per_minute", 700),
            )
        selected_model.initial_samples = self.initial_samples
        selected_model.statistics = self.statistics
        selected_model.llm_settings = self.llm_settings
        return selected_model

    def get_optimization_method(self) -> OptimizationStrategy:
        optimization_method = None
        logger.debug(f"Optimization method: {self.optimization_method}")
        if self.optimization_method == "mohollm":
            optimization_method: mohollm = mohollm()
            optimization_method.warmstarter = self.warmstarter
            optimization_method.candidate_sampler = self.candidate_sampler
            optimization_method.acquisition_function = self.acquisition_function
            optimization_method.surrogate_model = self.surrogate_model
            optimization_method.statistics = self.statistics
            optimization_method.benchmark = self.benchmark
            optimization_method.initial_samples = self.initial_samples
            optimization_method.n_trials = self.n_trials
            optimization_method.top_k = self.top_k
            optimization_method.initialize()
            logger.debug("Initialized new mohollm instance with configured components")
        elif self.optimization_method == "SpacePartitioning":
            optimization_method: SpacePartitioningmohollm = SpacePartitioningmohollm()
            optimization_method.benchmark = self.benchmark
            optimization_method.statistics = self.statistics
            optimization_method.warmstarter = self.warmstarter
            optimization_method.n_trials = self.n_trials
            optimization_method.acquisition_function = self.acquisition_function
            optimization_method.surrogate_model = self.surrogate_model
            optimization_method.candidate_sampler = self.candidate_sampler
            optimization_method.top_k = self.top_k
            optimization_method.partitions_per_trial = (
                self.space_partitioning_settings.get("partitions_per_trial", 5)
            )
            optimization_method.use_clustering = self.space_partitioning_settings.get(
                "use_clustering", False
            )

            # Use the correct region acquisition function
            region_acq_strategy = self.space_partitioning_settings.get(
                "region_acquisition_strategy", "VolumeRegionACQ"
            )
            region_acquisition_function = self.get_region_acquisition_function(
                region_acq_strategy
            )
            optimization_method.region_acquisition_function = (
                region_acquisition_function
            )

            # Get the space partitioning strategy using our method
            search_space_partitioning_strategy = self.get_space_partitioning_strategy(
                self.space_partitioning_settings.get("partitioning_strategy", "voronoi")
            )
            search_space_partitioning_strategy.space_partitioning_settings = (
                self.space_partitioning_settings
            )

            optimization_method.search_space_partitioning = (
                search_space_partitioning_strategy
            )
            optimization_method.use_pareto_front_as_regions = (
                self.space_partitioning_settings.get(
                    "use_pareto_front_as_regions", False
                )
            )

            optimization_method.initialize()
            return optimization_method
        else:
            logger.warning(
                f"No valid optimization method specified: {self.optimization_method}"
            )
            return

        logger.debug(f"Optimization method: {self.optimization_method}")

        return optimization_method

    def build(self):
        """
        Constructs and initializes a mohollm object with the builder's configurations.

        This method sets up a mohollm instance by assigning it various components
        such as the warmstarter, candidate sampler, acquisition function, surrogate
        model, statistics, benchmark, initial samples, and number of trials. If
        optimization is to continue from a checkpoint, it adjusts the number of
        trials accordingly. It also calls the initialization method on the mohollm
        instance to prepare it for optimization.

        Returns:
            mohollm: The fully constructed and initialized mohollm object.
        """
        mohollm: OptimizationStrategy = self.get_optimization_method()
        return mohollm

    def get_space_partitioning_strategy(
        self, partitioning_strategy: str
    ) -> SPACE_PARTITIONING_STRATEGY:
        """
        Retrieves and configures a space partitioning strategy based on the provided name.

        This method first checks if a custom space partitioning strategy was provided at initialization.
        If not, it looks up the partitioning strategy specified by the 'partitioning_strategy'
        argument in the SPACE_PARTITIONING_STRATEGIES dictionary. If found, it configures the
        strategy with the necessary attributes from the current object's context.

        Args:
            partitioning_strategy (str): The name of the partitioning strategy to retrieve.

        Returns:
            SPACE_PARTITIONING_STRATEGY: An instance of the configured partitioning strategy.
        """
        # Check if a custom space partitioning strategy was provided
        if self._custom_space_partitioning_strategy is not None:
            logger.debug("Using custom space partitioning strategy")
            selected_strategy = self._custom_space_partitioning_strategy
        else:
            selected_strategy = SPACE_PARTITIONING_STRATEGIES.get(
                partitioning_strategy, None
            )
            if selected_strategy is None:
                logger.warning(
                    f"Invalid partitioning strategy specified: {partitioning_strategy}, using Voronoi partitioning"
                )
                selected_strategy = SPACE_PARTITIONING_STRATEGIES.get("voronoi")

        # Configure the bounding box
        bounding_box = BoundingBox(
            volume=0.0,
            boundaries=self.parameter_constraints,
            range_parameter_keys=self.range_parameter_keys,
        )
        bounding_box.calculate_volume()

        # Configure the strategy
        selected_strategy.bounding_box = bounding_box
        selected_strategy.range_parameter_keys = self.range_parameter_keys
        selected_strategy.integer_parameter_keys = self.integer_parameter_keys
        selected_strategy.float_parameter_keys = self.float_parameter_keys
        selected_strategy.statistics = self.statistics

        return selected_strategy

    def get_region_acquisition_function(self, region_acq_function: str) -> RegionACQ:
        """
        Retrieves and configures a region acquisition function based on the provided name.
        This method looks up the region acquisition function in REGION_ACQUISITION_FUNCTIONS.
        """
        if self._custom_region_acquisition_function is not None:
            logger.debug("Using custom region acquisition function")
            selected_region_acq = self._custom_region_acquisition_function
        else:
            selected_region_acq = REGION_ACQUISITION_FUNCTIONS.get(
                region_acq_function, None
            )
            if selected_region_acq is None:
                logger.warning(
                    f"No region acquisition function specified: {region_acq_function}"
                )
                return
        # Set any additional attributes here if needed
        selected_region_acq.metrics_targets = self.metrics_targets
        selected_region_acq.space_partitioning_settings = (
            self.space_partitioning_settings
        )
        selected_region_acq.candidates_per_request = self.candidates_per_request
        selected_region_acq.n_trials = self.total_trials
        selected_region_acq.alpha = self.space_partitioning_settings.get(
            "region_acquisition_alpha", 0.5
        )
        selected_region_acq.statistics = self.statistics
        # Scheduler setup if needed
        scheduler_name = self.space_partitioning_settings.get(
            "scheduler_settings", {}
        ).get("scheduler")
        if scheduler_name:
            selected_region_acq.scheduler = SCHEDULERS.get(scheduler_name)
            selected_region_acq.scheduler.apply_settings(
                self.space_partitioning_settings.get("scheduler_settings", {})
            )
        return selected_region_acq
