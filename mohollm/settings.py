from mohollm.llm.models.local import LOCAL
from mohollm.llm.models.gpt import GPT
from mohollm.llm.models.dummy_nb201 import DUMMY_NB201
from mohollm.llm.models.gemini import GEMINI
from mohollm.llm.models.groq import GROQ
from mohollm.llm.models.huggingface import HUGGINGFACE
from mohollm.llm.models.deepseek import DEEPSEEK
from mohollm.llm.models.nebius import NEBIUS
from mohollm.llm.models.openrouter import OPENROUTER

from mohollm.space_partitioning.voronoi_partitioning import VoronoiPartitioning
from mohollm.space_partitioning.kd_tree_partitioning import KDTreePartitioning

from mohollm.acquisition_functions.hypervolume_improvement import HypervolumeImprovement
from mohollm.acquisition_functions.hypervolume_improvement_batch import (
    HypervolumeImprovementBatch,
)

from mohollm.acquisition_functions.random_acq import RandomACQ
from mohollm.acquisition_functions.function_value import FunctionValueACQ
from mohollm.region_acquisition_functions.volume_region_acq import VolumeRegionACQ
from mohollm.region_acquisition_functions.vis_he_region_acq import VISHERegionACQ
from mohollm.region_acquisition_functions.vis_he_scheduling_region_acq import (
    VISHESchedulingRegionACQ,
)
from mohollm.region_acquisition_functions.vis_region_acq import VISRegionACQ
from mohollm.region_acquisition_functions.score_region_acq import ScoreRegionACQ
from mohollm.region_acquisition_functions.score_region_acq_hv import ScoreRegionHVC
from mohollm.region_acquisition_functions.score_region_acq_gdp import ScoreRegionGDP
from mohollm.region_acquisition_functions.score_region_acq_rhvc import ScoreRegionRHVC
from mohollm.region_acquisition_functions._score_region_acq_ablations import (
    ScoreRegionACQAblations,
)

from mohollm.surrogate_models.llm_surrogate_batch import LLM_Surrogate_batch
from mohollm.surrogate_models.llm_surrogate import LLM_Surrogate
from mohollm.surrogate_models.gaussian_process import GaussianProcessSurrogate
from mohollm.surrogate_models.tabpfn import TabPFNSurrogate
from mohollm.surrogate_models.gaussian_process_botorch import (
    GaussianProcessBoTorchSurrogate,
)
from mohollm.candidate_sampler.LLM_sampler import LLM_SAMPLER
from mohollm.candidate_sampler.random_search_sampler import RandomSearchSampler
from mohollm.candidate_sampler.qLogEHVI_sampler import qLogEHVISampler
from mohollm.warmstarter.random_warmstarter import RANDOM_WARMSTARTER
from mohollm.warmstarter.zero_shot_warmstarter import ZERO_SHOT_WARMSTARTER
from mohollm.statistics.context_limit_strategy.lastN import LastN
from mohollm.statistics.context_limit_strategy.random import Random
from mohollm.schedulers.constant_scheduler import ConstantScheduler
from mohollm.schedulers.step_wise_decay_scheduler import StepWiseDecayScheduler
from mohollm.schedulers.linear_decay_scheduler import LinearDecayScheduler
from mohollm.schedulers.epsilon_greedy_scheduler import EpsilonGreedyScheduler
from mohollm.schedulers.epsilon_greedy_decay_scheduler import EpsilonDecayScheduler
from mohollm.schedulers.cosine_annealing_scheduler import CosineAnnealingScheduler
from mohollm.schedulers.cosine_decay_scheduler import CosineDecayScheduler
from mohollm.surrogate_models.random_search_surrogate import RandomSearchSurrogate


MODELS = {
    "casperhansen/llama-3-8b-instruct-awq": LOCAL,
    "solidrust/gemma-2-9b-it-AWQ": LOCAL,
    "Qwen/Qwen2.5-7B-Instruct-AWQ": LOCAL,
    "Qwen/Qwen2.5-14B-Instruct-AWQ": LOCAL,
    "Qwen/Qwen2.5-32B-Instruct-AWQ": LOCAL,
    "Qwen/QwQ-32B-AWQ": LOCAL,
    "stelterlab/phi-4-AWQ": LOCAL,
    "gpt-4o-mini": GPT,
    "gpt-4.1-nano": GPT,
    "gpt-5-nano": GPT,
    "gemini-1.5-flash": GEMINI,
    "gemini-2.0-flash": GEMINI,
    "gemini-2.5-flash-lite": GEMINI,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": DEEPSEEK,
    "Dummy_nb201": DUMMY_NB201,
    "Qwen/Qwen3-4B-fast": NEBIUS,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": NEBIUS,
    "meta-llama/Llama-3.3-70B-Instruct": NEBIUS,
    "openai/gpt-oss-120b": NEBIUS,
    "openai/gpt-oss-20b": NEBIUS,
    "google/gemma-2-2b-it": NEBIUS,
    "google/gemma-2-9b-it-fast": NEBIUS,
    "Qwen/Qwen3-14B": NEBIUS,
    "Qwen/Qwen3-32B": NEBIUS,
    "Qwen/Qwen3-32B-fast": NEBIUS,
    "Qwen/QwQ-32B": NEBIUS,
    "openai/gpt-oss-20b:free": OPENROUTER,
}

WARMSTARTERS = {
    "RANDOM_WARMSTARTER": RANDOM_WARMSTARTER,
    "ZERO_SHOT_WARMSTARTER": ZERO_SHOT_WARMSTARTER,
}
CANDIDATE_SAMPLERS = {
    "LLM_SAMPLER": LLM_SAMPLER,
    "RANDOM_SEARCH_SAMPLER": RandomSearchSampler,
    "qLogEHVISampler": qLogEHVISampler,
}
SURROGATE_MODELS = {
    "LLM_SUR": LLM_Surrogate,
    "LLM_SUR_BATCH": LLM_Surrogate_batch,
    "RANDOM_SEARCH_SURROGATE": RandomSearchSurrogate,
    "GAUSSIAN_PROCESS_SURROGATE": GaussianProcessSurrogate,
    "TABPFN_SURROGATE": TabPFNSurrogate,
    "GAUSSIAN_PROCESS_SURROGATE_BOTORCH": GaussianProcessBoTorchSurrogate,
}
ACQUISITION_FUNCTIONS = {
    "HypervolumeImprovement": HypervolumeImprovement,  # Multi-objective
    "HypervolumeImprovementBatch": HypervolumeImprovementBatch,  # Please use the non-batch version
    "RandomACQ": RandomACQ,  # Single and multi-objective
    "FunctionValueACQ": FunctionValueACQ,  # Single objective
}

REGION_ACQUISITION_FUNCTIONS = {
    "VOLUME": VolumeRegionACQ(),
    "VIS_HE": VISHERegionACQ(),
    "VIS_HE_SCHEDULER": VISHESchedulingRegionACQ(),
    "VIS": VISRegionACQ(),
    "ScoreRegion": ScoreRegionACQ(),
    "ScoreRegionHVC": ScoreRegionHVC(),
    "ScoreRegionRHVC": ScoreRegionRHVC(),
    "ScoreRegionGDP": ScoreRegionGDP(),
    # Just for ablation usage with more configuration options
    "ScoreRegionAblations": ScoreRegionACQAblations(),
}

SCHEDULERS = {
    "CONSTANT_SCHEDULER": ConstantScheduler(),
    "STEP_WISE_DECAY_SCHEDULER": StepWiseDecayScheduler(),
    "LINEAR_DECAY_SCHEDULER": LinearDecayScheduler(),
    "EPSILON_GREEDY_SCHEDULER": EpsilonGreedyScheduler(),
    "EPSILON_DECAY_SCHEDULER": EpsilonDecayScheduler(),
    "COSINE_ANNEALING_SCHEDULER": CosineAnnealingScheduler(),
    "COSINE_DECAY_SCHEDULER": CosineDecayScheduler(),
}

CONTEXT_LIMIT_STRATEGIES = {
    "LastN": LastN,
    "Random": Random,
}
# Space partitioning strategies
SPACE_PARTITIONING_STRATEGIES = {
    "voronoi": VoronoiPartitioning(),
    "kdtree": KDTreePartitioning(),
}
