import os
import logging
from datetime import datetime
import random

timestamp = (
    f"{datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}_{random.randint(1000, 9999)}"
)


LOGDIR = ".mohollm/log/"
LOGFILE_TEMPLATE = "mohollm_{}.log"
LOGFILE = LOGFILE_TEMPLATE.format(timestamp)
LOGPATH = os.path.join(LOGDIR, LOGFILE)

os.makedirs(LOGDIR, exist_ok=True)


LOGGING_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
DEFAULT_LOG_LEVEL = 0  # 0 equals to NOTSET level

# Estimate maxBytes based on average log entry size (assume 100 bytes per entry)
MAX_BYTES = 10000 * 100  # ~1 MB assuming each entry is about 100 bytes

# Add a the logger name to this list to be shown in the output logs and console
WHITELISTED_LOGGERS = list(
    dict.fromkeys(
        [
    "mohollm",
    "LLMInterface",
    "Builder",
    "ACQUISITION_FUNCTION",
    "LOCAL_LLM",
    "Dummy_LLM",
    "mohollm_ACQ",
    "RateLimiter",
    "Hypervolume",
    "SURROGATE_MODEL",
    "CANDIDATE_SAMPLER",
    "LLM_SAMPLER",
    "WARMSTARTER",
    "RANDOM_WARMSTARTER",
    "ZERO_SHOT_WARMSTARTER",
    "PromptBuilder",
    "STATISTICS",
    "RandomACQ",
    "KDE",
    "GPT",
    "ZDT",
    "BENCHMARK",
    "SURROGATE_MODEL_BATCH",
    "HypervolumeImprovementBatch",
    "SpacePartitioningmohollm",
    "RegionACQ",
    "Threadedmohollm",
    "VoronoiPartitioning",
    "WELDED_BEAM",
    "KDTreePartitioning",
    "GEMINI",
    "HUGGINGFACE",
    "Groq",
    "COSINE_ANNEALING_SCHEDULER",
    "LLM_Surrogate",
    "LLM_Surrogate_batch",
    "ScoreRegionACQ",
    "ScoreMORegionACQ",
    "RandomSearchSampler",
    "RandomSearchSurrogate",
    "CarSideImpact",
    "NEBIUS",
    "GaussianProcessSurrogate",
    "TabPFNSurrogateModel",
    "OpenRouter",
    "Kursawe",
    "qLogEHVISampler",
    "ScoreRegionACQAblations",
        ]
    )
)


class WhitelistFilter(logging.Filter):
    def __init__(self, whitelist=None):
        super().__init__()
        if whitelist is None:
            whitelist = []
        self.whitelist = whitelist

    def filter(self, record):
        return record.name in self.whitelist


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": LOGGING_FORMAT}},
    "filters": {
        "whitelist_filter": {
            "()": WhitelistFilter,
            "whitelist": WHITELISTED_LOGGERS,
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
            "filters": ["whitelist_filter"],
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "simple",
            "filename": LOGPATH,
            "filters": ["whitelist_filter"],
        },
    },
    "loggers": {"root": {"level": "DEBUG", "handlers": ["stdout", "file"]}},
}
