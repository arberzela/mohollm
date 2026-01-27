COLORS_FALLBACK = [
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#469990",
    "#469990",
    "#51c36f",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#000000",
    "#808000",
    "#000075",
]


# COLORS_SURROGATE_ACC = ["#003049", "#d62828", "#f77f00", "#fcbf49"]
COLORS_SURROGATE_ACC = ["red", "green", "blue", "black"]


LINE_STYLE_HV = [
    "-",  # Solid
    # "--",  # Dashed
    # ":",  # Dotted
    # "-.",  # Dash-Dot
]


MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]


# My old color palette
# COLORS = {
#     # --- LLM Methods (Warm and Vibrant) ---
#     "MOHOLLM (Gemini 2.0 Flash) (Context)": "#E6194B",  # Vibrant Red
#     "MOHOLLM (Gemini 2.0 Flash)": "#E6194B",  # Vibrant Red
#     "mohollm (Gemini 2.0 Flash)": "#F58231",  # Bright Orange
#     # --- BASELINES (Cool Colors) ---
#     "CTAEA": "#0082C8",  # Strong Blue
#     "EpsNSGA2": "#469990",  # Teal
#     "IBEA": "#3CB44B",  # Green
#     "MOEAD": "#4363d8",  # Sapphire Blue
#     "NSGA2": "#911eb4",  # Purple
#     "NSGA3": "#f032e6",  # Magenta
#     "PESA2": "#008080",  # Dark Teal
#     "RNSGA2": "#42d4f4",  # Cyan
#     "RVEA": "#aaffc3",  # Mint Green
#     "SMSEMOA": "#800000",  # Maroon
#     "SPEA2": "#808000",  # Olive
#     "UNSGA3": "#bfef45",  # Lime Green
#     "qLogEHVI": "#00a691",
#     "EHVI": "#0085a4",
#     "GDE3": "#f7aa34",
#     # --- Add a fallback for any other methods ---
#     # You can add more specific assignments here if needed
# }

# https://sashamaps.net/docs/resources/20-color
COLORS = {
    # --- LLM Methods (Warm and Vibrant) ---
    "MOHOLLM (Gemini 2.0 Flash) (Context)": "#e6194B",  # Vibrant Red
    "MOHOLLM (Gemini 2.0 Flash)": "dodgerblue",
    # "MOHOLLM (Gemini 2.0 Flash)": "#469990",  # Testing
    "mohollm (Gemini 2.0 Flash)": "#f56805",  # f58231",  # Bright Orange
    "RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context)": "purple",
    "RS + LLM Surrogate (Gemini 2.0 Flash) (Context)": "forestgreen",
    "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context)": "dodgerblue",
    "MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context)": "darkgreen",
    # --- BASELINES (Cool Colors) ---
    "CTAEA": "#3cb44b",
    "EpsNSGA2": "#ffe119",  
    "IBEA": "#4363d8",
    "MOEAD": "#911eb4",
    "NSGA2": "#42d4f4",
    "NSGA3": "#f032e6",
    "PESA2": "#bfef45",
    "RNSGA2": "#469990",
    "RVEA": "#469990",
    "SMSEMOA": "#51c36f",
    "SPEA2": "#9A6324",
    "UNSGA3": "#fffac8",
    "qLogEHVI": "#800000",
    "EHVI": "#000000",
    "GDE3": "#808000",
    "UNSGA3": "#000075",
    # --- Add a fallback for any other methods ---
    # You can add more specific assignments here if needed
    # Alpha sweep labels for MOHOLLM
    "MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0": "#c6dbef",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4": "#6baed6",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8": "#2171b5",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0": "#08306b",
    # Candidates per request sweep labels for MOHOLLM
    "MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1": "#c6dbef",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3": "#6baed6",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7": "#2171b5",
    # M0 sweep labels for MOHOLLM
    "MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1": "#c6dbef",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3": "#6baed6",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10": "#2171b5",
    # "MOHOLLM (Gemini 2.0 Flash) (Context)": r"MOHOLLM ($K = 5$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=1": "#c6dbef",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=3": "#6baed6",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=7": "#2171b5",
    # Beta sweep labels for MOHOLLM
    "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0)": "#3cb44b",
    "MOHOLLM (Gemini 2.0 Flash) (Beta=0.25)": "#ffe119",
    "MOHOLLM (Gemini 2.0 Flash) (Beta=0.5)": "#4363d8",
    "MOHOLLM (Gemini 2.0 Flash) (Beta=0.75)": "#911eb4",
    "MOHOLLM (Gemini 2.0 Flash) (Beta=1.0)": "#42d4f4",
    # Beta sweep labels for mohollm -> displayed as LLM
    "mohollm (Gemini 2.0 Flash) (Beta=0.0)": "#469990",
    "mohollm (Gemini 2.0 Flash) (Beta=0.25)": "#800000",
    "mohollm (Gemini 2.0 Flash) (Beta=0.5)": "#000000",
    "mohollm (Gemini 2.0 Flash) (Beta=0.75)": "#808000",
    "mohollm (Gemini 2.0 Flash) (Beta=1.0)": "#000075",
    "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context)": "blue",  # TODO: Testing
}


# TODO: These values are empirically computed from all runs!
# 1% above the maximum value across all the runs
HV_REFERENCE_POINTS = {
    "DTLZ1": [510.57419, 528.80469],
    "DTLZ2": [2.2725, 2.2725],
    "DTLZ3": [1109.84052, 1109.84052],
    "BraninCurrin": [311.21029, 13.91174],
    "ChankongHaimes": [936.27, 180.3557],
    "GMM": [0.0, 0.0],
    "Poloni": [62.2463, 52.57454],
    "SchafferN1": [101.0, 145.44],
    "SchafferN2": [6.06, 101.0],
    "TestFunction4": [56.56, 9.595],
    "ToyRobust": [49.995, 37.36394],
    "Kursawe": [-4.91062, 24.01174],
    "Penicillin": [25.935, 57.612, 935.5],  # Known from benchmark
    "CarSideImpact": [45.4872, 4.5114, 13.3394, 10.3942],  # Known from benchmark
    "VehicleSafety": [1864.72022, 11.81993945, 0.2903999384],  # Known from benchmark
}

# The maximum possible hypervolume per benchmark
MAX_HV = {
    # Rest is unknown
    "Penicillin": 2183455.909507436,
    "CarSideImpact": 484.72654347642793,
    "VehicleSafety": 246.81607081187002,
}


def get_color(method: str, idx: int) -> str:
    if method not in COLORS:
        print(f"Method: {method} not in COLORS. Using fallback color ")
        return COLORS_FALLBACK[idx % len(COLORS_FALLBACK)]
    return COLORS[method]


LABEL_MAP_HV = {
    "MOHOLLM (Gemini 2.0 Flash)": "MOHOLLM (No Context)",
    "MOHOLLM (Gemini 2.0 Flash) (Context)": "MOHOLLM",  # "MOHOLLM (Context)",
    "MOHOLLM (Gemini 2.0 Flash) (minimal)": "MOHOLLM (Minimal)",  # "MOHOLLM (Context)",
    "mohollm (Gemini 2.0 Flash)": "LLM",
    # Ablations
    # "MOHOLLM (Gemini 2.0 Flash) (Context)": r"MOHOLLM ($\alpha_{max} = 1.0 $)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0": r"MOHOLLM ($\alpha_{max} = 0 $)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4": r"MOHOLLM ($\alpha_{max} = 0.4$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8": r"MOHOLLM ($\alpha_{max} = 0.8$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0": r"MOHOLLM ($\alpha_{max} = 2.0$)",
    # "MOHOLLM (Gemini 2.0 Flash) (Context)": r"MOHOLLM ($M = 5$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1": r"MOHOLLM ($N = 1$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3": r"MOHOLLM ($N = 3$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7": r"MOHOLLM ($N = 7$)",
    # "MOHOLLM (Gemini 2.0 Flash) (Context)": r"MOHOLLM ($m_0 = 5$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1": r"MOHOLLM ($m_0 = 1$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3": r"MOHOLLM ($m_0 = 3$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10": r"MOHOLLM ($m_0 = 10$)",
    # "MOHOLLM (Gemini 2.0 Flash) (Context)": r"MOHOLLM ($K = 5$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=1": r"MOHOLLM ($k = 1$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=3": r"MOHOLLM ($k = 3$)",
    "MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=7": r"MOHOLLM ($k = 7$)",
    # Ablations components
    "MOHOLLM + GP (Gemini 2.0 Flash) (Context)": "MOHOLLM (GP Surrogate)",
    "MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context)": "MOHOLLM (TabPFN Surrogate)",
    "RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context)": "MOHOLLM (Random Sampler)",
    "RS + LLM Surrogate (Gemini 2.0 Flash) (Context)": "LLM (Random Sampler)",
    "qLogEHVI (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context)": "qLogEHVI (KD-Tree) + LLM Surrogate",
    "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context)": "MOHOLLM (GP Surrogate)",
    # Alpha prompt ablations
    "MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.0": r"MOHOLLM ($\alpha_{prompt} = 0.0$)",
    "MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.25": r"MOHOLLM ($\alpha_{prompt} = 0.25$)",
    "MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.5": r"MOHOLLM ($\alpha_{prompt} = 0.5$)",
    "MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.75": r"MOHOLLM ($\alpha_{prompt} = 0.75$)",
    "MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=1.0": r"MOHOLLM ($\alpha_{prompt} = 1.0$)",
    "mohollm (Gemini 2.0 Flash) - alpha_prompt=0.0": r"LLM ($\alpha_{prompt} = 0.0$)",
    "mohollm (Gemini 2.0 Flash) - alpha_prompt=0.25": r"LLM ($\alpha_{prompt} = 0.25$)",
    "mohollm (Gemini 2.0 Flash) - alpha_prompt=0.5": r"LLM ($\alpha_{prompt} = 0.5$)",
    "mohollm (Gemini 2.0 Flash) - alpha_prompt=0.75": r"LLM ($\alpha_{prompt} = 0.75$)",
    "mohollm (Gemini 2.0 Flash) - alpha_prompt=1.0": r"LLM ($\alpha_{prompt} = 1.0$)",
    # Models:
    "MOHOLLM (gemma-2-9b-it-fast) (Context)": "MOHOLLM (Gemma-2-9B)",
    "MOHOLLM (gpt-oss-120b) (Context)": "MOHOLLM (GPT-OSS-120B)",
    "MOHOLLM (Llama-3.3-70B-Instruct) (Context)": "MOHOLLM (Llama-3.3-70B)",
    "MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context)": "MOHOLLM (Llama-3.1-8B)",
    "MOHOLLM (Qwen3-32B) (Context)": "MOHOLLM (Qwen-3-32B)",
    "MOHOLLM (gpt-4o-mini) (Context)": "MOHOLLM (GPT-4o-mini)",
    #"MOHOLLM (Gemini 2.0 Flash) (Context)": "MOHOLLM (Gemini 2.0 Flash)",
    "mohollm (gemma-2-9b-it) (Context)": "LLM (Gemma 2 9B)",
    "mohollm (gpt-oss-120b) (Context)": "LLM (GPT Oss 120B)",
    "mohollm (llama3-3-70B) (Context)": "LLM (Llama 3.3 70B)",
    "mohollm (llama3-1-8B) (Context)": "LLM (Llama 3.1 8B)",
    "mohollm (Gemini 2.0 Flash) (Context)": "LLM (Gemini 2.0 Flash)",
    "mohollm (Qwen3-32B) (Context)": "LLM (Qwen3 32B)",
    "mohollm (gpt-4o-mini) (Context)": "LLM (GPT 4o Mini)",
    # Beta sweep labels for MOHOLLM
    "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0)": r"MOHOLLM ($\beta=0.0$)",
    "MOHOLLM (Gemini 2.0 Flash) (Beta=0.25)": r"MOHOLLM ($\beta=0.25$)",
    "MOHOLLM (Gemini 2.0 Flash) (Beta=0.5)": r"MOHOLLM ($\beta=0.5$)",
    "MOHOLLM (Gemini 2.0 Flash) (Beta=0.75)": r"MOHOLLM ($\beta=0.75$)",
    "MOHOLLM (Gemini 2.0 Flash) (Beta=1.0)": r"MOHOLLM ($\beta=1.0$)",
    # Beta sweep labels for mohollm -> displayed as LLM
    "mohollm (Gemini 2.0 Flash) (Beta=0.0)": r"LLM  ($\beta=0.0$)",
    "mohollm (Gemini 2.0 Flash) (Beta=0.25)": r"LLM  ($\beta=0.25$)",
    "mohollm (Gemini 2.0 Flash) (Beta=0.5)": r"LLM  ($\beta=0.5$)",
    "mohollm (Gemini 2.0 Flash) (Beta=0.75)": r"LLM  ($\beta=0.75$)",
    "mohollm (Gemini 2.0 Flash) (Beta=1.0)": r"LLM  ($\beta=1.0$)",
}

MARKER_MAP = {
    "MOHOLLM (Gemini 2.0 Flash)": "^",
    "MOHOLLM (Gemini 2.0 Flash) (Context)": "s",
    "MOHOLLM (Gemini 2.0 Flash) (minimal)": "h",
    "mohollm (Gemini 2.0 Flash)": "8",
    "RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context)": "v",
    "RS + LLM Surrogate (Gemini 2.0 Flash) (Context)": ">",
    "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context)": "d",
    "MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context)": "p",
}

# Add a mapping for the methods names (folder name) to more readable labels
LABEL_MAP = {}
