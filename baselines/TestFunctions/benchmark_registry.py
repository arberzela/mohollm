from syne_tune.config_space import Float
from mohollm.benchmarks.simple_2d_mo.chankong_haimes import ChankongHaimes
from mohollm.benchmarks.simple_2d_mo.test_function_4 import TestFunction4
from mohollm.benchmarks.simple_2d_mo.poloni import Poloni
from mohollm.benchmarks.simple_2d_mo.schaffer_n1 import SchafferN1
from mohollm.benchmarks.simple_2d_mo.schaffer_n2 import SchafferN2
from mohollm.benchmarks.nb201.NB201Bench import NB201
from mohollm.benchmarks.branin_currin import BraninCurrinBenchmark
from mohollm.benchmarks.car_side_impact import CarSideImpactBenchmark
from mohollm.benchmarks.gmm import GMMBenchmark
from mohollm.benchmarks.penicillin import PenicillinBenchmark
from mohollm.benchmarks.toy_robust import ToyRobustBenchmark
from mohollm.benchmarks.vehicle_safety import VehicleSafetyBenchmark
from mohollm.benchmarks.dtlz.dtlz import DTLZ
from mohollm.benchmarks.simple_2d_mo.kursawe import Kursawe


def nb201_continuous_to_discrete(config):
    # Map [0,1] floats to nearest discrete value for each op key
    discrete_keys = [
        "op_0_to_1",
        "op_0_to_2",
        "op_0_to_3",
        "op_1_to_2",
        "op_1_to_3",
        "op_2_to_3",
    ]
    choices = [0, 1, 2, 3, 4]
    mapped = {}
    for k in config:
        if k in discrete_keys:
            # Map float in [0,1] to one of the choices
            idx = int(round(config[k] * (len(choices) - 1)))
            idx = max(0, min(idx, len(choices) - 1))
            mapped[k] = choices[idx]
        else:
            mapped[k] = config[k]
    return mapped


NB201_DEVICES = [
    "fpga_latency",
    "pixel3_latency",
    "raspi4_latency",
    "eyeriss_latency",
    "pixel2_latency",
    "1080ti_1_latency",
    "1080ti_32_latency",
    "1080ti_256_latency",
    "2080ti_1_latency",
    "2080ti_32_latency",
    "2080ti_256_latency",
    "titanx_1_latency",
    "titanx_32_latency",
    "titanx_256_latency",
    "titanxp_1_latency",
    "titanxp_32_latency",
    "titanxp_256_latency",
    "titan_rtx_1_latency",
    "titan_rtx_32_latency",
    "titan_rtx_256_latency",
    "essential_ph_1_latency",
    "gold_6226_latency",
    "gold_6240_latency",
    "samsung_a50_latency",
    "samsung_s7_latency",
    "silver_4114_latency",
    "silver_4210r_latency",
]
NB201_BENCHMARKS = ["cifar10", "cifar100", "imagenet16"]


def populate_nb201_benchmarks(benchmarks_dict):
    for dataset in NB201_BENCHMARKS:
        for device in NB201_DEVICES:
            key = f"nb201_{dataset}_{device}"
            benchmarks_dict[key] = {
                "class": NB201,
                "args": {
                    "metrics": ["F1", "F2"],
                    "dataset": dataset,
                    "device_metric": device,
                },
                "config_space": {
                    "op_0_to_1": Float(0, 1),
                    "op_0_to_2": Float(0, 1),
                    "op_0_to_3": Float(0, 1),
                    "op_1_to_2": Float(0, 1),
                    "op_1_to_3": Float(0, 1),
                    "op_2_to_3": Float(0, 1),
                },
                "result_folder": f"NB201/{dataset}/{device}",
                "continuous_to_discrete": nb201_continuous_to_discrete,
            }


BENCHMARKS = {
    "chankong_haimes": {
        "class": ChankongHaimes,
        "config_space": {"x": Float(-20, 20), "y": Float(-20, 20)},
        "result_folder": "ChankongHaimes",
    },
    "test_function_4": {
        "class": TestFunction4,
        "config_space": {"x": Float(-7, 4), "y": Float(-7, 4)},
        "result_folder": "TestFunction4",
    },
    "schaffer_n1": {
        "class": SchafferN1,
        "config_space": {"x": Float(-10, 10)},
        "result_folder": "SchafferN1",
    },
    "schaffer_n2": {
        "class": SchafferN2,
        "config_space": {"x": Float(-5, 10)},
        "result_folder": "SchafferN2",
    },
    "poloni": {
        "class": Poloni,
        "config_space": {"x": Float(-3.14159, 3.14159), "y": Float(-3.14159, 3.14159)},
        "result_folder": "Poloni",
    },
    "branin_currin": {
        "class": BraninCurrinBenchmark,
        "config_space": {"x0": Float(0, 1), "x1": Float(0, 1)},
        "result_folder": "BraninCurrin",
    },
    "car_side_impact": {  # This one has 4 objectives
        "class": CarSideImpactBenchmark,
        "config_space": {
            "x0": Float(0.5, 1.5),
            "x1": Float(0.45, 1.35),
            "x2": Float(0.5, 1.5),
            "x3": Float(0.5, 1.5),
            "x4": Float(0.875, 2.625),
            "x5": Float(0.4, 1.2),
            "x6": Float(0.4, 1.2),
        },
        "result_folder": "CarSideImpact",
    },
    "GMM": {
        "class": GMMBenchmark,
        "config_space": {
            "x0": Float(0, 1),
            "x1": Float(0, 1),
        },
        "result_folder": "GMM",
    },
    "penicillin": {  # This one has 3 objectives
        "class": PenicillinBenchmark,
        "config_space": {
            "x0": Float(60.0, 120.0),
            "x1": Float(0.05, 18.0),
            "x2": Float(293.0, 303.0),
            "x3": Float(0.05, 18.0),
            "x4": Float(0.01, 0.5),
            "x5": Float(500.0, 700.0),
            "x6": Float(5.0, 6.5),
        },
        "result_folder": "Penicillin",
    },
    "toy_robust": {
        "class": ToyRobustBenchmark,
        "config_space": {
            "x": Float(0.0, 0.7),
        },
        "result_folder": "ToyRobust",
    },
    "vehicle_safety": {  # This one has 3 objectives
        "class": VehicleSafetyBenchmark,
        "config_space": {
            "x0": Float(1.0, 3.0),
            "x1": Float(1.0, 3.0),
            "x2": Float(1.0, 3.0),
            "x3": Float(1.0, 3.0),
            "x4": Float(1.0, 3.0),
        },
        "result_folder": "VehicleSafety",
    },
    "dtlz1": {
        "class": DTLZ,
        "args": {"problem_id": "dtlz1"},
        "config_space": {
            "x0": Float(0.0, 1.0),
            "x1": Float(0.0, 1.0),
            "x2": Float(0.0, 1.0),
            "x3": Float(0.0, 1.0),
            "x4": Float(0.0, 1.0),
            "x5": Float(0.0, 1.0),
        },
        "result_folder": "DTLZ1",
    },
    "dtlz2": {
        "class": DTLZ,
        "args": {"problem_id": "dtlz2"},
        "config_space": {
            "x0": Float(0.0, 1.0),
            "x1": Float(0.0, 1.0),
            "x2": Float(0.0, 1.0),
            "x3": Float(0.0, 1.0),
            "x4": Float(0.0, 1.0),
            "x5": Float(0.0, 1.0),
        },
        "result_folder": "DTLZ2",
    },
    "dtlz3": {
        "class": DTLZ,
        "args": {"problem_id": "dtlz3"},
        "config_space": {
            "x0": Float(0.0, 1.0),
            "x1": Float(0.0, 1.0),
            "x2": Float(0.0, 1.0),
            "x3": Float(0.0, 1.0),
            "x4": Float(0.0, 1.0),
            "x5": Float(0.0, 1.0),
        },
        "result_folder": "DTLZ3",
    },
    "kursawe": {
        "class": Kursawe,
        "config_space": {
            "x0": Float(-5, 5),
            "x1": Float(-5, 5),
            "x2": Float(-5, 5),
        },
        "result_folder": "Kursawe",
    },
}

# Populate BENCHMARKS with all NB201 dataset/device combinations
populate_nb201_benchmarks(BENCHMARKS)
# print(f"Populated {len(BENCHMARKS)} benchmarks in total.")
# print("Available benchmarks:", list(BENCHMARKS.keys()))


def get_benchmark(problem_name, seed=None, model_name="baseline"):
    if problem_name not in BENCHMARKS:
        raise ValueError(f"Unknown problem: {problem_name}")
    bench = BENCHMARKS[problem_name]
    cls = bench["class"]
    config_space = bench["config_space"]
    mapping_fn = bench.get("continuous_to_discrete", None)
    if cls == NB201:
        # For NB201, we need to pass additional arguments
        args = bench.get("args", {})
        instance = cls(
            args.get("metrics", None),
            args.get("dataset", None),
            args.get("device_metric", None),
            seed,
            model_name,
            pkl_path="../../mohollm/benchmarks/nb201/nb201.pkl",  # We need to override the default path
        )
    if cls == DTLZ:
        # For DTLZ, we need to pass the problem_id
        instance = cls(
            model_name=model_name,
            seed=seed,
            problem_id=bench.get("args", {}).get("problem_id", None),
        )
    else:
        instance = cls(model_name=model_name, seed=seed)
    return instance, config_space, mapping_fn
