import logging
from mohollm.benchmarks.nb201.NB201Bench import NB201
from mohollm.benchmarks.welded_beam.welded_beam import WELDED_BEAM
from mohollm.benchmarks.dtlz.dtlz import DTLZ
from mohollm.benchmarks.vlmop.vlmop import VLMOP
from mohollm.benchmarks.simple_2d_mo.simple_2d import Simple2D
from mohollm.benchmarks.simple_2d_mo.chankong_haimes import ChankongHaimes
from mohollm.benchmarks.simple_2d_mo.test_function_4 import TestFunction4
from mohollm.benchmarks.simple_2d_mo.schaffer_n1 import SchafferN1
from mohollm.benchmarks.simple_2d_mo.schaffer_n2 import SchafferN2
from mohollm.benchmarks.simple_2d_mo.poloni import Poloni
from mohollm.benchmarks.zdt.zdt import ZDT
from mohollm.benchmarks.branin_currin import BraninCurrinBenchmark
from mohollm.benchmarks.gmm import GMMBenchmark
from mohollm.benchmarks.toy_robust import ToyRobustBenchmark
from mohollm.benchmarks.penicillin import PenicillinBenchmark
from mohollm.benchmarks.vehicle_safety import VehicleSafetyBenchmark
from mohollm.benchmarks.car_side_impact import CarSideImpactBenchmark
from mohollm.benchmarks.simple_2d_mo.kursawe import Kursawe

logger = logging.getLogger("Benchmark Initialization")


def get_benchmark_fn(config):
    benchmark_settings = config.get("benchmark_settings", None)
    benchmark_name = config.get("benchmark", None)
    seed = config.get("seed", None)
    metrics = config.get("metrics", None)
    model_name = config.get("llm_settings", {}).get("model", None)

    match benchmark_name:
        case "NB201":
            dataset = benchmark_settings.get("dataset", None)
            device_metric = benchmark_settings.get("device_metric", None)
            if dataset is None:
                logger.warning("No dataset specified for NB201 benchmark")
                raise ValueError("No dataset specified for NB201 benchmark")
            if device_metric is None:
                logger.warning("No device metric specified for NB201 benchmark")
                raise ValueError("No device metric specified for NB201 benchmark")

            return NB201(metrics, dataset, device_metric, seed, model_name)
        case "ZDT":
            problem_id = benchmark_settings.get("problem_id", 1)
            return ZDT(model_name, seed, problem_id)
        case "WELDED_BEAM":
            problem_id = benchmark_settings.get("problem_id", None)
            return WELDED_BEAM(model_name, seed, problem_id)
        case "VLMOP":
            return VLMOP(model_name, seed)
        case "DTLZ":
            problem_id = benchmark_settings.get("problem_id", None)
            return DTLZ(model_name, seed, problem_id=problem_id)
        case "Simple2D":
            return Simple2D(model_name, seed)
        case "ChankongHaimes":
            return ChankongHaimes(model_name, seed)
        case "TestFunction4":
            return TestFunction4(model_name, seed)
        case "SchafferN1":
            A = benchmark_settings.get("A", 10)
            return SchafferN1(model_name, seed, A)
        case "SchafferN2":
            return SchafferN2(model_name, seed)
        case "Poloni":
            return Poloni(model_name, seed)
        case "BraninCurrin":
            return BraninCurrinBenchmark(model_name, seed)
        case "GMM":
            return GMMBenchmark(model_name, seed)
        case "ToyRobust":
            return ToyRobustBenchmark(model_name, seed)
        case "penicillin":
            return PenicillinBenchmark(model_name, seed)
        case "vehicle_safety":
            return VehicleSafetyBenchmark(model_name, seed)
        case "car_side_impact":
            return CarSideImpactBenchmark(model_name, seed)
        case "Kursawe":
            return Kursawe(model_name, seed)
        case _:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
