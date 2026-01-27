from syne_tune import Reporter
import pickle
import torch
import logging
import ConfigSpace as CS

report = Reporter()

edge_mapping = {
    0: "none",
    1: "skip_connect",
    2: "avg_pool_3x3",
    3: "nor_conv_1x1",
    4: "nor_conv_3x3",
}


def objective(arch, metric, dataset):
    print(arch)
    mapped_arch = {key: edge_mapping[val] for key, val in arch.items()}
    print("After mapping", mapped_arch)
    arch_str = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*mapped_arch.values())
    print(arch_str)
    with open("./benchmark_all_hw_metrics.pkl", "rb") as f:
        data = pickle.load(f)
    error = 100 - data[arch_str][dataset]
    latency = data[arch_str][metric]
    print(error, latency)
    report(error=error, latency=latency)


if __name__ == "__main__":
    import logging
    import argparse

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    num_edges = 6
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, required=True)
    for i in range(num_edges):
        parser.add_argument(f"--edge{i}", type=int, default=0)
    args, _ = parser.parse_known_args()
    print(vars(args))
    arch = {}
    for i in range(num_edges):
        arch["edge" + str(i)] = getattr(args, f"edge{i}")
    objective(arch, args.metric, "cifar10")
