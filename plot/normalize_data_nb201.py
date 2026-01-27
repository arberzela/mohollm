import os
import pickle
import glob
import numpy as np
import pandas as pd


dataset = "cifar10"


def normalize_data_nb201(df, device_metric):
    with open("./mohollm/benchmarks/nb201/benchmark_all_hw_metrics.pkl", "rb") as f:
        data = pickle.load(f)

    true_errors = []
    true_latencies = []
    archs_true = []

    for arch in data.keys():
        true_errors.append(100 - data[arch][dataset])
        true_latencies.append(data[arch][device_metric])
        archs_true.append(arch)

    max_lat, min_lat = max(true_latencies), min(true_latencies)
    max_err, min_err = max(true_errors), min(true_errors)

    # TODO: This right now assumes that the dataframe has columns 'error' and 'latency'
    # df["error"] = (df["error"] - min_err) / (max_err - min_err)
    # df["latency"] = (df["latency"] - min_lat) / (max_lat - min_lat)

    # TODO: Make this flexible to different column names
    df["F1"] = (df["F1"] - min_err) / (max_err - min_err)
    df["F2"] = (df["F2"] - min_lat) / (max_lat - min_lat)

    return df
