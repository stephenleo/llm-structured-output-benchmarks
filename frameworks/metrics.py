import itertools

import numpy as np
import pandas as pd


def reliability_metric(percent_successful: dict[str, list[float]]):
    df = pd.DataFrame(percent_successful)
    df.columns = [col.replace("Framework", "") for col in df.columns]

    reliability = df.describe().loc["mean", :].to_frame(name="Reliability")
    reliability = reliability.round(3)
    reliability.sort_values(by="Reliability", ascending=False, inplace=True)
    return reliability


def latency_metric(latencies: dict[str, list[float]], percentile: int = 95):
    # Flatten the list of latencies
    latencies = {
        key: list(itertools.chain.from_iterable(value))
        for key, value in latencies.items()
    }

    # Calculate the latency percentiles
    latencies = {
        key.replace("Framework", ""): np.percentile(values, percentile)
        for key, values in latencies.items()
    }

    latency_percentile = pd.DataFrame(list(latencies.values()), index=latencies.keys(), columns=[f"Latency_p{percentile}(s)"])
    latency_percentile = latency_percentile.round(3)
    latency_percentile.sort_values(
        by=f"Latency_p{percentile}(s)", ascending=True, inplace=True
    )
    return latency_percentile


def ner_micro_metrics(results: dict[str, list[float]]):
    micro_metrics = {
            "framework": [],
            "micro_precision": [],
            "micro_recall": [],
            "micro_f1": []
        }
    
    for framework, values in results.items():
        tp_total, fp_total, fn_total = 0, 0, 0
        runs = values["ner"]["metrics"]

        for run in runs:
            for metric in run:
                tp_total += sum(metric["true_positives"].values())
                fp_total += sum(metric["false_positives"].values())
                fn_total += sum(metric["false_negatives"].values())

        micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0
        )

        micro_metrics["framework"].append(framework.replace("Framework", ""))
        micro_metrics["micro_precision"].append(micro_precision)
        micro_metrics["micro_recall"].append(micro_recall)
        micro_metrics["micro_f1"].append(micro_f1)

    return pd.DataFrame(micro_metrics)
