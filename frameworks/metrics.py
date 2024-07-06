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
