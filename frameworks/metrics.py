import itertools

import pandas as pd


def reliability_metric(percent_successful: dict[str, list[float]]):
    df = pd.DataFrame(percent_successful)
    df.columns = [col.replace("Framework", "") for col in df.columns]

    reliability = df.describe().loc["mean", :].to_frame(name="Reliability")
    reliability = reliability.round(3)
    reliability.sort_values(by="Reliability", ascending=False, inplace=True)
    return reliability


def latency_metric(latencies: dict[str, list[float]], percentile: int = 0.95):
    # Flatten the list of latencies
    latencies = {
        key: list(itertools.chain.from_iterable(value))
        for key, value in latencies.items()
    }

    df = pd.DataFrame(latencies)
    df.columns = [col.replace("Framework", "") for col in df.columns]

    # Calculate the desired percentile for each column
    latency_percentile = df.quantile(percentile).to_frame(name=f"Latency_p{int(percentile*100)}(s)")
    latency_percentile = latency_percentile.round(3)
    latency_percentile.sort_values(by=f"Latency_p{int(percentile*100)}(s)", ascending=True, inplace=True)
    return latency_percentile