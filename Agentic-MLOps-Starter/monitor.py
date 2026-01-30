import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def calculate_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.quantile(expected, quantiles)

    psi_value = 0
    for i in range(buckets):
        lower = breakpoints[i]
        upper = breakpoints[i + 1]

        expected_pct = ((expected >= lower) & (expected < upper)).mean()
        actual_pct = ((actual >= lower) & (actual < upper)).mean()

        expected_pct = max(expected_pct, 1e-6)
        actual_pct = max(actual_pct, 1e-6)

        psi_value += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    return psi_value

def calculate_ks(expected, actual):
    ks_stat, _ = ks_2samp(expected, actual)
    return ks_stat

def detect_drift(baseline_path, current_path, psi_threshold=0.2, ks_threshold=0.1):
    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)

    expected = baseline_df["trade_value"]
    actual = current_df["trade_value"]

    psi_value = calculate_psi(expected, actual)
    ks_value = calculate_ks(expected, actual)

    drift_detected = psi_value > psi_threshold or ks_value > ks_threshold

    return {
        "psi": psi_value,
        "ks": ks_value,
        "drift_detected": drift_detected
    }

if __name__ == "__main__":
    result = detect_drift("data/baseline_data.csv", "data/inference_data.csv")
    print(result)
