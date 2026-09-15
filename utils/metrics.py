"""RMSE, MAE, mean-normalized errors, and Willmott's agreement index."""

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = y_true.flatten().astype(np.float64)
    yp = y_pred.flatten().astype(np.float64)

    mse = np.mean((yt - yp) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(yt - yp))

    # Normalize by mean observed GHI.
    mean_true = np.mean(yt)
    nrmse = rmse / (mean_true + 1e-8)
    nmae = mae / (mean_true + 1e-8)

    # Willmott's Index of Agreement (IA)
    diff_sq = np.sum((yt - yp) ** 2)
    potential_error = np.sum((np.abs(yp - mean_true) + np.abs(yt - mean_true)) ** 2)
    ia = 1.0 - (diff_sq / (potential_error + 1e-8))

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "nRMSE": float(nrmse),
        "nMAE": float(nmae),
        "IA": float(ia),
        "Samples": int(len(yt))
    }


def compute_daytime_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 10.0) -> dict:
    """Evaluate forecast points with observed GHI above the threshold."""
    yt = y_true.flatten().astype(np.float64)
    yp = y_pred.flatten().astype(np.float64)

    mask = yt > threshold
    num_samples = int(np.sum(mask))
    if num_samples == 0:
        return {
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "nRMSE": float("nan"),
            "nMAE": float("nan"),
            "IA": float("nan"),
            "Samples": 0
        }

    return compute_metrics(yt[mask], yp[mask])


def print_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 10.0,
                        show_points: bool = True) -> None:
    m_all = compute_metrics(y_true, y_pred)
    m_day = compute_daytime_metrics(y_true, y_pred, threshold=threshold)

    def _fmt(val, spec):
        if np.isnan(val):
            return f"{'N/A':>{spec.split('.')[0] if '.' in spec else spec}}"
        return f"{val:{spec}}"

    points_header = f"{'Forecast points':>15} | " if show_points else ""
    header = f"{'Scope':<23} | {points_header} {'nRMSE':>10} | {'nMAE':>10} | {'IA':>8} | {'RMSE (W/m²)':>12} | {'MAE (W/m²)':>11} |"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)
    for scope, m in [("All-time (24h)", m_all), (f"Daytime (GHI > {threshold:g} W/m²)", m_day)]:
        points = f"{m['Samples']:>15,d} | " if show_points else ""
        print(f"{scope:<23} | {points} {_fmt(m['nRMSE'], '10.4f')} | {_fmt(m['nMAE'], '10.4f')} | {_fmt(m['IA'], '8.4f')} | {_fmt(m['RMSE'], '12.2f')} | {_fmt(m['MAE'], '11.2f')} |")
    print(divider)
