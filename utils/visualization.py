"""GHI forecast visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_sample_forecast(sample_en, sample_y, sample_pred, save_path: str = None, timestamps=None):
    """Plot a tensor sample; save if a path is given."""
    hist_ghi = sample_en[0, :, 0].detach().cpu().numpy()
    true_ghi = sample_y[0].detach().cpu().numpy()
    pred_ghi = sample_pred[0].detach().cpu().numpy()
    l_len, h_len = len(hist_ghi), len(true_ghi)
    x = np.arange(l_len + h_len)
    if timestamps is not None:
        x = pd.to_datetime(timestamps[:l_len + h_len])

    fig, ax = plt.subplots()
    ax.plot(x[:l_len], hist_ghi, label="Historical GHI")
    ax.plot(x[l_len - 1:], np.r_[hist_ghi[-1], true_ghi], "k--", label="Observed GHI")
    ax.plot(x[l_len - 1:], np.r_[hist_ghi[-1], pred_ghi], "r-", label="TAXA-Net")
    ax.axvline(x[l_len - 1], linestyle=":")
    ax.set(title="TAXA-Net GHI forecast", ylabel="GHI (W/m²)",
           xlabel="Date" if timestamps is not None else "Time step (15 min)")
    ax.legend()
    if timestamps is not None:
        fig.autofmt_xdate()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
