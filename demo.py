"""Run pretrained TAXA-Net inference and evaluation."""

import os
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from config import (
    SEQ_LEN, PRED_LEN, STRIDE, BATCH_SIZE,
    ENDO_COLS, EXO_COLS, DATA_PATH, CKPT_PATH, PLOT_PATH,
    MODEL_CONFIG, DAYTIME_THRESHOLD
)
from models.taxa_net import TAXA_Net
from utils.dataset import build_demo_dataloader
from utils.metrics import print_metrics_table
from utils.visualization import plot_sample_forecast


def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_file = os.path.join(CURRENT_DIR, CKPT_PATH)
    data_file = os.path.join(CURRENT_DIR, DATA_PATH)
    plot_file = os.path.join(CURRENT_DIR, PLOT_PATH)

    model = TAXA_Net(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        endo_cols=ENDO_COLS,
        exo_cols=EXO_COLS,
        **MODEL_CONFIG
    ).to(device)
    model.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=True))
    model.eval()

    dataset, dataloader = build_demo_dataloader(
        csv_path=data_file,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        endo_cols=ENDO_COLS,
        exo_cols=EXO_COLS
    )

    preds_list, trues_list = [], []
    first_sample = None

    with torch.no_grad():
        for idx, (b_en, b_ex, b_time, b_y) in enumerate(dataloader):
            b_en, b_ex, b_time = b_en.to(device), b_ex.to(device), b_time.to(device)
            pred = model(b_en, b_ex, b_time)

            if idx == 0:
                first_sample = (b_en[0:1], b_y[0:1], pred[0:1])

            preds_list.append(pred.cpu().numpy())
            trues_list.append(b_y.numpy())

    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(trues_list, axis=0)

    sample_ts = dataset.get_timestamps(0)
    sample_en, sample_y, sample_pred = first_sample
    plot_sample_forecast(sample_en, sample_y, sample_pred, save_path=plot_file, timestamps=sample_ts)

    print(f"Device: {device}; windows: {len(dataset):,}")
    print_metrics_table(y_true, y_pred, threshold=DAYTIME_THRESHOLD)

    print(f"Plot saved to: {plot_file}")


if __name__ == "__main__":
    run_demo()
