"""Configuration for the pretrained TAXA-Net demo."""

SEQ_LEN = 480       # 5 days at 15-minute intervals
PRED_LEN = 96       # 24 hours at 15-minute intervals
STRIDE = 1
BATCH_SIZE = 32

ENDO_COLS = ["Global_Horizontal_Radiation"]
EXO_COLS = [
    "Wind_Speed",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Weather_Daily_Rainfall"
]

DATA_PATH = "data/sample_as_15min.csv"
CKPT_PATH = "checkpoints/taxa_as_h96.pth"
PLOT_PATH = "demo_forecast.png"

# Pretrained AS configuration (H=96)
MODEL_CONFIG = {
    "d_model": 64,
    "d_ff": 256,
    "e_layers": 3,
    "n_heads": 4,
    "scales": [16],
    "dropout": 0.2,
}

DAYTIME_THRESHOLD = 10.0  # Daytime evaluation threshold (W/m²)
