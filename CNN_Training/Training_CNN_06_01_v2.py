#!/usr/bin/env python3

"""
Tunning_Model_06

CNN architecture:
Input (ROWS x COLS x 1)
→ Conv2D + ReLU + MaxPool
→ Conv2D + ReLU
→ Conv2D + ReLU + MaxPool
→ Conv2D + ReLU + MaxPool
→ GlobalAveragePooling2D
→ Dense + BatchNorm + Dropout
→ Dense
→ Linear output (N_OUTPUTS)

Optimizer: Adam (lr=1e-4)
Loss: L1-sum regression loss
"""


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback


import wandb
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# wandb.login()


# -----------------------------
# Config
# -----------------------------

ROWS = 18
COLS = 12
N_OUTPUTS = 28
FORCE_DECIMALS = 2

# Folder where THIS python file lives
SCRIPT_DIR = Path(__file__).resolve().parent

# Matched datasets live here
MATCHED_DIR = SCRIPT_DIR / "Dataset"

REAL_FILE = MATCHED_DIR / "Dataset_training_real.csv"
SIM_FILE = MATCHED_DIR / "Dataset_training_simulation.csv"

General_NAME= "Training_CNN_06_01_09_v2"
Keras_NAME= "Training_CNN_06_01_09_v2/best.keras"

# -----------------------------
# GPU memory growth (optional, safe)
# -----------------------------
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Folderpath = '/home/berith/.cache/wandb/'
        size_path = 0
        for path, dirs, files in os.walk('/home/berith01/.cache/wandb/'):
            for f in files:
                fp = os.path.join(path, f)
                size_path += os.path.getsize(fp)
        if size_path > 800000000:
            os.system('wandb artifact cache cleanup 0GB')
            print('---------------------------------------------------------------------------------------Vamos a borrar')

def cleanPath():
    search_dir = "/home/berith01/.local/share/wandb/artifacts/staging"
    if os.path.exists(search_dir):
        files = filter(os.path.isfile, os.listdir(search_dir))
        files = [os.path.join(search_dir, f) for f in files] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x))
        if len(files)>50:
            print('--------------------------------------------------------------------------------------- We are gona remove')
            files = files[0:-50]
            for i in files:
                os.remove(i)

def custom_loss_function(y_true, y_pred):
    # L1 sum over outputs (your style)
    return tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)

def load_and_align_data(real_file: Path, sim_file: Path):
    if not real_file.exists():
        raise FileNotFoundError(f"Missing file: {real_file}")
    if not sim_file.exists():
        raise FileNotFoundError(f"Missing file: {sim_file}")

    real_df = pd.read_csv(real_file, sep=None, engine="python")
    sim_df = pd.read_csv(sim_file, sep=None, engine="python")

    # ---- Identify columns ----
    # Real data columns:
    real_data_cols = [f"data_{i}" for i in range(N_OUTPUTS)]
    missing_real = [c for c in real_data_cols if c not in real_df.columns]
    if missing_real:
        raise ValueError(f"Matched_Real.csv missing columns: {missing_real}\nFound: {list(real_df.columns)}")

    # Simulation pixel columns:
    # Preferred: string digits "0".."11"
    pixel_cols = [str(i) for i in range(COLS)]
    if not all(c in sim_df.columns for c in pixel_cols):
        # fallback: int columns 0..11
        pixel_cols_int = list(range(COLS))
        if all(c in sim_df.columns for c in pixel_cols_int):
            pixel_cols = pixel_cols_int
        else:
            # last fallback: first 12 non-metadata columns
            meta = {"force", "path", "row_index", "test_location", "csv_name", "first_frame", "used_last_frame"}
            candidates = [c for c in sim_df.columns if c not in meta]
            if len(candidates) < COLS:
                raise ValueError(
                    f"Could not find {COLS} pixel columns in Matched_Simulation.csv.\n"
                    f"Columns: {list(sim_df.columns)}"
                )
            pixel_cols = candidates[:COLS]

    # Metadata columns for alignment
    # We expect sim has 'path' (because earlier script renames test_location to path)
    if "path" not in sim_df.columns and "test_location" in sim_df.columns:
        sim_df = sim_df.rename(columns={"test_location": "path"})
    if "path" not in sim_df.columns:
        raise ValueError(f"Matched_Simulation.csv must contain 'path' column. Found: {list(sim_df.columns)}")
    if "force" not in sim_df.columns:
        raise ValueError(f"Matched_Simulation.csv must contain 'force' column. Found: {list(sim_df.columns)}")

    if "path" not in real_df.columns:
        raise ValueError(f"Matched_Real.csv must contain 'path' column. Found: {list(real_df.columns)}")
    if "force" not in real_df.columns:
        raise ValueError(f"Matched_Real.csv must contain 'force' column. Found: {list(real_df.columns)}")

    # ---- Standardize key ----
    real_df = real_df.copy()
    sim_df = sim_df.copy()

    real_df["force_r"] = pd.to_numeric(real_df["force"], errors="coerce").round(FORCE_DECIMALS)
    sim_df["force_r"] = pd.to_numeric(sim_df["force"], errors="coerce").round(FORCE_DECIMALS)

    real_df["path_s"] = real_df["path"].astype(str).str.strip()
    sim_df["path_s"] = sim_df["path"].astype(str).str.strip()

    # ---- Build simulation samples grouped by (force_r, path_s) ----
    # Sort by row_index when available; otherwise keep original order
    if "row_index" in sim_df.columns:
        sim_df["row_index"] = pd.to_numeric(sim_df["row_index"], errors="coerce")
        sim_df = sim_df.sort_values(["force_r", "path_s", "row_index"])
    else:
        sim_df = sim_df.sort_values(["force_r", "path_s"])

    groups = sim_df.groupby(["force_r", "path_s"], sort=False)

    sim_map = {}
    for (f, p), g in groups:
        # Use first ROWS rows for that sample
        if len(g) < ROWS:
            continue
        grid = g.iloc[:ROWS][pixel_cols].to_numpy(dtype=np.float32)
        if grid.shape != (ROWS, COLS):
            continue
        sim_map[(float(f), str(p))] = grid

    # ---- Build aligned X, y in the order of Matched_Real rows ----
    X_list = []
    y_list = []
    used_keys = []
    missing = []

    for idx, r in real_df.iterrows():
        key = (float(r["force_r"]), str(r["path_s"]))
        grid = sim_map.get(key)
        if grid is None:
            missing.append(key)
            continue
        X_list.append(grid)
        y_list.append(r[real_data_cols].to_numpy(dtype=np.float32))
        used_keys.append(key)

    if not X_list:
        raise RuntimeError("No matched samples found when aligning Matched_Real with Matched_Simulation.")

    if missing:
        print(f"[WARN] {len(missing)} real samples had no sim match during training alignment.")
        print("       Example missing keys:", missing[:10])

    X = np.stack(X_list, axis=0)  # (N, ROWS, COLS)
    y = np.stack(y_list, axis=0)  # (N, 28)

    # Expand channel dim for Conv2D
    X = np.expand_dims(X, axis=-1)  # (N, ROWS, COLS, 1)

    return X, y

wandb.init(
    # set the wandb project where this run will be logged
    project="IsaaSim_Training",
    name = General_NAME,

    # track hyperparameters and run metadata with wandb.config
    config={
        "Conv2D_Init":64,
        "Conv2D_0":320,
        "Conv2D_1":512,
        "Conv2D_2":512,
        "Dense_0":480,
        "Dense_1":352,
        "learning_rate":0.0001,
        "DropoutRate_0": 0.1,
        "batch_size": 120,
        "epochs": 600,
        "Dataset": "Matched_Simulation/Matched_Real",
    }
)
config = wandb.config

path = str(Path(__file__).parent.absolute())
print(path)

# "  Conv2D_Initial: 64
#   Conv2D_0: 320
#   Conv2D_1: 512
#   Conv2D_2: 512
#   Dense_0: 480
#   DropoutRate_0: 0.1
#   Dense_1: 352
#   tuner/epochs: 100

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("[INFO] Loading + aligning Matched_Real and Matched_Simulation...")
X, y = load_and_align_data(REAL_FILE, SIM_FILE)
print("[INFO] X shape:", X.shape, " y shape:", y.shape)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Normalize inputs (by training max)
the_max = np.max(X_train)
if the_max == 0:
    raise RuntimeError("Training inputs max is 0; cannot normalize.")
X_train = (X_train / the_max).astype("float32")
X_test = (X_test / the_max).astype("float32")

#################################################################### Model
# Input (ROWS x COLS x 1)
# → Conv2D + ReLU + MaxPool
# → Conv2D + ReLU
# → Conv2D + ReLU + MaxPool
# → Conv2D + ReLU + MaxPool
# → GlobalAveragePooling2D
# → Dense + BatchNorm + Dropout
# → Dense
# → Linear output (N_OUTPUTS)


input_shape = (ROWS, COLS, 1)
input = keras.Input(shape=input_shape)
x = layers.Conv2D(config.Conv2D_Init, kernel_size=(2, 2), activation=None, padding="same", )(input)
x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
x = layers.Conv2D(config.Conv2D_0, kernel_size=(2, 2), activation="relu", padding="same", )(x)
x = layers.Conv2D(config.Conv2D_1, kernel_size=(2, 2), activation="relu", padding="same", )(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
x = layers.Conv2D(config.Conv2D_2, kernel_size=(2, 2), activation="relu", padding="same", )(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(units=config.Dense_0, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(config.DropoutRate_0)(x)
x = layers.Dense(units=config.Dense_1, activation='relu')(x)
Output = layers.Dense(units=28, activation='linear', name='output')(x)
model = tf.keras.Model(inputs=input, outputs=Output, name=General_NAME)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss=custom_loss_function,
    metrics=[custom_loss_function],  
)

# keras.utils.plot_model(model, path + "/Ret_Abaqus_V6_01.png", show_shapes=True)

#################################################################### Fit Model
os.makedirs(General_NAME, exist_ok=True)

callbacks = [
    WandbMetricsLogger(log_freq=5),
    WandbModelCheckpoint(
        filepath=Keras_NAME,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    ),
    ClearMemory(),
]

model.fit(
    X_train, y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_split=0.1,
    callbacks=callbacks,
)

# model.fit(X_train, y_train, batch_size=config.batch_size, epochs=config.epochs, validation_split=0.1, callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models_12.keras", save_best_only=True), ClearMemory()])
model.save(os.path.join(path, Keras_NAME))


#################################################################### Evaluate Model
score = model.evaluate(X_test, y_test, verbose=0)
print(model.metrics_names)
print("Test loss:", score[0])
print("Y1 loss:", score[1])


