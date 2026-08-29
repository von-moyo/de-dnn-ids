"""
================================================================================
 Differential Evolution Optimised Deep Neural Network for Intrusion Detection
================================================================================
 Author     : Florish Adekogbe (190805025)
 Project    : B.Sc. Computer Science, University of Lagos
 Supervisor : Dr. B. A. Sawyerr
 Dataset    : CSE-CIC-IDS2018  (https://www.unb.ca/cic/datasets/ids-2018.html)

 WHAT THIS FILE DOES
 -------------------
 This is the complete, runnable pipeline for the project. It:
   1. Loads the CSE-CIC-IDS2018 CSV files.
   2. Cleans and preprocesses the data (handles the inf/NaN values this
      dataset is known for, encodes labels, splits, scales, and optionally
      balances classes with SMOTE).
   3. Builds a configurable Keras DNN whose architecture and training
      hyperparameters are chosen by Differential Evolution.
   4. Runs a hand-written Differential Evolution optimiser (DE/rand/1/bin)
      that searches the hyperparameter space, using validation macro-F1 as
      the fitness signal.
   5. Retrains the best configuration found and evaluates it on the held-out
      test set (accuracy, precision, recall, F1, false-positive rate, ROC-AUC),
      then saves the confusion matrix, the DE convergence curve, the metrics
      as JSON, and the trained model.

 LEAKAGE DISCIPLINE
 ------------------
 The test set is touched exactly once, inside `evaluate()`. Specifically:
   - The MinMax scaler is fitted on the TRAINING split only and then applied
     to validation and test, so no test statistic reaches the model.
   - SMOTE is applied to the training split only, after the split.
   - Early stopping during the final retrain monitors a 10% slice carved out
     of train+val, never the test set.
   - Columns that leak identity (IPs, ports, timestamps, flow IDs) are dropped
     before anything else happens.

 HOW TO RUN ON THE REAL DATA
 ---------------------------
   - Put the 10 daily capture files in ./data/ (see data/README.md)
   - pip install -r requirements.txt
   - python de_dnn_ids.py --data_dir ./data --mode multiclass --max_per_class 20000
   A GPU is strongly recommended; on Google Colab set Runtime -> GPU. Native
   Windows cannot use a GPU at all on TensorFlow >= 2.11.

 NOTE ON SCALE
 -------------
 The raw CIC release is ~16 million flows; the deduplicated parquet build is
 ~6.7 million. Either way the default search budget of
 pop_size * (1 + generations) = 1,020 full network trainings is out of reach on
 CPU. Search on a class-capped subset, then retrain the winner larger:

   # stage 1 - search (hours)
   python de_dnn_ids.py --out_dir results/stage1 --max_per_class 20000 \
       --min_class_rows 200 --pop_size 10 --generations 10 --fitness_epochs 8
   # stage 2 - final model, reusing the winning config
   python de_dnn_ids.py --out_dir results/final --max_per_class 200000 \
       --min_class_rows 200 --load_config results/stage1/best_config.json

 --load_config skips only the SEARCH. The data flags are not replayed from the
 saved config, so stage 2 must repeat --max_per_class and --min_class_rows or
 it will silently evaluate a different class set than the one DE tuned for.

 Prefer --max_per_class over --sample_frac: the latter thins every class
 equally and starves the rare attack families that macro-F1 depends on.
================================================================================
"""

import os
import json
import random
import argparse
import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# TensorFlow is imported lazily inside build_dnn/fitness so the DE optimiser
# and decoder can be imported without paying the TF startup cost.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             classification_report)

# SMOTE is optional; the pipeline still runs without imbalanced-learn installed.
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False


def set_seeds(seed=42):
    """Seed every RNG in the stack so a run can be reproduced.

    Note: exact bit-for-bit reproducibility on GPU additionally requires
    TF_DETERMINISTIC_OPS=1, which is left off here because it disables the
    fast cuDNN kernels and roughly doubles the cost of the DE search.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------
def _read_one(path):
    """Single read attempt, dispatching on extension."""
    low = path.lower()
    if low.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if low.endswith(".parquet"):
        return pd.read_parquet(path)
    if low.endswith(".zip"):
        import zipfile, io
        with zipfile.ZipFile(path) as z:
            inner = [n for n in z.namelist()
                     if n.lower().endswith((".parquet", ".csv"))]
            if not inner:
                raise ValueError(f"{path} contains no .parquet or .csv member")
            buf = io.BytesIO(z.read(inner[0]))
        return (pd.read_parquet(buf) if inner[0].lower().endswith(".parquet")
                else pd.read_csv(buf, low_memory=False))
    raise ValueError(f"unsupported file type: {path}")


def read_capture_file(path, retries=4, backoff=3.0):
    """Read one capture file. Supports .csv, .parquet, and zipped parquet.

    The raw CIC release ships CSVs; the widely used cleaned redistributions
    (e.g. dhoogla/csecicids2018 on Kaggle) ship parquet, sometimes zipped.

    Reads are retried because on Windows an on-access antivirus scan can hold a
    transient lock on a file this large and surface it as PermissionError. That
    is worth surviving rather than crashing: every invocation re-reads all ten
    captures, so an unretried lock can kill a DE search hours after it started.
    """
    import time
    for attempt in range(retries):
        try:
            return _read_one(path)
        except PermissionError:
            if attempt == retries - 1:
                raise
            wait = backoff * (attempt + 1)
            print(f"[warn] {os.path.basename(path)} locked "
                  f"(attempt {attempt + 1}/{retries}); retrying in {wait:.0f}s")
            time.sleep(wait)


def class_key(df, label_col, mode):
    """The label the MODEL actually learns, as a Series aligned with `df`.

    Binary mode collapses every attack family into one Attack class, so any
    per-class operation -- capping, rare-class removal -- has to group on the
    collapsed target rather than the raw family. Grouping on the family in
    binary mode would give Benign a single share of the budget while each
    attack family took a share of its own, producing an attack-heavy training
    set: the inverse of this dataset's ~80% benign prior.
    """
    lab = df[label_col].astype(str).str.strip()
    if mode == "binary":
        return ~lab.str.lower().isin(["benign", "normal"])
    return lab


def cap_by_class(df, label_col, mode, max_per_class, random_state):
    """Subsample so no class exceeds `max_per_class`, keeping rarer ones whole.

    Iterates the groups rather than using .apply: pandas 2.x drops the grouping
    column from the frames handed to .apply.
    """
    parts = [g.sample(n=min(len(g), max_per_class), random_state=random_state)
             for _, g in df.groupby(class_key(df, label_col, mode),
                                    observed=True)]
    return pd.concat(parts)


def load_and_preprocess(data_dir, mode="multiclass", sample_frac=1.0,
                        use_smote=False, random_state=42, max_per_class=None,
                        min_class_rows=0):
    """Load CSE-CIC-IDS2018 captures and return train/val/test splits.

    Accepts the raw CIC CSVs (80 columns) and the cleaned parquet
    redistributions (78 columns) alike. Every capture file in `data_dir` is
    concatenated, columns that leak identity or carry no signal are dropped,
    the infinite / missing values the raw release is notorious for are
    repaired, the Label column is encoded, then a 60/20/20 stratified split is
    scaled using training statistics only.
    """
    found = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
             if f.lower().endswith((".csv", ".parquet", ".zip"))]
    # If an archive has already been extracted alongside itself, read the
    # extracted copy and skip the archive rather than loading both.
    plain = {f for f in found if not f.lower().endswith(".zip")}
    data_files = [f for f in found
                  if not (f.lower().endswith(".zip") and f[:-4] in plain)]
    if not data_files:
        raise FileNotFoundError(
            f"No .csv, .parquet or .zip files found in {data_dir}")

    frames = []
    for f in data_files:
        df = read_capture_file(f)
        # Normalise column names here rather than after the concat: some raw
        # CSVs ship columns with a leading space (" Flow Duration"), which
        # would otherwise concat into a second, near-duplicate column.
        df.columns = [c.strip() for c in df.columns]
        if sample_frac < 1.0:
            # Sample per file so no capture day (and therefore no attack
            # family) can be dropped entirely by the subsample.
            df = df.sample(frac=sample_frac, random_state=random_state)
        if max_per_class:
            # Pre-cap per file. The authoritative cap is applied once after the
            # concat; this pass exists purely to bound peak memory, because a
            # class may appear across several capture days and each day is
            # allowed to contribute up to the full cap. Without it the entire
            # 6.7M-row corpus has to be resident before any thinning happens.
            lc = "Label" if "Label" in df.columns else df.columns[-1]
            df = cap_by_class(df, lc, mode, max_per_class, random_state)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Drop columns that either leak the answer or carry no signal. Src/Dst IP
    # and Timestamp are the dangerous ones: the attacks were launched from a
    # fixed set of hosts at known times, so a model left with them learns
    # "traffic from 18.219.211.138 is bad" rather than what an attack is.
    #
    # Dst Port is deliberately NOT in this list -- it is genuine protocol
    # signal (SSH on 22, FTP on 21) and is kept whenever it is present. Note
    # that the cleaned parquet redistributions drop both Dst Port and Timestamp
    # upstream, so on those builds this is a no-op and the model trains without
    # port information at all. Check the printed feature count: 77 features
    # before constant-column removal means Dst Port is absent.
    drop_cols = [c for c in ["Timestamp", "Flow ID", "Src IP", "Dst IP",
                             "Source IP", "Destination IP", "Src Port",
                             "Source Port"] if c in data.columns]
    data = data.drop(columns=drop_cols, errors="ignore")

    # The Label column is sometimes named "Label" and sometimes "label".
    label_col = "Label" if "Label" in data.columns else data.columns[-1]

    # Repair inf / NaN: CIC flow features such as "Flow Bytes/s" divide by a
    # duration that can be zero, and some CSVs embed a repeated header row.
    feature_cols = [c for c in data.columns if c != label_col]
    # Only coerce columns that are not already numeric. Cleaned parquet builds
    # arrive correctly typed (int8/int16/float32); blanket to_numeric would
    # upcast every column to float64 and roughly triple peak memory.
    non_numeric = [c for c in feature_cols
                   if not pd.api.types.is_numeric_dtype(data[c])]
    if non_numeric:
        print(f"[data] coercing {len(non_numeric)} non-numeric feature columns")
        data[non_numeric] = data[non_numeric].apply(pd.to_numeric,
                                                    errors="coerce")
    float_cols = [c for c in feature_cols
                  if pd.api.types.is_float_dtype(data[c])]
    if float_cols:
        data[float_cols] = data[float_cols].replace([np.inf, -np.inf], np.nan)
    before = len(data)
    data = data.dropna()
    print(f"[data] dropped {before - len(data):,} rows containing inf/NaN "
          f"({100 * (before - len(data)) / max(before, 1):.3f}%)")

    # Remove zero-variance columns (they help nothing and slow training).
    nunique = data[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"[data] dropped {len(constant_cols)} constant columns: "
              f"{constant_cols}")
    data = data.drop(columns=constant_cols, errors="ignore")
    feature_cols = [c for c in data.columns if c != label_col]

    # Remove classes too small to be scored at all. Deduplicated builds of this
    # dataset collapse several attack families to a few dozen distinct flows
    # (FTP-BruteForce 53, DoS-SlowHTTPTest 55, SQL Injection 85), which at a
    # 20% test split leaves ~10 test rows each. A per-class F1 computed on 10
    # samples moves by 0.1 for every single misclassification, and because
    # fitness() optimises the MACRO average those classes inject pure noise
    # into the signal DE is searching on -- they do not merely make the report
    # unreliable, they corrupt the search. Dropping them is explicit and
    # opt-in, never silent: the counts are printed and recorded in metrics.json.
    if min_class_rows:
        key = class_key(data, label_col, mode)
        counts = key.value_counts()
        rare = counts[counts < min_class_rows]
        if len(rare):
            print(f"[data] dropping {len(rare)} class(es) below "
                  f"{min_class_rows:,} rows: "
                  f"{ {str(k): int(v) for k, v in rare.items()} }")
            data = data[~key.isin(rare.index)]
        del key

    # Class-aware cap. Unlike --sample_frac, which thins every class equally
    # and wipes out the tiny web-attack families, this keeps every row of a
    # rare class and only subsamples classes above the cap. That is what makes
    # macro-F1 meaningful while keeping the DE search affordable.
    if max_per_class:
        before = len(data)
        data = (cap_by_class(data, label_col, mode, max_per_class,
                             random_state)
                .sample(frac=1.0, random_state=random_state))
        print(f"[data] capped classes at {max_per_class:,}/class: "
              f"{before:,} -> {len(data):,} rows")

    # Binary vs multi-class target.
    y_raw = data[label_col].astype(str).str.strip()
    if mode == "binary":
        is_attack = ~y_raw.str.lower().isin(["benign", "normal"])
        # Encoded explicitly rather than via LabelEncoder so that the positive
        # class (label 1, the one the sigmoid scores) is Attack. Alphabetical
        # encoding would put Attack at 0 and silently invert every reported
        # per-class precision/recall and the ROC-AUC.
        y = is_attack.astype(int).values
        class_names = np.array(["Benign", "Attack"])
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = label_encoder.classes_

    X = data[feature_cols].values.astype(np.float32)

    # 60 / 20 / 20 stratified split. Stratification matters here because the
    # rarest classes (e.g. Infiltration) are well under 1% of the data and a
    # plain random split can leave a test set with none of them.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=random_state)

    # Scale AFTER splitting, fitting on train only. Min-max is used rather than
    # standardisation because most CIC features are zero-heavy and min-max
    # preserves those exact zeros. Values in val/test may fall slightly outside
    # [0, 1] when they exceed the training range; that is expected and is itself
    # informative (a flow longer than anything seen in training is anomalous).
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Optional class balancing on the training split only. Applying SMOTE
    # before the split would put points synthesised from test samples into
    # training, which is leakage.
    if use_smote:
        if not _HAS_SMOTE:
            print("[warn] --use_smote requested but imbalanced-learn is not "
                  "installed; continuing without it.")
        else:
            try:
                X_train, y_train = SMOTE(random_state=random_state).fit_resample(
                    X_train, y_train)
                print(f"[data] SMOTE resampled train to {len(y_train):,} rows")
            except Exception as e:
                print(f"[warn] SMOTE skipped: {e}")

    n_classes = len(class_names)
    print(f"[data] features={X_train.shape[1]}  classes={n_classes}  "
          f"train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}")

    # Per-class test counts. A class with a handful of test flows cannot be
    # scored meaningfully, and one with zero breaks ROC-AUC entirely -- this
    # is the signal that --sample_frac is too aggressive for the rare attacks.
    print("[data] test-set support by class:")
    for idx, name in enumerate(class_names):
        support = int((y_test == idx).sum())
        flag = "  <-- TOO FEW" if support < 30 else ""
        print(f"        {str(name):<24} {support:>8,}{flag}")
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            n_classes, class_names)


# -----------------------------------------------------------------------------
# 2. CONFIGURABLE DNN
# -----------------------------------------------------------------------------
ACTIVATIONS = ["relu", "tanh", "sigmoid"]
BATCH_SIZES = [16, 32, 64, 128, 256]


def decode(vector):
    """Decode a real-valued DE vector into concrete DNN hyperparameters.

    DE searches the continuous unit hypercube; a network needs discrete and
    categorical settings. This function is the bridge between the two.

    Vector layout (all components live in [0, 1] inside DE):
        0: number of hidden layers   -> 1..5
        1: neurons per layer         -> 32..512
        2: learning rate (log scale) -> 1e-4..1e-1
        3: dropout rate              -> 0.0..0.5
        4: batch size                -> {16,32,64,128,256}
        5: activation function       -> {relu, tanh, sigmoid}

    The learning rate is mapped logarithmically so that each order of
    magnitude receives an equal share of the slider; a linear map would spend
    99.9% of its range above 1e-3, where Adam rarely converges well.
    """
    n_layers = int(round(1 + vector[0] * 4))
    neurons = int(round(32 + vector[1] * (512 - 32)))
    lr = float(10 ** (-4 + vector[2] * 3))           # 1e-4 .. 1e-1
    dropout = float(vector[3] * 0.5)
    batch = int(BATCH_SIZES[min(len(BATCH_SIZES) - 1, int(vector[4] * len(BATCH_SIZES)))])
    act = ACTIVATIONS[min(len(ACTIVATIONS) - 1, int(vector[5] * len(ACTIVATIONS)))]
    return dict(n_layers=n_layers, neurons=neurons, lr=lr,
                dropout=dropout, batch=batch, activation=act)


def build_dnn(input_dim, n_classes, hp):
    """Build a Keras DNN from a decoded hyperparameter dict."""
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    for _ in range(hp["n_layers"]):
        model.add(layers.Dense(hp["neurons"], activation=hp["activation"],
                               kernel_initializer="glorot_uniform"))  # Xavier
        if hp["dropout"] > 0:
            model.add(layers.Dropout(hp["dropout"]))

    if n_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        # sparse_ avoids materialising a one-hot label matrix, which on the
        # full dataset would be ~10M x 15 floats of pure overhead.
        model.add(layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(optimizer=keras.optimizers.Adam(hp["lr"]),
                  loss=loss, metrics=["accuracy"])
    return model


def fitness(vector, data, n_classes, epochs=20, verbose=0):
    """Train a candidate DNN and return validation macro-F1 (to maximise).

    Macro-F1 weights every class equally, so a rare attack family such as
    Infiltration counts as much as DDoS. Optimising accuracy instead would
    reward a model that predicts "Benign" for everything, since ~83% of the
    dataset is benign.

    A candidate that fails to train (typically an out-of-memory configuration)
    scores 0.0 rather than aborting the search, so a multi-hour run is not lost
    to one bad individual.
    """
    from tensorflow import keras
    X_train, y_train, X_val, y_val = data
    hp = decode(vector)
    try:
        keras.backend.clear_session()
        model = build_dnn(X_train.shape[1], n_classes, hp)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                           restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=hp["batch"], verbose=verbose,
                  callbacks=[es])

        if n_classes == 2:
            y_pred = (model.predict(X_val, verbose=0).ravel() >= 0.5).astype(int)
        else:
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        return float(f1_score(y_val, y_pred, average="macro"))
    except Exception as e:
        print(f"[warn] candidate {hp} failed ({type(e).__name__}: {e}); "
              f"scoring 0.0")
        return 0.0


# -----------------------------------------------------------------------------
# 3. DIFFERENTIAL EVOLUTION (DE/rand/1/bin) - written from scratch
# -----------------------------------------------------------------------------
def _save_de_state(path, pop, scores, history, generation, rng):
    """Persist the DE population after a completed generation.

    The RNG's bit-generator state is saved alongside the population. Without
    it a resumed run would reseed and replay the same mutation and crossover
    draws it already made, which is not the search continuing -- it is a
    different, correlated one.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"pop": pop.tolist(), "scores": scores.tolist(),
                   "history": history, "generation": generation,
                   "rng": rng.bit_generator.state}, fh)
    os.replace(tmp, path)          # atomic: a kill mid-write cannot corrupt it


def differential_evolution(fitness_fn, dim=6, pop_size=20, generations=50,
                           F=0.8, CR=0.9, seed=42, checkpoint=None):
    """Classic DE/rand/1/bin maximising `fitness_fn` over the unit hypercube.

    The search is derivative-free, which it has to be: validation macro-F1 is
    not differentiable with respect to "number of layers". DE's step size is
    self-adapting because the mutation vector is drawn from the population's
    own spread -- large while the population is scattered, small once it has
    converged on a region.

    `checkpoint`, if given, is a path written after every generation and read
    back on startup to resume an interrupted search. This exists for hosted
    runtimes such as Colab, which disconnect on idle and cap session length:
    without it a drop at generation 7 of 10 discards every training run made
    so far. Delete the file to force a fresh search.

    Returns the best vector, its fitness, and the per-generation best-fitness
    history (for the convergence plot).
    """
    # DE/rand/1 draws three donors distinct from the target, so a population
    # below 4 cannot form a mutation vector at all.
    if pop_size < 4:
        raise ValueError(
            f"pop_size must be >= 4 for DE/rand/1 (got {pop_size}); the "
            f"mutation needs three donors distinct from the target. "
            f"Guidance is roughly 10x the dimensionality, i.e. ~60 here, "
            f"with 20 a reasonable compromise.")

    rng = np.random.default_rng(seed)
    start_gen = 1

    resumed = False
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as fh:
            state = json.load(fh)
        saved = np.array(state["pop"])
        if saved.shape != (pop_size, dim):
            # Refuse to graft a population of one shape onto a run configured
            # for another; silently reshaping would invalidate the whole search.
            raise ValueError(
                f"{checkpoint} holds a {saved.shape} population but this run "
                f"is configured for {(pop_size, dim)}. Delete the checkpoint "
                f"to start fresh, or restore the original --pop_size.")
        pop, scores = saved, np.array(state["scores"])
        history, start_gen = state["history"], state["generation"] + 1
        rng.bit_generator.state = state["rng"]
        resumed = True
        print(f"[DE] resumed from {checkpoint} at generation {start_gen}, "
              f"best so far {scores.max():.4f}")

    if not resumed:
        pop = rng.random((pop_size, dim))
        scores = np.array([fitness_fn(ind) for ind in pop])
        history = [float(scores.max())]
        print(f"[DE] gen 0  best macro-F1 = {scores.max():.4f}")
        if checkpoint:
            # Checkpoint gen 0 too: evaluating the initial population is
            # pop_size full trainings and is just as expensive to lose.
            _save_de_state(checkpoint, pop, scores, history, 0, rng)

    for g in range(start_gen, generations + 1):
        for i in range(pop_size):
            # --- mutation: v = x_r1 + F*(x_r2 - x_r3) ---
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            donor = pop[r1] + F * (pop[r2] - pop[r3])
            donor = np.clip(donor, 0.0, 1.0)

            # --- binomial crossover ---
            # j_rand forces at least one gene from the donor, so the trial can
            # never be an exact clone of its parent (a wasted evaluation).
            trial = pop[i].copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = donor[j]

            # --- greedy selection ---
            # The child replaces only its own parent, and only on >=, so the
            # incumbent best can never be lost and the curve is monotonic.
            trial_score = fitness_fn(trial)
            if trial_score >= scores[i]:
                pop[i] = trial
                scores[i] = trial_score

        history.append(float(scores.max()))
        print(f"[DE] gen {g}  best macro-F1 = {scores.max():.4f}", flush=True)
        if checkpoint:
            _save_de_state(checkpoint, pop, scores, history, g, rng)

    best = int(np.argmax(scores))
    return pop[best], float(scores[best]), history


def plot_training_loss(history, out_dir="results"):
    """Save the final model's training/validation loss curve.

    The gap between the two curves is the overfitting diagnostic, and the epoch
    at which validation loss bottoms out is the one early stopping restores --
    both are worth having on paper rather than only in console scrollback.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss, val = history.get("loss", []), history.get("val_loss", [])
    if not loss:
        return
    ep = range(1, len(loss) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, loss, marker="o", markersize=3, linewidth=1.6,
             color="#1f4e79", label="Training loss")
    if val:
        plt.plot(ep, val, marker="s", markersize=3, linewidth=1.6,
                 color="#c00000", label="Validation loss")
        best = int(np.argmin(val)) + 1
        plt.axvline(best, linestyle="--", linewidth=1.1, color="grey")
        plt.annotate(f"best validation loss (epoch {best}, {min(val):.4f})",
                     xy=(best, min(val)), xytext=(best + 2, max(loss) * 0.8),
                     fontsize=8, arrowprops=dict(arrowstyle="->", linewidth=0.8,
                                                 color="grey"))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss of the Final Model")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_loss.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[out] training loss curve -> {path}")


def plot_convergence(history, out_dir="results"):
    """Save the DE convergence curve (best macro-F1 per generation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(range(len(history)), history, marker="o", markersize=3,
             linewidth=1.6, color="#1f4e79")
    plt.xlabel("Generation")
    plt.ylabel("Best validation macro-F1")
    plt.title("Differential Evolution Convergence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "de_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[out] convergence curve -> {path}")


# -----------------------------------------------------------------------------
# 4. FINAL EVALUATION
# -----------------------------------------------------------------------------
def macro_ovr_auc(y_true, proba, n_classes):
    """Macro one-vs-rest ROC-AUC that tolerates classes missing from y_true.

    scikit-learn refuses to compute OVR AUC when a class has no positive
    samples, which happens on subsamples of this dataset because the
    web-attack classes are tiny. Rather than returning NaN for the whole run,
    restrict the score to the classes actually present and renormalise their
    probabilities so they still form a distribution.
    """
    present = np.unique(y_true)
    try:
        if len(present) == n_classes:
            return float(roc_auc_score(y_true, proba, multi_class="ovr",
                                       average="macro",
                                       labels=np.arange(n_classes)))
        missing = sorted(set(range(n_classes)) - set(present.tolist()))
        print(f"[warn] classes {missing} have no test samples; ROC-AUC is "
              f"computed over the {len(present)} classes that do")
        sub = proba[:, present]
        sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
        if len(present) == 2:
            return float(roc_auc_score((y_true == present[1]).astype(int),
                                       sub[:, 1]))
        return float(roc_auc_score(y_true, sub, multi_class="ovr",
                                   average="macro", labels=present))
    except Exception as e:
        print(f"[warn] ROC-AUC unavailable: {e}")
        return float("nan")


def evaluate(model, X_test, y_test, n_classes, class_names, out_dir="results",
             sampling=None):
    """Score the final model on the held-out test set and save artefacts.

    `sampling`, when given, is recorded verbatim in metrics.json so the file
    documents the distribution its numbers were measured on. That matters
    because --max_per_class rebalances the test set: macro precision/recall/F1
    weight classes equally and are unaffected, but accuracy and FPR are
    computed over whatever mix the cap produced, not real traffic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = np.arange(n_classes)
    if n_classes == 2:
        proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_test, proba))
    else:
        proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        auc = macro_ovr_auc(y_test, proba, n_classes)

    # `labels=` pins the matrix to n_classes x n_classes. Without it, a rare
    # class absent from BOTH y_test and y_pred silently shrinks the matrix,
    # which then mismatches class_names and raises in the heatmap and in
    # classification_report. On this dataset that is not hypothetical: on the
    # deduplicated build SQL Injection has 85 flows in the entire corpus,
    # DoS-SlowHTTPTest 55 and FTP-BruteForce 53.
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # Macro false-positive rate, derived one-vs-rest from the matrix. This is
    # the operationally decisive metric for an IDS: at enterprise traffic
    # volumes even a 1% FPR buries analysts in false alarms.
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fpr = float(np.mean(fp / np.clip(fp + tn, 1, None)))

    metrics = dict(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        fpr=fpr, auc=auc)

    # Operational false-alarm rate: of all genuinely benign flows, the fraction
    # the model flagged as SOME attack. This is the number the alert-fatigue
    # argument is actually about, and `fpr` above is not it in multiclass mode:
    # that one averages one-vs-rest FPR over every class, so a dozen easy
    # attack classes with near-zero false positives dilute it to ~0.007 even
    # when the model is misclassifying most benign traffic. Benign and
    # Infiltration are near-indistinguishable in CSE-CIC-IDS2018, so that gap
    # is wide here and reporting only the macro figure would hide it.
    benign_idx = [i for i, c in enumerate(class_names)
                  if str(c).strip().lower() in ("benign", "normal")]
    if benign_idx:
        b = benign_idx[0]
        n_benign = int((y_test == b).sum())
        if n_benign:
            metrics["benign_false_alarm_rate"] = float(
                int(((y_test == b) & (y_pred != b)).sum()) / n_benign)

    target_names = [str(c) for c in class_names]
    report = classification_report(y_test, y_pred, labels=labels,
                                   target_names=target_names, zero_division=0)

    print("\n==== TEST RESULTS ====")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print(report)

    payload = dict(metrics)
    if sampling:
        payload["_sampling"] = sampling
        print(f"\n[note] {sampling['caveat']}")
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as fh:
        fh.write(report)
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm,
               fmt="%d", delimiter=",")

    # Confusion matrix heatmap.
    plt.figure(figsize=(max(7, n_classes), max(6, n_classes * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix - DE-DNN on CSE-CIC-IDS2018")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[out] confusion matrix -> {path}")
    return metrics


# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Differential Evolution optimised DNN for intrusion "
                    "detection on CSE-CIC-IDS2018.")
    ap.add_argument("--data_dir", default="./data",
                    help="directory containing the CSE-CIC-IDS2018 CSVs")
    ap.add_argument("--out_dir", default="./results",
                    help="where figures, metrics and the model are written")
    ap.add_argument("--mode", choices=["binary", "multiclass"],
                    default="multiclass")
    ap.add_argument("--sample_frac", type=float, default=1.0,
                    help="per-file random subsample, e.g. 0.05 for 5%%. Thins "
                         "every class equally, so it starves rare attacks -- "
                         "prefer --max_per_class")
    ap.add_argument("--max_per_class", type=int, default=None,
                    help="cap each class at N rows, keeping ALL rows of any "
                         "class smaller than N. The right knob for this "
                         "dataset, e.g. --max_per_class 50000. 'Class' follows "
                         "--mode: attack family in multiclass, Benign/Attack "
                         "in binary (so binary caps to at most 1:1, never "
                         "attack-heavy)")
    ap.add_argument("--min_class_rows", type=int, default=0,
                    help="drop any class with fewer than N rows in the whole "
                         "corpus. Deduplicated builds leave FTP-BruteForce, "
                         "DoS-SlowHTTPTest and SQL Injection at ~50-85 rows, "
                         "too few to score and pure noise in the macro-F1 that "
                         "DE optimises. 200 removes exactly those three. "
                         "Off by default -- dropping data is never silent")
    ap.add_argument("--use_smote", action="store_true",
                    help="oversample minority classes in the training split")
    ap.add_argument("--pop_size", type=int, default=20)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--fitness_epochs", type=int, default=20,
                    help="epoch budget per DE candidate")
    ap.add_argument("--final_epochs", type=int, default=100,
                    help="epoch budget for the retrained winner")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load_config", default=None,
                    help="skip the DE search and retrain the configuration in "
                         "this JSON file (as written to best_config.json)")
    ap.add_argument("--checkpoint", default=None,
                    help="path to a DE checkpoint. Written after every "
                         "generation and resumed from if it already exists. "
                         "Use it on Colab and other hosted runtimes, which "
                         "disconnect on idle -- without it a drop late in the "
                         "search discards every training run made so far. "
                         "Delete the file to force a fresh search")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seeds(args.seed)

    (X_train, y_train, X_val, y_val, X_test, y_test,
     n_classes, class_names) = load_and_preprocess(
        args.data_dir, args.mode, args.sample_frac, args.use_smote, args.seed,
        max_per_class=args.max_per_class, min_class_rows=args.min_class_rows)

    from tensorflow import keras

    if args.load_config:
        with open(args.load_config) as fh:
            saved = json.load(fh)
        best_hp = saved["hyperparameters"]
        history = saved.get("convergence_history", [])
        print(f"[DE] skipped; loaded configuration from {args.load_config}")
    else:
        de_data = (X_train, y_train, X_val, y_val)

        def fit(v):
            return fitness(v, de_data, n_classes, epochs=args.fitness_epochs)

        best_vec, best_f1, history = differential_evolution(
            fit, pop_size=args.pop_size, generations=args.generations,
            seed=args.seed, checkpoint=args.checkpoint)

        best_hp = decode(best_vec)
        print(f"\n[DE] best validation macro-F1 = {best_f1:.4f}")
        print(f"[DE] best hyperparameters = {best_hp}")

        with open(os.path.join(args.out_dir, "best_config.json"), "w") as fh:
            json.dump({"hyperparameters": best_hp,
                       "vector": best_vec.tolist(),
                       "validation_macro_f1": best_f1,
                       "convergence_history": history,
                       "settings": vars(args)}, fh, indent=2)
        plot_convergence(history, args.out_dir)

    # Retrain the winner from scratch on train+val for the full budget. A 10%
    # stratified slice of that pool drives early stopping, so the test set
    # stays untouched until evaluate() -- using it here would mean selecting
    # weights on the same data the results are reported from.
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_full, y_full, test_size=0.10, stratify=y_full,
        random_state=args.seed)

    keras.backend.clear_session()
    final_model = build_dnn(X_train.shape[1], n_classes, best_hp)
    final_model.summary()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)
    hist = final_model.fit(X_fit, y_fit,
                           validation_data=(X_es, y_es),
                           epochs=args.final_epochs,
                           batch_size=best_hp["batch"],
                           verbose=2, callbacks=[es])

    # Persist and plot the learning curve. Without this the only record of how
    # the final model trained is console scrollback, which cannot be put in a
    # report or checked afterwards.
    with open(os.path.join(args.out_dir, "training_history.json"), "w") as fh:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()},
                  fh, indent=2)
    plot_training_loss(hist.history, args.out_dir)

    sampling = None
    if args.max_per_class or args.min_class_rows:
        sampling = dict(
            max_per_class=args.max_per_class,
            min_class_rows=args.min_class_rows or None,
            classes_evaluated=[str(c) for c in class_names],
            caveat="Test set was class-balanced by --max_per_class, so it does "
                   "not carry the ~80% benign prior of real traffic. Macro "
                   "precision/recall/F1 weight classes equally and are "
                   "unaffected; accuracy and fpr are measured on the "
                   "rebalanced mix and are NOT operational estimates.")

    evaluate(final_model, X_test, y_test, n_classes, class_names, args.out_dir,
             sampling=sampling)

    model_path = os.path.join(args.out_dir, "de_dnn_best.keras")
    final_model.save(model_path)
    print(f"[out] model -> {model_path}")


if __name__ == "__main__":
    main()
