 """
================================================================================
 Differential Evolution Optimized Deep Neural Network for Intrusion Detection
================================================================================
 Author : Florish Adekogbe (190805025)
 Project: B.Sc. Computer Science, University of Lagos
 Supervisor: Dr. B. A. Sawyerr
 Dataset: CSE-CIC-IDS2018  (https://www.unb.ca/cic/datasets/ids-2018.html)

 WHAT THIS FILE DOES
 -------------------
 This is the complete, runnable pipeline for the project. It:
   1. Loads the CSE-CIC-IDS2018 CSV files.
   2. Cleans and preprocesses the data (handles the inf/NaN values this
      dataset is known for, encodes labels, scales features, splits, and
      optionally balances classes with SMOTE).
   3. Builds a configurable Keras DNN whose architecture and training
      hyperparameters are chosen by Differential Evolution.
   4. Runs a hand-written Differential Evolution optimiser (DE/rand/1/bin)
      that searches the hyperparameter space, using validation macro-F1 as
      the fitness signal.
   5. Retrains the best configuration found and evaluates it on the held-out
      test set (accuracy, precision, recall, F1, false-positive rate, ROC-AUC),
      then saves the confusion matrix and the DE convergence curve.

 HOW TO RUN ON THE REAL DATA
 ---------------------------
   - Download CSE-CIC-IDS2018 (the 10 daily CSVs) into ./data/
   - pip install tensorflow scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
   - python de_dnn_ids.py --data_dir ./data --mode multiclass
   A GPU is strongly recommended; on Google Colab set Runtime -> GPU.

 NOTE ON SCALE
 -------------
 The full CSE-CIC-IDS2018 dataset is ~16 million flows. For development you
 can pass --sample_frac 0.1 to work on a stratified 10% subset, then run the
 full thing for your final results.
================================================================================
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

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


# -----------------------------------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------
def load_and_preprocess(data_dir, mode="multiclass", sample_frac=1.0,
                        use_smote=False, random_state=42):
    """Load CSE-CIC-IDS2018 CSVs and return train/val/test splits.

    The CIC flow CSVs share the same 80-column schema. We concatenate every
    CSV in `data_dir`, drop the columns that leak identity or are constant,
    repair the infinite / missing values the dataset is notorious for, encode
    the Label column, scale features to [0, 1], and split 60/20/20.
    """
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                 if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Normalise column names (the CSVs ship with inconsistent spacing/case).
    data.columns = [c.strip() for c in data.columns]

    # Drop columns that either leak the answer or carry no signal.
    drop_cols = [c for c in ["Timestamp", "Flow ID", "Src IP", "Dst IP",
                             "Source IP", "Destination IP", "Src Port",
                             "Source Port"] if c in data.columns]
    data = data.drop(columns=drop_cols, errors="ignore")

    # The Label column is sometimes named "Label" and sometimes "label".
    label_col = "Label" if "Label" in data.columns else data.columns[-1]

    # Repair inf / NaN: CIC flow features overflow on very short flows.
    feature_cols = [c for c in data.columns if c != label_col]
    data[feature_cols] = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Remove zero-variance columns (they help nothing and slow training).
    nunique = data[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    data = data.drop(columns=constant_cols, errors="ignore")
    feature_cols = [c for c in data.columns if c != label_col]

    # Binary vs multi-class target.
    y_raw = data[label_col].astype(str).str.strip()
    if mode == "binary":
        y_raw = y_raw.apply(lambda v: "Benign" if v.lower() in
                            ("benign", "normal") else "Attack")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X = data[feature_cols].values.astype(np.float32)

    # Scale to [0, 1] — min-max keeps the sparse flow features well behaved.
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 60 / 20 / 20 stratified split.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=random_state)

    # Optional class balancing on the training split only.
    if use_smote and _HAS_SMOTE:
        try:
            X_train, y_train = SMOTE(random_state=random_state).fit_resample(
                X_train, y_train)
        except Exception as e:
            print(f"[warn] SMOTE skipped: {e}")

    n_classes = len(label_encoder.classes_)
    print(f"[data] features={X.shape[1]}  classes={n_classes}  "
          f"train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            n_classes, label_encoder.classes_)


# -----------------------------------------------------------------------------
# 2. CONFIGURABLE DNN
# -----------------------------------------------------------------------------
ACTIVATIONS = ["relu", "tanh", "sigmoid"]


def decode(vector):
    """Decode a real-valued DE vector into concrete DNN hyperparameters.

    Vector layout (all components live in [0, 1] inside DE):
        0: number of hidden layers   -> 1..5
        1: neurons per layer         -> 32..512
        2: learning rate (log scale) -> 1e-4..1e-1
        3: dropout rate              -> 0.0..0.5
        4: batch size                -> {16,32,64,128,256}
        5: activation function       -> {relu, tanh, sigmoid}
    """
    n_layers = int(round(1 + vector[0] * 4))
    neurons = int(round(32 + vector[1] * (512 - 32)))
    lr = float(10 ** (-4 + vector[2] * 3))           # 1e-4 .. 1e-1
    dropout = float(vector[3] * 0.5)
    batch = int([16, 32, 64, 128, 256][min(4, int(vector[4] * 5))])
    act = ACTIVATIONS[min(2, int(vector[5] * 3))]
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
        model.add(layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(optimizer=keras.optimizers.Adam(hp["lr"]),
                  loss=loss, metrics=["accuracy"])
    return model


def fitness(vector, data, n_classes, epochs=20, verbose=0):
    """Train a candidate DNN and return validation macro-F1 (to maximise)."""
    from tensorflow import keras
    X_train, y_train, X_val, y_val = data
    hp = decode(vector)
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
    return f1_score(y_val, y_pred, average="macro")


# -----------------------------------------------------------------------------
# 3. DIFFERENTIAL EVOLUTION (DE/rand/1/bin) — written from scratch
# -----------------------------------------------------------------------------
def differential_evolution(fitness_fn, dim=6, pop_size=20, generations=50,
                           F=0.8, CR=0.9, seed=42):
    """Classic DE/rand/1/bin maximising `fitness_fn` over the unit hypercube.

    Returns the best vector, its fitness, and the per-generation best-fitness
    history (for the convergence plot in Chapter 4).
    """
    rng = np.random.default_rng(seed)
    pop = rng.random((pop_size, dim))
    scores = np.array([fitness_fn(ind) for ind in pop])
    history = [scores.max()]
    print(f"[DE] gen 0  best macro-F1 = {scores.max():.4f}")

    for g in range(1, generations + 1):
        for i in range(pop_size):
            # --- mutation: v = x_r1 + F*(x_r2 - x_r3) ---
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            donor = pop[r1] + F * (pop[r2] - pop[r3])
            donor = np.clip(donor, 0.0, 1.0)

            # --- binomial crossover ---
            trial = pop[i].copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = donor[j]

            # --- greedy selection ---
            trial_score = fitness_fn(trial)
            if trial_score >= scores[i]:
                pop[i] = trial
                scores[i] = trial_score

        history.append(scores.max())
        print(f"[DE] gen {g}  best macro-F1 = {scores.max():.4f}")

    best = int(np.argmax(scores))
    return pop[best], scores[best], history


# -----------------------------------------------------------------------------
# 4. FINAL EVALUATION
# -----------------------------------------------------------------------------
def evaluate(model, X_test, y_test, n_classes, class_names, out_dir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if n_classes == 2:
        proba = model.predict(X_test, verbose=0).ravel()
        y_pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
    else:
        proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        try:
            auc = roc_auc_score(y_test, proba, multi_class="ovr",
                                average="macro")
        except Exception:
            auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)
    # False-positive rate (benign flagged as attack), derived from the matrix.
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fpr = float(np.mean(fp / np.clip(fp + tn, 1, None)))

    metrics = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_test, y_pred, average="macro", zero_division=0),
        f1=f1_score(y_test, y_pred, average="macro", zero_division=0),
        fpr=fpr, auc=auc)

    print("\n==== TEST RESULTS ====")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names],
                                zero_division=0))

    # Confusion matrix heatmap.
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.title("Confusion Matrix — DE-DNN on CSE-CIC-IDS2018")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    return metrics


# -----------------------------------------------------------------------------
# 5. MAIN
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--mode", choices=["binary", "multiclass"], default="multiclass")
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--use_smote", action="store_true")
    ap.add_argument("--pop_size", type=int, default=20)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--fitness_epochs", type=int, default=20)
    ap.add_argument("--final_epochs", type=int, default=100)
    args = ap.parse_args()

    (X_train, y_train, X_val, y_val, X_test, y_test,
     n_classes, class_names) = load_and_preprocess(
        args.data_dir, args.mode, args.sample_frac, args.use_smote)

    from tensorflow import keras
    de_data = (X_train, y_train, X_val, y_val)
    fit = lambda v: fitness(v, de_data, n_classes, epochs=args.fitness_epochs)

    best_vec, best_f1, history = differential_evolution(
        fit, pop_size=args.pop_size, generations=args.generations)

    best_hp = decode(best_vec)
    print(f"\n[DE] best validation macro-F1 = {best_f1:.4f}")
    print(f"[DE] best hyperparameters = {best_hp}")

    # Retrain the winner from scratch for the full budget.
    keras.backend.clear_session()
    final_model = build_dnn(X_train.shape[1], n_classes, best_hp)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)
    final_model.fit(np.vstack([X_train, X_val]),
                    np.concatenate([y_train, y_val]),
                    validation_data=(X_test, y_test),
                    epochs=args.final_epochs, batch_size=best_hp["batch"],
                    verbose=2, callbacks=[es])

    evaluate(final_model, X_test, y_test, n_classes, class_names)


if __name__ == "__main__":
    main()
