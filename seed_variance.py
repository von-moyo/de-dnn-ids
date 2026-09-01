"""Measure how far macro-F1 moves between identical runs.

Before a DE-found configuration can be said to beat a hand-tuned one, you have
to know how large a difference the experiment can actually resolve. This script
trains the same configuration several times over, changing only the weight
initialisation and the batch shuffling, and reports the spread.

The point is a single number: the run-to-run standard deviation. If two
configurations differ by less than that, the difference is not evidence of
anything, and no amount of extra search generations will change it.

    python seed_variance.py --data_dir ./data --seeds 5 \
        --config results/stage1/best_config.json \
        --config baseline_config.json \
        --out_dir results/variance \
        --max_per_class 20000 --min_class_rows 200

The data split is held fixed across every run, so the only thing varying is
training stochasticity -- which is what makes two configurations hard to tell
apart in the first place. Vary --data_seed as well and you fold in split
variance too, a larger and separate question.

Give it one --config to get a noise floor, two or more to also get a pairwise
comparison with a Welch t-test.
"""
import argparse
import json
import os
import statistics

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from de_dnn_ids import build_dnn, load_and_preprocess, set_seeds


def load_hp(path):
    """Accept best_config.json, baseline_config.json, or a bare hp dict."""
    with open(path) as fh:
        blob = json.load(fh)
    hp = blob.get("hyperparameters", blob)
    missing = {"n_layers", "neurons", "lr", "dropout", "batch",
               "activation"} - set(hp)
    if missing:
        raise SystemExit(f"{path}: missing hyperparameters {sorted(missing)}")
    return {k: hp[k] for k in ("n_layers", "neurons", "lr", "dropout",
                               "batch", "activation")}


def train_once(hp, split, n_classes, seed, epochs):
    """One full training run, mirroring the final-model protocol in main().

    Same 90/10 split of train+val for early stopping, same patience, same
    untouched test set -- so the numbers here are comparable to the ones in
    metrics.json rather than being a separate protocol with its own biases.
    """
    from tensorflow import keras
    X_fit, y_fit, X_es, y_es, X_test, y_test = split

    set_seeds(seed)
    keras.backend.clear_session()
    model = build_dnn(X_fit.shape[1], n_classes, hp)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)
    hist = model.fit(X_fit, y_fit, validation_data=(X_es, y_es),
                     epochs=epochs, batch_size=hp["batch"], verbose=0,
                     callbacks=[es])

    if n_classes == 2:
        y_pred = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
    else:
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    return {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "per_class_f1": [float(x) for x in
                         f1_score(y_test, y_pred, average=None)],
        "epochs_run": len(hist.history["loss"]),
    }


def summarise(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "n": len(values),
    }


def welch(a, b):
    """Welch's t-test. Returns (t, p) or (t, None) if scipy is absent."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    va, vb = statistics.variance(a), statistics.variance(b)
    se = (va / na + vb / nb) ** 0.5
    if se == 0:
        return None, None
    t = (statistics.mean(a) - statistics.mean(b)) / se
    try:
        from scipy import stats
        df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        return t, float(2 * stats.t.sf(abs(t), df))
    except Exception:
        return t, None


def compare_runs(a, b, fa, fb):
    """Judge the gap between two configurations against their own spread.

    Returns (record, lines_to_print). Two standard deviations is the usual
    informal bar for "these are actually different"; the t-test is the formal
    version of the same question. Both have to agree before the difference is
    called real, because with five runs apiece neither is strong on its own.
    """
    gap = statistics.mean(fa) - statistics.mean(fb)
    pooled = (((statistics.variance(fa) + statistics.variance(fb)) / 2) ** 0.5
              if min(len(fa), len(fb)) > 1 else 0.0)
    t, p = welch(fa, fb)
    separable = bool(pooled > 0 and abs(gap) > 2 * pooled
                     and (p is None or p < 0.05))

    lines = ["=" * 68, f"{a}  vs  {b}", "=" * 68,
             f"  gap in mean macro-F1 : {gap:+.4f}",
             f"  pooled std           :  {pooled:.4f}"]
    if p is not None:
        lines.append(f"  Welch t              : {t:+.3f}   p = {p:.3f}")
    lines.append("")
    if separable:
        lines.append(f"  The gap clears the noise. {a} is genuinely "
                     f"{'better' if gap > 0 else 'worse'} than {b} here.")
    else:
        lines.append(f"  The gap does NOT clear the noise. A difference of "
                     f"{abs(gap):.4f} is what rerunning the same "
                     f"configuration under a different seed already produces, "
                     f"so this experiment cannot tell {a} and {b} apart.")
        lines.append(f"  Raising --pop_size or --generations will not fix "
                     f"this; it makes the search chase differences the "
                     f"measurement cannot see.")
    return ({"a": a, "b": b, "gap": gap, "pooled_std": pooled,
             "t": t, "p": p, "separable": separable}, lines)


def plot(results, out_dir):
    """Strip plot of every run, so the overlap between configurations is
    visible rather than having to be inferred from a standard deviation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] no plot ({type(e).__name__}: {e})")
        return

    # The JSON is already on disk by the time this runs, so a failure here
    # costs a figure, never the experiment.
    try:
        names = list(results)
        fig, ax = plt.subplots(figsize=(1.9 * len(names) + 3.2, 5))
        rng = np.random.default_rng(0)

        for i, name in enumerate(names):
            vals = results[name]["macro_f1_runs"]
            ax.scatter(i + rng.uniform(-0.07, 0.07, len(vals)), vals, s=52,
                       alpha=0.75, zorder=3, edgecolor="white", linewidth=.8)
            m, s = statistics.mean(vals), (statistics.stdev(vals)
                                           if len(vals) > 1 else 0.0)
            ax.hlines(m, i - 0.26, i + 0.26, color="black", linewidth=2,
                      zorder=4)
            if s:
                ax.add_patch(plt.Rectangle((i - 0.26, m - s), 0.52, 2 * s,
                                           facecolor="grey", alpha=0.16,
                                           zorder=1))
            ax.annotate(f"{m:.4f}\n+/-{s:.4f}", (i, m), xytext=(14, 0),
                        textcoords="offset points", va="center", fontsize=9)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel("test macro-F1")
        ax.set_title("Run-to-run variation at a fixed data split\n"
                     "(band = 1 standard deviation)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        path = os.path.join(out_dir, "seed_variance.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[out] plot -> {path}")
    except Exception as e:
        print(f"[warn] plot failed ({type(e).__name__}: {e}); "
              f"results are still saved")


def main():
    ap = argparse.ArgumentParser(
        description="Measure run-to-run variation in macro-F1, so a "
                    "difference between two configurations can be judged "
                    "against the noise it has to clear.")
    ap.add_argument("--config", action="append", required=True,
                    help="path to a config JSON. Repeat the flag to compare "
                         "several, e.g. the DE winner against the baseline")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./results/variance")
    ap.add_argument("--mode", choices=["binary", "multiclass"],
                    default="multiclass")
    ap.add_argument("--seeds", type=int, default=5,
                    help="training runs per configuration")
    ap.add_argument("--epochs", type=int, default=100,
                    help="epoch budget per run; match --final_epochs of the "
                         "run being checked")
    ap.add_argument("--data_seed", type=int, default=42,
                    help="seed for the split. Held fixed across runs so only "
                         "training stochasticity varies")
    ap.add_argument("--max_per_class", type=int, default=None)
    ap.add_argument("--min_class_rows", type=int, default=0)
    ap.add_argument("--sample_frac", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = {os.path.splitext(os.path.basename(p))[0]: load_hp(p)
               for p in args.config}
    if len(configs) != len(args.config):
        raise SystemExit("config files must have distinct basenames")

    (X_train, y_train, X_val, y_val, X_test, y_test,
     n_classes, class_names) = load_and_preprocess(
        args.data_dir, args.mode, args.sample_frac, False, args.data_seed,
        max_per_class=args.max_per_class, min_class_rows=args.min_class_rows)

    # Same pooling and 90/10 early-stopping split the final model uses.
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_full, y_full, test_size=0.10, stratify=y_full,
        random_state=args.data_seed)
    split = (X_fit, y_fit, X_es, y_es, X_test, y_test)

    total = len(configs) * args.seeds
    print(f"\n{len(configs)} configuration(s) x {args.seeds} seeds = {total} "
          f"trainings, {n_classes} classes\n")

    results = {}
    done = 0
    for name, hp in configs.items():
        print(f"--- {name}: {hp}")
        runs = []
        for s in range(args.seeds):
            seed = args.data_seed + 1000 * (s + 1)
            r = train_once(hp, split, n_classes, seed, args.epochs)
            runs.append(r)
            done += 1
            print(f"  [{done:>2}/{total}] seed {seed:<5} "
                  f"macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}  "
                  f"({r['epochs_run']} epochs)", flush=True)
        f1s = [r["macro_f1"] for r in runs]
        results[name] = {
            "hyperparameters": hp,
            "macro_f1_runs": f1s,
            "accuracy_runs": [r["accuracy"] for r in runs],
            "macro_f1": summarise(f1s),
            "per_class_f1_mean": [
                float(np.mean([r["per_class_f1"][i] for r in runs]))
                for i in range(len(runs[0]["per_class_f1"]))],
        }
        st = results[name]["macro_f1"]
        print(f"  => mean {st['mean']:.4f}  sd {st['std']:.4f}  "
              f"range {st['range']:.4f}\n")

    print("=" * 68)
    print("NOISE FLOOR")
    print("=" * 68)
    for name, r in results.items():
        st = r["macro_f1"]
        print(f"  {name:<28} {st['mean']:.4f} +/- {st['std']:.4f}   "
              f"[{st['min']:.4f}, {st['max']:.4f}]")

    comparisons = []
    names = list(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            record, lines = compare_runs(a, b, results[a]["macro_f1_runs"],
                                         results[b]["macro_f1_runs"])
            comparisons.append(record)
            print()
            for line in lines:
                print(line)

    out = os.path.join(args.out_dir, "seed_variance.json")
    with open(out, "w") as fh:
        json.dump({"settings": vars(args),
                   "class_names": [str(c) for c in class_names],
                   "results": results,
                   "comparisons": comparisons}, fh, indent=2)
    print(f"\n[out] {out}")
    plot(results, args.out_dir)


if __name__ == "__main__":
    main()
