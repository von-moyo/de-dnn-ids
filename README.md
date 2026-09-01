# DE-DNN-IDS

**Differential Evolution optimised Deep Neural Network for Network Intrusion Detection**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A hand-implemented **DE/rand/1/bin** differential evolution optimiser that searches
the architecture and training hyperparameters of a deep neural network for
network intrusion detection on **CSE-CIC-IDS2018**, using **validation macro-F1**
as the fitness signal.

> B.Sc. Computer Science final-year project · University of Lagos
> Author: **Florish Adekogbe** (190805025) · Supervisor: **Dr. B. A. Sawyerr**

---

## Why this project

Deep neural networks work well for intrusion detection, but their performance is
dominated by choices nobody can derive analytically: how deep, how wide, what
learning rate, how much dropout. Those choices are usually made by hand or by
grid search.

This project replaces that guesswork with **Differential Evolution**, a
population-based, derivative-free optimiser. DE is the right tool here because the
search space is hostile to gradient methods:

| Property of the search space | Consequence |
|---|---|
| Non-differentiable | You cannot take ∂(macro-F1)/∂(number of layers) |
| Non-convex | Many local optima; hill-climbing gets stuck |
| Mixed discrete + continuous | Layer count is an integer, learning rate is a real |
| Expensive to evaluate | Each fitness call is a full network training run |

The second design decision matters as much as the first: fitness is **macro-F1,
not accuracy**. CSE-CIC-IDS2018 is 80% benign traffic, so a model that predicts
"Benign" for every flow scores 80% accuracy while detecting nothing. Macro-F1
weights every class equally, forcing the search to care about rare attack
families — which are precisely the ones a real adversary uses.

---

## Pipeline

```mermaid
flowchart TD
    A[10 daily captures<br/>6.7M flows · 15 classes] --> B[Clean<br/>drop identity cols · repair inf/NaN<br/>drop zero-variance cols]
    B --> B2[Class-aware cap<br/>drop unscoreable classes]
    B2 --> C[Stratified split<br/>60 / 20 / 20]
    C --> D[MinMax scale<br/>fitted on TRAIN only]
    D --> E{Optional SMOTE<br/>train split only}
    E --> F[["DE/rand/1/bin<br/>mutate → crossover → select"]]
    F -->|"decode(vector) → hyperparameters"| G[Train candidate DNN<br/>score validation macro-F1]
    G -->|fitness| F
    F --> H[Best configuration]
    H --> I[Retrain on train+val<br/>early stopping on a 10% inner slice]
    I --> J[Evaluate ONCE on test<br/>accuracy · precision · recall · F1 · FPR · ROC-AUC]
```

### The search space

DE searches the continuous unit hypercube `[0,1]⁶`; `decode()` maps each vector
onto concrete network settings.

| Gene | Hyperparameter | Range | Mapping |
|---|---|---|---|
| 0 | Hidden layers | 1 – 5 | linear, rounded |
| 1 | Neurons per layer | 32 – 512 | linear, rounded |
| 2 | Learning rate | 1e-4 – 1e-1 | **logarithmic** |
| 3 | Dropout | 0.0 – 0.5 | linear |
| 4 | Batch size | {16, 32, 64, 128, 256} | index |
| 5 | Activation | {relu, tanh, sigmoid} | index |

The learning rate is mapped **logarithmically** on purpose. A linear map from
1e-4 to 1e-1 would spend 99.9% of the gene's range above 1e-3, where Adam rarely
converges well, so DE would almost never sample the region that actually works.

### DE operators

```
mutation    v_i = x_r1 + F · (x_r2 − x_r3)        F  = 0.8
crossover   binomial, gene-wise                    CR = 0.9, with a forced j_rand gene
selection   greedy, one-to-one, elitist            child replaces parent iff score ≥ parent
```

The mutation step is what makes DE self-adapting: the difference vector is drawn
from the **population's own spread**. Early on the population is scattered, so
steps are large and exploratory; once it converges on a region the differences
shrink and the steps become fine refinements. There is no cooling schedule to tune.

---

## Results

> ⚠️ **Preliminary.** The numbers below come from a reduced pilot run, not the
> final experiment. They are published here for transparency and will be replaced
> once the full-scale run completes.

**Configuration** — 5% stratified subsample of four capture days
(`02-14`, `02-15`, `02-22`, `03-02`), 9 classes, 70 features after cleaning.
239,000 flows: 143,400 train / 47,800 validation / 47,800 test.
DE budget: `pop_size=6`, `generations=5` → 36 fitness evaluations.

**Configuration selected by DE**

| Hyperparameter | Value |
|---|---|
| Hidden layers | 4 |
| Neurons per layer | 156 |
| Learning rate | 1.706e-3 |
| Dropout | 0.087 |
| Batch size | 32 |
| Activation | sigmoid |

**Test-set performance**

| Metric | Value |
|---|---|
| Accuracy | 0.9849 |
| Precision (macro) | 0.7333 |
| Recall (macro) | 0.7475 |
| **F1 (macro)** | **0.7481** |
| False-positive rate | 0.0110 |
| ROC-AUC (macro OVR) | *not computable — see below* |

<!-- Drop your figures into docs/images/ and these will render.
     Suggested filenames are already referenced below. -->

| DE convergence | Confusion matrix |
|---|---|
| ![DE convergence](docs/images/de_convergence.png) | ![Confusion matrix](docs/images/confusion_matrix.png) |

### Reading these results honestly

The **98.5% accuracy against a 0.75 macro-F1 is the whole story of this dataset**,
and the gap is the point rather than a defect. Benign traffic dominates, so
accuracy is nearly free; macro-F1 exposes that several rare attack classes are
still being missed. This is exactly why macro-F1, not accuracy, drives the
optimisation.

Two honest caveats on the pilot run:

1. **The DE search was too small to have converged.** Best fitness moved from
   0.6574 to 0.6575 across five generations — essentially flat. With a population
   of 6 in a 6-dimensional space, DE barely has enough diversity to form useful
   difference vectors. The published guidance is `pop_size ≈ 10 × dim`; the final
   run uses `pop_size=20, generations=50`.
2. **Two classes are starved at a 5% sample.** `SQL Injection` has ~87 flows in
   the entire corpus and `Brute Force -XSS` ~230. At 5% that is ~4 and ~11 flows
   respectively, leaving roughly **zero** in the test split. This is what makes
   ROC-AUC uncomputable — scikit-learn cannot score a one-vs-rest AUC for a class
   with no positive samples — and it drags macro-F1 down mechanically, because a
   class with no support scores 0.0 and still counts as 1/9 of the average.

The pipeline now prints per-class test support before training and flags any
class with fewer than 30 test flows, so this is visible up front rather than as a
failure at the reporting stage.

#### What the search is worth, measured

A single run cannot answer this. On stage 1 (`--max_per_class 20000`, 12
scoreable classes) the DE winner scored macro-F1 **0.9041** against a hand-tuned
baseline's **0.9019** — a gap of 0.0022 that looks like nothing at all.
Retraining both five times with different weight initialisations at a fixed data
split (`seed_variance.py`, 100-epoch budget) says otherwise:

| Configuration | mean macro-F1 | sd | range |
|---|---|---|---|
| **DE winner** — 4×209, tanh, dropout 0.0, lr 7.2e-4, batch 128 | **0.9060** | 0.0058 | 0.8960 – 0.9098 |
| **Hand-tuned** — 2×128, ReLU, dropout 0.2, lr 1e-3, batch 64 | 0.8876 | 0.0106 | 0.8775 – 0.9032 |

Gap **+0.0184**, pooled sd 0.0085, Welch *t* = 3.40, **p = 0.014**. The
difference clears the noise.

The single-run comparison understated it badly, and in the direction that
flatters the baseline: **0.9019 was a lucky draw**, essentially the top of that
configuration's own range (max 0.9032), while its typical run is 0.8876. Measure
the spread before quoting any gap — it can mislead in either direction.

Three things to keep in view when reading the number:

- **8 of the 12 classes already sit at F1 ≥ 0.99** (Bot, HOIC, LOIC-UDP,
  LOIC-HTTP, GoldenEye, Hulk, Slowloris, SSH-Bruteforce). Two-thirds of macro-F1
  is frozen and no hyperparameter moves it; the entire searchable range lives in
  Benign (0.66), Infiltration (0.77), Brute Force -Web (0.80) and -XSS (0.65).
  Whatever DE won, it won inside that third.
- **Brute Force -XSS has 46 test rows.** One row landing differently shifts its
  F1 by 0.018 and macro-F1 by 0.0015 — small against a 0.0184 gap, but it is
  why the five runs spread as far as they do.
- **Early stopping may be handicapping the baseline.** It stops on `val_loss`
  with patience 10, and the baseline halted at epoch 11 in four of five runs —
  meaning its best epoch was 1 and `restore_best_weights` discarded the rest.
  The DE winner trained 25–44 epochs; the single baseline run that reached 20
  scored that configuration's best result. Part of what DE selected for may be
  "trains stably under val_loss early stopping" rather than raw capacity. That
  is still a real property of the found configuration, but it is a narrower
  claim than "better architecture", and worth stating rather than glossing.

`seed_variance.py` is the instrument for all of this: it retrains one
configuration several times, reports the run-to-run spread, and judges a gap
against it with both the 2σ rule and a Welch *t*-test. Quote the spread
alongside the gap. Most published DE-for-IDS comparisons report a single run per
configuration and cannot distinguish a real improvement from a fortunate seed.

---

## Dataset build matters

CSE-CIC-IDS2018 circulates in two incompatible forms, and **which one you use
changes your results more than any hyperparameter DE will find.**

| | Raw CIC release (CSV) | Cleaned redistribution (parquet) |
|---|---|---|
| Source | [CIC](https://www.unb.ca/cic/datasets/ids-2018.html), AWS S3, `solarmainframe/ids-intrusion-csv` | `dhoogla/csecicids2018` |
| Rows | ~16,000,000 | **6,659,532** |
| Columns | 80 | **78** |
| `Dst Port` | present | **absent** |
| `Timestamp` | present | absent |
| Duplicate rows | many | **0** |
| `Infinity` / NaN | present | already repaired |

The pipeline reads both. The numbers in this README come from the **cleaned
parquet** build.

**Deduplication is the consequential difference.** Removing duplicate flows
collapses several attack families to a handful of distinct feature vectors:

| Class | Raw | Deduplicated |
|---|---:|---:|
| FTP-BruteForce | ~193,000 | **53** |
| DoS attacks-SlowHTTPTest | ~140,000 | **55** |
| SQL Injection | ~87 | 85 |

Those rows were never independent evidence — they were the same flow repeated.
But the consequence is that at a 20% test split these classes get ~10 test rows
each, too few to score and pure noise inside the macro-F1 that DE optimises.
`--min_class_rows 200` removes exactly those three, leaving 12 scoreable classes.

**This affects how you compare against published work.** Nearly all published
CSE-CIC-IDS2018 results use the raw build, where duplicated flows appear in both
the train and test splits — the model memorises rather than generalises, and the
reported figures are inflated. Results on the deduplicated build are **lower and
not directly comparable**. State which build you used; a reader comparing a
deduplicated macro-F1 against a raw-build table will misread it as
underperformance.

Do not mix the two, or fill gaps in one from the other: half a corpus
deduplicated and half not means preprocessing differs by capture day.

---

## Installation

```bash
git clone https://github.com/von-moyo/de-dnn-ids.git
cd de-dnn-ids

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Then fetch the dataset into `data/` — see [`data/README.md`](data/README.md).

> **Windows + GPU:** TensorFlow 2.11+ dropped native Windows GPU support. Use
> WSL2, `tensorflow-directml-plugin`, or Google Colab (Runtime → Change runtime
> type → GPU). On CPU the full search is not practical.

### Running it on Colab

[`colab_de_dnn_ids.ipynb`](colab_de_dnn_ids.ipynb) reproduces the whole study on
a free T4. Open it in Colab, set the runtime to GPU, and run the cells in order.

Nothing has to be set up beforehand: it clones this repo, downloads the dataset
from Kaggle's anonymous public endpoint, and runs both stages. There is no
Kaggle account, API token or `kaggle.json` involved, and no dataset to upload.
The Google Drive mount in section 5 is the only sign-in in the notebook, and it
is optional — set `USE_DRIVE = False` to keep results on local disk, at the cost
of being unable to resume the search after a Colab disconnect.

---

## Usage

```bash
# Multi-class over every attack family, class-capped so the search is affordable
python de_dnn_ids.py --data_dir ./data --mode multiclass --max_per_class 20000

# Binary detection: benign vs attack
python de_dnn_ids.py --data_dir ./data --mode binary --max_per_class 200000

# Fast smoke test — exercises the whole pipeline in a few minutes
python de_dnn_ids.py --mode binary --max_per_class 2000 \
    --pop_size 4 --generations 1 --fitness_epochs 3 --final_epochs 5

# Balance rare classes with SMOTE (training split only)
python de_dnn_ids.py --use_smote
```

`--pop_size 4` is the floor: DE/rand/1 needs three donor vectors distinct from
the target, so a smaller population cannot form a mutation at all.

### Recommended protocol

The default budget is `pop_size × (1 + generations) = 1,020` full network
trainings — days of compute on CPU, and not feasible at any realistic data size.
Search cheaply, then retrain the winner larger:

```bash
# Stage 0 — noise floor. How large a difference can the experiment resolve?
python seed_variance.py --seeds 5 --epochs 100 \
    --config results/stage1/best_config.json \
    --config baseline_config.json \
    --out_dir results/variance \
    --max_per_class 20000 --min_class_rows 200

# Stage 1 — search on a class-capped subset
python de_dnn_ids.py --out_dir results/stage1 \
    --max_per_class 20000 --min_class_rows 200 \
    --pop_size 20 --generations 12 --fitness_epochs 25

# Stage 2 — retrain the winning configuration once, on 10× the data (1–2 h)
python de_dnn_ids.py --out_dir results/final \
    --max_per_class 200000 --min_class_rows 200 \
    --load_config results/stage1/best_config.json
```

**Do stage 0 first.** It is ten trainings and it decides whether the rest is
worth running. If macro-F1 moves as much between two identical runs as it does
between the DE winner and the baseline, then the comparison cannot support any
claim, and a larger `--pop_size` only makes the search chase differences the
measurement cannot see. See [Reading these results
honestly](#reading-these-results-honestly).

Keep `--fitness_epochs` within reach of `--final_epochs`. `fitness()` stops
early on patience 5; a cap far below the final budget means the cap ends every
candidate instead, so DE ranks on *"best after N epochs"* rather than *"best
when trained out"* — which systematically favours fast-starting configurations
(high learning rate, small batch) over ones that win given the full budget.

This is a standard **fidelity-reduction** strategy: DE converges on the *relative
ranking* of hyperparameters, and that ranking is largely stable across sample
sizes even though absolute scores are not.

> `--load_config` skips only the **search**. Data flags are not replayed from the
> saved config, so stage 2 must repeat `--max_per_class` and `--min_class_rows`
> — otherwise it evaluates a different class set than the one DE tuned for.

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Directory containing the CSE-CIC-IDS2018 CSVs |
| `--out_dir` | `./results` | Where figures, metrics and the model are written |
| `--mode` | `multiclass` | `binary` or `multiclass` |
| `--sample_frac` | `1.0` | Stratified per-file subsample, e.g. `0.05`. Thins every class equally, so it starves rare attacks — prefer `--max_per_class` |
| `--max_per_class` | — | Cap each class at N rows, keeping **all** rows of any class below N. "Class" follows `--mode`: attack family in multiclass, Benign/Attack in binary |
| `--min_class_rows` | `0` (off) | Drop classes with fewer than N rows corpus-wide. `200` removes the three that deduplication left unscoreable |
| `--use_smote` | off | Oversample minority classes in the training split |
| `--pop_size` | `20` | DE population size |
| `--generations` | `50` | DE generations |
| `--fitness_epochs` | `20` | Epoch budget per DE candidate. Early stopping normally ends the fit first; set this far below `--final_epochs` and the cap bites instead, changing what DE optimises |
| `--fitness_repeats` | `1` | Train each candidate N times and average the macro-F1. Costs N×, cuts fitness noise by √N. Only worth it if `seed_variance.py` shows run-to-run spread near the gap between candidates |
| `--final_epochs` | `100` | Epoch budget for the retrained winner |
| `--seed` | `42` | Seeds Python, NumPy, TensorFlow and all splits |
| `--load_config` | — | Skip the search; retrain a saved `best_config.json` |

### Outputs

Every run writes to `--out_dir`:

| File | Contents |
|---|---|
| `best_config.json` | Winning hyperparameters, raw DE vector, full convergence history, and the exact run settings |
| `de_convergence.png` | Best validation macro-F1 per generation |
| `confusion_matrix.png` | Annotated heatmap |
| `confusion_matrix.csv` | Raw counts, for your own tables |
| `classification_report.txt` | Per-class precision / recall / F1 / support |
| `metrics.json` | Headline test metrics |
| `de_dnn_best.keras` | The trained final model |

---

## Methodology notes

These are the decisions an examiner is most likely to probe.

**Leakage control.** The test set is read exactly once, inside `evaluate()`.
Concretely:

- **Identity columns are dropped first.** `Src IP`, `Dst IP`, `Flow ID` and
  `Timestamp` are removed before anything else. The attacks in this dataset were
  launched from a fixed set of hosts during known windows, so a model that keeps
  them learns *"traffic from 18.219.211.138 is malicious"* rather than what an
  attack looks like — near-perfect on paper, useless in production. `Dst Port` is
  deliberately **kept** where it exists: it is genuine protocol signal.
  ⚠️ The cleaned parquet build drops `Dst Port` and `Timestamp` upstream, so on
  that build the model trains **without port information at all**. See
  [Dataset build](#dataset-build-matters).
- **The scaler is fitted on the training split only.** Splitting happens *before*
  scaling, and validation and test are transformed with training statistics.
- **SMOTE runs after the split, on training data only.** Applied beforehand it
  would synthesise training points by interpolating test samples.
- **Early stopping never sees the test set.** During the final retrain a 10%
  stratified slice is carved out of train+val to drive early stopping. Monitoring
  the test set there would mean selecting weights on the same data the results are
  reported from.

**Why macro-F1 as fitness.** Per-class F1, averaged with equal weight. A class
with 2,000 flows counts as much as one with 8,000,000. F1 itself is the *harmonic*
mean of precision and recall, so it cannot be gamed by flagging everything as an
attack.

**Why false-positive rate is reported.** At enterprise traffic volumes a 1% FPR
across 10M benign flows is 100,000 false alarms per day. Analysts stop reading
them — this is *alert fatigue*, and it is the documented reason the 2013 Target
breach went unactioned despite the IDS firing correctly. A model with 99% accuracy
and 2% FPR is operationally worse than one with 97% accuracy and 0.1% FPR.

Two false-positive figures are reported, and the distinction matters:

| Metric | Meaning |
|---|---|
| `fpr` | Macro one-vs-rest FPR, averaged over **all** classes |
| `benign_false_alarm_rate` | Of genuinely benign flows, the fraction flagged as **some** attack |

`fpr` is diluted in multiclass mode: a dozen easy attack classes with near-zero
false positives average it down to ~0.007 even when most benign traffic is being
misclassified. `benign_false_alarm_rate` is the one the alert-fatigue argument is
about — **quote that one operationally.**

**Benign vs Infiltration is the hard case.** `Infilteration` is near-
indistinguishable from normal traffic in this dataset, and the model reliably
confuses the two in both directions. Expect this to dominate your error budget;
it is a known property of CSE-CIC-IDS2018 rather than a defect in the pipeline.

**Robustness choices.** A candidate that fails to train (typically an
out-of-memory configuration such as 5 × 512 neurons at batch 16) scores 0.0 rather
than aborting the search, so a multi-hour run is not lost to one bad individual.
`keras.backend.clear_session()` is called before every candidate to stop the Keras
global graph accumulating across ~1,000 evaluations.

---

## Known limitations

- **Peak memory.** All captures are concatenated into one DataFrame before
  cleaning. With `--max_per_class` each file is pre-capped during the load loop,
  which bounds this; without it, the full corpus is resident at once and needs
  roughly 10–12 GB on the raw CSVs.
- **Rare-class support.** Three classes cannot be scored on the deduplicated
  build at any sample size — see [Dataset build](#dataset-build-matters).
- **Class-balanced metrics.** `--max_per_class` rebalances the test set, so
  reported `accuracy` and `fpr` are measured on that mix rather than on real
  traffic's ~80% benign prior. Macro precision/recall/F1 weight classes equally
  and are unaffected. The caveat is written into `metrics.json` under
  `_sampling` on every capped run.
- **DE cost.** Fitness evaluation is a full training run. Multi-fidelity methods
  (Hyperband, successive halving) would reach a comparable configuration for less
  compute — at the cost of the clean, interpretable DE convergence curve.
- **Determinism.** `--seed` fixes Python, NumPy, TensorFlow and every split, but
  bit-for-bit GPU reproducibility additionally requires `TF_DETERMINISTIC_OPS=1`,
  which disables the fast cuDNN kernels and roughly doubles search time.

---

## Repository layout

```
de-dnn-ids/
├── de_dnn_ids.py          # Complete pipeline: preprocessing, DNN, DE, evaluation
├── seed_variance.py       # Run-to-run spread: what gap can this experiment resolve?
├── baseline_config.json   # Hand-picked hyperparameters, for the comparison
├── colab_de_dnn_ids.ipynb # End-to-end run on a free Colab T4, no credentials
├── requirements.txt
├── data/
│   └── README.md          # How to obtain CSE-CIC-IDS2018 + its known defects
├── report/                # Measured artefacts: stage1/, final/, baseline/
├── results/               # Run artefacts (git-ignored except the CSV matrix)
├── docs/images/           # Figures used in this README
├── CITATION.cff
└── LICENSE
```

---

## Citation

```bibtex
@software{adekogbe2026dednnids,
  author  = {Adekogbe, Florish},
  title   = {Differential Evolution Optimised Deep Neural Network
             for Network Intrusion Detection},
  year    = {2026},
  school  = {University of Lagos},
  url     = {https://github.com/von-moyo/de-dnn-ids}
}
```

**Dataset:** Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018).
*Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization.* 4th International Conference on Information Systems Security
and Privacy (ICISSP), Portugal.

**Method:** Storn, R., & Price, K. (1997). *Differential Evolution – A Simple and
Efficient Heuristic for Global Optimization over Continuous Spaces.* Journal of
Global Optimization, 11(4), 341–359.

---

## Acknowledgements

Supervised by **Dr. B. A. Sawyerr**, Department of Computer Science, University
of Lagos. Dataset provided by the Canadian Institute for Cybersecurity (CIC) in
collaboration with the Communications Security Establishment (CSE).

## License

MIT — see [LICENSE](LICENSE).
