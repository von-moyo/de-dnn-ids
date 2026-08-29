"""Reconstruct the stage-2 artefacts locally.

metrics.json and the classification report are transcribed from the run log;
the confusion matrix is redrawn from the CSV downloaded from Drive, and the
loss curve from the epoch-by-epoch output. The pipeline now writes the loss
curve itself, but this run predates that change.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "final")
os.makedirs(OUT, exist_ok=True)

NAMES = json.load(open(os.path.join(HERE, "stage1", "metrics.json"))
                  )["_sampling"]["classes_evaluated"]

metrics = {
    "accuracy": 0.9241, "precision": 0.9447, "recall": 0.8706,
    "f1": 0.8923, "fpr": 0.0072, "auc": 0.9952,
    "benign_false_alarm_rate": 0.2938,
    "_sampling": {
        "max_per_class": 200000, "min_class_rows": 200,
        "classes_evaluated": NAMES,
        "caveat": ("Test set was class-balanced by --max_per_class, so it does "
                   "not carry the ~80% benign prior of real traffic. Macro "
                   "precision/recall/F1 weight classes equally and are "
                   "unaffected; accuracy and fpr are measured on the "
                   "rebalanced mix and are NOT operational estimates."),
    },
}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)

PERCLASS = [
    ("Benign", 0.84, 0.71, 0.77, 40000),
    ("Bot", 1.00, 1.00, 1.00, 28907),
    ("Brute Force -Web", 1.00, 0.48, 0.65, 113),
    ("Brute Force -XSS", 1.00, 0.52, 0.69, 46),
    ("DDOS attack-HOIC", 1.00, 1.00, 1.00, 39772),
    ("DDOS attack-LOIC-UDP", 0.90, 0.99, 0.94, 346),
    ("DDoS attacks-LOIC-HTTP", 1.00, 1.00, 1.00, 40000),
    ("DoS attacks-GoldenEye", 1.00, 1.00, 1.00, 8281),
    ("DoS attacks-Hulk", 1.00, 1.00, 1.00, 29040),
    ("DoS attacks-Slowloris", 1.00, 0.99, 1.00, 1982),
    ("Infilteration", 0.61, 0.77, 0.68, 23697),
    ("SSH-Bruteforce", 1.00, 1.00, 1.00, 18810),
]
with open(os.path.join(OUT, "classification_report.txt"), "w") as fh:
    fh.write("%-24s %9s %9s %9s %9s\n\n" % ("", "precision", "recall",
                                            "f1-score", "support"))
    for c, p, r, f, s in PERCLASS:
        fh.write("%-24s %9.2f %9.2f %9.2f %9d\n" % (c, p, r, f, s))
    fh.write("\n%-24s %9s %9s %9.2f %9d\n" % ("accuracy", "", "", 0.92, 230994))
    fh.write("%-24s %9.2f %9.2f %9.2f %9d\n" % ("macro avg", 0.94, 0.87, 0.89,
                                                230994))

# ------------------------------------------------------------ confusion matrix
cm = np.array([[int(x) for x in r] for r in
               csv.reader(open(os.path.join(OUT, "confusion_matrix.csv")))])
plt.figure(figsize=(12, 9.6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=NAMES, yticklabels=NAMES,
            annot_kws={"size": 7}, linewidths=0.4, linecolor="white")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix - Final DE-Optimised DNN on the Held-Out Test Set")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------- loss curve
loss = [0.1802, 0.1487, 0.1450, 0.1429, 0.1420, 0.1410, 0.1405, 0.1402,
        0.1395, 0.1390, 0.1388, 0.1388, 0.1383, 0.1379, 0.1380, 0.1373,
        0.1374, 0.1371, 0.1371, 0.1368, 0.1369, 0.1368, 0.1366, 0.1364,
        0.1361, 0.1363, 0.1359, 0.1362, 0.1360, 0.1360, 0.1359, 0.1358,
        0.1357, 0.1358, 0.1356, 0.1357, 0.1353, 0.1355, 0.1351, 0.1351,
        0.1349, 0.1351, 0.1352, 0.1347, 0.1349, 0.1350, 0.1347, 0.1343,
        0.1348, 0.1345, 0.1343, 0.1342, 0.1343, 0.1343, 0.1342, 0.1342]
val = [0.1534, 0.1447, 0.1447, 0.1436, 0.1417, 0.1395, 0.1412, 0.1465,
       0.1414, 0.1388, 0.1407, 0.1386, 0.1386, 0.1381, 0.1383, 0.1379,
       0.1381, 0.1402, 0.1395, 0.1379, 0.1394, 0.1363, 0.1361, 0.1372,
       0.1365, 0.1368, 0.1367, 0.1370, 0.1387, 0.1370, 0.1360, 0.1358,
       0.1377, 0.1370, 0.1366, 0.1354, 0.1360, 0.1372, 0.1355, 0.1357,
       0.1359, 0.1367, 0.1363, 0.1360, 0.1365, 0.1352, 0.1361, 0.1356,
       0.1352, 0.1364, 0.1354, 0.1371, 0.1356, 0.1353, 0.1358, 0.1378]
json.dump({"loss": loss, "val_loss": val},
          open(os.path.join(OUT, "training_history.json"), "w"), indent=2)

ep = range(1, len(loss) + 1)
best = int(np.argmin(val)) + 1
plt.figure(figsize=(7, 4.5))
plt.plot(ep, loss, marker="o", markersize=2.5, linewidth=1.5,
         color="#1f4e79", label="Training loss")
plt.plot(ep, val, marker="s", markersize=2.5, linewidth=1.5,
         color="#c00000", label="Validation loss")
plt.axvline(best, linestyle="--", linewidth=1.1, color="grey")
plt.annotate("best validation loss (epoch %d, %.4f)" % (best, min(val)),
             xy=(best, min(val)), xytext=(best - 26, 0.168), fontsize=8,
             arrowprops=dict(arrowstyle="->", linewidth=0.8, color="grey"))
plt.xlabel("Epoch")
plt.ylabel("Sparse categorical cross-entropy loss")
plt.title("Training and Validation Loss of the Final DE-Optimised DNN")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "training_loss.png"), dpi=200)
plt.close()

E = int(cm.sum() - np.trace(cm))
print("wrote final/ artefacts")
print("test flows %d, errors %d (%.2f%%)" % (cm.sum(), E, 100 * E / cm.sum()))
print("early stopping: best val at epoch %d (%.4f), ran %d epochs"
      % (best, min(val), len(loss)))
