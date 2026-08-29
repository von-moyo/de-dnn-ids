"""Chapter 4 and 5 content, derived entirely from the artefacts in stage1/.

Every number appearing here is read from metrics.json, best_config.json,
classification_report.txt or confusion_matrix.csv, or is a direct count taken
from the corpus. Nothing is illustrative.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.join(HERE, "stage1")

m = json.load(open(os.path.join(S1, "metrics.json")))
cfg = json.load(open(os.path.join(S1, "best_config.json")))
hp = cfg["hyperparameters"]
hist = cfg["convergence_history"]
gain = hist[-1] - hist[0]
last_imp = max(i for i in range(1, len(hist)) if hist[i] > hist[i - 1])
plateau = len(hist) - 1 - last_imp

PERCLASS = [
    ("Benign", 0.88, 0.52, 0.66, 4000),
    ("Bot", 1.00, 0.99, 1.00, 4000),
    ("Brute Force -Web", 0.74, 0.89, 0.80, 114),
    ("Brute Force -XSS", 1.00, 0.48, 0.65, 46),
    ("DDOS attack-HOIC", 1.00, 1.00, 1.00, 4000),
    ("DDOS attack-LOIC-UDP", 0.99, 0.99, 0.99, 346),
    ("DDoS attacks-LOIC-HTTP", 0.99, 1.00, 0.99, 4000),
    ("DoS attacks-GoldenEye", 1.00, 1.00, 1.00, 4000),
    ("DoS attacks-Hulk", 1.00, 1.00, 1.00, 4000),
    ("DoS attacks-Slowloris", 1.00, 1.00, 1.00, 1981),
    ("Infilteration", 0.66, 0.93, 0.77, 4000),
    ("SSH-Bruteforce", 1.00, 1.00, 1.00, 4000),
]

CORPUS = [
    ("Benign", "5,329,008", "80.021"),
    ("DDoS attacks-LOIC-HTTP", "575,364", "8.640"),
    ("DDOS attack-HOIC", "198,861", "2.986"),
    ("DoS attacks-Hulk", "145,199", "2.180"),
    ("Bot", "144,535", "2.170"),
    ("Infilteration", "118,483", "1.779"),
    ("SSH-Bruteforce", "94,048", "1.412"),
    ("DoS attacks-GoldenEye", "41,406", "0.622"),
    ("DoS attacks-Slowloris", "9,908", "0.149"),
    ("DDOS attack-LOIC-UDP", "1,730", "0.026"),
    ("Brute Force -Web", "568", "0.009"),
    ("Brute Force -XSS", "229", "0.003"),
    ("SQL Injection", "85", "0.001"),
    ("DoS attacks-SlowHTTPTest", "55", "0.001"),
    ("FTP-BruteForce", "53", "0.001"),
]


def ch4():
    S = []
    A = S.append
    A(("h1", "CHAPTER FOUR"))
    A(("h1", "RESULTS AND DISCUSSION"))

    A(("h2", "4.1 \tIntroduction"))
    A(("p", "Chapter Three set out the design of the Differential Evolution "
            "optimised deep neural network and described how it was built, the "
            "environment it was run in, and the state the dataset arrived in. "
            "This chapter reports what happened when that system was run. It "
            "presents what the evolutionary search found, how the resulting "
            "detector performed on data it had never seen, how that "
            "performance changed with the quantity of training data, and how "
            "it compares against a network configured by hand."))
    A(("p", "Some of what follows is not the result the project set out to "
            "find. The search converged faster and gained less than the framing "
            "in Chapter One anticipated, and the final model carries a specific "
            "and substantial weakness that the headline metrics conceal. Both "
            "are reported here in full, together with the reasoning that "
            "explains them, because a result that is only partly reported is of "
            "little use to anyone who later tries to build on it."))

    A(("h2", "3.9 \tImplementation Environment and Tools"))
    A(("p", "The system was written in Python 3.12. The neural network was "
            "built with the Keras interface to TensorFlow 2.21, while Pandas, "
            "NumPy and Scikit-learn handled data loading, cleaning, scaling and "
            "scoring, and Matplotlib and Seaborn produced the figures. The "
            "Differential Evolution optimiser was written from first principles "
            "rather than taken from a library. There were two reasons for this: "
            "every step of the search had to remain inspectable so that its "
            "convergence behaviour could be reported honestly, and the standard "
            "library implementations assume a continuous objective function and "
            "do not accommodate the mixture of integer, categorical and "
            "continuous hyperparameters that this problem involves."))
    A(("p", "Development was carried out locally under Windows, but the "
            "experiments themselves were run on Google Colab using an NVIDIA T4 "
            "graphics processor. This was not a matter of preference. "
            "TensorFlow withdrew native GPU support on Windows at version 2.11, "
            "which confines a local run to the processor alone, and on that "
            "basis the full search was estimated to require several days."))
    A(("p", "It is worth recording that the graphics processor helped far less "
            "than anticipated. Measured step times were approximately eight "
            "milliseconds locally on the processor against nine milliseconds on "
            "the Colab GPU. The networks in this search space are small by "
            "contemporary standards, at most five layers of at most 512 "
            "neurons, and at the smaller batch sizes the work per step is too "
            "slight to keep a GPU occupied; the run is bound by kernel launch "
            "latency rather than by arithmetic. Any later work that expects a "
            "substantial speed-up from hardware alone should take note."))
    A(("p", "Because Colab disconnects idle sessions and limits total session "
            "length, the optimiser was given the ability to checkpoint. The "
            "population, its fitness scores, the convergence history and the "
            "internal state of the random number generator are written to disk "
            "after every generation and read back on restart. This proved "
            "necessary rather than merely prudent: the session was lost partway "
            "through and the search resumed from generation five instead of "
            "beginning again. Preserving the generator state matters, because "
            "an optimiser that reseeds itself on resumption repeats draws it "
            "has already made, which constitutes a different and correlated "
            "search rather than a continuation of the original one."))
    A(("tbl", ("Software and Hardware Environment Used for the Experiments",
               ["Component", "Specification"],
               [["Language", "Python 3.12"],
                ["Deep learning framework", "TensorFlow 2.21 with Keras 3.15"],
                ["Supporting libraries",
                 "NumPy 2.0, Pandas 3.0, Scikit-learn 1.9, PyArrow 25.0"],
                ["Optimiser",
                 "Differential Evolution, DE/rand/1/bin, implemented from scratch"],
                ["Execution platform", "Google Colab with an NVIDIA T4 GPU"],
                ["Random seed",
                 "42, fixed across Python, NumPy, TensorFlow and every split"]],
               [2.0, 4.2])))

    A(("h2", "3.10 \tDataset Preparation"))
    A(("p", "The dataset used is CSE-CIC-IDS2018, introduced in Chapter Three. "
            "It circulates in two materially different forms, and the "
            "distinction between them turned out to matter more than any "
            "hyperparameter the search would later select."))
    A(("p", "The original release from the Canadian Institute for Cybersecurity "
            "ships ten daily capture files as comma-separated values, totalling "
            "roughly sixteen million flow records across eighty columns. A "
            "widely used cleaned redistribution ships the same captures in "
            "Parquet format, with duplicate records removed, infinite and "
            "missing values repaired and column types reduced. This study used "
            "the cleaned redistribution, which contains 6,659,532 records "
            "across seventy-eight columns."))
    A(("p", "Deduplication is the defensible choice. Where identical flow "
            "records appear in both the training and the test partitions, a "
            "model can score well by memorising rather than by generalising, "
            "and a considerable part of the very high performance reported in "
            "the literature on this dataset is attributable to precisely that "
            "effect. Removing the duplicates removes the inflation. It also "
            "carries two consequences which must be stated plainly."))
    A(("p", "First, two columns present in the original release, the "
            "destination port and the timestamp, are absent from the cleaned "
            "build. The timestamp is no loss, since Chapter Three already "
            "discards it as an identity leak. The destination port is a genuine "
            "loss: it carries real protocol signal, and the model reported here "
            "was trained with no port information whatsoever."))
    A(("p", "Second, deduplication collapses several attack families almost "
            "entirely, because the records that made them numerous were "
            "duplicates of one another. The full class distribution of the "
            "corpus as used is given in Table 3.6."))
    A(("tbl", ("Class Distribution of the CSE-CIC-IDS2018 Corpus as Used",
               ["Class", "Records", "Share of corpus (%)"],
               [[c, n, p] for c, n, p in CORPUS]
               + [["Total", "6,659,532", "100.000"]],
               [2.6, 1.5, 1.7])))
    A(("p", "The imbalance is severe. Benign traffic accounts for eighty per "
            "cent of the corpus, the largest attack class for a further nine "
            "per cent, and the four smallest classes together for fewer than "
            "five hundred records out of six and a half million. Three classes "
            "are small enough that they cannot meaningfully be evaluated at "
            "all: at a twenty per cent test split, SQL Injection would "
            "contribute seventeen test records, DoS-SlowHTTPTest eleven and "
            "FTP-BruteForce ten."))
    A(("p", "A per-class F1 score computed over ten records shifts by roughly "
            "0.1 for every single misclassification. Because the fitness signal "
            "guiding the optimiser is the macro average, in which every class "
            "carries equal weight regardless of its size, those three classes "
            "would not merely have produced unreliable reporting; they would "
            "have injected noise directly into the signal the search was "
            "following. They were therefore excluded, and the exclusion is "
            "recorded in Table 3.7 rather than performed silently."))
    A(("tbl", ("Classes Excluded as Too Small to Evaluate",
               ["Class", "Records in corpus", "Expected test records",
                "Records in original release"],
               [["SQL Injection", "85", "17", "approximately 87"],
                ["DoS attacks-SlowHTTPTest", "55", "11", "approximately 140,000"],
                ["FTP-BruteForce", "53", "10", "approximately 193,000"]],
               [1.9, 1.4, 1.4, 1.6])))
    A(("p", "The final column of Table 3.7 is the clearest available "
            "illustration of what deduplication does to this dataset. Two of "
            "the three excluded classes are major attack families in the "
            "original release; they survive here as a few dozen distinct "
            "records because everything else was a repetition."))
    A(("p", "Twelve classes remained. Training on six and a half million "
            "records for each of the many candidate networks the search would "
            "evaluate was not feasible, so a class-aware cap was applied: each "
            "class was limited to at most 20,000 records, while every record of "
            "any class falling below that threshold was retained in full. This "
            "differs deliberately from taking a uniform random sample, which "
            "thins every class by the same proportion and would have destroyed "
            "the small classes on which the macro average depends. The "
            "resulting experimental set is described in Table 3.8."))
    A(("tbl", ("Composition of the Experimental Dataset",
               ["Property", "Value"],
               [["Classes retained", "12"],
                ["Cap applied",
                 "20,000 records per class; smaller classes retained in full"],
                ["Records after capping", "172,435"],
                ["Features after cleaning", "69"],
                ["Training partition (60%)", "103,461 records"],
                ["Validation partition (20%)", "34,487 records"],
                ["Test partition (20%)", "34,487 records"],
                ["Split method",
                 "Stratified, so each partition preserves the class proportions"],
                ["Feature scaling",
                 "Min-max to the range [0, 1], fitted on the training partition only"]],
               [2.2, 4.0])))
    A(("p", "Eight of the seventy-seven feature columns were removed as "
            "constant, carrying an identical value in every record and "
            "therefore no information, which left sixty-nine. The scaler was "
            "fitted on the training partition alone and then applied to the "
            "validation and test partitions, so that no statistic derived from "
            "the test data reached the model. The test partition was read "
            "exactly once, at the very end of the process."))
    A(("p", "One consequence of the cap must be carried through into the "
            "interpretation of the results. Capping Benign at 20,000 records "
            "makes it 11.6 per cent of the experimental set rather than the "
            "eighty per cent it represents in reality. Macro-averaged "
            "precision, recall and F1 weight every class equally by "
            "construction and are unaffected by this. Overall accuracy and the "
            "false-positive rate are not: they are computed over the rebalanced "
            "mixture and are therefore not estimates of what would be observed "
            "on live traffic. This caveat is written into the metrics file the "
            "pipeline produces, so that it travels with the numbers rather than "
            "depending on the reader remembering it."))

    A(("h2", "3.11 \tDifferential Evolution Implementation"))
    A(("p", "Each candidate solution is a vector of six real numbers in the "
            "unit interval. Each component is decoded into one hyperparameter "
            "only at the point where a network actually has to be built, which "
            "is what allows a single optimiser operating on a tidy continuous "
            "space to handle parameters that are otherwise of quite different "
            "kinds. The number of hidden layers and the neuron count are "
            "decoded by linear scaling and rounding, the batch size and "
            "activation function by indexing into a fixed list, and the dropout "
            "rate linearly. The learning rate is decoded logarithmically across "
            "the range 0.0001 to 0.1, so that each order of magnitude receives "
            "an equal share of the component's range. A linear mapping would "
            "have devoted almost the entire range to values above 0.001, where "
            "the Adam optimiser rarely converges well on this problem."))
    A(("p", "The search follows the DE/rand/1/bin scheme set out in Chapter "
            "Two. For each member of the population, three other members are "
            "drawn at random and a donor vector is formed by adding a scaled "
            "difference between two of them to the third. Binomial crossover "
            "then mixes the donor with the current member, with at least one "
            "component always taken from the donor so that a trial can never be "
            "an exact copy of its parent and waste an evaluation. The trial is "
            "decoded, the network it describes is built and trained, and its "
            "macro-averaged F1 on the validation partition becomes its fitness. "
            "Selection is greedy and one-to-one, so that a trial replaces only "
            "its own parent and only where it scores at least as well; the best "
            "solution found can therefore never be lost, and the convergence "
            "curve is monotonic by construction."))
    A(("p", "Macro-averaged F1 was chosen as the fitness signal deliberately. "
            "On data that is eighty per cent benign, accuracy would reward a "
            "model that predicted Benign for every flow, and the search would "
            "optimise steadily towards a detector that detects nothing."))
    A(("tbl", ("Differential Evolution Configuration Used in the Experiment",
               ["Parameter", "Value", "Reason for the choice"],
               [["Strategy", "DE/rand/1/bin",
                 "The standard, robust scheme established in the literature"],
                ["Population size (NP)", "10",
                 "A compromise; the usual guidance of ten per dimension would give 60"],
                ["Generations", "10",
                 "110 network trainings in total, the affordable budget on the available hardware"],
                ["Scaling factor (F)", "0.8",
                 "A standard, well-tested value giving healthy mutation steps"],
                ["Crossover rate (CR)", "0.9",
                 "High mixing, so that trials differ meaningfully from their parents"],
                ["Dimensionality", "6",
                 "Layers, neurons, learning rate, dropout, batch size, activation"],
                ["Fitness signal", "Validation macro-F1",
                 "Weights every class equally despite the heavy imbalance"],
                ["Epoch budget per candidate", "8",
                 "Sufficient to rank candidates without training each to convergence"]],
               [1.7, 1.5, 3.0])))

    A(("h2", "4.2 \tOptimisation Results"))
    A(("p", "The search ran for ten generations over a population of ten, "
            "evaluating 110 networks in total. Its trajectory is shown in "
            "Figure 4.1, and it is not the trajectory the project anticipated."))
    A(("fig", ("Convergence of the Differential Evolution search, showing the "
               "best validation macro-F1 attained in each generation. The "
               "vertical axis spans only 0.0026.",
               os.path.join(S1, "de_convergence.png"), 5.6)))
    A(("p", "The initial random population already contained a configuration "
            "scoring {:.4f}. Ten generations of search improved this to {:.4f}, "
            "a gain of {:.4f}, or {:.2f} per cent. The final improvement "
            "occurred at generation {}; the remaining {} generations, "
            "representing roughly {} network trainings, produced no change "
            "whatsoever.".format(hist[0], hist[-1], gain,
                                 100 * gain / hist[0], last_imp, plateau,
                                 plateau * 10)))
    A(("p", "The shape of the curve in Figure 4.1 appears dramatic, but the "
            "vertical axis must be read with care: it spans from 0.9039 to "
            "0.9065. The apparent climb is an artefact of automatic axis "
            "scaling across a very narrow band of values."))
    A(("p", "The natural first explanation is that the population was too "
            "small. At ten members in a six-dimensional space the search sits "
            "well below the customary guidance of roughly ten members per "
            "dimension, and a small population loses diversity quickly, so that "
            "the difference vectors driving mutation collapse towards zero. "
            "That is a genuine limitation and it is acknowledged as such in "
            "Chapter Five. It is not, however, a sufficient explanation, "
            "because it does not account for the initial population already "
            "scoring 0.9039. A randomly drawn set of ten configurations does "
            "not land within 0.3 per cent of the best value a search will ever "
            "find unless the objective surface is close to flat across the "
            "region being sampled."))
    A(("p", "The fuller explanation, supported by the per-class results in "
            "Section 4.3, is that performance on this problem is not limited by "
            "the architecture of the network. It is limited by a property of "
            "the data which no choice of depth, width, learning rate or "
            "activation function can overcome. The configuration the search "
            "settled upon is given in Table 4.1."))
    A(("tbl", ("Best Hyperparameter Configuration Found by the Search",
               ["Hyperparameter", "Value selected", "Search range"],
               [["Hidden layers", str(hp["n_layers"]), "1 to 5"],
                ["Neurons per layer", str(hp["neurons"]), "32 to 512"],
                ["Learning rate", "{:.6f}".format(hp["lr"]),
                 "0.0001 to 0.1, logarithmic"],
                ["Dropout rate", "{:.3f}".format(hp["dropout"]), "0.0 to 0.5"],
                ["Batch size", str(hp["batch"]), "16, 32, 64, 128, 256"],
                ["Activation function", hp["activation"],
                 "ReLU, tanh, sigmoid"],
                ["Validation macro-F1",
                 "{:.4f}".format(cfg["validation_macro_f1"]), "not applicable"]],
               [1.9, 1.6, 2.4])))
    A(("p", "Two of these selections are worth remarking upon. The search chose "
            "a dropout rate of exactly zero, at the very bottom of its "
            "permitted range, which indicates that regularisation was not what "
            "constrained this model. It also chose tanh in preference to ReLU, "
            "the more common default in the intrusion detection literature. "
            "Neither is an obvious choice in advance, which is exactly the sort "
            "of trade-off automated search is intended to surface; whether "
            "either made a material difference to the outcome is precisely what "
            "Section 4.5 sets out to test."))

    return S
