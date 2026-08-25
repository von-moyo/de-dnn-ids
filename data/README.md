# Dataset

The CSE-CIC-IDS2018 capture files are **not committed** to this repository.

The pipeline reads every `.csv`, `.parquet` and `.zip` it finds in this
directory. If an archive has already been extracted alongside itself
(`X.parquet` next to `X.parquet.zip`), the archive is skipped rather than
loaded twice, so it is safe to leave the zips in place.

## Which build to use

Read [Dataset build matters](../README.md#dataset-build-matters) first — the
choice changes your results substantially and is not reversible halfway through.
In short:

| | Raw CIC release | Cleaned redistribution |
|---|---|---|
| Format | 10 daily CSVs, ~6.5 GB | 10 daily parquet, ~700 MB |
| Rows | ~16,000,000 | 6,659,532 |
| Columns | 80 | 78 (no `Dst Port`, no `Timestamp`) |
| Duplicates | present | removed |
| Comparable to published work | yes | no — results are lower and honest |

**Do not mix them.** Filling gaps in one build from the other means preprocessing
differs by capture day, which is indefensible in a writeup.

## Obtaining the data

**Raw CIC release** — hosted on AWS as a Registry of Open Data bucket:

```bash
aws s3 sync --no-sign-request --region eu-west-3 \
    "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" ./data/
```

Also mirrored on Kaggle as `solarmainframe/ids-intrusion-csv`.

**Cleaned parquet redistribution** — `dhoogla/csecicids2018`:

```bash
pip install kaggle          # token at ~/.kaggle/kaggle.json
kaggle datasets download -d dhoogla/csecicids2018 -p ./data --unzip
```

You do not need all ten files. The pipeline concatenates whatever it finds and
derives the class list from the labels present — but a missing day means a
missing attack family, so partial coverage must be stated in your results.

## The files

Counts below are measured on the **cleaned parquet** build, all ten days.

| File | Rows | Attacks present |
|---|---:|---|
| `Bruteforce_Wednesday_14_02_2018` | 619,346 | SSH-Bruteforce 94,048 · FTP-BruteForce 53 |
| `DoS1-Thursday-15-02-2018` | 794,812 | DoS-GoldenEye 41,406 · DoS-Slowloris 9,908 |
| `DoS2-Friday-16-02-2018` | 591,873 | DoS-Hulk 145,199 · DoS-SlowHTTPTest 55 |
| `DDoS1-Tuesday-20-02-2018` | 954,846 | DDoS-LOIC-HTTP 575,364 |
| `DDoS2-Wednesday-21-02-2018` | 561,396 | DDOS-HOIC 198,861 · DDOS-LOIC-UDP 1,730 |
| `Web1-Thursday-22-02-2018` | 830,224 | Brute Force -Web 228 · -XSS 79 · SQL Injection 34 |
| `Web2-Friday-23-02-2018` | 829,405 | Brute Force -Web 340 · -XSS 150 · SQL Injection 51 |
| `Infil1-Wednesday-28-02-2018` | 456,873 | Infilteration 56,449 |
| `Infil2-Thursday-01-03-2018` | 249,170 | Infilteration 62,034 |
| `Botnet-Friday-02-03-2018` | 771,587 | Bot 144,535 |
| **Total** | **6,659,532** | 15 classes, 80.02% Benign |

Two label quirks come from CIC itself, not from this pipeline: `Infilteration`
is misspelled, and casing is inconsistent between `DDoS attacks-LOIC-HTTP` and
`DDOS attack-HOIC`. They are distinct strings and are correctly treated as
distinct classes.

### Classes too small to score

On the deduplicated build, three classes fall below the ~30 test rows needed for
a meaningful per-class F1:

| Class | Corpus rows | Test rows @ 20% |
|---|---:|---:|
| SQL Injection | 85 | 17 |
| DoS attacks-SlowHTTPTest | 55 | 11 |
| FTP-BruteForce | 53 | 10 |

Because `fitness()` optimises the **macro** average, these contribute noise to
the signal DE searches on rather than merely making the report unreliable. Pass
`--min_class_rows 200` to exclude them, leaving 12 scoreable classes.

## Known defects in the raw CSVs

The pipeline handles all of these automatically; they are listed so you know why
the preprocessing code looks the way it does. The cleaned parquet build has
already repaired every one of them.

- **Infinities.** `Flow Bytes/s` and `Flow Packets/s` divide by a duration that
  is zero for single-packet flows, producing `Infinity`.
- **Embedded header rows.** Several files repeat the header line partway
  through, so numeric columns contain the literal string `"Flow Duration"`.
- **Inconsistent column names.** Some files ship columns with a leading space
  (`" Flow Duration"`), which breaks a naive `pd.concat`.
- **Extra columns on 20-02.** That day alone carries `Flow ID`, `Src IP`,
  `Src Port` and `Dst IP`. They are dropped before the `dropna()`, so the NaNs
  the concat introduces for the other nine days do not delete those rows.
- **Dead columns.** Around ten features (`Bwd PSH Flags`, `Fwd URG Flags`,
  `Bwd Avg Bytes/Bulk`, …) are zero for every row in every file.
- **Identity columns.** `Src IP`, `Dst IP`, `Flow ID` and `Timestamp` leak the
  answer, because the attacks came from a fixed set of hosts at known times.
- **Severe imbalance.** Benign is 80% of the corpus, while `SQL Injection` has
  85 flows and `FTP-BruteForce` 53.

## Windows note

Reads are retried with backoff: on-access antivirus scanning can hold a
transient lock on files this large and surface it as `PermissionError`. Without
the retry a lock can kill a DE search hours after it started.
