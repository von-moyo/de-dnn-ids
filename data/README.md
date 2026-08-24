# Dataset

The CSE-CIC-IDS2018 CSV files are **not committed** to this repository — the
full corpus is roughly 6.5 GB across 10 daily capture files.

## Obtaining the data

1. Dataset home page: <https://www.unb.ca/cic/datasets/ids-2018.html>
2. The files are hosted on AWS S3 as a Registry of Open Data bucket:

```bash
aws s3 sync --no-sign-request --region eu-west-3 \
    "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" ./data/
```

3. Place the resulting `*.csv` files directly in this directory. The pipeline
   reads every `.csv` it finds here.

## The files

Each CSV is one capture day with a distinct attack scenario:

| File | Attacks present |
|---|---|
| `Wednesday-14-02-2018` | FTP-BruteForce, SSH-BruteForce |
| `Thursday-15-02-2018`  | DoS-GoldenEye, DoS-Slowloris |
| `Friday-16-02-2018`    | DoS-SlowHTTPTest, DoS-Hulk |
| `Thursday-22-02-2018`  | Brute Force -Web, Brute Force -XSS, SQL Injection |
| `Friday-23-02-2018`    | Brute Force -Web, Brute Force -XSS, SQL Injection |
| `Wednesday-28-02-2018` | Infiltration |
| `Thursday-01-03-2018`  | Infiltration |
| `Friday-02-03-2018`    | Bot |
| `Friday-02-03-2018`    | DDoS-LOIC-HTTP, DDoS-HOIC, DDoS-LOIC-UDP |

You do not need all of them. Any subset works — the pipeline concatenates
whatever CSVs are present and derives the class list from the labels it finds.

## Known defects in these CSVs

The pipeline handles all of these automatically; they are listed so you know
why the preprocessing code looks the way it does.

- **Infinities.** `Flow Bytes/s` and `Flow Packets/s` divide by a duration that
  is zero for single-packet flows, producing `Infinity`.
- **Embedded header rows.** Several files repeat the header line partway
  through, so numeric columns contain the literal string `"Flow Duration"`.
- **Inconsistent column names.** Some files ship columns with a leading space
  (`" Flow Duration"`), which breaks a naive `pd.concat`.
- **Dead columns.** Around eight features (`Bwd PSH Flags`, `Fwd URG Flags`,
  `Bwd Avg Bytes/Bulk`, …) are zero for every row in every file.
- **Identity columns.** `Src IP`, `Dst IP`, `Flow ID` and `Timestamp` leak the
  answer, because the attacks came from a fixed set of hosts at known times.
- **Severe imbalance.** Benign is ~83% of the corpus, while `SQL Injection` has
  roughly 87 flows in total and `Brute Force -XSS` roughly 230.
