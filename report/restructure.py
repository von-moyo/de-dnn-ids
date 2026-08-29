"""Move the implementation sections from Chapter 4 into Chapter 3.

The approved chapter plan is:
    3. Methodology & Implementation   - what I did and how I built it
    4. Results & Discussion           - what happened when I tested it

Chapter 4 was drafted as "Implementation and Results", so three sections sit in
the wrong chapter. This renumbers the section headings and every in-text
Table / Figure / Section cross-reference to match the plan. Run once.

    old 4.2 Implementation Environment  -> 3.9
    old 4.3 Dataset Preparation         -> 3.10
    old 4.4 DE Implementation           -> 3.11
    old 4.5 Optimisation Results        -> 4.2
    old 4.6 Performance of Final Model  -> 4.3
    old 4.7 Effect of Sampling Ratio    -> 4.4
    old 4.8 Baseline Comparison         -> 4.5
    old 4.9 Discussion                  -> 4.6
"""
import io
import re

# (file, [(exact old, exact new)], {ref old: ref new})
JOBS = [
    ("content_ch45.py",
     [('"4.2 \\tImplementation Environment and Tools"',
       '"3.9 \\tImplementation Environment and Tools"'),
      ('"4.3 \\tDataset Preparation"', '"3.10 \\tDataset Preparation"'),
      ('"4.4 \\tDifferential Evolution Implementation"',
       '"3.11 \\tDifferential Evolution Implementation"'),
      ('"4.5 \\tOptimisation Results"', '"4.2 \\tOptimisation Results"')],
     {"Table 4.2": "Table 3.6", "Table 4.3": "Table 3.7",
      "Table 4.4": "Table 3.8", "Table 4.6": "Table 4.1"}),

    ("content_ch6.py",
     [('"4.6 \\tPerformance of the Final Model"',
       '"4.3 \\tPerformance of the Final Model"'),
      ('"4.7 \\tEffect of Sampling Ratio on Detection Behaviour"',
       '"4.4 \\tEffect of Sampling Ratio on Detection Behaviour"')],
     {"Table 4.6": "Table 4.1", "Table 4.7": "Table 4.2",
      "Table 4.8": "Table 4.3", "Table 4.9": "Table 4.4",
      "Table 4.10": "Table 4.5", "Table 4.11": "Table 4.6",
      "Section 4.4": "Section 3.11", "Section 4.5": "Section 4.2",
      "Section 4.6": "Section 4.3", "Section 4.7": "Section 4.4"}),

    ("content_ch5.py",
     [('"4.8 \\tComparison with a Manually Configured Baseline"',
       '"4.5 \\tComparison with a Manually Configured Baseline"'),
      ('"4.9 \\tDiscussion"', '"4.6 \\tDiscussion"')],
     {"Table 4.12": "Table 4.7", "Table 4.13": "Table 4.8",
      "Section 4.5": "Section 4.2", "Section 4.6": "Section 4.3",
      "Section 4.7": "Section 4.4"}),
]

for path, exacts, refs in JOBS:
    s = io.open(path, encoding="utf8").read()
    n_exact = 0
    for old, new in exacts:
        if old in s:
            s = s.replace(old, new)
            n_exact += 1
        else:
            print("  !! %s: heading not found -> %s" % (path, old))

    # Single regex pass so replacements cannot cascade into one another
    # (a naive sequential replace would turn "Table 4.10" into "Table 4.50").
    if refs:
        pat = re.compile(r"\b(" + "|".join(
            re.escape(k) for k in sorted(refs, key=len, reverse=True)) + r")\b")
        s, n_ref = pat.subn(lambda m: refs[m.group(1)], s)
    else:
        n_ref = 0
    io.open(path, "w", encoding="utf8").write(s)
    print("%-18s %d heading(s), %d cross-reference(s)" % (path, n_exact, n_ref))

print("\nremaining bare 'Table 4.' / 'Section 4.' references, for eyeballing:")
for path, _, _ in JOBS:
    s = io.open(path, encoding="utf8").read()
    for m in sorted(set(re.findall(r"(?:Table|Figure|Section) [34]\.\d+", s))):
        print("   %-18s %s" % (path, m))
