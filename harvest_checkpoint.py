"""Write best_config.json from a DE checkpoint without finishing the search.

differential_evolution() writes best_config.json only when it returns, so a
search stopped at generation 7 of 12 leaves every training it did stranded in
the checkpoint with nothing downstream able to use it. Colab sessions time out,
and a 260-evaluation search does not reliably fit in one.

This reads the checkpoint, takes the best vector in the saved population, and
writes the same best_config.json a completed run would have written -- so
`--load_config` can pick it up for stage 2.

    python harvest_checkpoint.py --checkpoint /content/drive/MyDrive/de-dnn-ids/de_checkpoint.json \
        --out_dir /content/drive/MyDrive/de-dnn-ids/stage1

The search is a greedy one-to-one replacement, so the population's best only
ever improves: an incomplete search yields a genuinely valid configuration, just
a less-searched one than the full budget would have found. Say which generation
it came from when reporting it.
"""
import argparse
import json
import os

from de_dnn_ids import decode


def main():
    ap = argparse.ArgumentParser(
        description="Turn a DE checkpoint into a best_config.json that "
                    "--load_config can consume.")
    ap.add_argument("--checkpoint", required=True,
                    help="the --checkpoint file the search was writing")
    ap.add_argument("--out_dir", required=True,
                    help="where to write best_config.json")
    ap.add_argument("--name", default="best_config.json")
    args = ap.parse_args()

    with open(args.checkpoint) as fh:
        state = json.load(fh)

    pop = state["pop"]
    scores = state["scores"]
    generation = state.get("generation")
    history = state.get("history", [])
    if len(pop) != len(scores):
        raise SystemExit("checkpoint is inconsistent: %d vectors, %d scores"
                         % (len(pop), len(scores)))

    best = max(range(len(scores)), key=lambda i: scores[i])
    vector = pop[best]
    hp = decode(vector)

    print("checkpoint: generation %s, population %d" % (generation, len(pop)))
    print("best macro-F1 in population: %.4f" % scores[best])
    print("hyperparameters:", hp)
    if history:
        print("convergence: " + " ".join("%.4f" % h for h in history))
        if len(history) > 3 and history[-1] == history[-4]:
            print("  (flat for the last 3 generations -- little is being "
                  "gained by continuing)")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.name)
    with open(out, "w") as fh:
        json.dump({
            "hyperparameters": hp,
            "vector": vector,
            "validation_macro_f1": scores[best],
            "convergence_history": history,
            "settings": {
                "harvested_from": os.path.abspath(args.checkpoint),
                "generation": generation,
                "pop_size": len(pop),
                "note": "Harvested from an unfinished DE search. The best "
                        "vector in the population at this generation, which "
                        "greedy selection guarantees is the best seen so far. "
                        "Report the generation it stopped at.",
            },
        }, fh, indent=2)
    print("\n[out]", out)


if __name__ == "__main__":
    main()
