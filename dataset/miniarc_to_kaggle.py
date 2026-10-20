"""Convert Mini-ARC (Kim et al., KSB21ST/MINI-ARC) into the consolidated
challenges/solutions format that `build_arc_dataset.py` expects.

Mini-ARC: 149 tasks, EVERY grid exactly 5x5 (1,354 grids,
colors 0-9, 2-8 demo pairs per task, exactly 1 test pair per task, 0 malformed).
Source: a clone of github.com/KSB21ST/MINI-ARC (data/MiniARC/*.json).
One subset "miniarc", a task's "train"
examples -> TRAIN split, its "test" example -> TEST split, same puzzle_identifier
(transductive; the per-puzzle embedding carries the transformation).

Task ids = filename stems, sanitized: build_arc_dataset appends aug tags with the
"|||" separator, so ids must not contain it (none do; asserted anyway).
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_data/MINI-ARC/data/MiniARC")
    ap.add_argument("--out", default="raw_data/miniarc")
    a = ap.parse_args()
    SRC, OUT = a.src, a.out
    challenges, solutions = {}, {}
    for path in sorted(glob.glob(f"{SRC}/*.json")):
        pid = os.path.splitext(os.path.basename(path))[0]
        assert "|||" not in pid, f"id contains PuzzleIdSeparator: {pid}"
        d = json.load(open(path))
        assert pid not in challenges, f"duplicate task id {pid}"
        for split in ("train", "test"):
            for ex in d[split]:
                for k in ("input", "output"):
                    g = ex[k]
                    assert len(g) == 5 and all(len(r) == 5 for r in g), \
                        f"{pid}: non-5x5 grid in {split}/{k}"
        challenges[pid] = {
            "train": [{"input": e["input"], "output": e["output"]}
                      for e in d["train"]],
            "test": [{"input": e["input"]} for e in d["test"]],
        }
        solutions[pid] = [e["output"] for e in d["test"]]

    json.dump(challenges, open(f"{OUT}_miniarc_challenges.json", "w"))
    json.dump(solutions, open(f"{OUT}_miniarc_solutions.json", "w"))
    ntr = sum(len(v["train"]) for v in challenges.values())
    nte = sum(len(v["test"]) for v in challenges.values())
    print(f"tasks {len(challenges)}   demo pairs {ntr}   test pairs {nte}")
    print(f"wrote {OUT}_miniarc_challenges.json / _solutions.json")


if __name__ == "__main__":
    main()
