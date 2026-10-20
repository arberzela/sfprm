"""Build the Mini-ARC dataset in this repo's puzzle format.

Usage: python -m dataset.build_miniarc_dataset [--num-aug 100]
Input:  raw_data/miniarc_miniarc_{challenges,solutions}.json (miniarc_to_kaggle.py)
Output: data/miniarc-aug-<N>/{train,test}/... + identifiers.json + test_puzzles.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset.build_arc_dataset as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-aug", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    B.ARCMaxGridSize = 5                      # 5x5 grids -> seq_len 25
    cfg = B.DataProcessConfig(
        input_file_prefix="raw_data/miniarc",
        output_dir=a.out or f"data/miniarc-aug-{a.num_aug}",
        subsets=["miniarc"],
        test_set_name="miniarc",
        seed=a.seed,
        num_aug=a.num_aug,
    )
    B.convert_dataset(cfg)
    print(f"BUILD_MINIARC_DONE -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
