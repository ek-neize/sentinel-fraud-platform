"""Local version of Step 6: train champion-style XGBoost + LR baseline, evaluate on test.

Examples
  python scripts/evaluate_local.py --mock --bootstrap 200     # quick check (CI)
  python scripts/evaluate_local.py --out docs/results_local.md

Uses the same code as notebooks/databricks/06_evaluate.py. Databricks numbers are the
reference; this is for development and for checking the pipeline end to end.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_local import load_frame  # noqa: E402

from sentinel.common.config import load_assumptions  # noqa: E402
from sentinel.evaluation.report import run_business_evaluation  # noqa: E402
from sentinel.models.training import ExperimentConfig, run_experiments  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mock", action="store_true")
    p.add_argument("--bootstrap", type=int, default=None, help="iterations (default: config)")
    p.add_argument("--out", help="write the Markdown report here")
    args = p.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = load_assumptions()
    df = load_frame(args.mock)
    train, valid, test = (
        df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")
    )
    configs = [
        ExperimentConfig("champion", "xgb", "full", {"max_depth": 6, "learning_rate": 0.05}),
        ExperimentConfig("baseline", "logreg", "full", {"C": 0.1}),
    ]
    _, models = run_experiments(train, valid, configs)
    scores = {
        name: {
            split: m["pipeline"].predict_proba(part)[:, 1]
            for split, part in (("valid", valid), ("test", test))
        }
        for name, m in models.items()
    }
    result = run_business_evaluation(valid, test, scores, cfg, bootstrap_n=args.bootstrap)
    print(result.markdown)
    if args.out:
        Path(args.out).write_text(result.markdown, encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
