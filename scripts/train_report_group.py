"""Train one of the four report networks with explicit annotation-group exclusions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from solarfil.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--role", required=True, choices=("n", "w", "record", "union"))
    parser.add_argument("--fold", required=True, type=int, choices=(0, 2, 3, 4))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    ds = Dataset()
    stems = ds.train_stems
    folds = json.loads((ROOT / "configs/report_group_folds.json").read_text())["group"]
    if set(folds) != set(stems) or len(stems) != 707 or set(folds.values()) != {0, 2, 3, 4}:
        raise ValueError("The report requires the complete four-block manifest")
    train = [s for s in stems if folds[s] != args.fold]
    held = [s for s in stems if folds[s] == args.fold]
    if {ds.observations[s].group for s in train} & {ds.observations[s].group for s in held}:
        raise ValueError("An annotation group crosses the training boundary")
    identity = dict(role=args.role, fold=args.fold, training_stems=train, heldout_stems=held)
    if args.validate_only:
        print(json.dumps(dict(role=args.role, fold=args.fold, training_frames=len(train),
                              heldout_frames=len(held), model_constructed=False)))
        return

    def report_folds(scheme):
        if scheme != "group":
            raise ValueError("The report training entrypoint requires group folds")
        return dict(folds)

    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / f"report_{args.role}_fold{args.fold}.json"
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError("This output belongs to a different training pool")
    receipt.write_text(json.dumps(identity, indent=2) + "\n")
    if args.role in ("n", "w"):
        import solarfil.train as trainer
        trainer.load_folds = report_folds
        cfg = trainer.load_config(ROOT / f"configs/report_group_{args.role}.json")
        trainer.run(cfg, args.fold, args.out, args.cache)
    else:
        import train_instances as trainer
        trainer.load_folds = report_folds
        sys.argv = ["train_instances.py", "--fold", str(args.fold), "--scheme", "group",
                    "--labels", args.role, "--out", str(args.out), "--cache", str(args.cache)]
        trainer.main()


if __name__ == "__main__":
    main()
