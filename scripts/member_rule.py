"""Two dense runs or blends judged by the release rule (`solarfil.rule`), each through its own decision layer.

Each brings its own candidate bank (`build_bank.py --splits`) and, with `--inst`, its own columns of the detector
sets (`instance_study.py --build`, side by side in `<sets>_split/`; `member_test.sh` makes both). The hurdle scorer
is cross-fitted on each with the same partitions, both select at each tau, and the frame bootstrap pairs the two on
the same frames, so the gain is the tested run's over the base run.

`--slots` gives the tested run the draw-augmented scorer instead (`draw_study.py`: one slot per recipe, each node once
per combination of draws, the utilities averaged), while the base keeps its sets side by side (`--base-inst`).
`--first-partition` and `--first-group` start the partitions at fresh seeds, for a test whose parts were chosen on
the usual ones.

Usage: member_rule.py --base ens_s0_win --run ens_win_wins2 --inst inst_maskrcnn,inst_maskrcnn_grp,inst_maskrcnn_union
       member_rule.py --base e3_effv2s_n_long_lean_grp --run e3_effv2s_n_long_lean_grp_hierem   (descriptors only)
       member_rule.py --base ens_s0_win --base-inst inst_maskrcnn,inst_maskrcnn_grp,inst_maskrcnn_union --run <blend>
           --slots inst_maskrcnn,inst_maskrcnn_grp inst_maskrcnn_union,inst_maskrcnn_union_grp
           --first-partition 10 --first-group 5
       [--partitions 5] [--group-partitions 5] [--out <json>] [--rows <npz>]   (writes <run>/member_rule_vs_<base>.json)
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from decide import crossfit, crossfit_draws, sweep
from solarfil.bank import load_banks
from solarfil.paths import RUNS
from solarfil.rule import TAUS, partitions, release_rule


def banks_with_columns(run: Path, sets_dir: str | None) -> list:
    """The run's banks, with the detector sets' columns after the descriptors when `sets_dir` names them."""
    banks = load_banks(run / "bank_split")
    for b in banks if sets_dir else []:
        A = np.load(run / sets_dir / f"{b.stem}.npz")["A"]
        assert len(A) == b.tree.n, (run.name, b.stem, len(A), b.tree.n)
        b.X = np.concatenate([b.X, A], 1)
    return banks


def main():
    """Cross-fit both arms on the same partitions, sweep tau and print and save the release rule's verdict."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the release's blend (a run folder)")
    ap.add_argument("--run", required=True, help="the blend under test")
    ap.add_argument("--inst", default=None,
                    help="the detector sets, comma-separated, as instance_study.py names them (none: descriptors only)")
    ap.add_argument("--base-inst", default=None, help="the base's sets side by side (default: --inst)")
    ap.add_argument("--slots", nargs="+", default=None,
                    help="the tested run's draw slots, one argument per slot, its draws comma-separated")
    ap.add_argument("--first-partition", type=int, default=0)
    ap.add_argument("--first-group", type=int, default=0)
    ap.add_argument("--partitions", type=int, default=5)
    ap.add_argument("--group-partitions", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="the JSON (default <run>/member_rule_vs_<base>.json)")
    ap.add_argument("--rows", type=Path, default=None,
                    help="also save the per-frame totals (an .npz, keys scheme|rep|arm|tau), for side readings")
    a = ap.parse_args()
    def sets_dir(inst):
        """The folder of the detector sets' combined columns, `<a>+<b>_split`, or None without sets."""
        return "+".join(inst.split(",")) + "_split" if inst else None

    arms = {"base": banks_with_columns(RUNS / a.base, sets_dir(a.base_inst or a.inst)),
            "agree": banks_with_columns(RUNS / a.run, None if a.slots else sets_dir(a.inst))}
    stems = [b.stem for b in arms["base"]]
    assert stems == [b.stem for b in arms["agree"]], "the two blends' banks hold different frames"
    draws = None
    if a.slots:  # each combination of draws, one per slot, after the descriptors (draw_study.py)
        slots = [s.split(",") for s in a.slots]
        cols = {n: [np.load(RUNS / a.run / f"{n}_split" / f"{b.stem}.npz")["A"] for b in arms["agree"]]
                for n in sum(slots, [])}
        draws = [[np.concatenate([b.X, *(cols[n][i] for n in c)], 1) for i, b in enumerate(arms["agree"])]
                 for c in itertools.product(*slots)]
    rows = {}  # (scheme, rep, arm) -> {tau: (frames, 4) totals}
    for scheme, rep, folds in partitions(stems, a.partitions, a.group_partitions, first=a.first_partition,
                                         first_group=a.first_group):
        for arm, banks in arms.items():
            scores = (crossfit_draws(banks, draws, folds) if arm == "agree" and draws is not None
                      else crossfit(banks, folds, "hurdle"))
            swept = sweep(banks, scores, taus=TAUS)
            rows[scheme, rep, arm] = {t: v[1] for t, v in swept.items()}
    out = dict(base=a.base, run=a.run, inst=a.inst, base_inst=a.base_inst, slots=a.slots, frames=len(stems),
               partitions=list(range(a.first_partition, a.first_partition + a.partitions)),
               group_partitions=list(range(a.first_group, a.first_group + a.group_partitions)))
    out.update(release_rule(rows, len(stems)))
    if a.rows:
        np.savez_compressed(a.rows, stems=np.array(stems),
                            **{f"{s}|{r}|{arm}|{t}": v for (s, r, arm), d in rows.items() for t, v in d.items()})
    print(json.dumps(out, indent=1))
    (a.out or RUNS / a.run / f"member_rule_vs_{a.base}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
