"""LaTeX tables + number macros for the report, generated from logged data. Usage: .venv/bin/python report/tables.py"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import data as D

OUT = Path(__file__).parent / "gen"
OUT.mkdir(exist_ok=True)
KAGGLE = {"random": "0.02$^\\dagger$", "T10": "0.00", "T11": "0.00", "T12": "0.00", "T20": "0.00", "T21": "0.05",
          "T37": "0.07"}
TRAIN_PREFIX = {"T10": "TEAM-10-dqn-base", "T11": "TEAM-11-dqn-novelty", "T12": "TEAM-12-dqn-novelty-mask",
                "T20": "TEAM-20-ppo-base", "T21": "TEAM-21-ppo-rnd", "T22": "TEAM-22-ppo-rnd-lstm"}


def fmt(r, sd=True):
    if r is None:
        return "--"
    if r["mean"] == 0:
        return "0"
    s = f"{r['mean']:.4f}" if r["mean"] < 0.1 else f"{r['mean']:.3f}"
    if sd:
        s += f"\\,$\\pm$\\,{r['std']:.4f}" if r["mean"] < 0.1 else f"\\,$\\pm$\\,{r['std']:.3f}"
    return s


def first_level_actions(key):
    """{game: [actions at which level 1 was completed, one per solving run]} on dev."""
    v = next(x for x in D.VERSIONS if x[0] == key)
    out = {}
    for d in D.run_dirs(v[3], v[4]):
        by = {}
        for row in D.load_run(d)[2]:
            if int(row["levels"]) >= 1:
                k = (row["seed"], row["game"])
                by[k] = min(by.get(k, 10 ** 9), int(row["action"]))
        for (_, g), a in by.items():
            out.setdefault(g, []).append(a)
    return out


def solved_str(acts, n_runs):
    if not acts:
        return "--"
    parts = []
    for g in sorted(acts):
        a = acts[g]
        rng = f"{min(a)}" if min(a) == max(a) else f"{min(a)}--{max(a)}"
        parts.append(f"{g} {len(a)}/{n_runs} @{rng}")
    return "; ".join(parts)


def master(keys, name, caption_note=""):
    lines = [r"\begin{tabular}{@{}llrrlrrr@{}}", r"\toprule",
             r"Ver. & Change & Dev score & Lv & Level~1 on dev: runs @ actions & Train & Held-out & Kaggle \\",
             r"\midrule"]
    prev_fam = None
    for k in keys:
        fam = D.FAMILY[k]
        if prev_fam and fam != prev_fam:
            lines.append(r"\midrule")
        prev_fam = fam
        r = D.version_results(k)
        if r is None:          # version not evaluated (yet): leave it out rather than invent a row
            continue
        games = solved_str(first_level_actions(k), r["n_runs"])
        tr, ho = D.version_results(k, "train"), D.version_results(k, "heldout")
        lab = D.LABEL[k]
        ver, _, interv = lab.partition(" ")
        ver = ver.replace("TEAM-", "T").replace("Random", "Rand.")
        lines.append(f"{ver} & {interv} & {fmt(r)} & {r['levels']} & {games} & {fmt(tr, False)} & "
                     f"{fmt(ho, False)} & {KAGGLE.get(k, '--')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute():
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Version & Train steps & Train h/seed (T4) & Eval s/train seed (CPU) & Train lv (s0/s1/s2) \\",
             r"\midrule"]
    for k, p in TRAIN_PREFIX.items():
        ts = D.train_summary(p)
        r = D.version_results(k)
        ev = r["eval_seconds"] / 3
        lines.append(f"{D.LABEL[k]} & {ts[0]['env_steps']:,} & {min(t['train_hours'] for t in ts):.2f}--"
                     f"{max(t['train_hours'] for t in ts):.2f} & {ev:,.0f} & "
                     f"{'/'.join(str(t['levels_total']) for t in ts)} \\\\")
    for k in ["T30", "T37", "T45"]:
        r = D.version_results(k)
        per = r["eval_seconds"] / (1 if k == "T30" else 3)
        lines.append(f"{D.LABEL[k]} & 0 (+T21 prior) & -- & {per:,.0f} & -- \\\\" if k != "T30" else
                     f"{D.LABEL[k]} & 0 & -- & {per:,.0f} (9 seeds) & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "compute.tex").write_text("\n".join(lines).replace(",", "{,}") + "\n", encoding="utf-8")


def numbers():
    m = {}
    for k in ["random", "T10", "T11", "T12", "T12f", "T20", "T21", "T22", "G1", "T30", "T31", "T35", "T35n", "T36",
              "T37", "T38", "T41", "T42", "T43", "T44", "T45", "T45n", "T46", "T47"]:
        r = D.version_results(k)
        # LaTeX macro names cannot hold digits: 0-9 -> a-j, ablation suffix n -> X, frozen f -> F (T45n -> Tefx)
        n = "Random" if k == "random" else k.replace("n", "X").replace("f", "F")
        n = "".join("abcdefghij"[int(c)] if c.isdigit() else c for c in n)
        m[f"dev{n}"] = fmt(r)
        m[f"lv{n}"] = str(r["levels"])
        for sp in ["heldout", "train"]:
            x = D.version_results(k, sp)
            m[f"{sp[:2]}{n}"] = fmt(x, True) if x else "pending"
            m[f"{sp[:2]}lv{n}"] = str(x["levels"]) if x else "?"
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items())
    (OUT / "numbers.tex").write_text(body + "\n", encoding="utf-8")
    return m


if __name__ == "__main__":
    master(["random", "T10", "T11", "T12", "T12a", "T20", "T21", "T22", "T30", "T31", "T37", "T43", "T45", "T45n"], "master")
    master([v[0] for v in D.VERSIONS], "master_full")
    compute()
    for k, v in numbers().items():
        print(k, v)
