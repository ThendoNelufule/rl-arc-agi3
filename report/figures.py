"""All report figures (vector PDF) from logged data only. Usage: .venv/bin/python report/figures.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

sys.path.insert(0, str(Path(__file__).parent))
import data as D

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)
import os
PREVIEW = os.environ.get("FIG_PREVIEW")  # optional PNG copies for checking layout
W = 5.5  # RLJ \textwidth (in)

# one colour per family (Okabe-Ito based; validated for CVD with the dataviz validator)
FAM = {"dqn": "#0072B2", "ppo": "#E69F00", "graph": "#009E73", "franzen": "#8E5BB5", "random": "#8C8C8C"}
FAMNAME = {"dqn": "DQN chain", "ppo": "PPO chain", "graph": "Graph explorer", "franzen": "Franzen M2 fork",
           "random": "Random"}
INK, MUTED, GRID = "#222222", "#666666", "#E4E4E4"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.edgecolor": "#999999", "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def shade(c, f):
    """f<0 darker, f>0 lighter (same hue): within-family versions."""
    r = np.array(to_rgb(c))
    return tuple(r + (1 - r) * f) if f > 0 else tuple(r * (1 + f))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=140)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. improvement ladder (dev)
def fig_ladder():
    groups = [("DQN chain", ["random", "T10", "T11", "T12"]), ("PPO chain", ["random", "T20", "T21", "T22"]),
              ("Graph explorer (key steps)", ["T30", "T31", "T37", "T43", "T45"])]
    short = {"random": "Rand.", "T10": "T10\nbase", "T11": "T11\nnovel.", "T12": "T12\nmask", "T20": "T20\nbase",
             "T21": "T21\nRND", "T22": "T22\nLSTM", "T30": "T30\nstats", "T31": "T31\nprior",
             "T37": "T37\ninv.", "T43": "T43\nobj.", "T45": "T45\ncomp."}
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.05), sharey=True, gridspec_kw=dict(width_ratios=[4, 4, 5]))
    floor = 1e-4
    for ax, (title, keys) in zip(axes, groups):
        for i, k in enumerate(keys):
            r = D.version_results(k)
            c = FAM[D.FAMILY[k]]
            m = max(r["mean"], floor)
            ax.bar(i, m - floor, bottom=floor, width=0.62, color=c, alpha=0.85, edgecolor="white", linewidth=1)
            if r["per_train_seed"]:
                lo = max(r["mean"] - r["std"], floor)
                ax.errorbar(i, m, yerr=[[m - lo], [r["std"]]], color=INK, lw=0.9, capsize=2.5)
            pts = np.maximum(np.array(r["runs"]), floor)
            jit = np.linspace(-0.2, 0.2, len(pts))
            ax.scatter(i + jit, pts, s=7, color="white", edgecolor=INK, linewidth=0.5, zorder=3)
            ax.text(i, 2.4, f"{r['levels']}", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(keys)), [short[k] for k in keys], fontsize=7)
        ax.set_title(title, loc="left", color=INK)
        ax.set_yscale("log")
        ax.set_ylim(floor, 4)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Official score (dev, log)")
    save(fig, "ladder")


# ---------------------------------------------------------------- 2. training curves
def fig_training():
    chains = [("DQN chain", [("TEAM-10-dqn-base", "T10 base"), ("TEAM-11-dqn-novelty", "T11 +novelty"),
                             ("TEAM-12-dqn-novelty-mask", "T12 +effect mask")], "dqn"),
              ("PPO chain", [("TEAM-20-ppo-base", "T20 base"), ("TEAM-21-ppo-rnd", "T21 +RND"),
                             ("TEAM-22-ppo-rnd-lstm", "T22 +LSTM")], "ppo")]
    # one row: [DQN levels, PPO levels, DQN frame-change, PPO frame-change]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.95))
    styles = [(0.45, "-"), (0.0, "--"), (-0.35, "-")]
    for col, (title, vers, fam) in enumerate(chains):
        for (prefix, lab), (f, ls) in zip(vers, styles):
            ms = D.train_metrics(prefix)
            c = shade(FAM[fam], f)
            for row, key in enumerate(["levels_total", "frame_change_rate"]):
                n = min(len(m) for m in ms)
                x = np.array([float(r["env_steps"]) for r in ms[0][:n]]) / 1e3
                y = np.array([[float(r[key]) for r in m[:n]] for m in ms])
                ax = axes[2 * row + col]
                ax.plot(x, y.mean(0), color=c, ls=ls, lw=1.3, label=lab)
                ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=c, alpha=0.18, lw=0)
        axes[col].set_title(f"{title[:3]}: levels", loc="left", fontsize=8)
        axes[2 + col].set_title(f"{title[:3]}: frame change", loc="left", fontsize=8)
        axes[col].set_ylim(0, 12.5)
        axes[col].legend(loc="upper left", fontsize=6, handlelength=1.6, labelspacing=0.2)
    for ax in axes:
        ax.set_xlabel("Steps (k)", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Cumulative levels", fontsize=7.5)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "training")


# ---------------------------------------------------------------- 3. levels vs actions (dev)
def fig_levels_actions():
    keys = [("random", 0), ("T12", 0), ("T21", 0), ("T22", 0.4), ("T30", 0.45), ("T37", 0), ("T45", -0.35)]
    ls = {"T22": "--", "T30": "--"}
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.0))
    for k, f in keys:
        v = next(x for x in D.VERSIONS if x[0] == k)
        cur = D.level_curve(D.run_dirs(v[3], v[4]))
        m, s = cur.mean(0), cur.std(0)
        c = shade(FAM[D.FAMILY[k]], f)
        x = np.arange(len(m))
        ax.plot(x, m, color=c, ls=ls.get(k, "-"), label=f"{D.LABEL[k]} (n={len(cur)})")
        ax.fill_between(x, np.maximum(m - s, 0), m + s, color=c, alpha=0.15, lw=0)
    ax.set_xlim(0, 505)
    ax.set_xlabel("Actions used per game (budget 500)")
    ax.set_ylabel("Levels completed\n(sum over 5 dev games)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), fontsize=8, labelspacing=0.35)
    save(fig, "levels_actions")


# ---------------------------------------------------------------- 4. generalisation gap
KAGGLE_SCORE = {"random": 0.02, "T10": 0.00, "T11": 0.00, "T12": 0.00, "T20": 0.00, "T21": 0.05, "T37": 0.07}


def fig_generalisation():
    keys = ["random", "T10", "T12", "T20", "T21", "T22", "T37", "T45", "T45n"]
    splits = [("train", "Train split (15 games)", 0.5), ("dev", "Dev (5)", 0.0), ("heldout", "Held-out (5)", -0.4),
              ("kaggle", "Kaggle public LB (hidden games)", None)]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    bw, floor = 0.2, 1e-4
    for i, k in enumerate(keys):
        fam = D.FAMILY[k]
        for j, (sp, lab, f) in enumerate(splits):
            xo = i + (j - 1.5) * bw
            if sp == "kaggle":
                if k not in KAGGLE_SCORE:
                    continue                      # never submitted: blank slot
                val = KAGGLE_SCORE[k]
                ax.bar(xo, max(val, floor) - floor, bottom=floor, width=bw, color="white", edgecolor=FAM[fam],
                       hatch="////", linewidth=0.8)
                if val == 0:
                    ax.text(xo, floor * 1.25, "0", ha="center", va="bottom", fontsize=6, color=INK)
                continue
            r = D.version_results(k, sp)
            if r is None:
                continue                          # not run: blank slot
            m = max(r["mean"], floor)
            ax.bar(xo, m - floor, bottom=floor, width=bw, color=shade(FAM[fam], f), edgecolor="white", lw=0.5)
            ax.scatter(np.full(len(r["runs"]), xo), np.maximum(r["runs"], floor), s=4, color=INK, zorder=3)
            if r["mean"] == 0:
                ax.text(xo, floor * 1.25, "0", ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(floor, 3)
    short = {"random": "Rand.", "T10": "T10\nDQN", "T12": "T12\n+mask", "T20": "T20\nPPO", "T21": "T21\n+RND",
             "T22": "T22\n+LSTM", "T37": "T37\n+inverse", "T45": "T45\n+compon.", "T45n": "T45n\nno prior"}
    ax.set_xticks(range(len(keys)), [short[k] for k in keys], fontsize=7)
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Official score (log)")
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=shade("#777777", f), label=lab) for _, lab, f in splits[:3]]
    hs.append(Patch(facecolor="white", edgecolor="#777777", hatch="////", label=splits[3][1]))
    ax.legend(handles=hs, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7, handlelength=1.4,
              columnspacing=1.0)
    save(fig, "generalisation")


# ---------------------------------------------------------------- 5. Kaggle timeline + leaderboard variance
def fig_kaggle():
    import datetime as dt
    subs = D.kaggle_submissions()
    fig, (a, b, b2) = plt.subplots(1, 3, figsize=(W, 2.15), gridspec_kw=dict(width_ratios=[1.3, 0.8, 0.8]))
    seen = set()
    for s in subs:
        t = dt.datetime.fromisoformat(s["date_utc"][:19])
        y = float(s["public_score"])
        fam = s["family"]
        kw = dict(color=FAM[fam], s=26, zorder=3, edgecolor="white", linewidth=0.6)
        lab = FAMNAME[fam] if fam not in seen else None
        seen.add(fam)
        if y == 0 and fam == "franzen":
            a.scatter(t, y, marker="X", s=40, color=FAM[fam], zorder=4, edgecolor=INK, linewidth=0.4)
        elif s["submission_id"] == "57556604":
            a.scatter(t, y, marker="*", s=150, color=FAM[fam], edgecolor=INK, linewidth=0.6, zorder=5, label=lab)
            a.annotate("nominated\n29.15", (t, y), xytext=(-9, 4), textcoords="offset points", fontsize=7,
                       ha="right", color=INK)
        else:
            a.scatter(t, y, label=lab, **kw)
    a.scatter([], [], marker="X", color=FAM["franzen"], edgecolor=INK, linewidth=0.4, s=40,
              label="Franzen fork, 0.00 (failed run)")
    a.set_ylabel("Kaggle public score")
    a.set_ylim(-2, 34)
    a.xaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=[1, 15]))
    a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    a.tick_params(axis="x", rotation=0, labelsize=7)
    a.legend(loc="center left", fontsize=6.3, handletextpad=0.2, bbox_to_anchor=(0, 0.52))
    a.set_title("(a) All 16 submissions", loc="left", fontsize=8.5)
    # (b) our Franzen-family runs
    fr = [float(s["public_score"]) for s in subs if s["family"] == "franzen"]
    nz = np.array([x for x in fr if x > 0])
    bins = np.arange(-1, 32, 2)
    b.hist([x for x in fr if x == 0], bins=bins, color="white", edgecolor=FAM["franzen"], hatch="xxx",
           label="0.00 (3)")
    b.hist(nz, bins=bins, color=FAM["franzen"], alpha=0.85, edgecolor="white", label=f"non-zero ({len(nz)})")
    b.set_xlabel("Kaggle public score")
    b.set_ylabel("Runs")
    b.set_ylim(0, 4.2)
    b.set_xlim(-2, 33)
    b.legend(loc="upper left", fontsize=6.5, handlelength=1.2)
    b.set_title("(b) Our 9 fork runs", loc="left", fontsize=8.5)
    # (c) expected best of N under the spread of public copies of the notebook
    rng = np.random.default_rng(0)
    Ns = np.unique(np.round(np.logspace(0, np.log10(120), 30)).astype(int))
    mu, sd = 25.77, 3.93   # public copies of Franzen's notebook (thread 745062)
    sim = rng.normal(mu, sd, (20000, Ns.max()))
    b2.plot(Ns, [np.maximum.accumulate(sim, 1)[:, n - 1].mean() for n in Ns], color=FAM["franzen"], lw=1.5,
            label=r"$\mathcal{N}(25.8, 3.9^2)$")
    b2.scatter([106], [38.85], marker="D", s=20, color=INK, zorder=3, label="Franzen (106 entries)")
    b2.scatter([9], [29.15], marker="*", s=70, color=FAM["franzen"], edgecolor=INK, lw=0.5, zorder=3,
               label="ours (9 fork runs)")
    b2.set_xscale("log")
    b2.set_ylim(24, 49)
    b2.set_xlabel("Entries N")
    b2.set_ylabel("Expected best of N")
    b2.legend(loc="upper left", fontsize=6.3, handletextpad=0.3, borderaxespad=0.2)
    b2.set_title("(c) Best-of-N", loc="left", fontsize=8.5)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "kaggle")


# ---------------------------------------------------------------- 6. MOD-1 vs MOD-2 proxy (supplementary)
def fig_proxy():
    m1, m2 = D.proxy("proxy-mod1"), D.proxy("proxy-mod2")
    games = sorted(m1, key=lambda g: m2[g]["score"] - m1[g]["score"])
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(len(games))
    for i, g in enumerate(games):
        ax.plot([i, i], [m1[g]["score"], m2[g]["score"]], color="#BBBBBB", lw=1, zorder=1)
    ax.scatter(x, [m1[g]["score"] for g in games], color=FAM["franzen"], s=18, zorder=3,
               label=f"MOD-1 (mean {np.mean([m1[g]['score'] for g in games]):.2f})")
    ax.scatter(x, [m2[g]["score"] for g in games], facecolor="white", edgecolor=FAM["franzen"], s=18, zorder=3,
               linewidth=1, label=f"MOD-2 (mean {np.mean([m2[g]['score'] for g in games]):.2f})")
    ax.set_xticks(x, games, rotation=90, fontsize=7)
    ax.set_ylabel("Game score (public-25 proxy)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.07, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, "proxy")


# ---------------------------------------------------------------- 7. graph-explorer progression (supplementary)
def fig_graph_progression():
    keys = ["T30", "T31", "T32", "T33", "T34", "T35", "T35n", "T36", "T37", "T38", "T39", "T40", "T41", "T42",
            "T43", "T44", "T45", "T45n", "T46", "T47"]
    adopted = {"T30", "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T43", "T44", "T45"}
    abl = {"T35n", "T45n"}
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for i, k in enumerate(keys):
        r = D.version_results(k)
        c = FAM["graph"]
        if k in adopted:
            ax.errorbar(i, r["mean"], yerr=r["std"], fmt="o", color=c, ms=5, capsize=2, lw=0.9)
        elif k in abl:
            ax.errorbar(i, r["mean"], yerr=r["std"], fmt="s", mfc="white", color=c, ms=4.5, capsize=2, lw=0.9)
        else:
            ax.errorbar(i, r["mean"], yerr=r["std"], fmt="x", color="#C0392B" if k in {"T38", "T42", "T47"} else MUTED,
                        ms=5.5, capsize=2, lw=0.9)
        ax.text(i, r["mean"] + r["std"] + 0.03, str(r["levels"]), ha="center", fontsize=6, color=MUTED)
    best = np.maximum.accumulate([D.version_results(k)["mean"] if k in adopted else 0 for k in keys])
    ax.step(range(len(keys)), best, where="mid", color=shade(FAM["graph"], 0.5), lw=1, zorder=0)
    ax.set_xticks(range(len(keys)), keys, rotation=90, fontsize=7)
    ax.set_ylabel("Official score (dev)")
    ax.set_ylim(0, 1.0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FAM["graph"], label="adopted"),
                       Line2D([], [], marker="s", ls="", mfc="white", color=FAM["graph"], label="no-prior ablation"),
                       Line2D([], [], marker="x", ls="", color=MUTED, label="not adopted (no gain / mixed)"),
                       Line2D([], [], marker="x", ls="", color="#C0392B", label="not adopted (regression)")],
              loc="upper left", ncol=2, fontsize=7)
    ax.grid(axis="x", visible=False)
    save(fig, "graph_progression")


# ---------------------------------------------------------------- 8. trajectories
def fig_trajectories():
    from arc_agi.rendering import COLOR_MAP
    cmap = ListedColormap([COLOR_MAP[i][:7] for i in range(16)])

    def load(tag, game, seed):
        z = np.load(D.RUNS / tag / "recordings" / f"{game}_s{seed}.npz")
        rows = [r for r in D.load_run(D.RUNS / tag)[2] if r["game"] == game and int(r["seed"]) == seed]
        return z["frames"], z["actions"], rows

    # success: TEAM-45 seed 0 on lf52, the eval seed with the earliest level-1 solve
    tag = "TEAM-45-graph-components-s0-dev"
    best = None
    for s in range(3):
        fr, ac, cur = load(tag, "lf52", s)
        t1 = min([int(r["action"]) for r in cur if int(r["levels"]) >= 1] or [10 ** 9])
        if best is None or t1 < best[0]:
            best = (t1, s, fr)
    t1, s_ok, fr_ok = best
    # stall: same agent on dc22 (never solved), eval seed 0
    fr_st, ac_st, _ = load(tag, "dc22", 0)
    # distinct frames with the step-counter rows masked (rows that change in >= 50% of steps, as the agent does)
    chg = (np.diff(fr_st.astype(int), axis=0) != 0).any(axis=2).mean(0)
    keep = chg < 0.5
    uniq = [len({fr_st[i][keep].tobytes() for i in range(t + 1)}) for t in range(len(fr_st))]
    fig, axes = plt.subplots(2, 4, figsize=(W, 2.95))
    picks_ok = [0, t1 // 2, t1 - 1, t1]
    picks_st = [0, 100, 300, 500]
    for j, t in enumerate(picks_ok):
        axes[0, j].imshow(fr_ok[t], cmap=cmap, vmin=0, vmax=15, interpolation="nearest")
        axes[0, j].set_title(f"t={t}" + (" (level 1 done)" if t == t1 else ""), fontsize=7.5)
    for j, t in enumerate(picks_st):
        axes[1, j].imshow(fr_st[t], cmap=cmap, vmin=0, vmax=15, interpolation="nearest")
        axes[1, j].set_title(f"t={t}  ({uniq[t]} state{'s' if uniq[t] > 1 else ''})", fontsize=7.5)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_color("#BBBBBB")
    axes[0, 0].set_ylabel(f"lf52 (solved)\nT45, eval seed {s_ok}", fontsize=7.5)
    axes[1, 0].set_ylabel("dc22 (stall)\nT45, eval seed 0", fontsize=7.5)
    fig.subplots_adjust(wspace=0.05, hspace=0.25)
    save(fig, "trajectories")
    return dict(lf52_t1=t1, lf52_seed=s_ok, dc22_distinct=uniq[-1])


if __name__ == "__main__":
    only = sys.argv[1:]
    for f in [fig_ladder, fig_training, fig_levels_actions, fig_generalisation, fig_kaggle, fig_proxy,
              fig_graph_progression, fig_trajectories]:
        if not only or f.__name__ in only:
            r = f()
            if r:
                print(r)
