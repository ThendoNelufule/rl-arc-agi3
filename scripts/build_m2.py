"""Build our attributed forks of Daniel Franzen's ARC-AGI-3 Milestone-2 solution.

Inputs (pulled with `kaggle kernels pull <ref> -p recon/nb/<owner>__<slug> -m`, not committed):
  dfranzen/arc-agi-3-milestone-2-solution                 (base; built on Tufa Labs' Duck harness)
  heliosli/try-merging-it-you-might-get-lucky-and-win-gold (stagnation restart, reused verbatim)
  shiiin9/affectify-arc-31-54-in-a-single-sub              (D' slot priority, reused verbatim)
Outputs: kaggle_sub/m2-base/ and kaggle_sub/m2-mod1/ (notebook + kernel-metadata.json, pinned docker image).

BASE  = original notebook + our attribution cell; Save & Run path reduced to a smoke test
        (3 public games, 1 pass, 5 min/game). Competition-rerun path byte-identical.
MOD-1 = BASE + stagnation restart (after the harness patch) + D' scheduler (after the config cell).
Reused public competition code is OSI-licensed under competition rule 2.b; all credits in the notebooks.

Usage: .venv/bin/python scripts/build_m2.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "recon" / "nb"
OUT = ROOT / "kaggle_sub"
SMOKE_GAMES = ["lp85", "tu93", "cd82"]
PUBLIC = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
          "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]

CREDITS = """## Team Vladmir_Putin (Wits COMS4061A/7071A): attributed fork, base for our final ARC-AGI-3 submission

**This notebook is not our original work.** It is a fork of
[dfranzen/arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution)
by **Daniel Franzen** (Milestone-2 solution; [write-up](https://www.kaggle.com/writeups/dfranzen/arc-agi-3-milestone-2),
[GitHub](https://github.com/da-fr/arc-agi-3-solution)), which builds on **Tufa Labs' Duck harness**
([notebook](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner),
[GitHub](https://github.com/Tufalabs/duck-harness), Jeroen Cottaar et al.). Model: Qwen3.8-Flash-Next (Qwen team),
Intel AutoRound W4A16 quantization, Albucino's MTP draft checkpoint; inference via Pennyroyal (John Pezzulli's SGLang fork)
with patches by Mamy Ratsimbazafy and Gabriel Olympie. All third-party code and weights remain under their own licenses.

"""
BASE_NOTE = """**Our changes in this version (BASE):** none to the agent or to the competition-rerun path (byte-identical to the original,
same pinned Docker image, same inputs, RTX PRO 6000). The only change is the *non-submission* (Save & Run) path,
reduced to a cheap smoke test: 3 public games (lp85, tu93, cd82), 1 pass, 5 min per game.
Any later modifications are made in a separate notebook and documented there.
"""
MOD1_NOTE = """**This version (MOD-1): base + two REUSED public add-ons, both used verbatim and credited:**
1. *Stagnation restart*, from [heliosli/try-merging-it-you-might-get-lucky-and-win-gold](https://www.kaggle.com/code/heliosli/try-merging-it-you-might-get-lucky-and-win-gold):
   one fresh LLM context per level after 96k generated tokens without completing it.
2. *D' slot priority*, from [shiiin9/affectify-arc-31-54-in-a-single-sub](https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub)
   (Osaka Metropolitan University Affectify ARC branch): a different formula for which game gets GPU time next.
Both were shared publicly in this competition, which under competition rule 2.b licenses them under an OSI-approved licence.
**Our own contribution in this version is only the selection and integration** (placement after the harness patch /
after the config cell, removal of heliosli's submission-only gate); the agent logic is theirs.
Save & Run = smoke test only (3 public games, 1 pass, 5 min); the competition path otherwise matches the original.
"""


def load(owner_slug):
    d = NB / owner_slug
    return json.load(open(next(d.glob("*.ipynb")), encoding="utf-8")), json.load(open(d / "kernel-metadata.json"))


def code(c):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c}


def find(cells, pred):
    return next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and pred("".join(c["source"])))


def build_base():
    nb, meta = load("dfranzen__arc-agi-3-milestone-2-solution")
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": CREDITS + BASE_NOTE})
    i = find(nb["cells"], lambda s: s.startswith("# Make one-off changes to `bm`"))
    src = "".join(nb["cells"][i]["source"])
    old = "demo_excluded_games = [] if TRUE_SUBMISSION else []"
    src = src.replace(old, f"""# [Team Vladmir_Putin] Save & Run = smoke test only (3 public games); competition path unchanged.
SMOKE_GAMES = {SMOKE_GAMES!r}
demo_excluded_games = [] if TRUE_SUBMISSION else [g for g in {PUBLIC!r} if g not in SMOKE_GAMES]""")
    old2 = "        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110"
    src = src.replace(old2, old2 + """
    # [Team Vladmir_Putin] smoke test: 1 pass, 5 min per game
    bm.n_passes = 1
    bm.solver.concurrency = len(SMOKE_GAMES)
    bm.solver.max_runtime_s_per_game = 5 * 60""")
    nb["cells"][i]["source"] = src
    meta = {k: v for k, v in meta.items() if k != "id_no"}
    return nb, meta


def save(nb, meta, slug):
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    name = f"arc3-{slug}.ipynb"
    json.dump(nb, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = dict(meta, id=f"thendonelufule/arc3-{slug}", title=f"arc3-{slug}", code_file=name,
                is_private=True, keywords=[])
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=2)
    print("wrote", out)


def build_mod1():
    nb, meta = build_base()
    nb["cells"][0]["source"] = CREDITS + MOD1_NOTE
    h, _ = load("heliosli__try-merging-it-you-might-get-lucky-and-win-gold")
    d, _ = load("shiiin9__affectify-arc-31-54-in-a-single-sub")
    hc = ["".join(c["source"]) for c in h["cells"] if c["cell_type"] == "code"]
    dc = ["".join(c["source"]) for c in d["cells"] if c["cell_type"] == "code"]
    s = next(c for c in hc if "patch_stagnation_restart.py" in c and c.startswith("if TRUE_SUBMISSION:\n"))
    body = "\n".join(l[4:] if l.startswith("    ") else l for l in s.split("\n")[1:])
    stag = ("# [Team Vladmir_Putin MOD 1] Stagnation restart, reused verbatim (gate removed) from\n"
            "# heliosli/try-merging-it-you-might-get-lucky-and-win-gold (V4). Clears the LLM context and retained\n"
            "# functions ONCE per level after 96*1024 generated tokens without finishing it. Applied after the\n"
            "# harness patch; SHA-checks that tool_agent.py is exactly the patched Franzen version.\n" + body)
    dsrc = next(c for c in dc if "install_d(" in c)
    dprime = ("# [Team Vladmir_Putin MOD 2] D' slot priority, reused verbatim from\n"
              "# shiiin9/affectify-arc-31-54-in-a-single-sub (Osaka Metropolitan Univ. Affectify ARC branch).\n" + dsrc)
    cells = nb["cells"]
    i_env = find(cells, lambda s: 'git", "apply"' in s)
    cells.insert(i_env + 1, code(stag))
    i_cfg = find(cells, lambda s: s.startswith("# Make one-off changes to `bm`"))
    cells.insert(i_cfg + 1, code(dprime))
    return nb, meta


if __name__ == "__main__":
    save(*build_base(), "m2-base")
    save(*build_mod1(), "m2-mod1")


SUB_NOTE = """**Submission packaging (our change):** Save & Run does no GPU work: it only writes a placeholder
`submission.parquet` (the full solver runs only in the competition rerun, `KAGGLE_IS_COMPETITION_RERUN`).
Every code cell except the `%%writefile` patch is wrapped in `if TRUE_SUBMISSION:` (same pattern as
heliosli's notebook); inside, the code is unchanged except that two `!rm`/`!cp` shell lines are
replaced by equivalent `subprocess.run` calls; the SGLang launcher cell (its multi-line strings must not be
re-indented) is kept byte-identical and run via `exec(..., globals())`. Agent logic, inputs, docker image and GPU unchanged.
"""

TOP_CELL = '''import os
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
if not TRUE_SUBMISSION:
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]
                 ).to_parquet("/kaggle/working/submission.parquet", index=False)
    print("Save & Run: placeholder submission only; the full solver runs in the competition rerun.")
'''


def _consts(src):
    import ast
    t = compile(src, "c", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | ast.PyCF_ONLY_AST)
    return [n.value for n in ast.walk(t) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def gate(nb, note):
    """Submission packaging: wrap all heavy cells in `if TRUE_SUBMISSION:`."""
    import ast
    cells = nb["cells"]
    cells[0]["source"] = "".join(cells[0]["source"]) + "\n" + note
    for c in cells:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.startswith("%%writefile"):
            continue
        s = s.replace("!rm -Rf $BUNDLE_DIR", 'subprocess.run(["rm", "-Rf", str(BUNDLE_DIR)], check=True)')
        s = s.replace("!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR",
                      'subprocess.run(["cp", "-a", str(ORIG_BUNDLE_DIR), str(BUNDLE_DIR)], check=True)')
        assert not any(l.strip().startswith(("!", "%")) for l in s.split("\n")), "unhandled IPython line"
        wrapped = "if TRUE_SUBMISSION:\n" + "\n".join(("    " + l) if l.strip() else l for l in s.split("\n"))
        if _consts(wrapped) != _consts(s):
            # re-indenting would alter multi-line string literals (e.g. the SGLang launcher's embedded
            # files): keep the cell byte-identical and run it with exec (no await/magics in such cells)
            assert "await " not in s
            wrapped = ("if TRUE_SUBMISSION:\n    _CELL_SRC = " + repr(s) +
                       "\n    exec(compile(_CELL_SRC, '<cell>', 'exec'), globals())\n")
        compile(wrapped, "cell", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | ast.PyCF_ONLY_AST)
        c["source"] = wrapped
    first_code = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    cells.insert(first_code, code(TOP_CELL))
    return nb


def build_sub(builder):
    nb, meta = builder()
    return gate(nb, SUB_NOTE), meta


if __name__ == "__main__":
    save(*build_sub(build_base), "m2-base-sub")
    save(*build_sub(build_mod1), "m2-mod1-sub")


# ---------------------------------------------------------------------------------------------------------
# MOD-2 = MOD-1 + Fable layer (facts only) + reinvestment priority decay, and matched 25-game proxy variants.

MOD2_NOTE = """**This version (MOD-2) = MOD-1 + two more changes:**
3. *Fable layer* (REUSED, credited): [leoprovorov/fable-astra-play-arc-3-handbook-harness](https://www.kaggle.com/code/leoprovorov/fable-astra-play-arc-3-handbook-harness),
   cell F001 from dataset leoprovorov/arc-agi-3-fable-harness-analytical-handbook (Apache-2.0): harness-computed facts
   (indicator notes after 15k tokens on a level, pinned LEVEL LOG after 30k, budget alert). **Our modification:** its five
   PROTOCOL_LINES (advice text) are removed, keeping facts only (evidence: advice blocks hurt in rellik13's tests, thread 744792).
4. *Reinvestment priority decay* (OURS): after 130k generated tokens on the current level, a game's scheduling priority is
   multiplied by max(0.2, exp(-(tokens-130k)/100k)), so GPU slots shift to games that are progressing (no game is stopped).
"""

REINVEST_CELL = '''# [Team Vladmir_Putin MOD-2, OUR CODE] Reinvestment: decay the scheduling priority of a game that has spent many
# tokens on its current level (likely a dead end; ~26-29% of tokens go to never-solved levels in Franzen's v3 runs).
# Wraps whatever priority_value is installed (D' from the previous cell). The gate looks the name up at call time.
import math as _ri_math
import inference.agent.tool_agent as _ri_ta
REINVEST_K, REINVEST_SCALE, REINVEST_FLOOR = 130_000, 100_000, 0.2
_ri_orig = _ri_ta.priority_value
def _ri_priority(state, *args, **kwargs):
    v = _ri_orig(state, *args, **kwargs)
    t = float(getattr(state, "tokens", 0.0) or 0.0)
    if t <= REINVEST_K:
        return v
    f = max(REINVEST_FLOOR, _ri_math.exp(-(t - REINVEST_K) / REINVEST_SCALE))
    return max(1, int(v * f))
_ri_priority._ours_reinvest = True
_ri_ta.priority_value = _ri_priority
print("#OURS_REINVEST ok K=%d scale=%d floor=%.2f wraps=%s" % (REINVEST_K, REINVEST_SCALE, REINVEST_FLOOR,
      getattr(_ri_orig, "__name__", _ri_orig)), flush=True)
'''


def fable_cell():
    nb = json.load(open(ROOT / "recon" / "fable" / "F001_fable_layer_notebook.ipynb", encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "".join(c["source"]).startswith('# F001: the "Fable layer"'))
    a = src.index("PROTOCOL_LINES = [")
    b = src.index("]\n", a) + 2
    src = src[:a] + "PROTOCOL_LINES = []  # [Team Vladmir_Putin] advice lines removed: facts only\n" + src[b:]
    return ("# [Team Vladmir_Putin MOD-2] Fable layer, REUSED from leoprovorov's F001 notebook (Apache-2.0), unchanged\n"
            "# except PROTOCOL_LINES emptied (facts only). Applied after the stagnation patch (tested: both apply).\n" + src)


def build_mod2():
    nb, meta = build_mod1()
    nb["cells"][0]["source"] = "".join(nb["cells"][0]["source"]) + "\n" + MOD2_NOTE
    cells = nb["cells"]
    i_stag = find(cells, lambda s: "patch_stagnation_restart.py" in s)
    cells.insert(i_stag + 1, code(fable_cell()))
    i_d = find(cells, lambda s: "install_d(" in s)
    cells.insert(i_d + 1, code(REINVEST_CELL))
    return nb, meta


PROXY_NOTE = """**PROXY build (not for submission):** Save & Run plays ALL 25 public games, 1 pass, concurrency 25, per-game cap
532*60*25/110 s (same per-game compute share as the competition and as Franzen's v3 runs). Used for matched A/B tests.
"""


def proxyify(builder):
    nb, meta = builder()
    nb["cells"][0]["source"] = "".join(nb["cells"][0]["source"]) + "\n" + PROXY_NOTE
    i = find(nb["cells"], lambda s: s.startswith("# Make one-off changes to `bm`"))
    src = "".join(nb["cells"][i]["source"])
    a = src.index("# [Team Vladmir_Putin] Save & Run = smoke test only")
    b = src.index("\n", src.index("demo_excluded_games =", a)) + 1
    src = src[:a] + "# [Team Vladmir_Putin] PROXY: all 25 public games\ndemo_excluded_games = []\n" + src[b:]
    a = src.index("    # [Team Vladmir_Putin] smoke test: 1 pass, 5 min per game")
    b = src.index("    bm.solver.max_runtime_s_per_game = 5 * 60\n", a) + len("    bm.solver.max_runtime_s_per_game = 5 * 60\n")
    src = src[:a] + ("    # [Team Vladmir_Putin] PROXY: 1 pass, 25 concurrent games, competition-equivalent per-game share\n"
                     "    bm.n_passes = 1\n    bm.solver.concurrency = 25\n"
                     "    bm.solver.max_runtime_s_per_game = 532*60 * 25 // 110\n") + src[b:]
    nb["cells"][i]["source"] = src
    return nb, meta


if __name__ == "__main__":
    save(*build_mod2(), "m2-mod2")
    save(*proxyify(build_mod1), "m2-mod1-proxy")
    save(*proxyify(build_mod2), "m2-mod2-proxy")
