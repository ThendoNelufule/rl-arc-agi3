"""Failure-mode decomposition of Franzen v3 benchmark (25 public games x 4 passes).
Input: recon/analysis/franzen_v3_trajectories.json (extracted from skarin trajectory-audit notebook).
All points are on the 0-100 benchmark scale (mean over 25 games, averaged over 4 passes)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = json.load(open(Path(__file__).with_name('franzen_v3_trajectories.json')))
df = pd.DataFrame(D['records'])
G = df.game_id.nunique()
pd.set_option('display.width', 200)


def credit(h, a):
    return min(115.0, 100.0 * (h / max(1.0, a)) ** 2)


def score(h, a, n):
    w = np.arange(1, n + 1, dtype=float)
    k = len(a)
    c = sum(credit(h[i], a[i]) * w[i] for i in range(k)) / w.sum()
    return min(c, 100 * w[:k].sum() / w.sum())


rows = []
for r in df.itertuples():
    n, k = r.number_of_levels, r.levels_completed
    h, a = r.human_actions, r.agent_actions[:k]
    w = np.arange(1, n + 1, dtype=float)
    s = score(h, a, n)
    assert abs(s - r.score) < 1e-6, (r.game_id, s, r.score)
    ceil = 100 * w[:k].sum() / w.sum()
    rows.append(dict(game=r.game_id, pass_id=r.pass_id, n=n, k=k, score=s, ceil=ceil,
                     lost_incomplete=100 - ceil, lost_ineff=ceil - s, tokens=r.generated_tokens,
                     actions=r.actions, act_completed=sum(a), act_stuck=r.actions - sum(a),
                     tail_nonzero=sum(x > 0 for x in r.agent_actions[k:])))
T = pd.DataFrame(rows)
print('records', len(T), 'mean score', round(T.score.mean(), 2), 'mean ceiling', round(T.ceil.mean(), 2))
print('tail agent_actions nonzero:', T.tail_nonzero.sum())

# (a) incomplete levels
T['mode'] = np.where(T.k == 0, 'L1 never', np.where(T.k == T.n, 'won', 'stalled later'))
a = T.groupby('mode').lost_incomplete.sum() / (4 * G)
print('\n(a) points lost to incomplete levels by mode:\n', a.round(2).to_string(), '\ntotal', round(a.sum(), 2))
print('trajectory counts by mode', T['mode'].value_counts().to_dict())
per_game = T.groupby('game').agg(n=('n', 'first'), kmin=('k', 'min'), kmax=('k', 'max'), kmean=('k', 'mean'),
                                 mean=('score', 'mean'), sd=('score', 'std'), best=('score', 'max'),
                                 worst=('score', 'min'), lost_inc=('lost_incomplete', 'mean'),
                                 lost_ineff=('lost_ineff', 'mean'), tok=('tokens', 'mean'),
                                 act=('actions', 'mean'), act_stuck=('act_stuck', 'mean')).sort_values('mean')
print(per_game.round(1).to_string())
stall = T[T.k < T.n].groupby('game').k.apply(lambda s: sorted((s + 1).tolist()))
print('\nfailing level index per non-won pass (1-based):\n', stall.to_string())

# (b) inefficiency
print('\n(b) points lost to inefficiency on completed levels:', round(T.lost_ineff.sum() / (4 * G), 2))
lv = []
for r in df.itertuples():
    for i in range(r.levels_completed):
        lv.append(dict(game=r.game_id, pass_id=r.pass_id, lvl=i + 1, n=r.number_of_levels,
                       h=r.human_actions[i], a=r.agent_actions[i],
                       cr=credit(r.human_actions[i], r.agent_actions[i])))
L = pd.DataFrame(lv)
L['ratio'] = L.a / L.h
print('completed levels', len(L), 'median agent/human', round(L.ratio.median(), 2), 'mean credit',
      round(L.cr.mean(), 1), 'share credit>=100', round((L.cr >= 100).mean(), 2),
      'share credit<25', round((L.cr < 25).mean(), 2))
print('ineff by game (benchmark pts):\n',
      (T.groupby('game').lost_ineff.mean() / G).sort_values(ascending=False).head(8).round(2).to_string())

# (c) variance
mm = per_game['mean'].mean()
b4 = per_game['best'].mean()
w4 = per_game['worst'].mean()
print(f'\n(c) mean {mm:.2f}  best-of-4 {b4:.2f} (+{b4 - mm:.2f})  worst-of-4 {w4:.2f}')
pm = T.groupby('pass_id').score.mean()
print('per-pass benchmark scores', pm.round(2).tolist(), 'sd', round(pm.std(), 2))
b2 = np.mean([np.mean([max(v[i], v[j]) for i in range(4) for j in range(i + 1, 4)])
              for v in T.groupby('game').score.apply(list)])
print(f'expected best-of-2 {b2:.2f} (+{b2 - mm:.2f})')
cf = per_game[(per_game.best - per_game.worst) >= 30]
print('coin flips (span>=30):\n', cf[['kmin', 'kmax', 'n', 'mean', 'best', 'worst', 'sd']].round(1).to_string())
print('variance share by game (contribution of best-mean, pts):\n',
      ((per_game.best - per_game['mean']) / G).sort_values(ascending=False).head(8).round(2).to_string())

# (d) tokens / time
print('\n(d) total tokens', T.tokens.sum(), 'mean/trajectory', int(T.tokens.mean()))
print('corr(tokens, score)', round(np.corrcoef(T.tokens, T.score)[0, 1], 2),
      ' corr(tokens, k)', round(np.corrcoef(T.tokens, T.k)[0, 1], 2))
print('actions on completed levels', T.act_completed.sum(), ' on final unsolved level', T.act_stuck.sum(),
      f'share {T.act_stuck.sum() / T.actions.sum():.2f}')
tok_stuck = (T.tokens * T.act_stuck / T.actions.clip(lower=1)).sum()
print(f'token proxy (action-share) on unsolved level: {tok_stuck / T.tokens.sum():.2f} of all tokens')
weak = per_game[per_game['mean'] < 20].index
print('weak games (<20 mean):', list(weak), 'token share',
      round(T[T.game.isin(weak)].tokens.sum() / T.tokens.sum(), 2),
      'points', round(T[T.game.isin(weak)].score.sum() / (4 * G), 2))
won = T[T.k == T.n]
nw = T[T.k < T.n]
print('mean tokens: won', int(won.tokens.mean()), ' not won', int(nw.tokens.mean()),
      ' L1-never', int(T[T.k == 0].tokens.mean()))
T['tok_per_level'] = T.tokens / T.k.replace(0, np.nan)
print('tokens per completed level: median', int(T.tok_per_level.median()))
# regress tokens ~ k + stuck (per-level split not in data)
X = np.c_[np.ones(len(T)), T.k, (T.k < T.n).astype(float)]
coef = np.linalg.lstsq(X, T.tokens, rcond=None)[0]
print('OLS tokens = %.0f + %.0f*levels_completed + %.0f*[not won]' % tuple(coef))
print('agent/human action ratio by level index:\n',
      L.groupby('lvl').agg(n=('ratio', 'size'), med_ratio=('ratio', 'median'), mean_cr=('cr', 'mean')).round(2).to_string())
st = pd.to_datetime(df.started_at)
print('start spread (s)', (st.max() - st.min()).total_seconds(), ' job runtime h',
      round(D['protocol']['job_runtime_seconds'] / 3600, 2), 'protocol', D['protocol'])

# (e) between-level carry-over
L = L.sort_values(['game', 'pass_id', 'lvl'])
L['logr'] = np.log(L.ratio)
L['prev_logr'] = L.groupby(['game', 'pass_id']).logr.shift()
x = L.dropna(subset=['prev_logr'])
xd_l = x.logr - x.groupby('game').logr.transform('mean')
xd_p = x.prev_logr - x.groupby('game').prev_logr.transform('mean')
print('\n(e) corr(log ratio_k, log ratio_k+1) raw', round(np.corrcoef(x.logr, x.prev_logr)[0, 1], 2),
      ' within-game', round(np.corrcoef(xd_l, xd_p)[0, 1], 2), 'n', len(x))
L['rel'] = L.logr - L.groupby('game').logr.transform('mean')
print('within-game mean log(agent/human) by level index (neg = better):',
      L.groupby('lvl').rel.mean().round(2).to_dict())
nxt = []
for r in df.itertuples():
    for i in range(r.levels_completed):
        if i + 1 < r.number_of_levels:
            nxt.append(dict(eff=r.agent_actions[i] <= 1.5 * r.human_actions[i],
                            solved_next=i + 1 < r.levels_completed))
N = pd.DataFrame(nxt)
print('P(solve k+1 | level k <=1.5x human)', round(N[N.eff].solved_next.mean(), 2), 'n', int(N.eff.sum()),
      ' | >1.5x', round(N[~N.eff].solved_next.mean(), 2), 'n', int((~N.eff).sum()))
# level-1 vs later efficiency for the same game: does it get relatively easier?
first_vs_rest = L.groupby(['game', 'pass_id']).apply(
    lambda g: pd.Series(dict(l1=g[g.lvl == 1].logr.mean(), rest=g[g.lvl > 1].logr.mean()))).dropna()
print('mean log ratio L1', round(first_vs_rest.l1.mean(), 2), ' later levels', round(first_vs_rest.rest.mean(), 2))


# (f) credit >100 and cap
print('\n(f) levels with credit >100:', int((L.cr > 100).sum()), ' trajectories where game cap binds:',
      int((T.ceil - T.score < 1e-9).sum() & 0) if False else int(((T.ceil - T.score).abs() < 1e-9).sum()))


# upside
def plus_one(r, cr):
    n, k = r.number_of_levels, r.levels_completed
    if k >= n:
        return r.score
    h = r.human_actions
    a_ = list(r.agent_actions[:k])
    w = np.arange(1, n + 1, dtype=float)
    c = sum(credit(h[i], a_[i]) * w[i] for i in range(k)) + cr * w[k]
    return min(c / w.sum(), 100 * w[:k + 1].sum() / w.sum())


avg_cr = L.cr.mean()
base = T.score.mean()
for lab, cr in [('mean credit %.0f' % avg_cr, avg_cr), ('credit 25 (2x human)', 25.0)]:
    up = np.mean([plus_one(r, cr) for r in df.itertuples()]) - base
    print(f'UPSIDE every non-won trajectory clears +1 level at {lab}: +{up:.2f}')
    ups = np.mean([plus_one(r, cr) if r.levels_completed > 0 else r.score for r in df.itertuples()]) - base
    upl = np.mean([plus_one(r, cr) if r.levels_completed == 0 else r.score for r in df.itertuples()]) - base
    print(f'   stalled-later +{ups:.2f}, L1-never +{upl:.2f}')
T['next_pts'] = [((r.levels_completed + 1) / (r.number_of_levels * (r.number_of_levels + 1) / 2) * avg_cr / G)
                 if r.levels_completed < r.number_of_levels else 0 for r in df.itertuples()]
print('marginal benchmark pts of next level (mean over passes):\n',
      T.groupby('game').next_pts.mean().sort_values(ascending=False).head(10).round(2).to_string())
# inefficiency upside: all completed levels at >= 1.5x human (credit 44) floor
up_eff = np.mean([score(r.human_actions, [min(a_, 1.5 * h_) for a_, h_ in zip(r.agent_actions[:r.levels_completed], r.human_actions)], r.number_of_levels) for r in df.itertuples()]) - base
print(f'UPSIDE all completed levels capped at 1.5x human actions: +{up_eff:.2f}')
