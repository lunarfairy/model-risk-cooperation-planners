"""Revised calibration (revision 1).

Changes relative to calib.py (all prompted by referee comments):
  * Neighbours whose choices are missing from the node file (dropouts replaced by automated players, or
    never-filled slots) are kept in the network; their actions are decoded from the edge-level Edge_Type
    field, so degree k is the full structural degree and affordability uses the true cost c*k.
  * Full covariance matrices from a central-difference Hessian; delta-method SE of s1 = beta_k + beta_n.
  * Group-level (87-group) cluster bootstrap of the main models; parameter draws are saved for propagation.
  * Additional specifications: lagged own choice, condition fixed effects, condition-specific round trends,
    per-condition fits, baseline-only fits, baseline-only fit with contemporaneous neighbour cooperation
    (to explain the published parameters), and leave-one-condition-out (LOCO) fits.
  * Cross-validation folds start from a fixed, data-independent initial value.
  * Acceptance: pooled logit, + decider's own action, + decider random intercept (GLMM), within/between
    decomposition of the decider-degree effect, and group bootstrap of the main acceptance model.
Usage: python calib2.py <osf_dir> <out_dir> <nproc> [nboot]
"""
import sys, json, itertools, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from multiprocessing import Pool

C_COST = 0.05
PAIRS = np.array(list(itertools.combinations(range(16), 2)))
TYPE2S = {'C-C': 2, 'C-D': 1, 'D-D': 0}


def decode_actions(act_t, et_t):
    """Fill missing actions of round t from edge types (same-round actions of both endpoints)."""
    a = act_t.copy()
    for _ in range(3):
        miss = np.isnan(a)
        if not miss.any():
            break
        for e, (i, j) in enumerate(PAIRS):
            s = TYPE2S.get(et_t[e])
            if s is None:
                continue
            if np.isnan(a[i]) and not np.isnan(a[j]):
                a[i] = s - a[j]
            elif np.isnan(a[j]) and not np.isnan(a[i]):
                a[j] = s - a[i]
    return a


def load(OSF):
    rows, recs, audit = [], [], []
    for k in ['baseline', 'evaluation', 'validation']:
        cdf = pd.read_csv(f'{OSF}/{k}_cooperation_data.csv')
        gdf = pd.read_csv(f'{OSF}/{k}_graph_struct_data.csv')
        for (g, cond), cg in cdf.groupby(['Group', 'Condition']):
            cond = cond.replace('Max connectivity', 'Max. connectivity')
            gg = gdf[gdf.Group == g]
            act = cg.pivot(index='Round', columns='Player', values='Cooperation').reindex(index=range(15), columns=range(16)).values
            cap = cg.pivot(index='Round', columns='Player', values='Capital').reindex(index=range(15), columns=range(16)).values
            S = gg.pivot(index='Round', columns='Edge', values='Prev_Status').reindex(index=range(15), columns=range(120)).fillna(0).values
            ET = gg.pivot(index='Round', columns='Edge', values='Edge_Type').reindex(index=range(15), columns=range(120)).values
            human = ~np.isnan(act)
            full = np.array([decode_actions(act[t], ET[t]) for t in range(15)])
            n_dec = int((np.isnan(act) & ~np.isnan(full)).sum())
            audit.append((k, cond, g, int(human.any(0).sum()), int((~human).sum()), n_dec, int(np.isnan(full).sum())))
            for t in range(15):
                A = np.zeros((16, 16)); A[PAIRS[:, 0], PAIRS[:, 1]] = S[t]; A = A + A.T
                known = ~np.isnan(full[t])
                A = A * known[None, :] * known[:, None]          # slots with undecodable actions are inert
                deg = A.sum(1)
                xn_c = A @ np.nan_to_num(full[t])              # contemporaneous (for the published-model check)
                if t == 0:
                    for i in np.flatnonzero(human[t]):
                        rows.append((k, cond, g, i, t, act[t, i], np.nan, deg[i], np.nan, np.nan, 1.0, np.nan, np.nan))
                    continue
                prev = np.nan_to_num(full[t - 1])
                xn = A @ prev
                for i in np.flatnonzero(human[t] & ~np.isnan(full[t - 1])):
                    afford = float(C_COST * deg[i] <= cap[t - 1, i] + 1e-9) if not np.isnan(cap[t - 1, i]) else 1.0
                    xr = xn[i] / deg[i] if deg[i] > 0 else 0.0
                    xc = xn_c[i] - 0.0
                    rows.append((k, cond, g, i, t, act[t, i], full[t - 1, i], deg[i], xn[i], xr, afford, xc,
                                 xc / deg[i] if deg[i] > 0 else 0.0))
            if 'Accepted' in gg.columns:
                r = gg.dropna(subset=['Accepted'])
                for _, e in r.iterrows():
                    t = int(e.Round)
                    A = np.zeros((16, 16)); A[PAIRS[:, 0], PAIRS[:, 1]] = S[t]; A = A + A.T
                    deg = A.sum(1)
                    dp, rp = int(e.Deciding_Player), int(e.Referent_Player)
                    recs.append((cond, g, t, int(e.Recommendation), int(e.Deciding_Player_Coop), int(e.Referent_Player_Coop),
                                 deg[dp], deg[rp], int(e.Accepted), f'{k}_{g}_{dp}'))
    D = pd.DataFrame(rows, columns=['study', 'cond', 'group', 'player', 't', 'a', 'a_prev', 'deg', 'xn', 'xr', 'afford', 'xn_c', 'xr_c'])
    D['pid'] = D.study + '_' + D.group.astype(str) + '_' + D.player.astype(str)
    D['gid'] = D.study + '_' + D.group.astype(str)
    R = pd.DataFrame(recs, columns=['cond', 'group', 't', 'val', 'dec_coop', 'ref_coop', 'deg_dec', 'deg_ref', 'acc', 'pid'])
    AU = pd.DataFrame(audit, columns=['study', 'cond', 'group', 'slots_with_human', 'nonhuman_slot_rounds', 'decoded', 'undecodable'])
    return D, R, AU


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / GH_W.sum()
CONDS = ['Static network', 'Random rec.', 'Coop. clustering', 'GraphNet planner', 'Encourag. planner', 'Neutral planner', 'Max. connectivity']


def design(D, fam):
    one = np.ones(len(D))
    if fam == 'count':
        return np.column_stack([one, D.deg, D.xn, D.xr])
    if fam == 'count_t':
        return np.column_stack([one, D.deg, D.xn, D.xr, D.t])
    if fam == 'frac_t':
        return np.column_stack([one, np.log1p(D.deg), D.xr, D.t])
    if fam == 'count_lag_t':
        return np.column_stack([one, D.deg, D.xn, D.xr, D.t, D.a_prev])
    if fam == 'count_contemp':
        return np.column_stack([one, D.deg, D.xn_c, D.xr_c])
    if fam == 'count_t_condfe':
        X = [one, D.deg, D.xn, D.xr, D.t] + [(D.cond == c).astype(float).values for c in CONDS[1:]]
        return np.column_stack(X)
    if fam == 'count_condtrend':
        X = [one, D.deg, D.xn, D.xr] + [(D.cond == c).astype(float).values * D.t.values for c in CONDS] + \
            [(D.cond == c).astype(float).values for c in CONDS[1:]]
        return np.column_stack(X)
    raise ValueError(fam)


class GLMM:
    """Round 1: logit = a0 + a1*theta; later rounds: logit = X b + theta; theta ~ N(0, s^2)."""

    def __init__(self, D, fam):
        D = D[(D.t == 0) | (D.afford == 1)].copy()
        self.D = D
        self.codes, self.pids = pd.factorize(D.pid)
        self.first = (D.t == 0).values
        self.y = D.a.values.astype(float)
        Xl = design(D[~self.first].fillna(0), fam)
        self.X = np.zeros((len(D), Xl.shape[1])); self.X[~self.first] = Xl
        self.k = Xl.shape[1] + 3
        self.np_ = len(self.pids)

    def per_player(self, par):
        a0, a1, ls = par[:3]
        s = np.exp(ls)
        th = s * GH_X[None, :]
        lin = np.where(self.first[:, None], a0 + a1 * th, (self.X @ par[3:])[:, None] + th)
        ll = log_expit((2 * self.y[:, None] - 1) * lin)
        per = np.zeros((self.np_, len(GH_X)))
        np.add.at(per, self.codes, ll)
        m = per.max(1, keepdims=True)
        return m[:, 0] + np.log(np.exp(per - m) @ GH_W)

    def nll(self, par, idx=None):
        lp = self.per_player(par)
        return -(lp[idx].sum() if idx is not None else lp.sum())

    def x0(self):
        return np.r_[1.5, 0.7, np.log(3.0), np.zeros(self.k - 3)]

    def fit(self, idx=None, x0=None):
        return minimize(self.nll, self.x0() if x0 is None else x0, args=(idx,), method='L-BFGS-B',
                        options={'maxiter': 3000})

    def cov(self, par, eps=2e-4):
        k = len(par); H = np.zeros((k, k)); I = np.eye(k)
        for i in range(k):
            for j in range(i, k):
                f = lambda a, b: self.nll(par + eps * (a * I[i] + b * I[j]))
                H[i, j] = H[j, i] = (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * eps ** 2)
        return np.linalg.inv(H)


def fit_summary(D, fam, cv=False):
    m = GLMM(D, fam)
    r = m.fit()
    V = m.cov(r.x)
    out = dict(par=r.x.tolist(), se=np.sqrt(np.diag(V)).tolist(), cov=V.tolist(), nll=float(r.fun), k=m.k,
               aic=float(2 * r.fun + 2 * m.k), n_obs=int(len(m.y)), n_players=int(m.np_), converged=bool(r.success))
    if fam.startswith('count') and fam != 'count_contemp':
        g = np.zeros(m.k); g[4] = 1; g[5] = 1
        out['s1'] = float(r.x[4] + r.x[5]); out['s1_se'] = float(np.sqrt(g @ V @ g))
        out['corr_bk_bn'] = float(V[4, 5] / np.sqrt(V[4, 4] * V[5, 5]))
    if cv:
        pg = m.D.groupby('pid').gid.first().reindex(m.pids).values
        ug = np.unique(pg); rng = np.random.default_rng(1)
        fold = dict(zip(ug, rng.permutation(len(ug)) % 10)); pf = np.array([fold[x] for x in pg])
        nob = np.bincount(m.codes, minlength=m.np_); ll = 0.0; nn = 0; folds = []
        for f in range(10):
            tr, te = np.flatnonzero(pf != f), np.flatnonzero(pf == f)
            rf = m.fit(idx=tr)                                   # independent initial value
            l = -m.nll(rf.x, idx=te); ll += l; nn += nob[te].sum(); folds.append(l / nob[te].sum())
        out['cv_ll_per_decision'] = float(ll / nn); out['cv_fold_sd'] = float(np.std(folds, ddof=1))
    return out, r.x


# --------------------------------------------------------------- bootstrap workers
_D = None


def _init(D, R=None):
    global _D, _R
    _D, _R = D, R


def boot_task(args):
    fam, seed = args
    rng = np.random.default_rng(seed)
    gids = _D.gid.unique()
    pick = rng.choice(gids, len(gids), replace=True)
    parts = []
    for c, gname in enumerate(pick):
        d = _D[_D.gid == gname].copy(); d['pid'] = d.pid + f'#{c}'; parts.append(d)
    Db = pd.concat(parts)
    m = GLMM(Db, fam)
    r = m.fit()
    return r.x.tolist()


def fit_task(args):
    name, fam, mask_kind, val = args
    D = _D
    if mask_kind == 'cond':
        DD = D[D.cond == val]
    elif mask_kind == 'loco':
        DD = D[D.cond != val]
    elif mask_kind == 'study':
        DD = D[D.study == val]
    else:
        DD = D
    o, _ = fit_summary(DD, fam)
    o.pop('cov', None) if mask_kind in ('cond',) else None
    return name, o


# --------------------------------------------------------------- acceptance models
def acc_design(R, deg_terms=True, dec_action=False, within=False):
    add = (R.val == 1).astype(float).values
    rc = R.ref_coop.values.astype(float)
    cols = [add * rc, add * (1 - rc), (1 - add) * rc, (1 - add) * (1 - rc)]
    if deg_terms:
        if within:
            mu = R.groupby('pid').deg_dec.transform('mean').values
            dev = R.deg_dec.values - mu
            cols += [add * dev, (1 - add) * dev, add * mu, (1 - add) * mu]
        else:
            cols += [add * R.deg_dec.values, (1 - add) * R.deg_dec.values]
        cols += [add * R.deg_ref.values, (1 - add) * R.deg_ref.values]
    if dec_action:
        dc = R.dec_coop.values.astype(float)
        cols += [add * dc, (1 - add) * dc]
    return np.column_stack(cols)


def fit_logit(X, y):
    def f(b):
        z = X @ b
        return -(y * log_expit(z) + (1 - y) * log_expit(-z)).sum(), -(X.T @ (y - expit(z)))
    r = minimize(f, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B')
    return r.x, r.fun


def fit_logit_re(X, y, codes):
    """Logit with a decider random intercept (Gauss-Hermite)."""
    npid = codes.max() + 1

    def nll(p):
        b, s = p[:-1], np.exp(p[-1])
        z = (X @ b)[:, None] + s * GH_X[None, :]
        ll = log_expit((2 * y[:, None] - 1) * z)
        per = np.zeros((npid, len(GH_X))); np.add.at(per, codes, ll)
        m = per.max(1, keepdims=True)
        return -(m[:, 0] + np.log(np.exp(per - m) @ GH_W)).sum()
    b0, _ = fit_logit(X, y)
    r = minimize(nll, np.r_[b0, np.log(0.5)], method='L-BFGS-B')
    return r.x, r.fun


def acc_boot_task(seed):
    R = _R
    rng = np.random.default_rng(seed)
    ug = R.group.unique()
    Rb = pd.concat([R[R.group == g] for g in rng.choice(ug, len(ug), replace=True)])
    b, _ = fit_logit(acc_design(Rb), Rb.acc.values.astype(float))
    b2, _ = fit_logit(acc_design(Rb, within=True), Rb.acc.values.astype(float))
    return b.tolist(), b2.tolist()


_R = None


def main(OSF, OUT, nproc=32, nboot=200):
    t0 = time.time()
    global _D, _R
    D, R, AU = load(OSF)
    _D, _R = D, R
    D.to_csv(f'{OUT}/decisions_v2.csv.gz', index=False); R.to_csv(f'{OUT}/recommendations_v2.csv.gz', index=False)
    AU.to_csv(f'{OUT}/reconstruction_audit.csv', index=False)
    fit_rows = D[(D.t == 0) | (D.afford == 1)]
    out = {'counts': dict(rows_total=int(len(D)), round1=int((D.t == 0).sum()), rounds2_15=int((D.t > 0).sum()),
                          infeasible=int(((D.t > 0) & (D.afford == 0)).sum()), fitted=int(len(fit_rows)),
                          fitted_rounds2_15=int(((fit_rows.t > 0)).sum()), players=int(D.pid.nunique()),
                          groups=int(D.gid.nunique()), decoded_neighbour_actions=int(AU.decoded.sum()),
                          undecodable=int(AU.undecodable.sum()),
                          deg_by_cond=D[D.t > 0].groupby('cond').deg.agg(['mean', 'median', 'max']).round(2).reset_index().to_dict('records'))}
    print('loaded', out['counts'], round(time.time() - t0), flush=True)
    models = {}
    for fam in ['count', 'count_t', 'frac_t', 'count_lag_t', 'count_t_condfe', 'count_condtrend']:
        o, _ = fit_summary(D, fam, cv=fam in ('count', 'count_t', 'frac_t', 'count_lag_t'))
        models[fam] = o
        print(fam, np.round(o['par'], 3), 's1', round(o.get('s1', np.nan), 4), '+-', round(o.get('s1_se', np.nan), 4),
              'AIC', round(o['aic'], 1), round(time.time() - t0), flush=True)
    # published parameters evaluated on the same data
    Dp = D[(D.t == 0) | (D.afford == 1)]
    m = GLMM(D, 'count')
    th = -0.304 + 2.41 * GH_X[None, :]
    lin = np.where(m.first[:, None], 1.807 + 0.818 * th, (m.X @ np.array([-0.010, -0.193, 0.370, 1.521]))[:, None] + th)
    ll = log_expit((2 * m.y[:, None] - 1) * lin); per = np.zeros((m.np_, len(GH_X))); np.add.at(per, m.codes, ll)
    mx = per.max(1, keepdims=True)
    models['published'] = dict(nll=float(-(mx[:, 0] + np.log(np.exp(per - mx) @ GH_W)).sum()), s1=0.177)
    out['coop_models'] = models
    # baseline-only, contemporaneous, per condition, LOCO
    tasks = [('baseline_count_t', 'count_t', 'study', 'baseline'), ('baseline_count', 'count', 'study', 'baseline'),
             ('baseline_frac_t', 'frac_t', 'study', 'baseline'),
             ('baseline_contemp', 'count_contemp', 'study', 'baseline')]
    tasks += [(f'cond_{c}', 'count_t', 'cond', c) for c in CONDS]
    tasks += [(f'loco_count_t_{c}', 'count_t', 'loco', c) for c in CONDS]
    tasks += [(f'loco_frac_t_{c}', 'frac_t', 'loco', c) for c in CONDS]
    with Pool(min(nproc, len(tasks)), initializer=_init, initargs=(D,)) as P:
        extra = dict(P.map(fit_task, tasks, chunksize=1))
    out['extra_fits'] = extra
    print('extra fits', round(time.time() - t0), flush=True)
    # bootstrap
    btasks = [(fam, 1000 + i) for fam in ['count_t', 'frac_t'] for i in range(nboot)]
    with Pool(nproc, initializer=_init, initargs=(D,)) as P:
        bres = P.map(boot_task, btasks, chunksize=1)
    boot = {'count_t': [b for (f, _), b in zip(btasks, bres) if f == 'count_t'],
            'frac_t': [b for (f, _), b in zip(btasks, bres) if f == 'frac_t']}
    bc = np.array(boot['count_t']); s1b = bc[:, 4] + bc[:, 5]
    out['bootstrap'] = dict(n=nboot, s1_mean=float(s1b.mean()), s1_sd=float(s1b.std(ddof=1)),
                            s1_q=np.percentile(s1b, [2.5, 5, 50, 95, 97.5]).tolist(), p_s1_pos=float((s1b > 0).mean()))
    json.dump(boot, open(f'{OUT}/bootstrap_draws.json', 'w'))
    print('bootstrap', out['bootstrap'], round(time.time() - t0), flush=True)
    # acceptance
    y = R.acc.values.astype(float)
    acc = {}
    for name, kw in [('const', dict(deg_terms=False)), ('deg', {}), ('deg_dec', dict(dec_action=True)),
                     ('within', dict(within=True)), ('within_dec', dict(within=True, dec_action=True))]:
        X = acc_design(R, **kw); b, f = fit_logit(X, y)
        acc[name] = dict(par=b.tolist(), nll=float(f), k=X.shape[1])
    codes, _ = pd.factorize(R.pid)
    b, f = fit_logit_re(acc_design(R), y, codes)
    acc['deg_re'] = dict(par=b.tolist(), nll=float(f), sigma=float(np.exp(b[-1])))
    b, f = fit_logit_re(acc_design(R, within=True), y, codes)
    acc['within_re'] = dict(par=b.tolist(), nll=float(f), sigma=float(np.exp(b[-1])))
    acc['rates'] = R.groupby(['val', 'ref_coop', 'dec_coop']).acc.agg(['mean', 'size']).reset_index().to_dict('records')
    with Pool(nproc, initializer=_init, initargs=(D, R)) as P:
        ab = P.map(acc_boot_task, range(5000, 5000 + nboot), chunksize=4)
    A1 = np.array([a for a, _ in ab]); A2 = np.array([b for _, b in ab])
    acc['deg_boot_ci'] = np.percentile(A1, [2.5, 97.5], axis=0).tolist()
    acc['within_boot_ci'] = np.percentile(A2, [2.5, 97.5], axis=0).tolist()
    json.dump({'deg': A1.tolist(), 'within': A2.tolist()}, open(f'{OUT}/accept_boot_draws.json', 'w'))
    acc['names_deg'] = ['add_C', 'add_D', 'del_C', 'del_D', 'add_degdec', 'del_degdec', 'add_degref', 'del_degref']
    acc['names_within'] = ['add_C', 'add_D', 'del_C', 'del_D', 'add_dev', 'del_dev', 'add_mean', 'del_mean', 'add_degref', 'del_degref']
    acc['n'] = int(len(R)); acc['n_groups'] = int(R.group.nunique()); acc['deg_dec_mean'] = float(R.deg_dec.mean())
    out['accept'] = acc
    # human outcomes and recommendations actually issued in the human experiments
    go = pd.concat([pd.read_csv(f'{OSF}/{k}_group_outcomes_data.csv') for k in ['baseline', 'evaluation', 'validation']])
    go['Condition'] = go.Condition.str.replace('Max connectivity', 'Max. connectivity', regex=False)
    fin = go[go.Round == 14].groupby('Condition').Mean_Cooperation.agg(['mean', 'sem', 'count'])
    out['human_final'] = fin.reset_index().to_dict('records')
    gs = pd.concat([pd.read_csv(f'{OSF}/{k}_graph_struct_data.csv') for k in ['baseline', 'evaluation', 'validation']])
    gs['Condition'] = gs.Condition.str.replace('Max connectivity', 'Max. connectivity', regex=False)
    rp = gs[(gs.Recommendation != 0) & (gs.Round < 14)].groupby(['Condition', 'Group', 'Round']).size().groupby('Condition').mean() / 16
    out['human_recs_per_person_round'] = rp.round(3).to_dict()
    go.groupby(['Condition', 'Round']).agg(coop=('Mean_Cooperation', 'mean'), conn=('Connectivity', 'mean')).reset_index().to_csv(f'{OUT}/human_traj.csv', index=False)
    json.dump(out, open(f'{OUT}/calib_v2.json', 'w'), indent=1, default=float)
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32, int(sys.argv[4]) if len(sys.argv) > 4 else 200)
