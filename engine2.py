"""Revised simulator (revision 1) of the McKee et al. (2023) networked donation game with constrained planners.

Revision changes (each answers a referee comment):
  * One behavioural-model class (BModel) is shared by the simulated worlds and by the planners' internal
    models, so a planner can be given exactly the model that generates its world ("matched" planners),
    including the round trend, the fraction form and degree-dependent acceptance.
  * Private (randomized-response) planners: the current-round action posterior now conditions on the
    current report once, using the pre-report predictive probability (the previous code re-used the report).
  * Edit values are exact expectations over independent endpoint actions, including the correlation
    between acceptance and the decider's own gain (the previous code multiplied two separate means).
  * Embedded planners receive an explicit local information state (radius-r view, no history, no capital);
    a separate "two-hop candidates with global valuation" planner reproduces the previous design.
  * The planner no longer observes after the final round (14 reports in a 15-round game).
  * New controls: deterministic-rule parent, random order within the deterministic pool, D-D-first within the
    stochastic encouragement pool, D-D-only rule, exact-label planner without capital information.
  * Graph families (ER, heavy-tailed configuration model, stochastic block model), horizon T, initial mean
    degree, and an affordability switch for mechanism decomposition.
  * Additional outcome metrics: isolates, cooperation among connected players, degree-weighted cooperation,
    surplus, low-capital quantile, recommendations issued and accepted, cap exceedance.
"""
import numpy as np
from scipy.special import expit

B, C, D0 = 0.10, 0.05, 1.0
PHI_PUB = np.array([0.909, 0.287, 0.085, 0.774])   # add->C, add->D, del->C, del->D (SI Table 3)
ENC_CC = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, .010], [1, .010],
                   [1, .011], [1, .028], [.991, .035], [.954, .073], [1, .108]])
ENC_CD = np.array([[.993, .048], [.973, .029], [.914, .145], [.791, .213], [.644, .318], [.594, .508],
                   [.463, .608], [.429, .745], [.366, .802], [.372, .753], [.361, .741], [.371, .774],
                   [.328, .706], [.408, .722]])
ENC_DD = np.tile([0.0, 1.0], (14, 1))
NEUTRAL = np.array([[.891, .119], [.841, .054], [.656, .084], [.642, .102], [.608, .117], [.549, .204],
                    [.545, .215], [.538, .224], [.520, .239], [.504, .213], [.532, .215], [.518, .237],
                    [.529, .232], [.522, .317]])
CD_SWITCH = 6


def table_row(t, T):
    """Map decision round t (1..T-1) onto the 14-row published tables."""
    if T == 15:
        return t - 1
    return int(round(13 * (t - 1) / max(T - 2, 1)))


# =============================================================== behavioural model
class BModel:
    """family: 'count' (beta = b0, b_k, b_n, b_r), 'frac' (b0, b_l, b_r), 'count_lag' (b0, b_k, b_n, b_r, b_lag);
    beta_t: per-round trend on the 0-indexed round; r1: round-1 (a0, a1); theta ~ N(mu, sd^2).
    acc: 'const' (published rates) or a parameter vector for the logistic acceptance model:
      [add_C, add_D, del_C, del_D, add_degdec, del_degdec, add_degref, del_degref (, add_decC, del_decC)]."""

    def __init__(self, family, beta, beta_t=0.0, r1=(1.807, 0.818), mu=-0.304, sd=2.41, acc='const', kappa=1.0,
                 deg_center=0.0):
        self.family, self.beta, self.beta_t = family, np.asarray(beta, float), float(beta_t)
        self.r1, self.mu, self.sd, self.kappa = r1, mu, sd, kappa
        self.acc = acc if isinstance(acc, str) else np.asarray(acc, float)
        self.deg_center = deg_center

    def eta(self, deg, xn, t0, prev_def=None, a_prev=None):
        """Later-round linear predictor (without theta). t0 = 0-indexed round of the decision."""
        xr = np.where(deg > 0, xn / np.maximum(deg, 1), 0.0)
        k = 1.0 if prev_def is None else np.where(prev_def, self.kappa, 1.0)
        b = self.beta
        if self.family == 'frac':
            z = b[0] + b[1] * np.log1p(deg) + k * (b[2] * xr)
        else:
            z = b[0] + b[1] * deg + k * (b[2] * xn + b[3] * xr)
            if self.family == 'count_lag' and a_prev is not None:
                z = z + b[4] * a_prev
        return z + self.beta_t * t0

    def accept(self, add, ref_act, deg_dec, deg_ref, dec_act=None):
        add = np.asarray(add, bool)
        if isinstance(self.acc, str):
            return np.where(add, np.where(ref_act == 1, PHI_PUB[0], PHI_PUB[1]) + 0 * deg_dec,
                            np.where(ref_act == 1, PHI_PUB[2], PHI_PUB[3]) + 0 * deg_dec)
        p = self.acc
        base = np.where(add, ref_act * p[0] + (1 - ref_act) * p[1], ref_act * p[2] + (1 - ref_act) * p[3])
        dd = deg_dec - self.deg_center
        z = base + np.where(add, p[4] * dd + p[6] * deg_ref, p[5] * dd + p[7] * deg_ref)
        if len(p) > 8 and dec_act is not None:
            z = z + np.where(add, p[8], p[9]) * dec_act
        return expit(z)


PUB = dict(family='count', beta=[-0.010, -0.193, 0.370, 1.521], beta_t=0.0, r1=(1.807, 0.818), mu=-0.304, sd=2.41, acc='const')


def model_from_par(fam, par, acc, kappa=1.0):
    par = np.asarray(par, float)
    r1, sd = (par[0], par[1]), float(np.exp(par[2]))
    if fam == 'count_t':
        return BModel('count', par[3:7], par[7], r1, 0.0, sd, acc, kappa)
    if fam == 'frac_t':
        return BModel('frac', par[3:6], par[6], r1, 0.0, sd, acc, kappa)
    if fam == 'count_lag_t':
        return BModel('count_lag', np.r_[par[3:7], par[8]], par[7], r1, 0.0, sd, acc, kappa)
    if fam == 'count':
        return BModel('count', par[3:7], 0.0, r1, 0.0, sd, acc, kappa)
    raise ValueError(fam)


class World:
    def __init__(self, name, model, w_imit=0.0, temp=0.1, rule='fitted', afford=True):
        self.name, self.m, self.w_imit, self.temp, self.rule, self.afford = name, model, w_imit, temp, rule, afford


def make_worlds(cal):
    cm, acc = cal['coop_models'], cal['accept']
    acc_deg = acc['deg']['par']
    W = {'pub': World('pub', BModel(**PUB)),
         'cnt': World('cnt', model_from_par('count_t', cm['count_t']['par'], acc_deg)),
         'frac': World('frac', model_from_par('frac_t', cm['frac_t']['par'], acc_deg))}
    W['frac0'] = World('frac0', model_from_par('frac_t', cm['frac_t']['par'], 'const'))
    W['lag'] = World('lag', model_from_par('count_lag_t', cm['count_lag_t']['par'], acc_deg))
    W['base'] = World('base', model_from_par('count_t', cal['extra_fits']['baseline_count_t']['par'], acc_deg))
    W['pub_k05'] = World('pub_k05', BModel(**{**PUB, 'kappa': 0.5}))
    W['pub_w03'] = World('pub_w03', BModel(**PUB), w_imit=0.3)
    W['pub_unan'] = World('pub_unan', BModel(**PUB), rule='unanimity')
    return W


# =============================================================== graphs
def make_graph(rng, n, mean_deg=4.5, p=None, kind='er'):
    if p is not None or kind == 'er':
        pp = (mean_deg / (n - 1)) if p is None else p
        A = np.triu(rng.random((n, n)) < pp, 1)
    elif kind == 'ht':      # heavy-tailed: Chung-Lu with Pareto(2.5) weights scaled to the mean degree
        w = (rng.pareto(1.5, n) + 1.0)
        w = w * mean_deg / w.mean()
        P = np.minimum(np.outer(w, w) / w.sum(), 1.0)
        A = np.triu(rng.random((n, n)) < P, 1)
    elif kind == 'sbm':     # 4 equal communities, within/between edge-probability ratio 10
        q = 4
        lab = np.arange(n) % q
        same = lab[:, None] == lab[None, :]
        n_in = n / q - 1; n_out = n - n / q
        pout = mean_deg / (10 * n_in + n_out); pin = 10 * pout
        A = np.triu(rng.random((n, n)) < np.where(same, pin, pout), 1)
    else:
        raise ValueError(kind)
    return A | A.T


# =============================================================== game state
class Game:
    def __init__(self, world, n, seed, T=15, p=None, mean_deg=4.5, graph='er'):
        self.w, self.n, self.T, self.seed = world, n, T, seed
        rng = np.random.default_rng([seed, 0, 0])
        self.A = make_graph(rng, n, mean_deg, p, graph)
        m = world.m
        self.theta = rng.normal(m.mu, m.sd, n)
        self.d = np.full(n, D0)
        self.t = 0
        self.a = None
        self.pay = np.zeros(n)
        self.hist, self.coop, self.coop_conn = [], [], []
        self.nrec = self.nacc = 0
        self.rec_types = np.zeros(5)
        self.rep = 0
        self.deg0 = self.A.sum(1).copy()
        self.transfers = 0.0

    def rng(self, t, stream):
        return np.random.default_rng([self.seed, t, stream, self.rep])

    def copy(self):
        g = Game.__new__(Game)
        g.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else v))
                      for k, v in self.__dict__.items()}
        return g

    def play_round(self):
        self.t += 1
        t, n, w, m = self.t, self.n, self.w, self.w.m
        r = self.rng(t, 1)
        u_act = r.random(n)
        deg = self.A.sum(1).astype(float)
        if t == 1:
            pc = expit(m.r1[0] + m.r1[1] * self.theta)
        else:
            xn = self.A.astype(float) @ self.a
            pc = expit(m.eta(deg, xn, t - 1, self.a == 0, self.a) + self.theta)
            if w.rule == 'unanimity':
                unan = (deg > 0) & (xn == deg)
                pc = np.where(unan, np.where(self.a == 1, 0.9, 0.7), 0.02)
                pc = np.where(deg == 0, 0.5, pc)
        a = (u_act < pc).astype(int)
        if t > 1 and w.w_imit > 0:
            use = r.random(n) < w.w_imit
            upick, uacc = r.random(n), r.random(n)
            for i in np.flatnonzero(use):
                nb = np.flatnonzero(self.A[i])
                if len(nb) == 0:
                    continue
                j = nb[int(upick[i] * len(nb))]
                a[i] = self.a[j] if uacc[i] < expit((self.pay[j] - self.pay[i]) / w.temp) else self.a[i]
        if w.afford:
            a = np.where(C * deg <= self.d + 1e-12, a, 0)
        pay = B * (self.A.astype(float) @ a) - C * a * deg
        self.transfers += float((a * deg).sum())
        self.d = self.d + pay
        self.a, self.pay = a, pay
        self.hist.append(a)
        self.coop.append(a.mean())
        conn = deg > 0
        self.coop_conn.append(a[conn].mean() if conn.any() else np.nan)

    def apply_recs(self, i, j, val):
        if len(i) == 0:
            return
        t, n = self.t, self.n
        r = self.rng(t, 2)
        U_route, U_acc = r.random((n, n)), r.random((n, n))
        swap = U_route[i, j] < 0.5
        dec = np.where(swap, j, i); ref = np.where(swap, i, j)
        deg = self.A.sum(1)
        phi = self.w.m.accept(val == 1, self.a[ref], deg[dec], deg[ref], self.a[dec])
        acc = U_acc[i, j] < phi
        s = self.a[i] + self.a[j]
        self.rec_types += [np.sum((val == 1) & (s == 2)), np.sum((val == -1) & (s == 0)), np.sum((val == 1) & (s == 1)),
                           np.sum((val == -1) & (s == 1)), np.sum(((val == -1) & (s == 2)) | ((val == 1) & (s == 0)))]
        self.nrec += len(i); self.nacc += int(acc.sum())
        ia, ja, va = i[acc], j[acc], val[acc] == 1
        self.A[ia, ja] = va; self.A[ja, ia] = va

    def summary(self, dcap=None):
        d = self.d; x = np.sort(d); n = len(x)
        gini = (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()) if x.sum() > 0 else 0.0
        deg = self.A.sum(1); a = self.hist[-1]
        rt = self.rec_types / max(self.rec_types.sum(), 1)
        out = dict(final=self.coop[-1], mean=float(np.mean(self.coop)), capital=float(d.mean()), gini=float(gini),
                   cap_p10=float(np.percentile(d, 10)), surplus=float((d - D0).mean()),
                   final_conn=float(self.coop_conn[-1]) if not np.isnan(self.coop_conn[-1]) else np.nan,
                   final_degw=float((a * deg).sum() / max(deg.sum(), 1)), isolates=float((deg == 0).mean()),
                   mean_deg=float(deg.mean()), max_deg=int(deg.max()), rec_per_cap=self.nrec / (self.n * (self.T - 1)),
                   acc_per_cap=self.nacc / (self.n * (self.T - 1)), acc_rate=self.nacc / max(self.nrec, 1),
                   sh_ccadd=rt[0], sh_ddcut=rt[1], sh_cdadd=rt[2], sh_cdcut=rt[3], sh_other=rt[4],
                   transfers_per_cap=self.transfers / (self.n * self.T))
        if dcap is not None:
            out['over_cap0'] = float((self.deg0 > dcap).mean()); out['over_cap_end'] = float((deg > dcap).mean())
        return out


# =============================================================== planner's internal model and beliefs
GRID_Z = np.linspace(-3, 3, 25)


class Belief:
    """Grid posterior over each player's disposition under an internal BModel. Observations are exact or
    epsilon-LDP randomized responses."""

    def __init__(self, n, model, eps=None, use_hist=True, capital_known=True):
        self.n, self.m, self.eps, self.use_hist, self.capital_known = n, model, eps, use_hist, capital_known
        self.grid = model.mu + model.sd * GRID_Z
        pr = np.exp(-0.5 * GRID_Z ** 2)
        self.logprior = np.log(pr / pr.sum())
        self.logw = np.tile(self.logprior, (n, 1))
        self.ahat = None
        self.a_prev_hat = None
        self.q = None if eps is None else np.exp(eps) / (1 + np.exp(eps))

    def p_grid(self, deg, xn, t1, a_prev=None):
        """Cooperation probability on the theta grid for a decision in 1-indexed round t1."""
        if t1 == 1:
            return expit(self.m.r1[0] + self.m.r1[1] * self.grid)[None, :].repeat(len(deg), 0)
        z = self.m.eta(deg, xn, t1 - 1, None, a_prev)
        return expit(z[:, None] + self.grid[None, :])

    def weights(self):
        lw = self.logw if self.use_hist else np.tile(self.logprior, (self.n, 1))
        w = np.exp(lw - lw.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True)

    def observe(self, g, rng):
        t = g.t
        deg = g.A.sum(1).astype(float)
        xn = g.A.astype(float) @ self.ahat if (t > 1 and self.ahat is not None) else np.zeros(self.n)
        P = self.p_grid(deg, xn, t, self.ahat)
        if self.capital_known:
            P = P * (C * deg <= g.d - g.pay + 1e-12)[:, None]
        if self.eps is None:
            o = g.a
            lik = np.where(o[:, None] == 1, P, 1 - P)
            self.logw += np.log(np.clip(lik, 1e-12, 1))
            self.a_prev_hat, self.ahat = self.ahat, o.astype(float)
            return
        flip = rng.random(self.n) >= self.q
        o = np.where(flip, 1 - g.a, g.a)
        self.obs = o
        q = self.q
        w_pre = self.weights()                         # weights BEFORE this report
        pbar = (w_pre * P).sum(1)                      # predictive P(a_t = 1 | history)
        r1 = np.where(o == 1, q, 1 - q); r0 = np.where(o == 1, 1 - q, q)
        lik = r1[:, None] * P + r0[:, None] * (1 - P)
        self.logw += np.log(np.clip(lik, 1e-12, 1))    # theta update uses the report once
        self.a_prev_hat = self.ahat
        if self.use_hist == 'naive':
            self.ahat = o.astype(float)
        else:
            self.ahat = r1 * pbar / np.maximum(r1 * pbar + r0 * (1 - pbar), 1e-12)

    def deltas(self, g, ahat=None, deg=None, xn=None, capital=True):
        """Expected change in next-round cooperation probability for the 4 tie edits (columns:
        add a cooperating partner, add a defecting partner, cut a cooperating partner, cut a defecting partner)."""
        ah = self.ahat if ahat is None else ahat
        if deg is None:
            deg = g.A.sum(1).astype(float)
        if xn is None:
            xn = g.A.astype(float) @ ah
        w = self.weights()
        cap = g.d if (capital and self.capital_known) else np.full(len(deg), 1e9)
        t1 = g.t + 1
        def P(dg, x):
            p = self.p_grid(dg, x, t1, ah) * (C * dg <= cap + 1e-12)[:, None]
            return (w * p).sum(1)
        p0 = P(deg, xn)
        return np.stack([P(deg + 1, xn + 1) - p0, P(deg + 1, xn) - p0,
                         P(np.maximum(deg - 1, 0), np.maximum(xn - 1, 0)) - p0, P(np.maximum(deg - 1, 0), xn) - p0], 1)

    def value_matrix(self, g, ahat=None, dl=None):
        """Exact expected one-step gain in cooperators from recommending edit (i, j), averaging over the random
        decider and over independent endpoint actions with probabilities ahat."""
        ah = self.ahat if ahat is None else ahat
        if dl is None:
            dl = self.deltas(g, ah)
        A = g.A; n = A.shape[0]
        deg = A.sum(1).astype(float)
        add = ~A
        V = np.zeros((n, n))
        for ai in (0, 1):
            for aj in (0, 1):
                w = (ah if ai else 1 - ah)[:, None] * (ah if aj else 1 - ah)[None, :]
                # gain of i given partner j's action aj, and of j given i's action ai
                gi = np.where(add, dl[:, 1 - aj][:, None], dl[:, 3 - aj][:, None]) * np.ones((1, n))
                gj = np.where(add, dl[:, 1 - ai][None, :], dl[:, 3 - ai][None, :])
                gain = gi + gj
                # decider i (referent j) and decider j (referent i)
                phi_i = self.m.accept(add, np.full((n, n), aj), deg[:, None] * np.ones((1, n)), deg[None, :] * np.ones((n, 1)),
                                      np.full((n, n), ai))
                phi_j = self.m.accept(add, np.full((n, n), ai), deg[None, :] * np.ones((n, 1)), deg[:, None] * np.ones((1, n)),
                                      np.full((n, n), aj))
                V += w * 0.5 * (phi_i + phi_j) * gain
        np.fill_diagonal(V, -np.inf)
        return V


# =============================================================== planners
def finalize(g, i, j, score, K, dcap):
    order = np.argsort(-score, kind='stable')
    i, j = i[order], j[order]
    val = np.where(g.A[i, j], -1, 1)
    if dcap is None:
        if K is not None:
            i, j, val = i[:K], j[:K], val[:K]
        return i, j, val
    quota = dcap - g.A.sum(1)
    keep = []; lim = len(i) if K is None else K
    for k in range(len(i)):
        if len(keep) >= lim:
            break
        if val[k] == -1:
            keep.append(k)
        elif quota[i[k]] > 0 and quota[j[k]] > 0:
            quota[i[k]] -= 1; quota[j[k]] -= 1; keep.append(k)
    keep = np.array(keep, int)
    return i[keep], j[keep], val[keep]


def rule_candidates(g):
    A, a, t = g.A, g.a, g.t
    s = a[:, None] + a[None, :]
    sw = CD_SWITCH if g.T == 15 else int(round(CD_SWITCH * (g.T - 1) / 14))
    return (~A) & (s == 2), A & (s == 0), (s == 1) & ((~A) if t <= sw else A)


def enc_sample(g, r):
    A, n = g.A, g.n
    row = table_row(g.t, g.T)
    s = g.a[:, None] + g.a[None, :]
    padd = np.where(s == 2, ENC_CC[row, 0], np.where(s == 1, ENC_CD[row, 0], ENC_DD[row, 0]))
    pdel = np.where(s == 2, ENC_CC[row, 1], np.where(s == 1, ENC_CD[row, 1], ENC_DD[row, 1]))
    u = r.random((n, n))
    R = np.zeros((n, n), np.int8)
    R[(~A) & (u < padd)] = 1; R[A & (u < pdel)] = -1
    return np.triu(R, 1)


class Planner:
    """kind: static, random, maxconn, neutral, clustering, enc_full, rand_thin, enc_dd, det_full, det_rand, type_dd,
    type_cc, dd_only, mb, cand2 (two-hop candidates, global valuation), local (radius-r information state), lev."""

    def __init__(self, kind, K=None, model=None, eps=None, use_hist=True, capital_known=True, radius=None, lev=None,
                 dcap=None, lev_pool=None):
        self.kind, self.K, self.model, self.eps = kind, K, model, eps
        self.use_hist, self.capital_known, self.radius, self.lev = use_hist, capital_known, radius, lev
        self.dcap, self.lev_pool = dcap, lev_pool

    def reset(self, g):
        self.focal = None
        m = self.model if self.model is not None else BModel(**PUB)
        self.belief = Belief(g.n, m, eps=self.eps, use_hist=self.use_hist,
                             capital_known=(self.capital_known and self.eps is None))

    def needs_belief(self):
        return self.kind in ('mb', 'lev', 'cand2', 'local')

    def observe(self, g):
        if self.needs_belief():
            self.belief.observe(g, g.rng(g.t, 3))

    def blocked(self, g):
        if self.dcap is None:
            return np.zeros(g.A.shape, bool)
        deg = g.A.sum(1)
        return (~g.A) & ((deg[:, None] >= self.dcap) | (deg[None, :] >= self.dcap))

    def recommend(self, g):
        n, A, t, K = g.n, g.A, g.t, self.K
        r = g.rng(t, 4)
        noise = r.random((n, n))
        k = self.kind
        E0 = (np.array([], int),) * 3
        if k == 'static':
            return E0
        if k == 'random':
            i, j = np.nonzero(np.triu(r.random((n, n)) < 0.30, 1))
            return i, j, np.where(A[i, j], -1, 1)
        if k == 'maxconn':
            i, j = np.nonzero(np.triu(~A, 1)); return i, j, np.ones(len(i), int)
        if k == 'neutral':
            row = table_row(t, g.T); u = r.random((n, n))
            R = np.zeros((n, n), np.int8); R[(~A) & (u < NEUTRAL[row, 0])] = 1; R[A & (u < NEUTRAL[row, 1])] = -1
            i, j = np.nonzero(np.triu(R, 1)); return i, j, R[i, j].astype(int)
        if k == 'clustering':
            return self.clustering(g, r)
        if k in ('enc_full', 'rand_thin', 'enc_dd'):
            R = enc_sample(g, r)
            R[(R == 1) & np.triu(self.blocked(g), 1)] = 0
            i, j = np.nonzero(R)
            if k == 'enc_dd':           # D-D cuts first within the sampled pool, then random order
                dd = (g.a[i] + g.a[j] == 0) & (R[i, j] == -1)
                return finalize(g, i, j, dd * 2.0 + noise[i, j], K, self.dcap)
            return finalize(g, i, j, noise[i, j], K if k == 'rand_thin' else None, self.dcap)
        if k in ('type_dd', 'type_cc', 'det_full', 'det_rand', 'dd_only'):
            cc_add, dd_cut, cd = rule_candidates(g)
            if k == 'dd_only':
                order = [dd_cut]
            elif k == 'type_cc':
                order = [cc_add, dd_cut, cd]
            else:
                order = [dd_cut, cc_add, cd]
            score = np.zeros((n, n))
            for rank, M in enumerate(order):
                sc = (3 - rank + 0.5 * noise) if k != 'det_rand' else (1 + noise)
                score = np.where(M & (score == 0), sc, score)
            score = np.where(self.blocked(g), 0, score)
            i, j = np.nonzero(np.triu(score, 1) > 0)
            KK = None if k == 'det_full' else K
            return finalize(g, i, j, score[i, j], KK, self.dcap)
        if k in ('mb', 'cand2'):
            V = self.belief.value_matrix(g)
            if k == 'cand2':
                Af = A.astype(float); reach = A | ((Af @ Af) > 0); np.fill_diagonal(reach, False)
                return self.one_per_node(g, np.where(reach, V, -np.inf), noise)
            V = np.where(self.blocked(g), -np.inf, V) + 1e-9 * noise
            i, j = np.nonzero(np.triu(V, 1) > 1e-12)
            if self.dcap is None and K is not None and len(i) > K:
                v = V[i, j]; sel = np.argpartition(-v, K - 1)[:K]; i, j = i[sel], j[sel]
            return finalize(g, i, j, V[i, j], K, self.dcap)
        if k == 'local':
            return self.local(g, noise)
        if k == 'lev':
            return self.lev_recommend(g, noise)
        raise ValueError(k)

    def one_per_node(self, g, V, noise):
        n = g.n
        V = np.where(self.blocked(g), -np.inf, V) + 1e-9 * noise
        np.fill_diagonal(V, -np.inf)
        j = np.argmax(V, 1); best = V[np.arange(n), j]; ok = best > 1e-12
        i = np.arange(n)[ok]; j = j[ok]
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        key = np.unique(lo * n + hi); lo, hi = key // n, key % n
        return finalize(g, lo, hi, np.maximum(V[lo, hi], V[hi, lo]), self.K, self.dcap)

    # ------------------------------------------------ embedded planners with an explicit local information state
    def local_view(self, g, i, dist):
        """Information available to focal i under radius r (protocol stated in Methods): the actions of nodes within
        distance r, the edges among them, and the degree (not the neighbours) of nodes at distance r. For r = 1 the
        focal also knows that its neighbours' neighbours exist (for introductions) but nothing else about them.
        Unseen actions are imputed with the cooperation share of the focal's view. No history and no capital."""
        r = self.radius
        a = g.a.astype(float); A = g.A; deg = A.sum(1).astype(float)
        if r is None:
            view = np.ones(g.n, bool)
        else:
            view = dist[i] <= r
        share = a[view & (np.arange(g.n) != i)].mean() if (view.sum() > 1) else 0.5
        ah = np.where(view, a, share)
        Av = A & view[None, :]                                    # edges to visible nodes
        deg_v = deg.copy()
        xn_v = Av.astype(float) @ a + (deg - Av.sum(1)) * share    # unseen neighbours imputed
        return ah, deg_v, xn_v, view, share

    def local(self, g, noise):
        n, A = g.n, g.A
        b = self.belief
        r = self.radius
        from scipy.sparse.csgraph import shortest_path
        dist = shortest_path(A.astype(float), unweighted=True, directed=False) if r is not None else None
        props_i, props_j, props_v = [], [], []
        degA = A.sum(1).astype(float)
        for i in range(n):
            if r is None:
                cand = np.flatnonzero(np.arange(n) != i)
                ah, deg_v, xn_v, view, share = self.local_view(g, i, None)
            else:
                ah, deg_v, xn_v, view, share = self.local_view(g, i, dist)
                if r == 1:
                    fof = np.flatnonzero((dist[i] == 2))
                    cand = np.r_[np.flatnonzero(A[i]), fof]
                    # friends-of-friends: identity known through introduction, action and degree unknown
                    nb_deg = degA[A[i]].mean() if A[i].any() else degA.mean()
                    deg_v = deg_v.copy(); xn_v = xn_v.copy(); ah = ah.copy()
                    deg_v[fof] = nb_deg; xn_v[fof] = nb_deg * share; ah[fof] = share
                else:
                    cand = np.flatnonzero((dist[i] >= 1) & (dist[i] <= r))
            if len(cand) == 0:
                continue
            dl = b.deltas(g, ahat=ah, deg=deg_v, xn=xn_v, capital=False)
            vals = self._pair_values(i, cand, ah, deg_v, dl, A)
            vals = vals + 1e-9 * noise[i, cand]
            if self.dcap is not None:
                blk = (~A[i, cand]) & ((degA[i] >= self.dcap) | (degA[cand] >= self.dcap))
                vals = np.where(blk, -np.inf, vals)
            k = int(np.argmax(vals))
            if vals[k] > 1e-12:
                props_i.append(min(i, cand[k])); props_j.append(max(i, cand[k])); props_v.append(vals[k])
        if not props_i:
            return (np.array([], int),) * 3
        key = np.array(props_i) * n + np.array(props_j)
        uk, idx = np.unique(key, return_index=True)
        lo, hi, vv = uk // n, uk % n, np.array(props_v)[idx]
        return finalize(g, lo, hi, vv, self.K, self.dcap)

    def _pair_values(self, i, cand, ah, deg, dl, A):
        m = self.belief.m
        add = ~A[i, cand]
        v = np.zeros(len(cand))
        for ai in (0, 1):
            for aj in (0, 1):
                w = (ah[i] if ai else 1 - ah[i]) * np.where(aj, ah[cand], 1 - ah[cand])
                gi = np.where(add, dl[i, 1 - aj], dl[i, 3 - aj])
                gj = np.where(add, dl[cand, 1 - ai], dl[cand, 3 - ai])
                pi = m.accept(add, np.full(len(cand), aj), np.full(len(cand), deg[i]), deg[cand], np.full(len(cand), ai))
                pj = m.accept(add, np.full(len(cand), ai), deg[cand], np.full(len(cand), deg[i]), np.full(len(cand), aj))
                v += w * 0.5 * (pi + pj) * (gi + gj)
        return v

    # ------------------------------------------------ Shirado & Christakis (2020)-style cooperative clustering
    def clustering(self, g, r):
        n, A, a = g.n, g.A, g.a
        if self.focal is None or g.t == 1:
            self.focal = list(r.choice(n, min(5, n), replace=False))
        R = np.zeros((n, n), np.int8); new = []
        for f in self.focal:
            nb = np.flatnonzero(A[f])
            if a[f] == 1:
                dnb = nb[a[nb] == 0]
                if len(dnb):
                    j = r.choice(dnb); R[min(f, j), max(f, j)] = -1
                else:
                    cand = np.flatnonzero((a == 1) & (~A[f])); cand = cand[cand != f]
                    if len(cand):
                        j = r.choice(cand); R[min(f, j), max(f, j)] = 1
                new.append(f)
            else:
                cand = [k for k in np.flatnonzero(a == 1) if k not in self.focal and k not in new]
                new.append(r.choice(cand) if cand else f)
        self.focal = new
        tog = np.triu(r.random((n, n)) < 0.05, 1)
        R = np.triu(np.where((R == 0) & tog, np.where(A, -1, 1), R).astype(np.int8), 1)
        i, j = np.nonzero(R)
        return i, j, R[i, j].astype(int)

    def lev_recommend(self, g, noise):
        n, K = g.n, self.K
        V0 = self.belief.value_matrix(g)
        V = np.where(self.blocked(g), -np.inf, V0)
        iu, ju = np.triu_indices(n, 1)
        X = edge_features(g, self.belief, iu, ju, V0)
        pred = self.lev.predict(X) + 1e-9 * noise[iu, ju]
        pred = np.where(np.isfinite(V[iu, ju]), pred, -np.inf)
        keep = np.flatnonzero(pred > 0)
        if self.dcap is None and K is not None and len(keep) > K:
            keep = keep[np.argpartition(-pred[keep], K - 1)[:K]]
        return finalize(g, iu[keep], ju[keep], pred[keep], K, self.dcap)


FEATURES = ['add', 'n_c', 'cd', 't_frac', 't_left', 'deg_u', 'deg_v', 'xr_u', 'xr_v', 'h_u', 'h_v', 'cap_u', 'cap_v',
            'common', 'common_c', 'glob_c', 'mb_val', 'd_u', 'd_v', 'phi']


def edge_features(g, b, i, j, V=None):
    A = g.A; Af = A.astype(float); ah = b.ahat
    deg = A.sum(1).astype(float)
    xr = np.where(deg > 0, (Af @ ah) / np.maximum(deg, 1), 0.0)
    H = np.mean(g.hist, 0) if len(g.hist) else np.zeros(g.n)
    if V is None:
        V = b.value_matrix(g)
    dl = b.deltas(g)
    swap = (ah[j] > ah[i]) | ((ah[j] == ah[i]) & (deg[j] > deg[i]))
    u = np.where(swap, j, i); v = np.where(swap, i, j)
    add = (~A[i, j]).astype(float)
    common = (Af[u] * Af[v]).sum(1); common_c = (Af[u] * Af[v] * ah[None, :]).sum(1)
    def dsel(x, other):
        return np.where(add == 1, dl[x, 0] * ah[other] + dl[x, 1] * (1 - ah[other]), dl[x, 2] * ah[other] + dl[x, 3] * (1 - ah[other]))
    phi = 0.5 * (b.m.accept(add == 1, ah[v], deg[u], deg[v]) + b.m.accept(add == 1, ah[u], deg[v], deg[u]))
    return np.column_stack([add, ah[u] + ah[v], (np.abs(ah[u] - ah[v]) > 0.5).astype(float), np.full(len(i), g.t / g.T),
                            np.full(len(i), g.T - g.t), deg[u], deg[v], xr[u], xr[v], H[u], H[v], g.d[u], g.d[v],
                            common, common_c, np.full(len(i), ah.mean()), V[i, j], dsel(u, v), dsel(v, u), phi])


# =============================================================== run
def run_game(world, n, seed, planner, T=15, p=None, mean_deg=4.5, graph='er'):
    g = Game(world, n, seed, T=T, p=p, mean_deg=mean_deg, graph=graph)
    planner.reset(g)
    for t in range(1, T + 1):
        g.play_round()
        if t == T:
            break
        planner.observe(g)
        i, j, val = planner.recommend(g)
        g.apply_recs(i, j, val)
    out = g.summary(planner.dcap)
    out['coop_traj'] = list(map(float, g.coop))
    return out


def clone_planner(pl):
    q = Planner.__new__(Planner); q.__dict__ = dict(pl.__dict__)
    if hasattr(pl, 'belief'):
        b = Belief.__new__(Belief)
        b.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pl.belief.__dict__.items()}
        q.belief = b
    return q


def continue_game(g, planner, extras=()):
    """Continue a copied state (after play_round and observe at g.t, before recommendations)."""
    first = True
    while True:
        i, j, val = planner.recommend(g)
        if first and len(extras):
            for ei, ej in extras:
                if not ((i == ei) & (j == ej)).any():
                    i = np.r_[i, ei]; j = np.r_[j, ej]; val = np.r_[val, -1 if g.A[ei, ej] else 1]
        first = False
        g.apply_recs(i, j, val)
        g.play_round()
        if g.t == g.T:
            return g
        planner.observe(g)


def onestep_value(g, b, recs):
    """Exact expected number of next-round cooperators under the planner's model for the set `recs` (acceptance
    events independent; endpoint actions known, i.e. exact-label planner)."""
    ah = b.ahat; A = g.A
    deg = A.sum(1).astype(float); xn = A.astype(float) @ ah
    w = b.weights(); cap = g.d; t1 = g.t + 1
    inc = {}
    for (i, j) in recs:
        add = not A[i, j]
        phi = 0.5 * (b.m.accept(np.array([add]), np.array([ah[j]]), np.array([deg[i]]), np.array([deg[j]]), np.array([ah[i]]))[0] +
                     b.m.accept(np.array([add]), np.array([ah[i]]), np.array([deg[j]]), np.array([deg[i]]), np.array([ah[j]]))[0])
        sgn = 1 if add else -1
        inc.setdefault(i, []).append((phi, sgn, sgn * ah[j])); inc.setdefault(j, []).append((phi, sgn, sgn * ah[i]))
    base = (w * (b.p_grid(deg, xn, t1, ah) * (C * deg <= cap + 1e-12)[:, None])).sum(1)
    tot = base.sum()
    for i, L in inc.items():
        m = len(L)
        ph = np.array([x[0] for x in L]); dd = np.array([x[1] for x in L]); dx = np.array([x[2] for x in L])
        masks = ((np.arange(2 ** m)[:, None] >> np.arange(m)[None, :]) & 1).astype(float)
        pr = np.prod(np.where(masks == 1, ph[None, :], 1 - ph[None, :]), 1)
        dg = deg[i] + masks @ dd; x = xn[i] + masks @ dx
        p = b.p_grid(dg, x, t1, np.full(len(dg), ah[i])) * (C * dg <= cap[i] + 1e-12)[:, None]
        tot += (pr * (p @ w[i])).sum() - base[i]
    return tot
