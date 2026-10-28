"""Revised experiment driver (v2).  Usage: python runs2.py <calib_v2_dir> <outdir> <phase> [smoke] [nproc]
phases: levdata, levfit, submod, eval, all
The calibration directory must contain calib_v2.json, bootstrap_draws.json and accept_boot_draws.json.
"""
import sys, os, json, time, pickle, itertools, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
import engine2 as E

CDIR, OUT, PHASE = sys.argv[1], sys.argv[2], sys.argv[3]
SMOKE = len(sys.argv) > 4 and sys.argv[4] == 'smoke'
NPROC = int(sys.argv[5]) if len(sys.argv) > 5 else 64
os.makedirs(OUT, exist_ok=True)
CAL = json.load(open(f'{CDIR}/calib_v2.json'))
BOOT = json.load(open(f'{CDIR}/bootstrap_draws.json'))
ABOOT = json.load(open(f'{CDIR}/accept_boot_draws.json'))
WORLDS = E.make_worlds(CAL)
ACC = CAL['accept']
COND_PLANNER = {'Static network': 'static', 'Random rec.': 'random', 'Coop. clustering': 'clustering',
                'Encourag. planner': 'enc_full', 'Neutral planner': 'neutral', 'Max. connectivity': 'maxconn'}
_LEV, _WC = {}, {}
RHOS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]


def get_lev(name):
    if name not in _LEV:
        _LEV[name] = pickle.load(open(f'{OUT}/{name}.pkl', 'rb'))
    return _LEV[name]


def acc_vector(variant):
    """Acceptance parameter vector for the simulator (BModel layout) and degree centring."""
    if variant == 'const':
        return 'const', 0.0
    if variant == 'deg':
        return ACC['deg']['par'], 0.0
    if variant == 'dec':
        return ACC['deg_dec']['par'], 0.0
    if variant == 'within':      # within-person degree slope; between-person component frozen at the data mean
        p = np.asarray(ACC['within']['par']); c = ACC['deg_dec_mean']
        base = p[:4] + np.r_[p[6], p[6], p[7], p[7]] * c
        return np.r_[base, p[4], p[5], p[8], p[9]].tolist(), c
    raise ValueError(variant)


def coop_par(name):
    cm = CAL['coop_models']
    if name == 'cnt':
        return 'count_t', cm['count_t']['par']
    if name == 'frac':
        return 'frac_t', cm['frac_t']['par']
    if name == 'base':
        return 'count_t', CAL['extra_fits']['baseline_count_t']['par']
    raise ValueError(name)


def world_of(key):
    """Standard world names, or composite keys:
    boot:<fam>:<i>   bootstrap draw i of the cooperation model (fam count_t|frac_t) paired with acceptance draw i
    loco:<fam>:<cond> model refitted without condition <cond>
    s1:<path>:<val>  count world with density response s1 set to <val> by moving beta_k (path k) or beta_n (path n)
    mech:<coop>:<acc>:<afford>  coop in {pub,cnt,frac}, acc in {const,deg}, afford in {0,1}
    acc:<coop>:<variant>         coop in {cnt,frac}, acceptance variant in {const,deg,dec,within}"""
    if key in WORLDS:
        return WORLDS[key]
    if key in _WC:
        return _WC[key]
    f = key.split(':')
    if f[0] == 'boot':
        fam, i = f[1], int(f[2])
        acc = ABOOT['deg'][i % len(ABOOT['deg'])]
        w = E.World(key, E.model_from_par(fam, BOOT[fam][i], acc))
    elif f[0] == 'loco':
        fam, cond = f[1], f[2]
        w = E.World(key, E.model_from_par(fam, CAL['extra_fits'][f'loco_{fam}_{cond}']['par'], ACC['deg']['par']))
    elif f[0] == 's1':
        path, val = f[1], float(f[2])
        par = np.array(CAL['coop_models']['count_t']['par'], float)
        s1 = par[4] + par[5]
        if path == 'k':
            par[4] += val - s1
        else:
            par[5] += val - s1
        w = E.World(key, E.model_from_par('count_t', par, ACC['deg']['par']))
    elif f[0] == 'mech':
        coop, acc, aff = f[1], f[2], f[3] == '1'
        av, c = acc_vector(acc)
        if coop == 'pub':
            m = E.BModel(**{**E.PUB, 'acc': av})
        else:
            fam, par = coop_par(coop); m = E.model_from_par(fam, par, av)
        m.deg_center = c
        w = E.World(key, m, afford=aff)
    elif f[0] == 'acc':
        coop, var = f[1], f[2]
        av, c = acc_vector(var)
        fam, par = coop_par(coop); m = E.model_from_par(fam, par, av); m.deg_center = c
        w = E.World(key, m)
    else:
        raise ValueError(key)
    _WC[key] = w
    return w


def seed_of(skey, n, g, fam=0):
    return int(fam * 10 ** 9 + (zlib.crc32(skey.encode()) % 997) * 10 ** 6 + n * 10 ** 3 + g) % (2 ** 62)


def make_planner(spec, n):
    k = spec['kind']
    K = spec.get('K')
    if spec.get('rho') is not None:
        K = max(1, int(round(spec['rho'] * n)))
    dcap = spec.get('dcap')
    model = None
    if spec.get('model') == 'match':
        model = world_of(spec['world']).m
    if k == 'lev':
        return E.Planner('lev', K=K, lev=get_lev(spec['lev']), dcap=dcap)
    if k in ('mb', 'local', 'cand2'):
        return E.Planner(k, K=K, model=model, eps=spec.get('eps'), use_hist=spec.get('use_hist', True),
                         capital_known=spec.get('capital_known', True), radius=spec.get('radius'), dcap=dcap)
    return E.Planner(k, K=K, dcap=dcap)


def eval_task(args):
    spec, g0, g1 = args
    w = world_of(spec['world'])
    n = spec['n']
    rows = []
    for gi in range(g0, g1):
        pl = make_planner(spec, n)
        sk = spec.get('skey', spec['world']) + f"|{spec.get('graph', 'er')}|{spec.get('mean_deg', 4.5)}|{spec.get('T', 15)}"
        out = E.run_game(w, n, seed_of(sk, n, gi, fam=spec.get('fam', 1)), pl, T=spec.get('T', 15), p=spec.get('p'),
                         mean_deg=spec.get('mean_deg', 4.5), graph=spec.get('graph', 'er'))
        traj = out.pop('coop_traj')
        if spec.get('keep_traj'):
            out['traj'] = ';'.join(f'{x:.4f}' for x in traj)
        r = {k: v for k, v in spec.items() if k != 'keep_traj'}
        r.update(out)
        r['game'] = gi
        r['K_eff'] = pl.K if pl.K is not None else -1
        rows.append(r)
    return rows


# ====================================================================== experiment grid
def grid():
    S = []
    G = lambda x: (2 if SMOKE else x)
    add = lambda exp, world, n, kind, ng, **kw: S.append(dict(exp=exp, world=world, n=n, kind=kind, ngames=G(ng), **kw))
    basic = ['static', 'random', 'clustering', 'enc_full', 'neutral', 'maxconn']
    # E0 validation: calibrated worlds, baseline-only out-of-sample world, lagged-action world, LOCO closed loop
    for w in ['pub', 'cnt', 'frac', 'base', 'lag', 'frac0']:
        for k in basic:
            add('E0', w, 16, k, 1000, keep_traj=w in ('pub', 'cnt', 'frac', 'base'))
    for fam in ['count_t', 'frac_t']:
        for cond, k in COND_PLANNER.items():
            add('E0loco', f'loco:{fam}:{cond}', 16, k, 500, cond=cond, fam_fit=fam, keep_traj=True)
    # E1 budget curves at n=16 with matched candidate-pool controls and matched internal models
    for w in ['pub', 'cnt', 'frac']:
        for k in ['static', 'enc_full', 'det_full']:
            add('E1', w, 16, k, 1000)
        for K in [1, 2, 4, 8, 16, 32, 64]:
            for k in ['rand_thin', 'enc_dd', 'det_rand', 'type_dd', 'type_cc', 'dd_only']:
                add('E1', w, 16, k, 1000, K=K)
            add('E1', w, 16, 'mb', 1000, K=K, model='pub')
            if w != 'pub':
                add('E1', w, 16, 'mb', 1000, K=K, model='match')
            add('E1', w, 16, 'lev', 600, K=K, lev='lev16')
        add('E1', w, 16, 'mb', 1000, K=None, model='pub')
        if w != 'pub':
            add('E1', w, 16, 'mb', 1000, K=None, model='match')
    # E2 behavioural robustness worlds
    for w in ['pub_k05', 'pub_w03', 'pub_unan']:
        for k in ['static', 'enc_full', 'det_full']:
            add('E2', w, 16, k, 600)
        for K in [2, 4, 8, 16]:
            for k in ['rand_thin', 'enc_dd', 'type_dd', 'type_cc']:
                add('E2', w, 16, k, 600, K=K)
            add('E2', w, 16, 'mb', 600, K=K, model='pub')
            add('E2', w, 16, 'lev', 400, K=K, lev='lev16')
    # E3 information factorial (candidate set x valuation information), internal model matched to the world
    for w in ['pub', 'cnt', 'frac']:
        for n, ng in [(16, 600), (64, 200)]:
            base = dict(K=n, dcap=15, model='match')
            add('E3', w, n, 'static', ng, info='static')
            add('E3', w, n, 'rand_thin', ng, K=n, dcap=15, info='random')
            add('E3', w, n, 'mb', ng, info='central_full', **base)
            add('E3', w, n, 'mb', ng, info='central_exact_nocap', capital_known=False, **base)
            add('E3', w, n, 'mb', ng, info='central_nohist_nocap', capital_known=False, use_hist=False, **base)
            add('E3', w, n, 'local', ng, info='embedded_global', radius=None, capital_known=False, use_hist=False, **base)
            add('E3', w, n, 'local', ng, info='embedded_r2', radius=2, capital_known=False, use_hist=False, **base)
            add('E3', w, n, 'local', ng, info='embedded_r1', radius=1, capital_known=False, use_hist=False, **base)
            add('E3', w, n, 'cand2', ng, info='cand2_global', **base)
            for eps in [0.25, 0.5, 1, 2, 4]:
                add('E3', w, n, 'mb', ng, info='ldp_bayes', eps=eps, **base)
                add('E3', w, n, 'mb', ng, info='ldp_plugin', eps=eps, use_hist='naive', **base)
    # E4 scaling
    ngs = {16: 400, 32: 300, 64: 200, 128: 120, 256: 80, 512: 60}
    for w in ['pub', 'cnt', 'frac']:
        for n, ng in ngs.items():
            add('E4', w, n, 'static', ng)
            add('E4', w, n, 'enc_full', ng)
            add('E4', w, n, 'enc_full', ng, dcap=15)
            add('E4', w, n, 'det_full', ng)
            for rho in RHOS:
                add('E4', w, n, 'type_dd', ng, rho=rho)
                add('E4', w, n, 'mb', ng, rho=rho, model='pub')
                if w != 'pub':
                    add('E4', w, n, 'mb', ng, rho=rho, model='match')
                add('E4', w, n, 'mb', ng, rho=rho, model='match' if w != 'pub' else 'pub', dcap=15)
    # E5 fixed density p=0.3
    for w in ['pub', 'cnt', 'frac']:
        for n, ng in [(16, 400), (32, 300), (64, 200), (128, 100)]:
            for k in ['static', 'enc_full']:
                add('E5', w, n, k, ng, p=0.3)
            add('E5', w, n, 'rand_thin', ng, K=n, p=0.3)
            add('E5', w, n, 'mb', ng, K=n, p=0.3, model='match' if w != 'pub' else 'pub')
    # E6 degree-cap sweep (admission rule)
    for w in ['pub', 'cnt', 'frac']:
        for n, ng in [(64, 150), (256, 80)]:
            for cap in [3, 4, 5, 6, 8, 10, 15, 20, 30, 50, None]:
                add('E6', w, n, 'enc_full', ng, dcap=cap)
                add('E6', w, n, 'det_full', ng, dcap=cap)
                add('E6', w, n, 'type_dd', ng, rho=1, dcap=cap)
                add('E6', w, n, 'mb', ng, rho=1, dcap=cap, model='pub')
                if w != 'pub':
                    add('E6', w, n, 'mb', ng, rho=1, dcap=cap, model='match')
    # E7 density-response sweep along two parameter paths
    s1g = [-0.10, -0.075, -0.05, -0.035, -0.0198, -0.01, 0.0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.1, 0.125, 0.15, 0.177]
    for path in ['k', 'n']:
        for v in s1g:
            wk = f's1:{path}:{v}'
            for n, ng in [(64, 150), (256, 50)]:
                for k, kw in [('static', {}), ('enc_full', {}), ('enc_full', {'dcap': 15}), ('det_full', {})]:
                    add('E7', wk, n, k, ng, path=path, s1=v, skey='s1sweep', **kw)
                for rho in RHOS[:5]:
                    add('E7', wk, n, 'type_dd', ng, rho=rho, path=path, s1=v, skey='s1sweep')
    # E8 propagation of estimation uncertainty: bootstrap parameter draws (cooperation x acceptance)
    nb = 4 if SMOKE else min(len(BOOT['count_t']), 200)
    for fam in ['count_t', 'frac_t']:
        for i in range(nb):
            wk = f'boot:{fam}:{i}'
            for k, kw in [('static', {}), ('enc_full', {}), ('det_full', {}), ('type_dd', {'K': 2}), ('mb', {'K': 2, 'model': 'match'}),
                          ('mb', {'K': 2, 'model': 'pub'})]:
                add('E8', wk, 16, k, 100, draw=i, fam_fit=fam, skey='boot', **kw)
            for k, kw in [('static', {}), ('enc_full', {}), ('enc_full', {'dcap': 15}), ('type_dd', {'rho': 1 / 8}),
                          ('mb', {'rho': 1, 'model': 'match'}), ('mb', {'rho': 1, 'model': 'match', 'dcap': 15})]:
                add('E8', wk, 256, k, 10, draw=i, fam_fit=fam, skey='boot', **kw)
    # E9 mechanism decomposition
    for coop in ['pub', 'cnt', 'frac']:
        for acc in ['const', 'deg']:
            for aff in ['1', '0']:
                wk = f'mech:{coop}:{acc}:{aff}'
                for n, ng in [(128, 100), (256, 60)]:
                    for k, kw in [('static', {}), ('enc_full', {}), ('det_full', {}), ('type_dd', {'rho': 1 / 8}),
                                  ('mb', {'rho': 1, 'model': 'match'})]:
                        add('E9', wk, n, k, ng, coop=coop, acc=acc, afford=int(aff), skey='mech', **kw)
    # E10 context sensitivity at n=256
    ctx = [dict(ctx='er3', mean_deg=3.0), dict(ctx='er4.5'), dict(ctx='er8', mean_deg=8.0), dict(ctx='ht4.5', graph='ht'),
           dict(ctx='sbm4.5', graph='sbm'), dict(ctx='T10', T=10), dict(ctx='T25', T=25)]
    for w in ['pub', 'cnt', 'frac']:
        for c in ctx:
            for k, kw in [('static', {}), ('enc_full', {}), ('enc_full', {'dcap': 6}), ('enc_full', {'dcap': 8}),
                          ('enc_full', {'dcap': 15}), ('mb', {'rho': 1, 'dcap': 8, 'model': 'match' if w != 'pub' else 'pub'})]:
                add('E10', w, 256, k, 60, **c, **kw)
            for rho in RHOS[:5]:
                add('E10', w, 256, 'type_dd', 60, rho=rho, **c)
    # E11 acceptance-model sensitivity
    for coop in ['cnt', 'frac']:
        for var in ['const', 'deg', 'dec', 'within']:
            wk = f'acc:{coop}:{var}'
            for k, kw in [('static', {}), ('enc_full', {}), ('det_full', {}), ('type_dd', {'K': 2}), ('mb', {'K': 2, 'model': 'match'})]:
                add('E11', wk, 16, k, 600, coop=coop, accv=var, skey='accsens', **kw)
            for k, kw in [('static', {}), ('enc_full', {}), ('enc_full', {'dcap': 15}), ('type_dd', {'rho': 1 / 8}),
                          ('mb', {'rho': 1, 'model': 'match'}), ('mb', {'rho': 1 / 8, 'model': 'match'})]:
                add('E11', wk, 256, k, 60, coop=coop, accv=var, skey='accsens', **kw)
    return S


def cost(spec):
    n = spec['n']
    c = (n / 16) ** 1.6 * spec.get('T', 15) / 15
    if spec['kind'] in ('mb', 'lev', 'cand2'):
        c *= 3
    if spec['kind'] == 'local':
        c *= 20 * (n / 16)
    return c


def run_eval():
    specs = grid()
    if SMOKE:
        specs = [s for s in specs if s['n'] <= 64 or s['exp'] in ('E4',)][::1]
    tasks = []
    for s in specs:
        ng = s.pop('ngames')
        per = max(1, int(round(40 / cost(s))))
        for g0 in range(0, ng, per):
            tasks.append((s, g0, min(ng, g0 + per)))
    tasks.sort(key=lambda t: -cost(t[0]) * (t[2] - t[1]))
    print('eval specs', len(specs), 'tasks', len(tasks), 'games', sum(t[2] - t[1] for t in tasks), flush=True)
    t0 = time.time(); rows = []
    with Pool(NPROC) as P:
        for i, rr in enumerate(P.imap_unordered(eval_task, tasks, chunksize=1)):
            rows.extend(rr)
            if i % 2000 == 0:
                print('eval', i, len(tasks), round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/eval_games_v2.csv.gz', index=False)
    print('eval done', len(df), round(time.time() - t0), flush=True)
    return df


# ====================================================================== LEV (training data with split-half replicate labels)
def levdata_task(args):
    n, ep0, nep, reps = args
    rng = np.random.default_rng(10 ** 6 + n * 1000 + ep0)
    w = WORLDS['pub']
    rows = []
    for ep in range(ep0, ep0 + nep):
        Kb = int(rng.choice([1, 2, 4, 8, 16])) * max(1, n // 16)
        pl = E.Planner('mb', K=Kb)
        g = E.Game(w, n, seed_of('lev', n, ep, fam=7))
        pl.reset(g)
        tstar = int(rng.integers(1, g.T))
        for t in range(1, tstar + 1):
            g.play_round(); pl.observe(g)
            if t < tstar:
                i, j, v = pl.recommend(g); g.apply_recs(i, j, v)
        bi, bj, _ = pl.recommend(g)
        V = pl.belief.value_matrix(g)
        iu, ju = np.triu_indices(n, 1)
        chosen = set(zip(bi.tolist(), bj.tolist()))
        order = np.argsort(-V[iu, ju])
        top = [k for k in order[:3 * Kb + 10] if (iu[k], ju[k]) not in chosen]
        if rng.random() < 0.5 and top:
            k = int(rng.choice(top))
        else:
            k = int(rng.integers(len(iu)))
            if (iu[k], ju[k]) in chosen:
                continue
        e = (int(iu[k]), int(ju[k]))
        X = E.edge_features(g, pl.belief, np.array([e[0]]), np.array([e[1]]), V)[0]
        dfin = []
        for r in range(reps):
            g0 = g.copy(); g0.rep = r + 1
            g1 = g.copy(); g1.rep = r + 1
            b0 = E.continue_game(g0, E.clone_planner(pl))
            b1 = E.continue_game(g1, E.clone_planner(pl), [e])
            dfin.append(b1.hist[-1].sum() - b0.hist[-1].sum())
        dfin = np.array(dfin, float)
        rows.append(np.r_[X, dfin.mean(), dfin[0::2].mean(), dfin[1::2].mean(), n, tstar, Kb, ep])
    return rows


def run_levdata(n, neps, reps):
    chunk = 25
    tasks = [(n, s, min(chunk, neps - s), reps) for s in range(0, neps, chunk)]
    with Pool(NPROC) as P:
        res = P.map(levdata_task, tasks, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr], columns=E.FEATURES + ['y_final', 'y_half1', 'y_half2', 'n', 'tstar', 'Kb', 'episode'])
    df.to_csv(f'{OUT}/levdata_n{n}.csv.gz', index=False)
    return df


def run_levfit():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    df = pd.read_csv(f'{OUT}/levdata_n16.csv.gz')
    X, y = df[E.FEATURES].values, df.y_final.values
    rng = np.random.default_rng(0)
    te = rng.random(len(df)) < 0.2
    mk = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                               l2_regularization=1.0, random_state=0)
    m = mk(); m.fit(X[~te], y[~te]); pr = m.predict(X[te])
    a = np.polyfit(df.mb_val.values[~te], y[~te], 1); pb = np.polyval(a, df.mb_val.values[te])
    r2 = lambda p: float(1 - np.mean((y[te] - p) ** 2) / np.var(y[te]))
    h1, h2 = df.y_half1.values, df.y_half2.values
    rel = float(np.corrcoef(h1, h2)[0, 1])
    rep = dict(n_samples=int(len(df)), n_test=int(te.sum()), r2_lev=r2(pr), r2_mb=r2(pb),
               corr_lev=float(np.corrcoef(pr, y[te])[0, 1]), corr_mb=float(np.corrcoef(pb, y[te])[0, 1]),
               y_mean=float(y.mean()), y_sd=float(y.std()), split_half_r=rel, spearman_brown=float(2 * rel / (1 + rel)),
               ceiling_r2_approx=float(2 * rel / (1 + rel)))
    # held-out permutation importance (model fitted on the training split only)
    pi = permutation_importance(m, X[te], y[te], n_repeats=10, random_state=0)
    rep['importance_heldout'] = {f: [float(a_), float(b_)] for f, a_, b_ in zip(E.FEATURES, pi.importances_mean, pi.importances_std)}
    m2 = mk(); m2.fit(X, y)
    pickle.dump(m2, open(f'{OUT}/lev16.pkl', 'wb'))
    json.dump(rep, open(f'{OUT}/levfit_v2.json', 'w'), indent=1)
    return rep


# ====================================================================== interaction / submodularity diagnostics
def submod_task(args):
    wname, n, s0, ns = args
    rng = np.random.default_rng(5 * 10 ** 6 + n * 1000 + s0 + (0 if wname == 'pub' else 777))
    w = WORLDS[wname]
    model = None if wname == 'pub' else w.m
    rows, prs = [], []
    for s in range(s0, s0 + ns):
        pl = E.Planner('mb', K=int(rng.choice([2, 4, 8])), model=model)
        g = E.Game(w, n, seed_of('submod' + wname, n, s, fam=8))
        pl.reset(g)
        tstar = int(rng.integers(1, g.T))
        for t in range(1, tstar + 1):
            g.play_round(); pl.observe(g)
            if t < tstar:
                i, j, v = pl.recommend(g); g.apply_recs(i, j, v)
        b = pl.belief
        V = b.value_matrix(g)
        iu, ju = np.triu_indices(n, 1)
        vv = V[iu, ju]
        top = np.argsort(-vv)[:20]
        cand = [(int(iu[k]), int(ju[k])) for k in top]
        f = lambda S: E.onestep_value(g, b, list(S))
        f0 = f([])
        single = {e: f([e]) - f0 for e in cand}
        npos = sum(v_ > 1e-12 for v_ in single.values())
        for K in [2, 3]:
            combs = [()] + [S for k_ in range(1, K + 1) for S in itertools.combinations(cand, k_)]
            vals = {S: f(S) - f0 for S in combs}
            best_le = max(vals.values())
            best_eq = max(v_ for S, v_ in vals.items() if len(S) == K)
            topk = sorted(cand, key=lambda e: -single[e])[:K]
            ftop = f(topk) - f0
            S = []
            for _ in range(K):        # greedy with stopping when the best marginal gain is non-positive
                fS = f(S)
                e = max([c for c in cand if c not in S], key=lambda c: f(S + [c]))
                if f(S + [e]) - fS <= 1e-12:
                    break
                S.append(e)
            fgr = f(S) - f0
            rows.append(dict(kind='opt', world=wname, n=n, t=tstar, K=K, f_opt_le=best_le, f_opt_eq=best_eq, f_topk=ftop,
                             f_greedy=fgr, greedy_size=len(S), n_pos_single=npos))
        for k in [2, 4, 8]:
            for _ in range(3):
                idx = rng.permutation(len(cand))
                Sset = [cand[x] for x in idx[:k]]
                L = [cand[x] for x in idx[k:2 * k]]
                fS = f(Sset)
                margs = [f(Sset + [e]) - fS for e in L]
                num = sum(margs)
                den = f(Sset + L) - fS
                nodesS = {u for e in Sset for u in e}
                shared = [len(set(e) & nodesS) > 0 for e in L]
                rows.append(dict(kind='gamma', world=wname, n=n, t=tstar, K=k, num=num, den=den,
                                 frac_shared=float(np.mean(shared)), n_neg_marg=int(sum(m_ < -1e-12 for m_ in margs)),
                                 n_neg_marg_pos_single=int(sum((m_ < -1e-12) and single[e] > 1e-12 for m_, e in zip(margs, L)))))
        # pairwise interactions: one-step (model) and multi-step under static and adaptive continuations (CRN)
        for _ in range(2):
            a_, b_ = rng.choice(10, 2, replace=False)
            e1, e2 = cand[a_], cand[b_]
            shared = len({*e1} & {*e2}) > 0
            I1 = f([e1, e2]) - f([e1]) - f([e2]) + f0
            for cont in ['static', 'type_dd', 'mb']:
                res = {k: [] for k in ['0', '1', '2', '12']}
                for r in range(32):
                    for key, ex in [('0', []), ('1', [e1]), ('2', [e2]), ('12', [e1, e2])]:
                        gg = g.copy(); gg.rep = r + 1
                        if cont == 'mb':
                            pc = E.clone_planner(pl)
                        else:
                            pc = E.Planner(cont, K=2 if cont == 'type_dd' else None); pc.reset(gg)
                        gg2 = E.continue_game(gg, pc, ex)
                        res[key].append(gg2.hist[-1].sum())
                d = np.array(res['12']) - np.array(res['1']) - np.array(res['2']) + np.array(res['0'])
                m_ = {k: np.mean(v_) for k, v_ in res.items()}
                prs.append(dict(world=wname, n=n, t=tstar, shared=shared, cont=cont, I_onestep=I1, I_final=d.mean(),
                                I_final_se=d.std(ddof=1) / np.sqrt(len(d)), d1=m_['1'] - m_['0'], d2=m_['2'] - m_['0'],
                                d12=m_['12'] - m_['0'], s1=single[e1], s2=single[e2]))
    return rows, prs


def run_submod(ns):
    chunk = 5
    tasks = [(w, 16, s, min(chunk, ns - s)) for w in ['pub', 'cnt'] for s in range(0, ns, chunk)]
    with Pool(NPROC) as P:
        res = P.map(submod_task, tasks, chunksize=1)
    pd.DataFrame([r for rr, _ in res for r in rr]).to_csv(f'{OUT}/submod_v2.csv', index=False)
    pd.DataFrame([r for _, pp in res for r in pp]).to_csv(f'{OUT}/interactions_v2.csv', index=False)


if __name__ == '__main__':
    t0 = time.time()
    ph = PHASE
    if ph in ('levdata', 'all'):
        run_levdata(16, 40 if SMOKE else 40000, 2 if SMOKE else 8)
        print('levdata', round(time.time() - t0), flush=True)
    if ph in ('levfit', 'all'):
        print('levfit', run_levfit(), round(time.time() - t0), flush=True)
    if ph in ('submod', 'all'):
        run_submod(4 if SMOKE else 600)
        print('submod', round(time.time() - t0), flush=True)
    if ph in ('eval', 'all'):
        run_eval()
    print('ALL DONE', round(time.time() - t0), flush=True)
