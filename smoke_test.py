"""Smoke test for the simulator and planners (runs in about one minute on one core).
Usage: python smoke_test.py [calib_v2.json]
Checks: (1) exact first-round LDP posterior; (2) the vectorised edit-value matrix equals the exact one-step objective
for single edits; (3) a radius-2 embedded planner is invariant to actions beyond two hops; (4) every planner kind runs.
"""
import sys, json
from types import SimpleNamespace
import numpy as np
import engine2 as E


def main():
    W = E.make_worlds(json.load(open(sys.argv[1]))) if len(sys.argv) > 1 else {'pub': E.World('pub', E.BModel(**E.PUB))}
    # (1) LDP posterior, first round, epsilon = 1, report = defect
    b = E.Belief(1, E.BModel(**E.PUB), eps=1.0, capital_known=False)
    p = float((b.weights() * b.p_grid(np.zeros(1), np.zeros(1), 1)).sum()); q = b.q
    g = SimpleNamespace(t=1, A=np.zeros((1, 1), bool), a=np.array([0]), d=np.ones(1), pay=np.zeros(1))

    class NoFlip:
        def random(self, n):
            return np.zeros(n)   # uniform draws of 0: every report is truthful
    b.observe(g, NoFlip())
    exact = (1 - q) * p / ((1 - q) * p + q * (1 - p))
    assert abs(b.ahat[0] - exact) < 1e-9, (b.ahat[0], exact)
    print('LDP posterior ok', round(exact, 4))
    # (2) value matrix vs one-step objective
    w = W.get('cnt', W['pub'])
    gm = E.Game(w, 16, 3); pl = E.Planner('mb', K=4, model=w.m); pl.reset(gm)
    for _ in range(3):
        gm.play_round(); pl.observe(gm); i, j, v = pl.recommend(gm); gm.apply_recs(i, j, v)
    gm.play_round(); pl.observe(gm)
    V = pl.belief.value_matrix(gm); f0 = E.onestep_value(gm, pl.belief, [])
    err = max(abs(V[i, j] - (E.onestep_value(gm, pl.belief, [(i, j)]) - f0)) for i in range(16) for j in range(i + 1, 16))
    assert err < 1e-9, err
    print('value matrix ok', err)
    # (3) locality invariance on a path graph 0-1-2-3-4
    from scipy.sparse.csgraph import shortest_path
    gg = E.Game(W['pub'], 5, 1); gg.A[:] = False
    for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        gg.A[i, j] = gg.A[j, i] = True
    gg.t = 3; gg.a = np.array([1, 1, 1, 0, 0]); gg.d = np.ones(5)
    plr = E.Planner('local', K=5, radius=2, use_hist=False, capital_known=False); plr.reset(gg); plr.belief.ahat = gg.a.astype(float)
    dist = shortest_path(gg.A.astype(float), unweighted=True, directed=False)

    def vals(game):
        ah, dv, xv, view, sh = plr.local_view(game, 0, dist)
        cand = np.flatnonzero((dist[0] >= 1) & (dist[0] <= 2))
        dl = plr.belief.deltas(game, ahat=ah, deg=dv, xn=xv, capital=False)
        return plr._pair_values(0, cand, ah, dv, dl, game.A)
    v1 = vals(gg); gg.a = np.array([1, 1, 1, 1, 0]); v2 = vals(gg); gg.a = np.array([1, 1, 1, 0, 1]); v3 = vals(gg)
    assert np.allclose(v1, v2) and np.allclose(v1, v3)
    print('locality invariance ok')
    # (4) every planner kind
    kinds = [('static', {}), ('random', {}), ('clustering', {}), ('neutral', {}), ('maxconn', {}), ('enc_full', {}),
             ('rand_thin', {'K': 2}), ('enc_dd', {'K': 2}), ('det_full', {}), ('det_rand', {'K': 2}), ('type_dd', {'K': 2}),
             ('type_cc', {'K': 2}), ('dd_only', {'K': 2}), ('mb', {'K': 4}), ('mb', {'K': 16, 'eps': 1.0}),
             ('mb', {'K': 16, 'eps': 1.0, 'use_hist': 'naive'}), ('cand2', {'K': 16}),
             ('local', {'K': 16, 'radius': 2, 'use_hist': False, 'capital_known': False, 'dcap': 15})]
    for k, kw in kinds:
        out = [E.run_game(w, 16, s, E.Planner(k, **kw))['final'] for s in range(5)]
        print(f'{k:12s} {str(kw):60s} mean final cooperation {np.mean(out):.3f}')
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
