"""Regenerate every figure and summary table of the revised manuscript from game-level outputs.
Usage: python figures.py <data_dir> <out_dir>
<data_dir> must contain eval_games_v2.csv.gz, calib_v2.json, bootstrap_draws.json, submod_v2.csv, interactions_v2.csv,
levfit_v2.json and levdata_n16.csv.gz.  Outputs: <out_dir>/fig*.pdf|png and <out_dir>/tables/*.csv
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA, OUT = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('data', 'figs')
os.makedirs(f'{OUT}/tables', exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'legend.fontsize': 6, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.titlelocation': 'left',
                     'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
                     'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']})
W2 = 7.0
WCOL = {'pub': '#7b3294', 'cnt': '#008837', 'frac': '#e66101', 'base': '#5e81b5', 'lag': '#999999'}
WNAME = {'pub': 'Published model', 'cnt': 'Count model (re-est.)', 'frac': 'Fraction model (re-est.)', 'base': 'Baseline-only count model'}
PCOL = {'static': '#9e9e9e', 'enc_full': '#000000', 'enc_cap': '#009E73', 'det_full': '#56B4E9', 'type_dd': '#CC79A7',
        'dd_only': '#882255', 'mb_pub': '#0072B2', 'mb_match': '#D55E00', 'rand_thin': '#999933', 'lev': '#6a6a6a'}
PNAME = {'static': 'Static network', 'enc_full': 'Encouragement, unrationed', 'enc_cap': 'Encouragement, cap 15',
         'det_full': 'Deterministic rule, unrationed', 'type_dd': 'D–D-first', 'dd_only': 'D–D cuts only',
         'mb_pub': 'Model-based (published)', 'mb_match': 'Model-based (matched)', 'rand_thin': 'Random thinning'}
COND = {'Static network': 'static', 'Random rec.': 'random', 'Coop. clustering': 'clustering', 'Encourag. planner': 'enc_full',
        'Neutral planner': 'neutral', 'Max. connectivity': 'maxconn'}


def letter(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')


def agg(d, keys, vals=('final',)):
    g = d.groupby(keys, dropna=False)
    o = g[list(vals)].mean()
    o['n_games'] = g.size()
    for v in vals:
        o[v + '_se'] = g[v].std() / np.sqrt(g.size())
    return o.reset_index()


def pair(d, a, b, val='final', on=('world', 'n', 'game')):
    A = d[a].set_index(list(on))[val]; B = d[b].set_index(list(on))[val]
    j = A.to_frame('a').join(B.to_frame('b'), how='inner'); x = j.a - j.b
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x)


def label_cols(d):
    d = d.copy()
    for c in ['K', 'dcap', 'rho', 'eps', 'radius', 'T', 'mean_deg', 'p']:
        if c in d:
            d[c] = d[c].astype(float)
    d['model'] = d['model'].fillna('-')
    d['Kc'] = d.K.fillna(-1); d['rhoc'] = d.rho.fillna(-1); d['dc'] = d.dcap.fillna(-1)
    d['lab'] = d.kind + np.where(d.model.isin(['pub', 'match']), '_' + d.model, '')
    return d


# ------------------------------------------------------------------ load
ev = label_cols(pd.read_csv(f'{DATA}/eval_games_v2.csv.gz', low_memory=False))
cal = json.load(open(f'{DATA}/calib_v2.json'))
boot = json.load(open(f'{DATA}/bootstrap_draws.json'))
H = {r['Condition']: r['mean'] for r in cal['human_final']}
HSE = {r['Condition']: r['sem'] for r in cal['human_final']}
T = {}

# ================================================================== Figure 1: model risk in the calibration
cm, ef = cal['coop_models'], cal['extra_fits']
rows = [('Published simulator', 0.177, np.nan), ('Count, no trend', cm['count']['s1'], cm['count']['s1_se']),
        ('Count + trend (main)', cm['count_t']['s1'], cm['count_t']['s1_se']),
        ('Count + trend + lagged action', cm['count_lag_t']['s1'], cm['count_lag_t']['s1_se']),
        ('+ condition intercepts', cm['count_t_condfe']['s1'], cm['count_t_condfe']['s1_se']),
        ('+ condition-specific trends', cm['count_condtrend']['s1'], cm['count_condtrend']['s1_se']),
        ('Baseline sessions only', ef['baseline_count_t']['s1'], ef['baseline_count_t'].get('s1_se', np.nan))]
cond_rows = [(f'{c} only', ef[f'cond_{c}']['s1'], ef[f'cond_{c}'].get('s1_se', np.nan)) for c in
             ['Static network', 'Random rec.', 'Coop. clustering', 'GraphNet planner', 'Encourag. planner', 'Neutral planner', 'Max. connectivity']]
S1 = pd.DataFrame(rows + cond_rows, columns=['spec', 's1', 'se'])
S1.to_csv(f'{OUT}/tables/s1_specifications.csv', index=False)
bc = np.array(boot['count_t']); s1b = bc[:, 4] + bc[:, 5]
T['s1_boot'] = dict(mean=float(s1b.mean()), sd=float(s1b.std(ddof=1)), q025=float(np.percentile(s1b, 2.5)),
                    q975=float(np.percentile(s1b, 97.5)), p_pos=float((s1b > 0).mean()), n=int(len(s1b)))

e0 = ev[ev.exp == 'E0']
a0 = agg(e0, ['world', 'kind'])
lo = agg(ev[ev.exp == 'E0loco'], ['fam_fit', 'cond', 'kind'])
val = []
for w in ['pub', 'cnt', 'frac', 'base', 'lag', 'frac0']:
    d = a0[a0.world == w].set_index('kind')
    p = np.array([d.loc[COND[c], 'final'] for c in COND]); h = np.array([H[c] for c in COND])
    val.append(dict(world=w, mae=float(np.mean(np.abs(p - h))), spearman=float(spearmanr(p, h).correlation),
                    enc_minus_static=float(d.loc['enc_full', 'final'] - d.loc['static', 'final']),
                    static=float(d.loc['static', 'final']), maxconn=float(d.loc['maxconn', 'final'])))
for fam, d in lo.groupby('fam_fit'):
    p = d.final.values; h = d.cond.map(H).values
    val.append(dict(world=f'loco_{fam}', mae=float(np.mean(np.abs(p - h))), spearman=float(spearmanr(p, h).correlation)))
VAL = pd.DataFrame(val); VAL.to_csv(f'{OUT}/tables/validation.csv', index=False)
T['human_enc_minus_static'] = H['Encourag. planner'] - H['Static network']

fig = plt.figure(figsize=(W2, 2.75))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.75, 1.05], wspace=0.5, left=0.205, right=0.985, bottom=0.17, top=0.86)
ax = fig.add_subplot(gs[0])
S = S1.iloc[::-1].reset_index(drop=True)
y = np.arange(len(S))
for i, r in S.iterrows():
    col = WCOL['pub'] if r.spec.startswith('Published') else ('#9e9e9e' if 'only' in r.spec and 'Baseline' not in r.spec else WCOL['cnt'])
    if np.isfinite(r.se):
        ax.errorbar(r.s1, i, xerr=1.96 * r.se, fmt='o', ms=3.2, color=col, elinewidth=0.9, capsize=0)
    else:
        ax.plot(r.s1, i, 'D', ms=3.6, color=col)
ax.axvline(0, color='k', lw=0.6, ls=':')
ax.axhline(len(cond_rows) - 0.5, color='#cccccc', lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(S.spec)
ax.set_xlabel(r'Density response $s_1=\beta_k+\beta_n$ (logit units)')
ax.set_title('The sign of the density response is uncertain', loc='left')
ax.margins(y=0.03)
letter(ax, 'A', dx=-0.62)
ax = fig.add_subplot(gs[1])
ax.hist(s1b, bins=24, color=WCOL['cnt'], alpha=0.85, edgecolor='white', lw=0.3)
ax.axvline(0, color='k', lw=0.6, ls=':')
ax.set_xlabel(r'$s_1$, group bootstrap draws')
ax.set_ylabel('Draws')
ax.set_title(f"P($s_1$>0) = {T['s1_boot']['p_pos']:.2f}", loc='left')
letter(ax, 'B', dx=-0.3)
ax = fig.add_subplot(gs[2])
h = np.array([H[c] for c in COND]); hs = np.array([HSE[c] for c in COND])
mk = {'pub': 'D', 'cnt': 'o', 'frac': 's', 'base': '^'}
for w in ['pub', 'cnt', 'frac', 'base']:
    d = a0[a0.world == w].set_index('kind'); p = np.array([d.loc[COND[c], 'final'] for c in COND])
    m = VAL.set_index('world').loc[w, 'mae']
    ax.scatter(h, p, s=13, marker=mk[w], color=WCOL[w], label=f'{WNAME[w]} (MAE {m:.2f})', zorder=3, edgecolor='white', lw=0.3)
d = lo[lo.fam_fit == 'count_t']
ax.scatter(d.cond.map(H), d.final, s=16, marker='o', facecolor='none', edgecolor=WCOL['cnt'], lw=0.8,
           label=f"Count, condition held out (MAE {VAL.set_index('world').loc['loco_count_t', 'mae']:.2f})", zorder=2)
ax.plot([0, 1], [0, 1], color='#bbbbbb', lw=0.7, zorder=1)
ax.set_xlim(0.05, 0.9); ax.set_ylim(0.05, 0.9)
ax.set_xlabel('Observed final cooperation'); ax.set_ylabel('Simulated final cooperation')
ax.set_title('Fit improves, but not via $s_1$', loc='left')
ax.legend(loc='lower right', fontsize=5.0, handletextpad=0.2, borderaxespad=0.1)
letter(ax, 'C', dx=-0.3)
fig.savefig(f'{OUT}/fig1_calibration.pdf'); fig.savefig(f'{OUT}/fig1_calibration.png')
plt.close(fig)

# ================================================================== Figure 2: budgets at n = 16 and the welfare trade-off
e1 = ev[ev.exp == 'E1']
a1 = agg(e1, ['world', 'lab', 'Kc'], ('final', 'capital', 'gini', 'isolates'))
a1.to_csv(f'{OUT}/tables/E1_budget_n16.csv', index=False)
pcs = []
for w in ['pub', 'cnt', 'frac']:
    d = e1[e1.world == w]
    sel = lambda lab, K: (d.lab == lab) & (d.Kc == K)
    for (la, Ka), (lb, Kb) in [(('type_dd', 2), ('enc_full', -1)), (('dd_only', 4), ('enc_full', -1)), (('mb_match', 4), ('enc_full', -1)),
                               (('mb_match', 4), ('type_dd', 2)), (('mb_match', -1), ('enc_full', -1)), (('mb_pub', -1), ('enc_full', -1)),
                               (('enc_dd', 2), ('type_dd', 2)), (('det_full', -1), ('enc_full', -1)), (('type_dd', 2), ('det_rand', 2)),
                               (('rand_thin', 2), ('det_rand', 2)), (('mb_pub', 4), ('type_dd', 2)), (('enc_full', -1), ('static', -1))]:
        if w == 'pub' and 'match' in la:
            continue
        for v in ['final', 'capital']:
            m, se, nn = pair(d, sel(la, Ka), sel(lb, Kb), v)
            pcs.append(dict(world=w, a=f'{la}_K{int(Ka)}', b=f'{lb}_K{int(Kb)}', metric=v, diff=m, se=se, n=nn))
PC1 = pd.DataFrame(pcs); PC1.to_csv(f'{OUT}/tables/E1_paired_contrasts.csv', index=False)

e8 = ev[ev.exp == 'E8'].copy()
e8['lab2'] = e8.lab + np.where(e8.dc > 0, '_cap', '') + np.where(e8.Kc > 0, '_K' + e8.Kc.astype(int).astype(str), '') + \
    np.where(e8.rhoc > 0, '_r' + e8.rhoc.astype(str), '')
m8 = e8.groupby(['fam_fit', 'draw', 'n', 'lab2'])[['final', 'capital']].mean().reset_index()
W8 = m8.pivot_table(index=['fam_fit', 'draw', 'n'], columns='lab2', values='final')
C8 = m8.pivot_table(index=['fam_fit', 'draw', 'n'], columns='lab2', values='capital')
contr = {16: [('type_dd_K2', 'enc_full'), ('mb_match_K2', 'enc_full'), ('mb_match_K2', 'type_dd_K2'), ('enc_full', 'static'), ('mb_pub_K2', 'type_dd_K2')],
         256: [('enc_full_cap', 'enc_full'), ('type_dd_r0.125', 'static'), ('enc_full', 'static'), ('mb_match_cap_r1.0', 'mb_match_r1.0'),
               ('enc_full_cap', 'static'), ('type_dd_r0.125', 'enc_full_cap')]}
s1d = {i: bc[i, 4] + bc[i, 5] for i in range(len(bc))}
rows = []
for (fam, n), g in W8.groupby(level=[0, 2]):
    for a, b in contr[n]:
        x = (g[a] - g[b]).values; xc = (C8.loc[g.index, a] - C8.loc[g.index, b]).values
        r = dict(fam=fam, n=n, contrast=f'{a} - {b}', mean=x.mean(), q025=np.percentile(x, 2.5), q975=np.percentile(x, 97.5),
                 p_pos=(x > 0).mean(), cap_mean=xc.mean(), cap_q025=np.percentile(xc, 2.5), cap_q975=np.percentile(xc, 97.5),
                 cap_p_pos=(xc > 0).mean(), draws=len(x))
        if fam == 'count_t':
            s = np.array([s1d[dd] for dd in g.index.get_level_values(1)]); r['corr_with_s1'] = np.corrcoef(s, x)[0, 1]
        rows.append(r)
P8 = pd.DataFrame(rows); P8.to_csv(f'{OUT}/tables/E8_bootstrap_propagation.csv', index=False)

fig = plt.figure(figsize=(W2, 4.3))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], width_ratios=[1, 1, 1, 0.72], hspace=0.62, wspace=0.12, left=0.075, right=0.985, bottom=0.085, top=0.93)
Ks = [1, 2, 4, 8, 16, 32, 64]
for k, w in enumerate(['pub', 'cnt', 'frac']):
    ax = fig.add_subplot(gs[0, k])
    d = a1[a1.world == w]
    for lab in ['rand_thin', 'type_dd', 'dd_only', 'mb_pub', 'mb_match']:
        q = d[(d.lab == lab) & (d.Kc > 0)].sort_values('Kc')
        if len(q) == 0:
            continue
        ax.plot(q.Kc, q.final, '-o', ms=2.4, color=PCOL[lab], lw=1.3 if lab == 'mb_match' else 1.0, label=PNAME[lab])
    for lab, ls in [('enc_full', '--'), ('static', ':')]:
        v = d[(d.lab == lab)].final.values[0]
        ax.axhline(v, color=PCOL[lab], ls=ls, lw=0.9, label=PNAME[lab])
    ax.set_xscale('log', base=2); ax.set_xticks(Ks); ax.set_xticklabels([str(x) for x in Ks])
    ax.set_ylim(0.56, 0.86)
    ax.set_xlabel('Edits per round, $K$')
    ax.set_title(WNAME[w], loc='left', color=WCOL[w])
    if k == 0:
        ax.set_ylabel('Final cooperation')
        letter(ax, 'A', dx=-0.25)
    else:
        ax.set_yticklabels([])
    if k == 2:
        hh, ll = ax.get_legend_handles_labels()
        lax = fig.add_subplot(gs[0, 3]); lax.axis('off')
        lax.legend(hh, ll, loc='center left', fontsize=5.6, handlelength=1.8, borderaxespad=0)
gsb = gs[1, :].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62)
ax = fig.add_subplot(gsb[0])
show = [('static', -1), ('enc_full', -1), ('det_full', -1), ('type_dd', 2), ('type_dd', 4), ('dd_only', 4), ('mb_pub', -1), ('mb_pub', 4),
        ('mb_match', 4), ('mb_match', -1)]
mkw = {'pub': 'D', 'cnt': 'o', 'frac': 's'}
for w in ['pub', 'cnt', 'frac']:
    d = a1[a1.world == w]
    for lab, K in show:
        q = d[(d.lab == lab) & (d.Kc == K)]
        if len(q) == 0:
            continue
        ax.scatter(q.final, q.capital, s=16, marker=mkw[w], color=PCOL[lab], edgecolor='white', lw=0.3, zorder=3)
from matplotlib.lines import Line2D
hl = [Line2D([], [], marker=mkw[w], ls='', color='#555555', ms=3.5, label=WNAME[w]) for w in mkw]
hl += [Line2D([], [], marker='o', ls='', color=PCOL[l], ms=3.5, label=PNAME[l]) for l in ['static', 'enc_full', 'det_full', 'type_dd', 'dd_only', 'mb_pub', 'mb_match']]
ax.legend(handles=hl, loc='upper left', fontsize=5.3, ncol=2, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
ax.set_xlabel('Final cooperation'); ax.set_ylabel('Mean final capital (welfare)')
ax.set_ylim(2.5, 12.5)
ax.set_title('Rationing raises cooperation but forgoes welfare', loc='left')
letter(ax, 'B', dx=-0.16)
ax = fig.add_subplot(gsb[1])
labs = [('type_dd_K2', 'enc_full', 'D–D-first\n− encouragement'),
        ('mb_match_K2', 'enc_full', 'Matched model-based\n− encouragement'),
        ('mb_match_K2', 'type_dd_K2', 'Matched model-based\n− D–D-first'),
        ('mb_pub_K2', 'type_dd_K2', 'Published model-based\n− D–D-first')]
for j, (a, b, name) in enumerate(labs):
    for off, fam, col in [(-0.14, 'count_t', WCOL['cnt']), (0.14, 'frac_t', WCOL['frac'])]:
        g = W8.xs((fam, 16), level=[0, 2])
        x = (g[a] - g[b]).values
        yy = j + off + np.random.default_rng(j).uniform(-0.07, 0.07, len(x))
        ax.scatter(x, yy, s=2, color=col, alpha=0.35, lw=0)
        ax.plot([np.median(x)] * 2, [j + off - 0.11, j + off + 0.11], color='k', lw=1.2)
ax.axvline(0, color='k', lw=0.6, ls=':')
ax.set_yticks(range(len(labs))); ax.set_yticklabels([l[2] for l in labs], fontsize=5.6)
ax.invert_yaxis()
ax.tick_params(axis='y', length=0, pad=2)
ax.set_xlabel('Difference in final cooperation per bootstrap draw ($K$=2)')
ax.set_title('Contrasts are stable across parameter draws', loc='left')
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=WCOL[w], ms=3, label=WNAME[w]) for w in ['cnt', 'frac']], loc='lower right', fontsize=5.4, borderaxespad=0.1)
letter(ax, 'C', dx=-0.42)
fig.savefig(f'{OUT}/fig2_budget.pdf'); fig.savefig(f'{OUT}/fig2_budget.png')
plt.close(fig)

# ================================================================== Figure 3: information
e3 = ev[ev.exp == 'E3'].copy(); e3['epsc'] = e3.eps.fillna(-1)
a3 = agg(e3, ['world', 'n', 'info', 'epsc'], ('final', 'capital'))
ret = []
for (w, n), g in a3.groupby(['world', 'n']):
    s = g[g['info'] == 'static'].final.values[0]; f = g[g['info'] == 'central_full'].final.values[0]
    for _, r in g.iterrows():
        ret.append(dict(world=w, n=n, info=r['info'], eps=r.epsc, final=r.final, final_se=r.final_se, retained=(r.final - s) / (f - s)))
RET = pd.DataFrame(ret); RET.to_csv(f'{OUT}/tables/E3_information.csv', index=False)
fig = plt.figure(figsize=(W2, 2.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.3, left=0.255, right=0.985, bottom=0.17, top=0.88)
ax = fig.add_subplot(gs[0])
order = [('central_full', 'Central: full history, capital, exact actions'), ('central_exact_nocap', 'Central: no capital'),
         ('central_nohist_nocap', 'Central: no history, no capital'), ('cand2_global', 'Two-hop candidates, global valuation'),
         ('embedded_global', 'Embedded: whole-network view'), ('embedded_r2', 'Embedded: two-hop view'),
         ('embedded_r1', 'Embedded: one-hop view'), ('random', 'Random recommendations')]
for j, (key, name) in enumerate(order):
    for off, w in [(-0.22, 'pub'), (0.0, 'cnt'), (0.22, 'frac')]:
        for n, mk_ in [(16, 'o'), (64, 's')]:
            q = RET[(RET.world == w) & (RET.n == n) & (RET['info'] == key)]
            ax.scatter(q.retained, j + off + (0.06 if n == 64 else -0.06), s=11, marker=mk_, color=WCOL[w],
                       facecolor=WCOL[w] if n == 16 else 'white', lw=0.8, zorder=3)
ax.axvline(1, color='#bbbbbb', lw=0.6); ax.axvline(0, color='k', lw=0.6, ls=':')
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[1] for o in order])
ax.invert_yaxis()
ax.set_xlabel('Benefit retained relative to full central planner')
ax.set_title('Local views retain most of the benefit', loc='left')
hl = [Line2D([], [], marker='o', ls='', color=WCOL[w], ms=3.5, label=WNAME[w]) for w in ['pub', 'cnt', 'frac']]
hl += [Line2D([], [], marker='o', ls='', color='#555555', ms=3.5, label='16 players'),
       Line2D([], [], marker='s', ls='', markerfacecolor='white', color='#555555', ms=3.5, label='64 players')]
ax.legend(handles=hl, loc='lower left', fontsize=5.4, borderaxespad=0.1)
ax.set_xlim(-0.65, 1.12)
letter(ax, 'A', dx=-0.62)
ax = fig.add_subplot(gs[1])
for w in ['pub', 'cnt', 'frac']:
    for key, ls, mk_ in [('ldp_bayes', '-', 'o'), ('ldp_plugin', '--', '^')]:
        q = RET[(RET.world == w) & (RET.n == 64) & (RET['info'] == key)].sort_values('eps')
        ax.plot(q.eps, q.retained, ls=ls, marker=mk_, ms=2.6, color=WCOL[w])
ax.set_xscale('log', base=2); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(['0.25', '0.5', '1', '2', '4'])
ax.axhline(1, color='#bbbbbb', lw=0.6); ax.axhline(0, color='k', lw=0.6, ls=':')
ax.set_xlabel(r'Per-report privacy budget $\varepsilon$ (64 players)')
ax.set_ylabel('Benefit retained')
ax.set_title('Exact Bayesian denoising dominates plug-in', loc='left')
ax.legend(handles=[Line2D([], [], color='#555555', ls='-', marker='o', ms=3, label='Bayesian posterior (corrected)'),
                   Line2D([], [], color='#555555', ls='--', marker='^', ms=3, label='Plug-in reports')], loc='lower right', fontsize=5.6)
letter(ax, 'B', dx=-0.22)
fig.savefig(f'{OUT}/fig3_information.pdf'); fig.savefig(f'{OUT}/fig3_information.png')
plt.close(fig)

# ================================================================== Figure 4: scale and mechanism
e4 = ev[ev.exp == 'E4'].copy()
e4['lab4'] = e4.lab + np.where(e4.dc > 0, '_cap', '')
a4 = agg(e4, ['world', 'n', 'lab4', 'rhoc'], ('final', 'capital', 'mean_deg', 'max_deg', 'isolates'))
a4.to_csv(f'{OUT}/tables/E4_scaling.csv', index=False)
e9 = ev[ev.exp == 'E9'].copy()
e9['lab9'] = e9.kind + np.where(e9.rhoc > 0, '_r' + e9.rhoc.astype(str), '')
a9 = agg(e9, ['coop', 'acc', 'afford', 'n', 'lab9'], ('final', 'capital', 'mean_deg'))
a9.to_csv(f'{OUT}/tables/E9_mechanism.csv', index=False)
fig = plt.figure(figsize=(W2, 4.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], width_ratios=[1, 1, 1, 0.66], hspace=0.6, wspace=0.12, left=0.075, right=0.985, bottom=0.09, top=0.93)
series = [('static', -1, 'static', ':'), ('enc_full', -1, 'enc_full', '-'), ('enc_full_cap', -1, 'enc_cap', '-'),
          ('type_dd', 0.125, 'type_dd', '-'), ('mb_match', 1.0, 'mb_match', '-'), ('mb_match_cap', 1.0, 'mb_match', '--'),
          ('mb_pub', 1.0, 'mb_pub', '-')]
lname = {'static': 'Static network', 'enc_full': 'Encouragement,\nunrationed', 'enc_full_cap': 'Encouragement, cap 15',
         'type_dd': r'D–D-first, $\rho$=1/8', 'mb_match': 'Model-based (matched),\n' + r'$\rho$=1', 'mb_match_cap': 'Model-based (matched),\n' + r'$\rho$=1, cap 15',
         'mb_pub': 'Model-based (published),\n' + r'$\rho$=1'}
for k, w in enumerate(['pub', 'cnt', 'frac']):
    ax = fig.add_subplot(gs[0, k])
    for lab, rho, col, ls in series:
        if w == 'pub' and lab.startswith('mb_match'):
            lab_ = 'mb_pub' + ('_cap' if lab.endswith('cap') else '')
            if lab == 'mb_match':
                continue
            q = a4[(a4.world == w) & (a4.lab4 == lab_) & (np.isclose(a4.rhoc, rho))].sort_values('n')
            ax.plot(q.n, q.final, ls=ls, marker='o', ms=2.2, color=PCOL['mb_pub'])
            continue
        q = a4[(a4.world == w) & (a4.lab4 == lab) & (np.isclose(a4.rhoc, rho))].sort_values('n')
        ax.plot(q.n, q.final, ls=ls, marker='o', ms=2.2, color=PCOL[col], label=lname[lab])
    ax.set_xscale('log', base=2); ax.set_xticks([16, 32, 64, 128, 256, 512]); ax.set_xticklabels(['16', '32', '64', '128', '256', '512'])
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel('Group size $n$')
    ax.set_title(WNAME[w], loc='left', color=WCOL[w])
    if k == 0:
        ax.set_ylabel('Final cooperation'); letter(ax, 'A', dx=-0.25)
    else:
        ax.set_yticklabels([])
    if k == 1:
        hh, ll = ax.get_legend_handles_labels()
        lax = fig.add_subplot(gs[0, 3]); lax.axis('off')
        lax.legend(hh, ll, loc='center left', fontsize=5.5, borderaxespad=0, handlelength=1.8)
gsb = gs[1, :].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.45)
ax = fig.add_subplot(gsb[0])
M = a9[(a9.n == 256) & (a9.lab9 == 'enc_full')].copy()
M['row'] = M.coop.map({'pub': 'Published', 'cnt': 'Count', 'frac': 'Fraction'})
M['col'] = M.acc.map({'const': 'constant acc.', 'deg': 'degree-dep. acc.'}) + np.where(M.afford == 1, ', budget rule', ', no budget rule')
piv = M.pivot_table(index='row', columns='col', values='final').reindex(['Published', 'Count', 'Fraction'])
cols = ['constant acc., no budget rule', 'constant acc., budget rule', 'degree-dep. acc., no budget rule', 'degree-dep. acc., budget rule']
piv = piv[cols]
im = ax.imshow(piv.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6, color='white' if v < 0.6 else 'black')
ax.set_xticks(range(4)); ax.set_xticklabels(['constant\nacceptance', 'constant\n+ affordability', 'degree-dep.\nacceptance', 'degree-dep.\n+ affordability'], fontsize=5.6)
ax.set_yticks(range(3)); ax.set_yticklabels(['Published', 'Count', 'Fraction'])
ax.set_ylabel('Cooperation equation')
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('Final cooperation', fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax.set_title('Unrationed collapse at $n$=256 needs two ingredients', loc='left')
letter(ax, 'B', dx=-0.2)
ax = fig.add_subplot(gsb[1])
for w in ['pub', 'cnt', 'frac']:
    for lab, rho, mk_ in [('enc_full', -1, 'o'), ('mb_match' if w != 'pub' else 'mb_pub', 1.0, 's')]:
        q = a4[(a4.world == w) & (a4.lab4 == lab) & (np.isclose(a4.rhoc, rho))].sort_values('n')
        ax.plot(q.n, q.max_deg, marker=mk_, ms=2.4, color=WCOL[w], ls='-' if lab == 'enc_full' else '--')
ax.plot([16, 512], [15, 15], color='#bbbbbb', lw=0.6)
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xticks([16, 32, 64, 128, 256, 512]); ax.set_xticklabels(['16', '32', '64', '128', '256', '512'])
ax.set_yticks([10, 30, 100, 300]); ax.set_yticklabels(['10', '30', '100', '300'])
ax.set_xlabel('Group size $n$'); ax.set_ylabel('Maximum final degree')
ax.set_title('Unrationed planners build hubs far beyond the data', loc='left')
ax.text(60, 11.2, 'largest degree in the experiment (15)', fontsize=5.5, color='#777777')
ax.legend(handles=[Line2D([], [], color='#555555', marker='o', ms=3, label='Encouragement (unrationed)'),
                   Line2D([], [], color='#555555', ls='--', marker='s', ms=3, label=r'Model-based, $\rho$=1')] +
          [Line2D([], [], color=WCOL[w], label=WNAME[w]) for w in ['pub', 'cnt', 'frac']], loc='upper left', fontsize=5.3, borderaxespad=0.1)
letter(ax, 'C', dx=-0.2)
fig.savefig(f'{OUT}/fig4_scale.pdf'); fig.savefig(f'{OUT}/fig4_scale.png')
plt.close(fig)

# ================================================================== Figure 5: caps, density sweep and decision under model risk
e6 = ev[ev.exp == 'E6'].copy(); e6['dc'] = e6.dcap.fillna(999)
a6 = agg(e6, ['world', 'n', 'lab', 'dc'], ('final', 'capital', 'over_cap0', 'over_cap_end', 'isolates'))
a6.to_csv(f'{OUT}/tables/E6_caps.csv', index=False)
reg = []
for n in [64, 256]:
    for met in ['final', 'capital']:
        t = a6[(a6.n == n) & (a6.lab == 'enc_full')].pivot_table(index='dc', columns='world', values=met)
        r = t.max() - t
        for dc in r.index:
            reg.append(dict(n=n, metric=met, cap=dc, **{f'regret_{w}': r.loc[dc, w] for w in r.columns}, max_regret=r.loc[dc].max(),
                            max_regret_calibrated=r.loc[dc, ['cnt', 'frac']].max()))
REG = pd.DataFrame(reg); REG.to_csv(f'{OUT}/tables/E6_cap_regret.csv', index=False)
e7 = ev[ev.exp == 'E7'].copy()
e7['lab7'] = e7.kind + np.where(e7.dc > 0, '_cap', '') + np.where(e7.rhoc > 0, '_r' + e7.rhoc.astype(str), '')
a7 = agg(e7, ['path', 'n', 's1', 'lab7'], ('final', 'capital'))
a7.to_csv(f'{OUT}/tables/E7_density_sweep.csv', index=False)
fig = plt.figure(figsize=(W2, 2.55))
gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.985, bottom=0.18, top=0.86)
caps = [3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 999]
for k, met in enumerate(['final', 'capital']):
    ax = fig.add_subplot(gs[k])
    for w in ['pub', 'cnt', 'frac']:
        q = a6[(a6.n == 256) & (a6.lab == 'enc_full') & (a6.world == w)].set_index('dc').reindex(caps)
        ax.plot(range(len(caps)), q[met], '-o', ms=2.4, color=WCOL[w], label=WNAME[w])
    ax.set_xticks(range(len(caps))); ax.set_xticklabels([str(c) if c < 999 else 'none' for c in caps], fontsize=5.6)
    ax.set_xlabel('Degree cap (admission rule), $n$=256')
    if met == 'final':
        ax.set_ylabel('Final cooperation'); ax.set_title('Cooperation favours tight caps', loc='left'); letter(ax, 'A', dx=-0.25)
        ax.legend(loc='lower left', fontsize=5.4, borderaxespad=0.1)
    else:
        ax.set_ylabel('Mean final capital'); ax.set_yscale('log'); ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(['1', '3', '10', '30'])
        ax.set_title('Welfare favours loose caps', loc='left'); letter(ax, 'B', dx=-0.25)
ax = fig.add_subplot(gs[2])
s1lo, s1hi = T['s1_boot']['q025'], T['s1_boot']['q975']
ax.axvspan(s1lo, s1hi, color=WCOL['cnt'], alpha=0.13, lw=0)
for lab, col, name in [('type_dd_r0.125', '#CC79A7', r'D–D-first, $\rho$=1/8'), ('type_dd_r0.5', '#a3507f', r'D–D-first, $\rho$=1/2'),
                       ('type_dd_r1.0', '#6b2a52', r'D–D-first, $\rho$=1'), ('enc_full_cap', PCOL['enc_cap'], 'Encouragement, cap 15'),
                       ('enc_full', PCOL['enc_full'], 'Encouragement (unrationed)'), ('static', PCOL['static'], 'Static network')]:
        q = a7[(a7.n == 256) & (a7.path == 'k') & (a7.lab7 == lab)].sort_values('s1')
        ax.plot(q.s1, q.final, marker='o', ms=1.8, color=col, label=name, ls=':' if lab == 'static' else '-')
ax.axvline(0, color='k', lw=0.6, ls=':')
ax.set_xlabel(r'Imposed $s_1$ (count world, $n$=256)'); ax.set_ylabel('Final cooperation')
ax.set_title('Budget tracks $s_1$; collapse does not', loc='left')
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.06), fontsize=5.0, borderaxespad=0.1)
ax.set_ylim(-0.03, 1.0)
letter(ax, 'C', dx=-0.25)
fig.savefig(f'{OUT}/fig5_caps_density.pdf'); fig.savefig(f'{OUT}/fig5_caps_density.png')
plt.close(fig)

# ================================================================== SI figures
# S1: greedy / interaction diagnostics
sb = pd.read_csv(f'{DATA}/submod_v2.csv'); it = pd.read_csv(f'{DATA}/interactions_v2.csv')
op = sb[sb.kind == 'opt'].copy(); pos = op.f_opt_le > 1e-9
op['r_greedy'] = np.where(pos, op.f_greedy / op.f_opt_le, np.nan); op['r_topk'] = np.where(pos, op.f_topk / op.f_opt_le, np.nan)
diag = []
for (w, K), g in op.groupby(['world', 'K']):
    diag.append(dict(world=w, K=K, states=len(g), share_opt_nonpos=float((~(g.f_opt_le > 1e-9)).mean()),
                     greedy_median=float(g.r_greedy.median()), greedy_p05=float(g.r_greedy.quantile(.05)), greedy_min=float(g.r_greedy.min()),
                     topk_median=float(g.r_topk.median()), topk_p05=float(g.r_topk.quantile(.05)), share_topk_negative=float((g.f_topk < -1e-9).mean())))
gm = sb[sb.kind == 'gamma']
for w, g in gm.groupby('world'):
    ok = g.den > 1e-6; r = (g.num / g.den)[ok]
    diag.append(dict(world=w, K='gamma', states=len(g), share_den_nonpos=float((g.den <= 1e-9).mean()),
                     neg_marginal_share=float(g.n_neg_marg.sum() / g.K.sum()), gamma_median=float(r.median()), gamma_p05=float(r.quantile(.05)),
                     gamma_p01=float(r.quantile(.01))))
it['z2'] = (it.I_final / it.I_final_se.replace(0, np.nan)) ** 2
for (w, c, s), g in it.groupby(['world', 'cont', 'shared']):
    diag.append(dict(world=w, K=f'interaction_{c}_{"shared" if s else "disjoint"}', states=len(g), I_mean=float(g.I_final.mean()),
                     I_mean_se=float(g.I_final.std() / np.sqrt(len(g))), I_abs_p95=float(g.I_final.abs().quantile(.95)),
                     mean_z2=float(np.nanmean(g.z2)), abs_single_effect=float(g.d1.abs().mean())))
pd.DataFrame(diag).to_csv(f'{OUT}/tables/SI_selection_diagnostics.csv', index=False)
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(wspace=1.05, left=0.07, right=0.975, bottom=0.2, top=0.85, width_ratios=[1, 1, 0.75]))
ax = axs[0]
for j, (w, K) in enumerate([('pub', 2), ('pub', 3), ('cnt', 2), ('cnt', 3)]):
    g = op[(op.world == w) & (op.K == K)]
    for off, col, key in [(-0.15, '#0072B2', 'r_greedy'), (0.15, '#D55E00', 'r_topk')]:
        x = g[key].clip(-1, 1.05).dropna()
        ax.scatter(np.full(len(x), j + off) + np.random.default_rng(j).uniform(-0.08, 0.08, len(x)), x, s=1.5, color=col, alpha=0.3, lw=0)
        ax.plot([j + off - 0.1, j + off + 0.1], [np.median(x)] * 2, color='k', lw=1.1)
ax.set_xticks(range(4)); ax.set_xticklabels(['Pub. K=2', 'Pub. K=3', 'Count K=2', 'Count K=3'], fontsize=5.6)
ax.set_ylabel('Gain / best gain (clipped at −1)')
ax.set_title('Stepwise greedy (blue) vs top-K (red)', loc='left'); letter(ax, 'A', dx=-0.25)
ax = axs[1]
for j, w in enumerate(['pub', 'cnt']):
    g = gm[(gm.world == w) & (gm.den > 1e-6)]
    r = (g.num / g.den).clip(0.5, 1.5)
    ax.hist(r, bins=40, alpha=0.6, color=WCOL[w], label=WNAME[w])
ax.axvline(1, color='k', lw=0.6, ls=':')
ax.set_xlabel('Sampled ratio of summed to joint marginal gains'); ax.set_ylabel('Samples')
ax.legend(fontsize=5.4, loc='upper right'); ax.set_title('Local submodularity ratios', loc='left'); letter(ax, 'B', dx=-0.25)
ax = axs[2]
labs = []
for j, ((w, c, s), g) in enumerate(it.groupby(['world', 'cont', 'shared'])):
    ax.errorbar(g.I_final.mean(), j, xerr=1.96 * g.I_final.std() / np.sqrt(len(g)), fmt='o', ms=2.5, color=WCOL[w])
    labs.append(f'{"Pub." if w == "pub" else "Count"}, {dict(mb="model-based", static="static", type_dd="D–D-first")[c]}, {"shared" if s else "disjoint"}')
ax.axvline(0, color='k', lw=0.6, ls=':')
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5.0); ax.invert_yaxis()
ax.set_xlabel('Mean pairwise interaction'); ax.set_title('Multi-round interactions', loc='left')
letter(ax, 'C', dx=-1.05)
fig.savefig(f'{OUT}/figS1_selection.pdf'); fig.savefig(f'{OUT}/figS1_selection.png'); plt.close(fig)

# S2: context and acceptance sensitivity
e10 = ev[ev.exp == 'E10'].copy()
e10['lab10'] = e10.kind + np.where(e10.dc > 0, '_cap' + e10.dc.astype(int).astype(str), '') + np.where(e10.rhoc > 0, '_r' + e10.rhoc.astype(str), '')
a10 = agg(e10, ['world', 'ctx', 'lab10'], ('final', 'capital')); a10.to_csv(f'{OUT}/tables/E10_context.csv', index=False)
e11 = ev[ev.exp == 'E11'].copy()
e11['lab11'] = e11.kind + np.where(e11.dc > 0, '_cap', '') + np.where(e11.rhoc > 0, '_r' + e11.rhoc.astype(str), '') + \
    np.where(e11.Kc > 0, '_K' + e11.Kc.astype(int).astype(str), '')
a11 = agg(e11, ['coop', 'accv', 'n', 'lab11'], ('final', 'capital', 'mean_deg')); a11.to_csv(f'{OUT}/tables/E11_acceptance.csv', index=False)
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(wspace=0.5, left=0.15, right=0.98, bottom=0.22, top=0.84, width_ratios=[1.5, 1]))
ax = axs[0]
ctxs = ['er3', 'er4.5', 'er8', 'ht4.5', 'sbm4.5', 'T10', 'T25']
rowsS = ['static', 'enc_full', 'enc_full_cap6', 'enc_full_cap8', 'enc_full_cap15', 'type_dd_r0.125', 'type_dd_r0.5', 'type_dd_r1.0', 'mb_cap8_r1.0']
rn = ['Static', 'Encouragement', 'Enc., cap 6', 'Enc., cap 8', 'Enc., cap 15', 'D–D-first ρ=1/8', 'D–D-first ρ=1/2', 'D–D-first ρ=1', 'Model-based ρ=1, cap 8']
mat = np.array([[a10[(a10.world == w) & (a10.ctx == c) & (a10.lab10 == r)].final.values[0] for w in ['pub', 'cnt', 'frac'] for c in ctxs] for r in rowsS])
im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1, aspect='auto')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f'{mat[i, j]:.2f}'[1:] if round(mat[i, j], 2) < 1 else '1', ha='center', va='center', fontsize=4.2, color='white' if mat[i, j] < 0.6 else 'black')
ax.set_xticks(range(len(ctxs) * 3)); ax.set_xticklabels(ctxs * 3, rotation=90, fontsize=5)
ax.set_yticks(range(len(rowsS))); ax.set_yticklabels(rn, fontsize=5.6)
for j, w in enumerate(['pub', 'cnt', 'frac']):
    ax.text(j * 7 + 3, -1.0, WNAME[w], ha='center', fontsize=6, color=WCOL[w])
    if j:
        ax.axvline(j * 7 - 0.5, color='white', lw=1.5)
ax.set_title('Context sensitivity, $n$=256 (final cooperation)', loc='left', pad=16); letter(ax, 'A', dx=-0.24, dy=1.1)
ax = axs[1]
rows11 = ['static', 'enc_full', 'enc_full_cap', 'type_dd_r0.125', 'mb_match_r1.0', 'mb_match_r0.125']
rn11 = ['Static', 'Encouragement', 'Enc., cap 15', 'D–D-first ρ=1/8', 'Model-based ρ=1', 'Model-based ρ=1/8']
e11['lab11'] = e11.lab + np.where(e11.dc > 0, '_cap', '') + np.where(e11.rhoc > 0, '_r' + e11.rhoc.astype(str), '')
a11b = agg(e11[e11.n == 256], ['coop', 'accv', 'lab11'])
accs = ['const', 'deg', 'dec', 'within']
mat = np.array([[a11b[(a11b.coop == c) & (a11b.accv == v) & (a11b.lab11 == r)].final.values[0] for c in ['cnt', 'frac'] for v in accs] for r in rows11])
im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=1, aspect='auto')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=4.8, color='white' if mat[i, j] < 0.6 else 'black')
ax.set_xticks(range(8)); ax.set_xticklabels(['constant', 'degree', '+decider act.', 'within-person'] * 2, rotation=90, fontsize=5)
ax.set_yticks(range(len(rows11))); ax.set_yticklabels(rn11, fontsize=5.6)
ax.axvline(3.5, color='white', lw=1.5)
ax.set_title('Acceptance sensitivity, $n$=256\n(left: count model; right: fraction model)', loc='left', pad=6); letter(ax, 'B', dx=-0.62, dy=1.1)
fig.savefig(f'{OUT}/figS2_sensitivity.pdf'); fig.savefig(f'{OUT}/figS2_sensitivity.png'); plt.close(fig)

# S3: robustness worlds, fixed density and learned edge values
e2 = ev[ev.exp == 'E2']; a2 = agg(e2, ['world', 'lab', 'Kc']); a2.to_csv(f'{OUT}/tables/E2_robustness_worlds.csv', index=False)
e5 = ev[ev.exp == 'E5']; a5 = agg(e5, ['world', 'n', 'kind']); a5.to_csv(f'{OUT}/tables/E5_fixed_density.csv', index=False)
lev = json.load(open(f'{DATA}/levfit_v2.json'))
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.42, left=0.07, right=0.985, bottom=0.2, top=0.85))
RW = {'pub_k05': 'Weaker neighbour response', 'pub_w03': 'Imitation (30%)', 'pub_unan': 'Unanimity acceptance (synthetic)'}
for w, col in zip(RW, ['#1b9e77', '#d95f02', '#7570b3']):
    for lab, ls in [('type_dd', '-'), ('rand_thin', '--')]:
        q = a2[(a2.world == w) & (a2.lab == lab)].sort_values('Kc')
        axs[0].plot(q.Kc, q.final, ls=ls, marker='o', ms=2.2, color=col, label=RW[w] if lab == 'type_dd' else None)
axs[0].set_xscale('log', base=2); axs[0].set_xticks([2, 4, 8, 16]); axs[0].set_xticklabels(['2', '4', '8', '16'])
axs[0].set_xlabel('Edits per round, $K$'); axs[0].set_ylabel('Final cooperation')
axs[0].set_title('D–D-first (solid) vs random thinning (dashed)', loc='left', fontsize=6.3); axs[0].legend(fontsize=5.2, loc='lower right')
letter(axs[0], 'A', dx=-0.25)
for w in ['pub', 'cnt', 'frac']:
    for k_, ls in [('enc_full', '-'), ('static', ':')]:
        q = a5[(a5.world == w) & (a5.kind == k_)].sort_values('n')
        axs[1].plot(q.n, q.final, ls=ls, marker='o', ms=2.2, color=WCOL[w])
axs[1].set_xscale('log', base=2); axs[1].set_xticks([16, 32, 64, 128]); axs[1].set_xticklabels(['16', '32', '64', '128'])
axs[1].set_xlabel('Group size $n$ (edge probability 0.3)'); axs[1].set_ylabel('Final cooperation')
axs[1].set_title('Fixed-density scaling (solid: encouragement)', loc='left', fontsize=6.3); letter(axs[1], 'B', dx=-0.25)
axs[1].legend(handles=[Line2D([], [], color=WCOL[w], label=WNAME[w]) for w in ['pub', 'cnt', 'frac']] + [Line2D([], [], color='#555555', ls=':', label='Static network')], fontsize=5.0, loc='lower left')
imp = lev['importance_heldout']; ks = sorted(imp, key=lambda k: -imp[k][0])[:8]
axs[2].barh(range(len(ks)), [imp[k][0] for k in ks], xerr=[imp[k][1] for k in ks], color='#777777', height=0.6)
axs[2].set_yticks(range(len(ks))); axs[2].set_yticklabels(ks, fontsize=5.6); axs[2].invert_yaxis()
axs[2].set_xlabel('Held-out permutation importance (ΔR²)')
axs[2].set_title(f"Learned edge value: split-half r = {lev['split_half_r']:.3f}", loc='left', fontsize=6.3); letter(axs[2], 'C', dx=-0.3)
fig.savefig(f'{OUT}/figS3_robustness_lev.pdf'); fig.savefig(f'{OUT}/figS3_robustness_lev.png'); plt.close(fig)

# ------------------------------------------------------------------ key-number table
T['validation'] = VAL.to_dict('records')
T['counts'] = cal['counts']
T['games_total'] = int(len(ev)); T['games_by_exp'] = ev.exp.value_counts().to_dict()
spec_cols = [c for c in ['exp', 'world', 'n', 'kind', 'K', 'rho', 'dcap', 'model', 'eps', 'use_hist', 'capital_known', 'radius', 'graph',
                         'mean_deg', 'T', 'p', 'lev', 'skey', 'fam']]
uq = ev.drop(columns=['exp', 'info', 'path', 's1', 'coop', 'acc', 'afford', 'accv', 'draw', 'fam_fit', 'cond', 'ctx'], errors='ignore')
key = [c for c in ['world', 'n', 'kind', 'K', 'rho', 'dcap', 'model', 'eps', 'use_hist', 'capital_known', 'radius', 'graph', 'mean_deg', 'T', 'p', 'lev',
                   'skey', 'game'] if c in uq]
T['games_unique'] = int(uq[key].astype(str).drop_duplicates().shape[0])
T['lev'] = {k: v for k, v in lev.items() if k != 'importance_heldout'}
json.dump(T, open(f'{OUT}/tables/key_numbers.json', 'w'), indent=1, default=float)
print('figures done', T['games_total'], T['games_unique'])
