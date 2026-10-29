"""Experiments E1-E13 of the paper, run on the saved predictions only (no GPU, no images).

Writes tables (csv) and figures to outputs/revision/.
"""
import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

import fusionlib as fl
from tea_models import LABELS, MODELS

OUT, A = Path("outputs"), Path("outputs/analysis")
R = OUT / "revision"
R.mkdir(exist_ok=True)
SEEDS = json.loads((OUT / "summary.json").read_text())["seeds"]
split = pd.read_csv(OUT / "split.csv")
CLASSES = [c.split(". ", 1)[-1] for c in sorted(split.cls.unique())]
K = len(CLASSES)
y_val = split.loc[split.split == "val", "label"].to_numpy()
y_test = split.loc[split.split == "test", "label"].to_numpy()
y_ext = np.load(A / "external_labels.npy")
SHARED = sorted(set(y_ext.tolist()))
GMB = [i for i in range(K) if i not in SHARED][0]
CONFIGS = [c for r in (1, 2, 3) for c in combinations(MODELS, r)]
TRIPLE, DN = tuple(MODELS), ("densenet121",)
BEST_PAIR = ("cnn", "densenet121")

P = {(s, m): np.load(OUT / f"probs_{m}_seed{s}.npz") for s in SEEDS for m in MODELS}
VAL = {(s, m): P[s, m]["val"] for s in SEEDS for m in MODELS}
TEST = {(s, m): P[s, m]["test"] for s in SEEDS for m in MODELS}
EXT = {(s, m): np.load(A / f"ext_probs_{m}_seed{s}.npy") for s in SEEDS for m in MODELS}
IDX = fl.bootstrap_indices(len(y_test), 2000, seed=0)

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE, LIGHT, GREY, RED = "#2b5876", "#9db4c0", "#7f8c8d", "#c0504d"


def name(c):
    return " + ".join(LABELS[m] for m in c)


def short(c):
    return "+".join({"cnn": "C", "densenet121": "D", "vgg16": "V"}[m] for m in c)


def weights(s, c, alpha=1.0):
    return fl.accuracy_weights([VAL[s, m] for m in c], y_val, alpha)


def fuse(s, c, src, alpha=1.0):
    d = {"val": VAL, "test": TEST, "ext": EXT}[src]
    return fl.weighted_average([d[s, m] for m in c], weights(s, c, alpha))


def ms(x, d=2):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}"


def save(df, name_):
    df.to_csv(R / name_, index=False)
    return df


numbers = {}

# ---------------------------------------------------------------- E1 main results with bootstrap CI
rows = []
for c in CONFIGS:
    preds = [fuse(s, c, "test").argmax(1) for s in SEEDS]
    acc = [accuracy_score(y_test, p) * 100 for p in preds]
    prf = [precision_recall_fscore_support(y_test, p, average="weighted", zero_division=0) for p in preds]
    mf1 = [f1_score(y_test, p, average="macro") * 100 for p in preds]
    lo, hi = fl.bootstrap_ci(np.mean([p == y_test for p in preds], 0), IDX)
    rows.append({"config": name(c), "accuracy": ms(acc), "ci95": f"{lo:.2f}-{hi:.2f}",
                 "precision_w": ms([x[0] * 100 for x in prf]), "recall_w": ms([x[1] * 100 for x in prf]),
                 "f1_w": ms([x[2] * 100 for x in prf]), "f1_macro": ms(mf1), "acc_mean": np.mean(acc)})
e1 = save(pd.DataFrame(rows), "e1_main_results.csv")
numbers["triple_acc"] = e1.acc_mean.iloc[-1]

# ---------------------------------------------------------------- E2 triple vs every other configuration
pv_all = {}
for s in SEEDS:
    pairs = list(combinations(CONFIGS, 2))
    ps = [fl.mcnemar(fuse(s, a, "test").argmax(1) == y_test, fuse(s, b, "test").argmax(1) == y_test)[2] for a, b in pairs]
    rej = fl.holm(ps)
    for (a, b), p, r in zip(pairs, ps, rej):
        pv_all[s, a, b] = (p, r)
rows = []
for c in CONFIGS[:-1]:
    ct = np.mean([fuse(s, TRIPLE, "test").argmax(1) == y_test for s in SEEDS], 0)
    cc = np.mean([fuse(s, c, "test").argmax(1) == y_test for s in SEEDS], 0)
    diffs = (ct[IDX] - cc[IDX]).mean(1) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pvals = [pv_all[s, c, TRIPLE][0] for s in SEEDS]
    holm_rej = [pv_all[s, c, TRIPLE][1] for s in SEEDS]
    rows.append({"triple_vs": name(c), "diff": f"{(ct - cc).mean() * 100:+.2f}", "ci95": f"{lo:+.2f} to {hi:+.2f}",
                 "p_per_seed": " / ".join(f"{p:.3f}" for p in pvals),
                 "seeds_sig": f"{sum(p < 0.05 for p in pvals)} / {sum(holm_rej)}"})
save(pd.DataFrame(rows), "e2_significance.csv")

# ---------------------------------------------------------------- E3 per-class F1
rows = []
for c in [DN, BEST_PAIR, TRIPLE]:
    f = np.array([f1_score(y_test, fuse(s, c, "test").argmax(1), average=None) * 100 for s in SEEDS])
    for k in range(K):
        rows.append({"config": name(c), "class": CLASSES[k], "f1_mean": f[:, k].mean(), "f1_sd": f[:, k].std(ddof=1)})
e3 = save(pd.DataFrame(rows), "e3_perclass_f1.csv")

# ---------------------------------------------------------------- E4 diversity, agreement
rows = []
for a, b in combinations(MODELS, 2):
    d = [fl.pair_diversity(TEST[s, a].argmax(1), TEST[s, b].argmax(1), y_test) for s in SEEDS]
    rows.append({"pair": f"{LABELS[a]} / {LABELS[b]}",
                 "disagreement": ms([x["disagreement"] * 100 for x in d]), "double_fault": ms([x["double_fault"] * 100 for x in d]),
                 "q": ms([x["q"] for x in d], 3), "error_corr": ms([x["error_corr"] for x in d], 3),
                 "q_mean": np.mean([x["q"] for x in d])})
e4 = save(pd.DataFrame(rows), "e4_diversity.csv")
ab = [fl.agreement_breakdown([TEST[s, m].argmax(1) for m in MODELS], fuse(s, TRIPLE, "test").argmax(1), y_test) for s in SEEDS]
agree = pd.DataFrame([{"members_right": n, "share": np.mean([x[n][0] for x in ab]) * 100,
                       "fused_acc": np.nanmean([x[n][1] for x in ab]) * 100} for n in range(4)])
save(agree, "e4_agreement.csv")

# ---------------------------------------------------------------- E5 oracle, fixed / broken vs DenseNet121
rows = []
for c in CONFIGS:
    fused, orac, fixed, broken = [], [], [], []
    for s in SEEDS:
        p = fuse(s, c, "test").argmax(1)
        d = TEST[s, "densenet121"].argmax(1)
        fused.append(accuracy_score(y_test, p) * 100)
        orac.append(fl.oracle_accuracy([TEST[s, m].argmax(1) for m in c], y_test) * 100)
        fixed.append(((p == y_test) & (d != y_test)).sum())
        broken.append(((p != y_test) & (d == y_test)).sum())
    rows.append({"config": name(c), "fused": ms(fused), "oracle": ms(orac) if len(c) > 1 else "",
                 "fixed": ms(fixed, 1), "broken": ms(broken, 1)})
save(pd.DataFrame(rows), "e5_oracle.csv")

# ---------------------------------------------------------------- E6 ten combination rules (triple)
RULES = ["Accuracy-weighted soft voting (original)", "Equal-weight soft voting", "Majority voting",
         "Weighted geometric mean (product rule)", "Max rule", "Grid-searched weights (val. accuracy)",
         "Log-loss-optimal weights (val.)", "Class-wise recall weights", "Temperature-scaled members + weighted voting",
         "Stacking (logistic regression)"]
res, cost, logw = {r: [] for r in RULES}, {r: [] for r in RULES}, []
for s in SEEDS:
    V = [VAL[s, m] for m in MODELS]
    T = [TEST[s, m] for m in MODELS]
    X = [EXT[s, m] for m in MODELS]
    w = weights(s, TRIPLE)
    for r in RULES:
        t0 = time.perf_counter()
        if r == RULES[0]:
            pt, px = fl.weighted_average(T, w), fl.weighted_average(X, w)
        elif r == RULES[1]:
            pt, px = fl.weighted_average(T, np.ones(3)), fl.weighted_average(X, np.ones(3))
        elif r == RULES[2]:
            pt, px = fl.majority_vote(T, w), fl.majority_vote(X, w)
        elif r == RULES[3]:
            pt, px = fl.weighted_product(T, w), fl.weighted_product(X, w)
        elif r == RULES[4]:
            pt, px = fl.max_rule(T), fl.max_rule(X)
        elif r == RULES[5]:
            g = fl.grid_weights(V, y_val)
            pt, px = fl.weighted_average(T, g), fl.weighted_average(X, g)
        elif r == RULES[6]:
            g = fl.logloss_weights(V, y_val)
            logw.append(g)
            pt, px = fl.weighted_average(T, g), fl.weighted_average(X, g)
        elif r == RULES[7]:
            wc = fl.classwise_weights(V, y_val, K)
            pt, px = fl.classwise_average(T, wc), fl.classwise_average(X, wc)
        elif r == RULES[8]:
            pt, px = fl.calibrate_then_fuse(V, y_val, T, w), fl.calibrate_then_fuse(V, y_val, X, w)
        else:
            st = fl.Stacker().fit(V, y_val)
            pt, px = st.predict_proba(T), st.predict_proba(X)
        cost[r].append((time.perf_counter() - t0) * 1000)
        res[r].append((pt, px))
rows = []
for r in RULES:
    lab = lambda q: q if q.ndim == 1 else q.argmax(1)
    acc = [accuracy_score(y_test, lab(pt)) * 100 for pt, _ in res[r]]
    mf1 = [f1_score(y_test, lab(pt), average="macro") * 100 for pt, _ in res[r]]
    ext = [accuracy_score(y_ext, lab(px)) * 100 for _, px in res[r]]
    prob = res[r][0][0].ndim == 2
    sig_in = sig_ext = "-"
    if r != RULES[0]:
        sig_in = sum(fl.mcnemar(lab(res[r][i][0]) == y_test, lab(res[RULES[0]][i][0]) == y_test)[2] < 0.05 for i in range(len(SEEDS)))
        sig_ext = sum(fl.mcnemar(lab(res[r][i][1]) == y_ext, lab(res[RULES[0]][i][1]) == y_ext)[2] < 0.05 for i in range(len(SEEDS)))
    rows.append({"rule": r, "accuracy": ms(acc), "macro_f1": ms(mf1),
                 "nll": f"{np.mean([fl.nll(pt, y_test) for pt, _ in res[r]]):.3f}" if prob else "-",
                 "ece": f"{np.mean([fl.ece(pt, y_test) for pt, _ in res[r]]):.3f}" if prob else "-",
                 "ext_nll": f"{np.mean([fl.nll(px, y_ext) for _, px in res[r]]):.3f}" if prob else "-",
                 "unseen_accuracy": ms(ext), "p_lt_005": f"{sig_in} / {sig_ext}" if r != RULES[0] else "-",
                 "acc_mean": np.mean(acc), "ext_mean": np.mean(ext), "cost_ms": np.mean(cost[r])})
e6 = save(pd.DataFrame(rows), "e6_rules.csv")
numbers["logloss_weight_densenet"] = float(np.mean([g[1] for g in logw]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, col, sdcol, title in [(axes[0], "acc_mean", "accuracy", "teaLeafBD test"), (axes[1], "ext_mean", "unseen_accuracy", "TeaLeafDiseaseBD (unseen)")]:
    sd = e6[sdcol].str.split(" ± ").str[1].astype(float)
    y = np.arange(len(RULES))[::-1]
    ax.errorbar(e6[col], y, xerr=sd, fmt="o", color=LIGHT, ecolor=LIGHT, capsize=3)
    ax.errorbar(e6[col].iloc[:1], y[:1], xerr=sd.iloc[:1], fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    ax.axvline(e6[col].iloc[0], color=BLUE, lw=0.6, ls="--")
    ax.set_title(title)
    ax.set_xlabel("Accuracy (%)")
axes[0].set_yticks(np.arange(len(RULES))[::-1], RULES, fontsize=9)
fig.tight_layout()
fig.savefig(R / "fig_rules.png")
plt.close(fig)

# ---------------------------------------------------------------- E7 weight simplex and alpha sweep
grid = fl.simplex_grid(3, 0.05)
gin = np.array([[accuracy_score(y_test, fl.weighted_average([TEST[s, m] for m in MODELS], w).argmax(1)) for s in SEEDS] for w in grid]).mean(1) * 100
gex = np.array([[accuracy_score(y_ext, fl.weighted_average([EXT[s, m] for m in MODELS], w).argmax(1)) for s in SEEDS] for w in grid]).mean(1) * 100
save(pd.DataFrame({"w_cnn": grid[:, 0], "w_densenet121": grid[:, 1], "w_vgg16": grid[:, 2], "acc_test": gin, "acc_ext": gex}), "e7_weight_grid.csv")
inner = (grid >= 0.2 - 1e-9).all(1)
numbers["grid_inner_range"] = (float(gin[inner].min()), float(gin[inner].max()))
alphas = [0, 0.5, 1, 2, 5, 10, 20, 50, 100]
alpha_rows = [{"alpha": a, "acc_test": np.mean([accuracy_score(y_test, fuse(s, TRIPLE, "test", a).argmax(1)) for s in SEEDS]) * 100,
               "acc_ext": np.mean([accuracy_score(y_ext, fuse(s, TRIPLE, "ext", a).argmax(1)) for s in SEEDS]) * 100} for a in alphas]
save(pd.DataFrame(alpha_rows), "e7_alpha.csv")
chosen = [fl.grid_weights([VAL[s, m] for m in MODELS], y_val) for s in SEEDS]
xy = lambda w: (w[..., 1] + 0.5 * w[..., 2], np.sqrt(3) / 2 * w[..., 2])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, z, title in [(axes[0], gin, "teaLeafBD test"), (axes[1], gex, "TeaLeafDiseaseBD (unseen)")]:
    x, y = xy(grid)
    tri = mtri.Triangulation(x, y)
    cs = ax.tricontourf(tri, z, levels=14, cmap="Blues")
    fig.colorbar(cs, ax=ax, shrink=0.8, label="Accuracy (%)")
    for w in chosen:
        cx, cy = xy(np.asarray(w))
        ax.plot(cx, cy, "o", color=RED, ms=5)
    ax.text(-0.04, -0.06, "CNN", ha="center")
    ax.text(1.04, -0.06, "DenseNet121", ha="center")
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "VGG16", ha="center")
    ax.set_title(title, pad=18)
    ax.set_aspect("equal")
    ax.axis("off")
fig.tight_layout()
fig.savefig(R / "fig_weight_simplex.png")
plt.close(fig)

# ---------------------------------------------------------------- E8 calibration
rows, bins_rows = [], []
for c in CONFIGS:
    vals = {k: [] for k in ["T", "ece", "ece_ts", "ece_iso", "nll", "nll_ts", "nll_iso", "brier", "brier_ts", "auc", "acc_iso",
                            "ext_ece", "ext_ece_ts", "ext_auc"]}
    for s in SEEDS:
        pv, pt, px = fuse(s, c, "val"), fuse(s, c, "test"), fuse(s, c, "ext")
        t = fl.fit_temperature(pv, y_val)
        iso = fl.Isotonic().fit(pv, y_val)
        pts, pti = fl.temperature_scale(pt, t), iso.predict(pt)
        for k, v in [("T", t), ("ece", fl.ece(pt, y_test)), ("ece_ts", fl.ece(pts, y_test)), ("ece_iso", fl.ece(pti, y_test)),
                     ("nll", fl.nll(pt, y_test)), ("nll_ts", fl.nll(pts, y_test)), ("nll_iso", fl.nll(pti, y_test)),
                     ("brier", fl.brier(pt, y_test)), ("brier_ts", fl.brier(pts, y_test)), ("auc", fl.macro_auc(pt, y_test)),
                     ("acc_iso", accuracy_score(y_test, pti.argmax(1)) * 100),
                     ("ext_ece", fl.ece(px, y_ext)), ("ext_ece_ts", fl.ece(fl.temperature_scale(px, t), y_ext)),
                     ("ext_auc", fl.macro_auc(px, y_ext, SHARED))]:
            vals[k].append(v)
        for b in (10, 15, 20):
            bins_rows.append({"config": name(c), "seed": s, "bins": b, "ece": fl.ece(pt, y_test, b), "ece_ts": fl.ece(pts, y_test, b)})
    rows.append({"config": name(c), **{k: round(float(np.mean(v)), 4) for k, v in vals.items()}})
e8 = save(pd.DataFrame(rows), "e8_calibration.csv")
save(pd.DataFrame(bins_rows).groupby(["config", "bins"]).mean(numeric_only=True).drop(columns="seed").round(4).reset_index(),
     "e7_calibration_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
s0 = SEEDS[0]
t0 = fl.fit_temperature(fuse(s0, TRIPLE, "val"), y_val)
iso0 = fl.Isotonic().fit(fuse(s0, TRIPLE, "val"), y_val)
edges = np.linspace(0, 1, 11)
for ax, src, y, title in [(axes[0], "test", y_test, "teaLeafBD test"), (axes[1], "ext", y_ext, "TeaLeafDiseaseBD (unseen)")]:
    p = fuse(s0, TRIPLE, src)
    for q, lab_, col, mk in [(p, "uncalibrated", LIGHT, "o"), (fl.temperature_scale(p, t0), f"temperature (T = {t0:.2f})", BLUE, "s"),
                             (iso0.predict(p), "isotonic", RED, "^")]:
        conf, ok = q.max(1), q.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 20:
                xs.append(conf[m].mean())
                ys.append(ok[m].mean())
        ax.plot(xs, ys, marker=mk, lw=1.4, color=col, label=lab_)
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.8)
    ax.axvline(0.8, ls=":", color=GREY, lw=0.8)
    ax.set(title=title, xlabel="Confidence", xlim=(0, 1), ylim=(0, 1))
axes[0].set_ylabel("Accuracy")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(R / "fig_reliability_cal.png")
plt.close(fig)

# ---------------------------------------------------------------- E9 advisory rule, risk-coverage, threshold sweep
def advisory(c, src, y, ts=False, agree=False):
    out = []
    for s in SEEDS:
        p = fuse(s, c, src)
        if ts:
            p = fl.temperature_scale(p, fl.fit_temperature(fuse(s, c, "val"), y_val))
        d = {"test": TEST, "ext": EXT}[src]
        ag = np.all([d[s, m].argmax(1) == p.argmax(1) for m in c], 0) if agree else None
        out.append(fl.selective_stats(p, y, fl.advise(p, 0.8, ag)))
    return out


rows = []
for label, c, ts, ag in [("DenseNet121, conf. ≥ 0.80", DN, False, False), ("CNN + DenseNet121, conf. ≥ 0.80", BEST_PAIR, False, False),
                         ("Triple, conf. ≥ 0.80", TRIPLE, False, False), ("Triple, temperature-scaled conf. ≥ 0.80", TRIPLE, True, False),
                         ("Triple, conf. ≥ 0.80 and all three agree", TRIPLE, False, True)]:
    a, b = advisory(c, "test", y_test, ts, ag), advisory(c, "ext", y_ext, ts, ag)
    row = {"rule": label}
    for tag, st in [("in", a), ("unseen", b)]:
        for k, kk in [("coverage", "advised"), ("selective_acc", "correct"), ("wrong_advice", "wrong_advice")]:
            row[f"{kk}_{tag}"] = ms([x[k] * 100 for x in st], 2)
    rows.append(row)
e9 = save(pd.DataFrame(rows), "e9_advisory.csv")

rows = []
for c in [DN, BEST_PAIR, TRIPLE]:
    for src, y in [("test", y_test), ("ext", y_ext)]:
        d = {"test": TEST, "ext": EXT}[src]
        res_ = {k: [] for k in ["max_prob", "margin", "neg_entropy", "agreement"]}
        cov1 = []
        for s in SEEDS:
            p = fuse(s, c, src)
            ok = p.argmax(1) == y
            agree_n = np.sum([d[s, m].argmax(1) == p.argmax(1) for m in c], 0) + p.max(1) * 1e-3
            for k, sc in [("max_prob", p.max(1)), ("margin", fl.margin(p)), ("neg_entropy", -fl.entropy(p)), ("agreement", agree_n)]:
                res_[k].append(fl.aurc(sc, ok))
            cov1.append(fl.coverage_at_risk(p.max(1), ok, 0.01))
        rows.append({"config": name(c), "data": "teaLeafBD" if src == "test" else "TeaLeafDiseaseBD",
                     **{f"aurc_{k}": round(float(np.mean(v)), 4) for k, v in res_.items()},
                     "coverage_at_1pct_risk": round(float(np.mean(cov1)) * 100, 1)})
e9rc = save(pd.DataFrame(rows), "e9_risk_coverage.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, src, y, title in [(axes[0], "test", y_test, "teaLeafBD test"), (axes[1], "ext", y_ext, "TeaLeafDiseaseBD (unseen)")]:
    for c, col in [(DN, GREY), (BEST_PAIR, LIGHT), (TRIPLE, BLUE)]:
        curves = []
        for s in SEEDS:
            p = fuse(s, c, src)
            cov, risk = fl.risk_coverage(p.max(1), p.argmax(1) == y)
            curves.append(np.interp(np.linspace(0.01, 1, 100), cov, risk))
        ax.plot(np.linspace(0.01, 1, 100) * 100, np.mean(curves, 0) * 100, color=col, lw=1.6, label=name(c))
    ax.set(title=title, xlabel="Coverage (%)", ylabel="Selective risk (%)")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(R / "fig_risk_coverage.png")
plt.close(fig)

ths = np.round(np.arange(0.40, 0.991, 0.01), 2)
rows = []
for th in ths:
    for src, y in [("test", y_test), ("ext", y_ext)]:
        st = [fl.selective_stats(fuse(s, TRIPLE, src), y, fl.advise(fuse(s, TRIPLE, src), th)) for s in SEEDS]
        rows.append({"threshold": th, "data": src, **{k: np.mean([x[k] for x in st]) * 100 for k in st[0]}})
sweep = save(pd.DataFrame(rows), "e9_threshold_sweep.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, src, title in [(axes[0], "test", "teaLeafBD test"), (axes[1], "ext", "TeaLeafDiseaseBD (unseen)")]:
    d = sweep[sweep.data == src]
    ax.plot(d.threshold, d.coverage, color=BLUE, lw=1.6, label="advised")
    ax.plot(d.threshold, d.selective_acc, color=LIGHT, lw=1.6, label="correct when advised")
    ax.plot(d.threshold, d.wrong_advice, color=RED, lw=1.6, label="wrong advice (all images)")
    ax.axvline(0.8, color=GREY, ls=":", lw=0.9)
    ax.set(title=title, xlabel="Confidence threshold")
axes[0].set_ylabel("%")
axes[0].legend(frameon=False, loc="center left")
fig.tight_layout()
fig.savefig(R / "fig_threshold_sweep.png")
plt.close(fig)

# ---------------------------------------------------------------- E10 cross-dataset
rows = []
for c in CONFIGS:
    acc_in = [accuracy_score(y_test, fuse(s, c, "test").argmax(1)) * 100 for s in SEEDS]
    px = [fuse(s, c, "ext") for s in SEEDS]
    acc = [accuracy_score(y_ext, p.argmax(1)) * 100 for p in px]
    mf1 = [f1_score(y_ext, p.argmax(1), labels=SHARED, average="macro", zero_division=0) * 100 for p in px]
    bal = [balanced_accuracy_score(y_ext, p.argmax(1)) * 100 for p in px]
    six = [accuracy_score(y_ext, np.array(SHARED)[p[:, SHARED].argmax(1)]) * 100 for p in px]
    gmb = [(p.argmax(1) == GMB).mean() * 100 for p in px]
    rows.append({"config": name(c), "teaLeafBD": ms(acc_in), "unseen_acc": ms(acc), "unseen_macro_f1": ms(mf1),
                 "balanced_acc": ms(bal), "six_class_acc": ms(six), "gmb_predicted": ms(gmb, 1)})
save(pd.DataFrame(rows), "e10_cross_dataset.csv")
cm = np.mean([pd.crosstab(pd.Categorical(y_ext, categories=range(K)), pd.Categorical(fuse(s, TRIPLE, "ext").argmax(1), categories=range(K)),
                          dropna=False).values for s in SEEDS], 0)
cmr = cm[SHARED] / cm[SHARED].sum(1, keepdims=True) * 100
save(pd.DataFrame(cmr, index=[CLASSES[i] for i in SHARED], columns=CLASSES).reset_index(names="true"), "e10_ext_confusion.csv")
numbers["ext_recall"] = {CLASSES[i]: float(cmr[j, i]) for j, i in enumerate(SHARED)}
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.imshow(cmr, cmap="Blues", vmin=0, vmax=100)
for i in range(cmr.shape[0]):
    for j in range(K):
        ax.text(j, i, f"{cmr[i, j]:.1f}", ha="center", va="center", fontsize=9, color="white" if cmr[i, j] > 50 else "black")
ax.set_xticks(range(K), CLASSES, rotation=35, ha="right")
ax.set_yticks(range(len(SHARED)), [CLASSES[i] for i in SHARED])
ax.set(xlabel="Predicted (teaLeafBD classes)", ylabel="True (TeaLeafDiseaseBD)")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(R / "fig_ext_confusion.png")
plt.close(fig)

# per-class recall on both datasets (triple)
rec_in = np.mean([[(fuse(s, TRIPLE, "test").argmax(1)[y_test == k] == k).mean() for k in range(K)] for s in SEEDS], 0) * 100
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(K)
ax.bar(x - 0.2, rec_in, 0.4, color=BLUE, label="teaLeafBD test")
ax.bar([i + 0.2 for i in SHARED], [numbers["ext_recall"][CLASSES[i]] for i in SHARED], 0.4, color=RED, label="TeaLeafDiseaseBD (unseen)")
for i, v in enumerate(rec_in):
    ax.text(i - 0.2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
for i in SHARED:
    v = numbers["ext_recall"][CLASSES[i]]
    ax.text(i + 0.2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
ax.text(GMB + 0.2, 3, "absent", ha="center", fontsize=8, color=GREY)
ax.set_xticks(x, CLASSES, rotation=20, ha="right")
ax.set(ylabel="Recall (%)", ylim=(0, 110))
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
fig.savefig(R / "fig_recall_both.png")
plt.close(fig)

# ---------------------------------------------------------------- E11 can confidence flag unseen images?
rows = []
for label, fn in [("1 - max prob., DenseNet121", lambda s, d: 1 - d[s, "densenet121"].max(1)),
                  ("1 - max prob., triple", lambda s, d: 1 - fl.weighted_average([d[s, m] for m in MODELS], weights(s, TRIPLE)).max(1)),
                  ("Entropy, DenseNet121", lambda s, d: fl.entropy(d[s, "densenet121"])),
                  ("Entropy, triple", lambda s, d: fl.entropy(fl.weighted_average([d[s, m] for m in MODELS], weights(s, TRIPLE)))),
                  ("1 - margin, triple", lambda s, d: 1 - fl.margin(fl.weighted_average([d[s, m] for m in MODELS], weights(s, TRIPLE)))),
                  ("Member disagreement (count)", lambda s, d: 3 - np.max([np.sum([d[s, m].argmax(1) == k for m in MODELS], 0) for k in range(K)], 0)),
                  ("Mean pairwise Jensen-Shannon divergence", lambda s, d: fl.mean_pairwise_js([d[s, m] for m in MODELS]))]:
    au = [fl.ood_auroc(fn(s, TEST), fn(s, EXT)) for s in SEEDS]
    fp = [fl.fpr_at_tpr(fn(s, TEST), fn(s, EXT)) * 100 for s in SEEDS]
    rows.append({"score": label, "auroc": ms(au, 3), "fpr_at_95tpr": ms(fp, 1)})
e11 = save(pd.DataFrame(rows), "e11_ood.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, sc, title in [(axes[0], lambda d: fl.mean_pairwise_js([d[s0, m] for m in MODELS]), "Mean pairwise JS divergence (triple)"),
                      (axes[1], lambda d: 1 - d[s0, "densenet121"].max(1), "1 - max probability (DenseNet121)")]:
    a, b = sc(TEST), sc(EXT)
    bins_ = np.linspace(0, max(a.max(), b.max()), 40)
    ax.hist(a, bins_, density=True, alpha=0.7, color=BLUE, label="teaLeafBD test")
    ax.hist(b, bins_, density=True, alpha=0.55, color=RED, label="TeaLeafDiseaseBD")
    ax.set(title=title, xlabel="Score", ylabel="Density")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(R / "fig_ood_hist.png")
plt.close(fig)

# ---------------------------------------------------------------- E12 label-free remedies
rows = []
for rem, tag in [("Baseline", None), ("Background removal (test time)", "cutout"), ("AdaBN (test time)", "adabn")]:
    for c in CONFIGS:
        acc, mf1 = [], []
        for s in SEEDS:
            src = EXT if tag is None else {(s, m): np.load(A / f"rem_{tag}_{m}_seed{s}.npy") for m in c}
            p = fl.weighted_average([src[s, m] for m in c], weights(s, c))
            acc.append(accuracy_score(y_ext, p.argmax(1)) * 100)
            mf1.append(f1_score(y_ext, p.argmax(1), labels=SHARED, average="macro", zero_division=0) * 100)
        rows.append({"remedy": rem, "config": name(c), "accuracy": ms(acc), "macro_f1": ms(mf1), "acc_mean": np.mean(acc)})
e12 = save(pd.DataFrame(rows), "e12_remedies.csv")
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(CONFIGS))
for k, (rem, col) in enumerate([("Baseline", BLUE), ("Background removal (test time)", LIGHT), ("AdaBN (test time)", RED)]):
    d = e12[e12.remedy == rem]
    ax.bar(x + (k - 1) * 0.27, d.acc_mean, 0.27, color=col, label=rem)
ax.set_xticks(x, [short(c) for c in CONFIGS])
ax.set(ylabel="TeaLeafDiseaseBD accuracy (%)", xlabel="Configuration (C = CNN, D = DenseNet121, V = VGG16)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
fig.savefig(R / "fig_remedies.png")
plt.close(fig)

# ---------------------------------------------------------------- E13 ablation
def acc_pair(c=None, rule=None):
    a_in, a_ex = [], []
    for s in SEEDS:
        if rule == "equal":
            pi, pe = [fl.weighted_average([d[s, m] for m in MODELS], np.ones(3)).argmax(1) for d in (TEST, EXT)]
        elif rule == "majority":
            w = weights(s, TRIPLE)
            pi, pe = fl.majority_vote([TEST[s, m] for m in MODELS], w), fl.majority_vote([EXT[s, m] for m in MODELS], w)
        else:
            pi, pe = fuse(s, c, "test").argmax(1), fuse(s, c, "ext").argmax(1)
        a_in.append(accuracy_score(y_test, pi) * 100)
        a_ex.append(accuracy_score(y_ext, pe) * 100)
    return a_in, a_ex


full_in, full_ex = acc_pair(TRIPLE)
rows = []
for label, kw in [("Full triple ensemble (accuracy-weighted soft voting)", {"c": TRIPLE}),
                  ("Without the CNN (DenseNet121 + VGG16)", {"c": ("densenet121", "vgg16")}),
                  ("Without DenseNet121 (CNN + VGG16)", {"c": ("cnn", "vgg16")}),
                  ("Without VGG16 (CNN + DenseNet121)", {"c": BEST_PAIR}),
                  ("Equal weights instead of accuracy weights", {"rule": "equal"}),
                  ("Majority voting instead of soft voting", {"rule": "majority"}),
                  ("All weight on the most accurate member (DenseNet121)", {"c": DN})]:
    a_in, a_ex = acc_pair(**kw)
    rows.append({"variant": label, "acc_in": ms(a_in), "change_in": f"{np.mean(a_in) - np.mean(full_in):+.2f}" if kw != {"c": TRIPLE} else "-",
                 "acc_unseen": ms(a_ex), "change_unseen": f"{np.mean(a_ex) - np.mean(full_ex):+.2f}" if kw != {"c": TRIPLE} else "-",
                 "d_in": np.mean(a_in) - np.mean(full_in), "d_ex": np.mean(a_ex) - np.mean(full_ex)})
e13 = save(pd.DataFrame(rows), "e13_ablation.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
lab_ = ["- CNN", "- DenseNet121", "- VGG16", "equal weights", "majority vote", "DenseNet121 only"]
for ax, col, title in [(axes[0], "d_in", "teaLeafBD test"), (axes[1], "d_ex", "TeaLeafDiseaseBD (unseen)")]:
    v = e13[col].iloc[1:].to_numpy()
    ax.barh(np.arange(6)[::-1], v, color=[RED if x < 0 else BLUE for x in v])
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set(title=title, xlabel="Change in accuracy (points)")
axes[0].set_yticks(np.arange(6)[::-1], lab_)
fig.tight_layout()
fig.savefig(R / "fig_ablation.png")
plt.close(fig)

# ---------------------------------------------------------------- extra figures and supplementary tables
# per-class F1 bars with sd
fig, ax = plt.subplots(figsize=(10, 4.4))
for k, (c, col) in enumerate([(DN, GREY), (BEST_PAIR, LIGHT), (TRIPLE, BLUE)]):
    d = e3[e3.config == name(c)]
    ax.bar(np.arange(K) + (k - 1) * 0.27, d.f1_mean, 0.27, yerr=d.f1_sd, capsize=2, color=col, label=name(c))
ax.set_xticks(np.arange(K), CLASSES, rotation=20, ha="right")
ax.set(ylabel="F1-score (%)", ylim=(75, 101))
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
fig.savefig(R / "fig_perclass_f1.png")
plt.close(fig)

# accuracy on both datasets
e10 = pd.read_csv(R / "e10_cross_dataset.csv")
m_in = e10.teaLeafBD.str.split(" ± ").str[0].astype(float)
s_in = e10.teaLeafBD.str.split(" ± ").str[1].astype(float)
m_ex = e10.unseen_acc.str.split(" ± ").str[0].astype(float)
s_ex = e10.unseen_acc.str.split(" ± ").str[1].astype(float)
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(CONFIGS))
ax.bar(x - 0.2, m_in, 0.4, yerr=s_in, capsize=2, color=BLUE, label="teaLeafBD test")
ax.bar(x + 0.2, m_ex, 0.4, yerr=s_ex, capsize=2, color=RED, label="TeaLeafDiseaseBD (unseen)")
for i in range(len(x)):
    ax.text(i - 0.2, m_in[i] + 1.5, f"{m_in[i]:.1f}", ha="center", fontsize=8)
    ax.text(i + 0.2, m_ex[i] + 1.5, f"{m_ex[i]:.1f}", ha="center", fontsize=8)
ax.set_xticks(x, [short(c) for c in CONFIGS])
ax.set(ylabel="Accuracy (%)", ylim=(0, 105), xlabel="Configuration (C = CNN, D = DenseNet121, V = VGG16)")
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
fig.savefig(R / "fig_accuracy_both.png")
plt.close(fig)

# diversity heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, key, title, fmt in [(axes[0], "q", "Yule's Q", "{:.2f}"), (axes[1], "double_fault", "Double fault (%)", "{:.1f}")]:
    mat = np.full((3, 3), np.nan)
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if i != j:
                v = np.mean([fl.pair_diversity(TEST[s, a].argmax(1), TEST[s, b].argmax(1), y_test)[key] for s in SEEDS])
                mat[i, j] = v * (100 if key == "double_fault" else 1)
    ax.imshow(mat, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "-" if i == j else fmt.format(mat[i, j]), ha="center", va="center")
    ax.set_xticks(range(3), [LABELS[m] for m in MODELS])
    ax.set_yticks(range(3), [LABELS[m] for m in MODELS])
    ax.set_title(title)
    ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(R / "fig_diversity.png")
plt.close(fig)

# agreement breakdown
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(agree.members_right, agree.share, color=[RED, LIGHT, BLUE, "#1d3c52"])
for _, r in agree.iterrows():
    ax.text(r.members_right, r.share + 1.5, f"{r.share:.2f}% of images\nfused acc. {r.fused_acc:.1f}%", ha="center", fontsize=8)
ax.set_xticks(range(4), ["0 of 3", "1 of 3", "2 of 3", "3 of 3"])
ax.set(xlabel="Members that are correct", ylabel="Share of test images (%)", ylim=(0, 95))
fig.tight_layout()
fig.savefig(R / "fig_agreement.png")
plt.close(fig)

# class distribution
counts = split.groupby(["cls", "split"]).size().unstack()[["train", "val", "test"]]
ext_counts = np.bincount(y_ext, minlength=K)
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(K)
bottom = np.zeros(K)
for sp_, col in [("train", BLUE), ("val", LIGHT), ("test", GREY)]:
    ax.bar(x - 0.2, counts[sp_].values, 0.4, bottom=bottom, color=col, label=f"teaLeafBD {sp_}")
    bottom += counts[sp_].values
ax.bar(x + 0.2, ext_counts, 0.4, color=RED, label="TeaLeafDiseaseBD (test only)")
ax.set_xticks(x, CLASSES, rotation=20, ha="right")
ax.set(ylabel="Images")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(R / "fig_class_distribution.png")
plt.close(fig)

# training time per model and seed
summ = json.loads((OUT / "summary.json").read_text())["models"]
fig, ax = plt.subplots(figsize=(8, 3.8))
for k, (m, col) in enumerate(zip(MODELS, [LIGHT, BLUE, GREY])):
    v = [summ[f"{LABELS[m]} seed {s}"]["train_minutes"] for s in SEEDS]
    ax.bar(np.arange(3) + (k - 1) * 0.27, v, 0.27, color=col, label=LABELS[m])
ax.set_xticks(range(3), [f"seed {s}" for s in SEEDS])
ax.set(ylabel="Training time (min)")
ax.legend(frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(R / "fig_training_time.png")
plt.close(fig)

# S1 rule cost, S8 confusion pairs
save(e6[["rule", "cost_ms"]].assign(cost_ms=lambda d: d.cost_ms.round(1)), "s1_rule_cost.csv")
pairs = []
for s in SEEDS:
    p = fuse(s, TRIPLE, "test").argmax(1)
    for t, q in zip(y_test[p != y_test], p[p != y_test]):
        pairs.append((CLASSES[t], CLASSES[q]))
conf = pd.Series(pairs).value_counts() / len(SEEDS)
save(pd.DataFrame({"true": [a for a, _ in conf.index], "predicted": [b for _, b in conf.index], "per_seed": conf.values.round(1)}).head(12),
     "s8_confusion_pairs.csv")

numbers.update({
    "all_correct": float(agree.share[3]), "all_wrong": float(agree.share[0]),
    "acc_two_right": float(agree.fused_acc[2]), "acc_one_right": float(agree.fused_acc[1]),
    "rules": e6.set_index("rule")[["acc_mean", "ext_mean"]].round(2).to_dict(orient="index"),
    "alpha": alpha_rows,
})
(R / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
print(json.dumps({k: v for k, v in numbers.items() if k not in ("rules", "alpha")}, indent=1, default=float))
print(e6[["rule", "accuracy", "macro_f1", "nll", "ece", "unseen_accuracy", "p_lt_005"]].to_string(index=False))
