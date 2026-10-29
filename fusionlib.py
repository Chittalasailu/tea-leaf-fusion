"""Fusion rules, calibration, diversity, significance tests and selective prediction.

Everything here works on saved probability arrays (n_images x n_classes), no PyTorch needed.
"""
from itertools import combinations, product

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-12


def check_probs(p, n_classes=None):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError("probabilities must be a non-empty 2-D array")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain NaN or inf")
    if (p < -1e-6).any() or not np.allclose(p.sum(1), 1, atol=1e-3):
        raise ValueError("rows must be non-negative and sum to 1")
    if n_classes is not None and p.shape[1] != n_classes:
        raise ValueError("wrong number of classes")
    return p


def check_labels(y, n, n_classes):
    y = np.asarray(y)
    if y.shape != (n,):
        raise ValueError("labels do not match the number of images")
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= n_classes:
        raise ValueError("invalid labels")
    return y


def normalise(p):
    p = np.clip(p, 0, None)
    return p / np.clip(p.sum(1, keepdims=True), EPS, None)


# --- fusion rules

def accuracy_weights(val_probs, y_val, alpha=1.0):
    """Validation accuracy of each member, raised to alpha and normalised (alpha=0 gives equal weights)."""
    acc = np.array([(check_probs(p).argmax(1) == y_val).mean() for p in val_probs])
    w = acc ** alpha
    return w / w.sum()


def weighted_average(probs, w):
    probs = [check_probs(p) for p in probs]
    if len({p.shape for p in probs}) != 1:
        raise ValueError("members have different shapes")
    w = np.asarray(w, dtype=float)
    if len(w) != len(probs) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("bad weights")
    w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, probs))


def weighted_product(probs, w):
    w = np.asarray(w, dtype=float) / np.sum(w)
    z = sum(wi * np.log(np.clip(check_probs(p), EPS, 1)) for wi, p in zip(w, probs))
    z -= z.max(1, keepdims=True)
    return normalise(np.exp(z))


def max_rule(probs):
    return normalise(np.max([check_probs(p) for p in probs], 0))


def majority_vote(probs, w):
    """Plurality of the members' labels; with no majority the weighted average decides. Returns labels."""
    votes = np.array([check_probs(p).argmax(1) for p in probs])
    fallback = weighted_average(probs, w).argmax(1)
    out = fallback.copy()
    for i in range(votes.shape[1]):
        vals, cnt = np.unique(votes[:, i], return_counts=True)
        if cnt.max() > 1:
            out[i] = vals[cnt.argmax()]
    return out


def simplex_grid(k, step=0.05):
    n = int(round(1 / step))
    return np.array([np.array(c) / n for c in product(range(n + 1), repeat=k) if sum(c) == n])


def grid_weights(val_probs, y_val, step=0.05):
    best, best_key = None, None
    for w in simplex_grid(len(val_probs), step):
        p = weighted_average(val_probs, w)
        key = ((p.argmax(1) == y_val).mean(), -nll(p, y_val))
        if best_key is None or key > best_key:
            best, best_key = w, key
    return best


def logloss_weights(val_probs, y_val):
    k = len(val_probs)
    res = minimize(lambda w: nll(weighted_average(val_probs, np.clip(w, 1e-9, None)), y_val), np.ones(k) / k,
                   method="SLSQP", bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def classwise_weights(val_probs, y_val, n_classes):
    """w[m, c] = validation recall of member m on class c, normalised over members."""
    rec = np.array([[(p.argmax(1)[y_val == c] == c).mean() if (y_val == c).any() else 0 for c in range(n_classes)]
                    for p in val_probs])
    return rec / np.clip(rec.sum(0, keepdims=True), EPS, None)


def classwise_average(probs, wc):
    return normalise(sum(wc[m][None, :] * check_probs(p) for m, p in enumerate(probs)))


def calibrate_then_fuse(val_probs, y_val, probs, w):
    ts = [fit_temperature(v, y_val) for v in val_probs]
    return weighted_average([temperature_scale(p, t) for p, t in zip(probs, ts)], w)


class Stacker:
    """Multinomial logistic regression on the members' log-probabilities, C picked by 5-fold CV."""

    def __init__(self, Cs=(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100), seed=0):
        self.Cs, self.seed = Cs, seed

    @staticmethod
    def features(probs):
        return np.log(np.clip(np.hstack([check_probs(p) for p in probs]), EPS, 1))

    def fit(self, val_probs, y_val):
        X = self.features(val_probs)
        cv = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        scores = []
        for C in self.Cs:
            ll = []
            for tr, te in cv.split(X, y_val):
                m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y_val[tr])
                q = np.zeros((len(te), X.shape[1] // len(val_probs)))
                q[:, m.classes_] = m.predict_proba(X[te])
                ll.append(nll(normalise(q + 1e-9), y_val[te]))
            scores.append(np.mean(ll))
        self.C = self.Cs[int(np.argmin(scores))]
        self.model = LogisticRegression(C=self.C, max_iter=5000).fit(X, y_val)
        self.n_classes = X.shape[1] // len(val_probs)
        return self

    def predict_proba(self, probs):
        q = np.zeros((len(probs[0]), self.n_classes))
        q[:, self.model.classes_] = self.model.predict_proba(self.features(probs))
        return q


# --- calibration and probabilistic metrics

def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean())


def brier(p, y):
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(1).mean())


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def temperature_scale(p, t):
    if t <= 0:
        raise ValueError("temperature must be positive")
    z = np.log(np.clip(check_probs(p), EPS, 1)) / t
    z -= z.max(1, keepdims=True)
    return normalise(np.exp(z))


def fit_temperature(p, y):
    return float(minimize_scalar(lambda t: nll(temperature_scale(p, t), y), bounds=(0.05, 10), method="bounded").x)


class Isotonic:
    """One-vs-rest isotonic regression per class, then renormalised."""

    def fit(self, p, y):
        self.models = [IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[:, k], y == k)
                       for k in range(p.shape[1])]
        return self

    def predict(self, p):
        q = np.column_stack([m.predict(p[:, k]) for k, m in enumerate(self.models)])
        return normalise(q)


def macro_auc(p, y, labels=None):
    from sklearn.metrics import roc_auc_score
    labels = sorted(set(y.tolist())) if labels is None else labels
    return float(np.mean([roc_auc_score(y == c, p[:, c]) for c in labels]))


# --- diversity

def pair_diversity(pred_a, pred_b, y):
    a, b = pred_a == y, pred_b == y
    n11, n00 = (a & b).sum(), (~a & ~b).sum()
    n10, n01 = (a & ~b).sum(), (~a & b).sum()
    den = n11 * n00 + n01 * n10
    q = (n11 * n00 - n01 * n10) / den if den else 0.0
    ea, eb = (~a).astype(float), (~b).astype(float)
    corr = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else 0.0
    n = len(y)
    return {"q": float(q), "disagreement": float((n01 + n10) / n), "double_fault": float(n00 / n), "error_corr": corr}


def oracle_accuracy(preds, y):
    return float(np.any([p == y for p in preds], 0).mean())


def agreement_breakdown(preds, fused_pred, y):
    """Share of images by how many members are right, and fused accuracy within each group."""
    k = np.sum([p == y for p in preds], 0)
    ok = fused_pred == y
    return {n: (float((k == n).mean()), float(ok[k == n].mean()) if (k == n).any() else float("nan"))
            for n in range(len(preds) + 1)}


# --- significance

def mcnemar(correct_a, correct_b):
    n1 = int((correct_a & ~correct_b).sum())
    n2 = int((~correct_a & correct_b).sum())
    p = binomtest(min(n1, n2), n1 + n2, 0.5).pvalue if n1 + n2 else 1.0
    return n1, n2, float(p)


def holm(pvals, alpha=0.05):
    """Boolean rejections in the original order."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    rej = np.zeros(len(pvals), bool)
    for i, j in enumerate(order):
        if pvals[j] <= alpha / (len(pvals) - i):
            rej[j] = True
        else:
            break
    return rej


def bootstrap_indices(n, reps=2000, seed=0):
    return np.random.default_rng(seed).integers(0, n, (reps, n))


def bootstrap_ci(correct, idx, level=0.95):
    """correct: n_images vector (can be a seed average). Percentile interval in %."""
    vals = np.asarray(correct, float)[idx].mean(1) * 100
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


# --- selective prediction

def advise(p, threshold=0.8, agree=None):
    keep = p.max(1) >= threshold
    if agree is not None:
        keep &= agree
    return keep


def selective_stats(p, y, keep):
    ok = p.argmax(1) == y
    return {"coverage": float(keep.mean()), "selective_acc": float(ok[keep].mean()) if keep.any() else float("nan"),
            "wrong_advice": float((keep & ~ok).mean())}


def risk_coverage(score, correct):
    """Risk of the k most confident images for k = 1..n (higher score = more confident)."""
    order = np.argsort(-np.asarray(score), kind="stable")
    err = (~np.asarray(correct, bool))[order]
    return np.arange(1, len(err) + 1) / len(err), np.cumsum(err) / np.arange(1, len(err) + 1)


def aurc(score, correct):
    _, risk = risk_coverage(score, correct)
    return float(risk.mean())


def coverage_at_risk(score, correct, max_risk=0.01):
    cov, risk = risk_coverage(score, correct)
    ok = risk <= max_risk
    return float(cov[ok].max()) if ok.any() else 0.0


# --- uncertainty scores for spotting unfamiliar images

def entropy(p):
    return -(p * np.log(np.clip(p, EPS, 1))).sum(1)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def mean_pairwise_js(probs):
    vals = []
    for a, b in combinations(probs, 2):
        m = (a + b) / 2
        vals.append(entropy(m) - (entropy(a) + entropy(b)) / 2)
    return np.mean(vals, 0)


def ood_auroc(score_in, score_out):
    """AUROC for flagging out-of-distribution images (higher score = more unusual)."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(score_in)), np.ones(len(score_out))]
    return float(roc_auc_score(y, np.r_[score_in, score_out]))


def fpr_at_tpr(score_in, score_out, tpr=0.95):
    thr = np.quantile(score_out, 1 - tpr)
    return float((np.asarray(score_in) >= thr).mean())
