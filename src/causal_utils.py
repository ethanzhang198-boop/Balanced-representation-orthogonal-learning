from __future__ import annotations

import numpy as np
import pandas as pd
import scipy._lib._util as scipy_util
from scipy.stats import norm
from sklearn.linear_model import LassoCV


if not hasattr(scipy_util, "_lazywhere"):
    def _lazywhere(cond, arrays, f, fillvalue=np.nan, f2=None):
        arrays = [np.asarray(arr) for arr in arrays]
        cond = np.asarray(cond, dtype=bool)
        out = np.full(cond.shape, fillvalue, dtype=np.result_type(*arrays))
        if np.any(cond):
            out[cond] = f(*[arr[cond] for arr in arrays])
        if f2 is not None:
            inv = ~cond
            if np.any(inv):
                out[inv] = f2(*[arr[inv] for arr in arrays])
        return out

    scipy_util._lazywhere = _lazywhere

import statsmodels.api as sm
from econml.dml import CausalForestDML


def dml_ate(X, Y, T, folds=3, seed=42):
    from sklearn.model_selection import KFold

    y_hat = np.zeros(len(Y))
    t_hat = np.zeros(len(T))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(X)):
        ym = LassoCV(cv=3, random_state=seed + fold)
        tm = LassoCV(cv=3, random_state=seed + fold + 100)
        ym.fit(X[tr], Y[tr])
        tm.fit(X[tr], T[tr])
        y_hat[te] = ym.predict(X[te])
        t_hat[te] = tm.predict(X[te])
    y_res = Y - y_hat
    t_res = T - t_hat
    fit = sm.OLS(y_res, sm.add_constant(pd.DataFrame({"t_res": t_res}), has_constant="add")).fit(cov_type="HC3")
    return {
        "estimate": float(fit.params["t_res"]),
        "std_err": float(fit.bse["t_res"]),
        "t_value": float(fit.tvalues["t_res"]),
        "p_value": float(fit.pvalues["t_res"]),
        "ci_low": float(fit.conf_int().loc["t_res", 0]),
        "ci_high": float(fit.conf_int().loc["t_res", 1]),
        "const": float(fit.params["const"]),
        "const_std_err": float(fit.bse["const"]),
        "const_t_value": float(fit.tvalues["const"]),
        "const_p_value": float(fit.pvalues["const"]),
        "const_ci_low": float(fit.conf_int().loc["const", 0]),
        "const_ci_high": float(fit.conf_int().loc["const", 1]),
        "nobs": int(fit.nobs),
    }


def fit_forest(train_X, train_Y, train_T, test_X, config):
    ym = LassoCV(cv=3, random_state=config.seed)
    tm = LassoCV(cv=3, random_state=config.seed + 1)
    forest = CausalForestDML(
        model_y=ym,
        model_t=tm,
        discrete_treatment=False,
        n_estimators=config.cf_n_estimators,
        min_samples_leaf=config.cf_min_samples_leaf,
        max_depth=config.cf_max_depth,
        honest=True,
        inference=True,
        cv=config.folds,
        random_state=config.seed,
        n_jobs=1,
    )
    forest.fit(Y=train_Y, T=train_T, X=train_X, W=None)
    cate = forest.effect(test_X)
    lb, ub = forest.effect_interval(test_X)
    ate_lb, ate_ub = forest.ate_interval(X=test_X)
    return {
        "cate": cate,
        "cate_lb": lb,
        "cate_ub": ub,
        "ate_hat": float(np.mean(cate)),
        "ate_ci_low": float(ate_lb),
        "ate_ci_high": float(ate_ub),
    }


def top_bottom_gap(cate, reps=200, seed=42):
    rng = np.random.default_rng(seed)
    q = pd.qcut(cate, 10, labels=False, duplicates="drop")
    top = cate[q == np.max(q)]
    bottom = cate[q == np.min(q)]
    gap = float(np.mean(top) - np.mean(bottom))
    boots = []
    for _ in range(reps):
        s1 = top[rng.integers(0, len(top), len(top))]
        s0 = bottom[rng.integers(0, len(bottom), len(bottom))]
        boots.append(np.mean(s1) - np.mean(s0))
    boots = np.asarray(boots)
    se = float(np.std(boots, ddof=1))
    z = gap / (se + 1e-12)
    p = float(2 * (1 - norm.cdf(abs(z))))
    return {
        "top_mean": float(np.mean(top)),
        "bottom_mean": float(np.mean(bottom)),
        "gap": gap,
        "se_boot": se,
        "p_value": p,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def subgroup_cate_summary(df, cate, group_cols, reps=500, seed=42):
    rng = np.random.default_rng(seed)
    work = df.copy().reset_index(drop=True)
    work["cate_hat"] = np.asarray(cate)
    rows = []
    for col in group_cols:
        if col not in work.columns:
            continue
        s = work[col]
        if set(pd.Series(s).dropna().unique()).issubset({0, 1}):
            g = (s == 1).astype(int)
            label_rule = "binary(1 vs 0)"
        else:
            cutoff = float(pd.Series(s).median())
            g = (s >= cutoff).astype(int)
            label_rule = f"median split >= {cutoff:.4f}"
        g1 = work.loc[g == 1, "cate_hat"].to_numpy()
        g0 = work.loc[g == 0, "cate_hat"].to_numpy()
        if len(g1) == 0 or len(g0) == 0:
            continue
        diff = float(g1.mean() - g0.mean())
        boots = []
        for _ in range(reps):
            b1 = g1[rng.integers(0, len(g1), len(g1))]
            b0 = g0[rng.integers(0, len(g0), len(g0))]
            boots.append(float(b1.mean() - b0.mean()))
        boots = np.asarray(boots)
        se = float(np.std(boots, ddof=1))
        z = diff / (se + 1e-12)
        p = float(2 * (1 - norm.cdf(abs(z))))
        rows.append(
            {
                "group_var": col,
                "split_rule": label_rule,
                "group1_mean_cate": float(g1.mean()),
                "group0_mean_cate": float(g0.mean()),
                "diff_g1_minus_g0": diff,
                "diff_se_boot": se,
                "diff_p_value": p,
                "diff_ci_low": float(np.quantile(boots, 0.025)),
                "diff_ci_high": float(np.quantile(boots, 0.975)),
                "n_group1": int(len(g1)),
                "n_group0": int(len(g0)),
            }
        )
    return pd.DataFrame(rows)
