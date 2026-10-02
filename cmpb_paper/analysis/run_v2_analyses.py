"""Further analyses on the corrected data, with the pipeline of run_paper_analysis.py:
paired bootstrap of XGBoost vs logistic regression (with net benefit), flexible calibration
curves (ICI, E50, E90, Emax), subgroups, and export of the released model to release/model/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paper_analysis as rpa  # noqa: E402

OUT = rpa.OUT
REL = rpa.ROOT / "release" / "model"
REL.mkdir(parents=True, exist_ok=True)
N_BOOT = 1000
N_BOOT_CAL = 200
PTS = [0.05, 0.08, 0.10, 0.15]
SEED = rpa.SEED
log = rpa.log


# 1. paired bootstrap
def nb(y, p, pt):
    pred = p >= pt; n = len(y)
    return np.sum(pred & (y == 1)) / n - np.sum(pred & (y == 0)) / n * pt / (1 - pt)


def treat_all(y, pt):
    prev = y.mean()
    return prev - (1 - prev) * pt / (1 - pt)


def stats(y, pa, pb):
    out = {"auc_xgb": roc_auc_score(y, pa), "auc_lr": roc_auc_score(y, pb),
           "ap_xgb": average_precision_score(y, pa), "ap_lr": average_precision_score(y, pb),
           "brier_xgb": np.mean((pa - y) ** 2), "brier_lr": np.mean((pb - y) ** 2)}
    for pt in PTS:
        out[f"nb_xgb_{pt}"] = nb(y, pa, pt); out[f"nb_lr_{pt}"] = nb(y, pb, pt)
        out[f"nb_all_{pt}"] = treat_all(y, pt)
    return out


def paired_bootstrap(y, pa, pb):
    point = stats(y, pa, pb)
    rng = np.random.default_rng(SEED); reps = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        reps.append(stats(y[i], pa[i], pb[i]))
    reps = pd.DataFrame(reps)
    rows = []
    for m in ("auc", "ap", "brier"):
        for who in ("xgb", "lr"):
            k = f"{m}_{who}"
            rows.append(dict(quantity=k, estimate=point[k], lo=reps[k].quantile(.025), hi=reps[k].quantile(.975)))
        d = reps[f"{m}_xgb"] - reps[f"{m}_lr"]
        rows.append(dict(quantity=f"delta_{m}_xgb_minus_lr", estimate=point[f"{m}_xgb"] - point[f"{m}_lr"],
                         lo=d.quantile(.025), hi=d.quantile(.975)))
    for pt in PTS:
        for who in ("xgb", "lr", "all"):
            k = f"nb_{who}_{pt}"
            rows.append(dict(quantity=k, estimate=point[k], lo=reps[k].quantile(.025), hi=reps[k].quantile(.975)))
        for a, b in (("xgb", "lr"), ("xgb", "all"), ("lr", "all")):
            d = reps[f"nb_{a}_{pt}"] - reps[f"nb_{b}_{pt}"]
            rows.append(dict(quantity=f"delta_nb_{a}_minus_{b}_{pt}",
                             estimate=point[f"nb_{a}_{pt}"] - point[f"nb_{b}_{pt}"],
                             lo=d.quantile(.025), hi=d.quantile(.975)))
    return pd.DataFrame(rows)


# 2. flexible calibration
GRID = np.linspace(0.005, 0.60, 120)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def flex_fit(y, p):
    lp = logit(p)
    knots = np.unique(np.quantile(lp, [0.05, 0.275, 0.5, 0.725, 0.95]))
    if len(knots) < 3:
        knots = np.linspace(lp.min(), lp.max(), 4)
    st = SplineTransformer(knots=knots.reshape(-1, 1), degree=3, extrapolation="linear").fit(lp.reshape(-1, 1))
    lr = LogisticRegression(C=np.inf, max_iter=5000).fit(st.transform(lp.reshape(-1, 1)), y)  # unpenalised
    return lambda q: lr.predict_proba(st.transform(logit(q).reshape(-1, 1)))[:, 1]


def flex_stats(y, p):
    f = flex_fit(y, p)
    err = np.abs(f(p) - p)
    return dict(ici=err.mean(), e50=np.median(err), e90=np.quantile(err, 0.9), emax=err.max()), f(GRID)


def flexible_calibration(y, p, label):
    point, curve = flex_stats(y, p)
    rng = np.random.default_rng(SEED); reps, curves = [], []
    for _ in range(N_BOOT_CAL):
        i = rng.integers(0, len(y), len(y))
        s, c = flex_stats(y[i], p[i]); reps.append(s); curves.append(c)
    reps = pd.DataFrame(reps); curves = np.array(curves)
    summ = {k: dict(estimate=v, lo=reps[k].quantile(.025), hi=reps[k].quantile(.975)) for k, v in point.items()}
    cdf = pd.DataFrame(dict(model=label, predicted=GRID, observed=curve, lo=np.quantile(curves, .025, axis=0),
                            hi=np.quantile(curves, .975, axis=0),
                            share_of_predictions_below=[(p <= g).mean() for g in GRID]))
    log(f"flexible calibration {label}: ICI={point['ici']:.4f} ({summ['ici']['lo']:.4f}-{summ['ici']['hi']:.4f}) "
        f"E50={point['e50']:.4f} E90={point['e90']:.4f} Emax={point['emax']:.3f}")
    return summ, cdf


# 3. subgroups
def subgroup_masks(te):
    sex = te["sex"].map({1: "Male", 2: "Female"})
    age = pd.cut(te["age"], [17, 44, 64, 79, 80], labels=["18-44", "45-64", "65-79", "80 (top-coded)"])
    race = te["race_ethnicity"].astype(float).map({1: "White, non-Hispanic", 2: "Black, non-Hispanic",
                                                   3: "Asian, non-Hispanic", 4: "AI/AN, non-Hispanic",
                                                   5: "Hispanic", 6: "Other or multiracial, non-Hispanic"})
    inc = pd.cut(te["income"].astype(float), [0, 4, 6, 8, 11],
                 labels=["< $25,000", "$25,000-49,999", "$50,000-99,999", ">= $100,000"])
    edu = pd.cut(te["education"].astype(float), [0, 3, 4, 5, 6],
                 labels=["Less than high school", "High school graduate", "Some college", "College graduate"])
    out = [("All", "All", pd.Series(True, index=te.index))]
    for dim, s, levels in (("Sex", sex, ["Female", "Male"]), ("Age", age, list(age.cat.categories)),
                           ("Race and ethnicity", race, ["White, non-Hispanic", "Black, non-Hispanic",
                                                         "Hispanic", "Asian, non-Hispanic", "AI/AN, non-Hispanic",
                                                         "Other or multiracial, non-Hispanic"]),
                           ("Household income", inc, list(inc.cat.categories)),
                           ("Education", edu, list(edu.cat.categories))):
        out += [(dim, lv, s == lv) for lv in levels]
        out.append((dim, "Missing", s.isna()))
    return out


def subgroups(te, y, p, t):
    rows = []
    for dim, lv, m in subgroup_masks(te):
        m = m.values
        if m.sum() == 0:
            continue
        yy, pp = y[m], p[m]
        if 0 < yy.sum() < len(yy):
            rng = np.random.default_rng(SEED); b = []
            for _ in range(500):
                i = rng.integers(0, len(yy), len(yy))
                if yy[i].min() != yy[i].max():
                    b.append(roc_auc_score(yy[i], pp[i]))
            auc, lo, hi = roc_auc_score(yy, pp), *np.percentile(b, [2.5, 97.5])
        else:
            auc = lo = hi = np.nan
        pred = pp >= t
        tp = np.sum(pred & (yy == 1)); fp = np.sum(pred & (yy == 0))
        rows.append(dict(dimension=dim, group=lv, n=int(m.sum()), cases=int(yy.sum()), prevalence=yy.mean(),
                         roc_auc=auc, roc_auc_lo=lo, roc_auc_hi=hi, observed_over_expected=yy.mean() / pp.mean(),
                         sensitivity=tp / max(yy.sum(), 1), specificity=1 - fp / max((yy == 0).sum(), 1),
                         ppv=tp / max(tp + fp, 1), flagged_share=pred.mean()))
    return pd.DataFrame(rows)


# 4. model export
def export_model(m, tr):
    ct = m.ct
    spec = {"features": m.feats, "numeric": [], "categorical": []}
    num = ct.named_transformers_["num"]; imp, sc = num.named_steps["imp"], num.named_steps["sc"]
    for c, med, mu, s in zip(ct.transformers_[0][2], imp.statistics_, sc.mean_, sc.scale_):
        spec["numeric"].append(dict(name=c, impute_median=float(med), mean=float(mu), scale=float(s)))
    cat = ct.named_transformers_["cat"]; imp, ohe = cat.named_steps["imp"], cat.named_steps["ohe"]
    for j, c in enumerate(ct.transformers_[1][2]):
        cats = [str(v) for v in ohe.categories_[j]]
        spec["categorical"].append(dict(name=c, impute_most_frequent=str(imp.statistics_[j]),
                                        categories=cats, dropped_reference=cats[0]))
    spec["output_columns"] = [str(n) for n in ct.get_feature_names_out()]
    (REL / "preprocessing.json").write_text(json.dumps(spec, indent=1))
    m.m.get_booster().save_model(str(REL / "xgb_pre_event.json"))
    import sklearn, xgboost
    (REL / "model_meta.json").write_text(json.dumps(dict(
        best_iteration=int(m.m.best_iteration), calibration=m.cal_name, n_training=len(tr),
        training_data="BRFSS 2023, 60% stratified training split (seed 42), corrected missing-value codes",
        xgboost_version=xgboost.__version__, sklearn_version=sklearn.__version__, params=m.params), indent=1))
    if m.cal_name == "isotonic":
        pd.DataFrame({"raw_score": m.cal_obj.X_thresholds_, "calibrated_risk": m.cal_obj.y_thresholds_}) \
            .to_csv(REL / "calibration_isotonic.csv", index=False)
    else:
        pd.DataFrame({"coef": m.cal_obj.coef_.ravel(), "intercept": m.cal_obj.intercept_}) \
            .to_csv(REL / "calibration_sigmoid.csv", index=False)
    pd.DataFrame([dict(tier=k, recall_target=rpa.TIERS[k], threshold=v) for k, v in m.thr.items()]) \
        .to_csv(REL / "thresholds.csv", index=False)
    log(f"exported model to {REL} (calibration: {m.cal_name}, best_iteration={m.m.best_iteration})")


# run
def main():
    t0 = time.time()
    params = rpa.stage2_params()
    d23 = pd.read_parquet(rpa.CACHE / "brfss_2023_stroke_rich_clean.parquet")
    d24 = pd.read_parquet(rpa.CACHE / "brfss_2024_stroke_rich_clean.parquet")
    full = [c for c in d23.columns if c != rpa.TARGET]
    pre = [c for c in full if c not in rpa.POST]
    pre_t = [c for c in pre if c not in rpa.ONLY_2023]
    tr_i, ca_i, th_i, te_i = rpa.four_way_split(d23)
    tr, ca, th, te = (d23.iloc[i].reset_index(drop=True) for i in (tr_i, ca_i, th_i, te_i))
    y, y24 = te[rpa.TARGET].values, d24[rpa.TARGET].values

    xgb = rpa.Model("xgb", pre, params).fit(tr, ca, th)
    lr = rpa.Model("lr", pre, params).fit(tr, ca, th)
    x24 = rpa.Model("xgb", pre_t, params).fit(tr, ca, th)
    px, pl, p24 = xgb.predict(te), lr.predict(te), x24.predict(d24)
    log(f"fitted: xgb_pre ROC={roc_auc_score(y, px):.4f} lr_pre ROC={roc_auc_score(y, pl):.4f} "
        f"xgb_pre_2024feats 2024 ROC={roc_auc_score(y24, p24):.4f}")
    numbers = {}

    pb = paired_bootstrap(y, px, pl)
    pb.to_csv(OUT / "v2_paired_bootstrap.csv", index=False)
    log("paired bootstrap:\n" + pb.round(4).to_string(index=False))

    cal_rows, curves = [], []
    for label, yy, pp in (("XGBoost, 2023 test", y, px), ("Logistic regression, 2023 test", y, pl),
                          ("XGBoost, BRFSS 2024", y24, p24)):
        s, c = flexible_calibration(yy, pp, label)
        cal_rows += [dict(model=label, measure=k) | v for k, v in s.items()]
        curves.append(c)
    pd.DataFrame(cal_rows).to_csv(OUT / "v2_flexible_calibration.csv", index=False)
    pd.concat(curves).to_csv(OUT / "v2_flexible_calibration_curves.csv", index=False)

    sub = subgroups(te, y, px, xgb.thr["primary"])
    sub.to_csv(OUT / "v2_subgroups.csv", index=False)
    log("subgroups:\n" + sub.round(3).to_string(index=False))

    export_model(xgb, tr)
    pd.DataFrame({"y": y, "xgb_pre": px}).iloc[:2000].to_csv(REL / "check_predictions.csv", index=False)
    te[pre].iloc[:2000].astype({c: "float64" for c in pre if te[c].dtype == "float32"}).to_csv(REL / "check_inputs.csv", index=False, float_format="%.17g")
    numbers.update(xgb_pre_thresholds=xgb.thr, xgb_pre_calibration=xgb.cal_name,
                   lr_pre_calibration=lr.cal_name, x24_calibration=x24.cal_name)
    (OUT / "v2_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
