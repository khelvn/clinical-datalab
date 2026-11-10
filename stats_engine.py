"""Statistics engine for the Clinical DataLab (no Streamlit code here, so it can be tested alone)."""
import re
import difflib
import numpy as np
import pandas as pd
from scipy import stats


ENGINE_VERSION = "2026-11-10"


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------- variable types and assumptions ----------
def classify(s, name=""):
    x = s.dropna()
    n, u = len(x), x.nunique()
    if n == 0:
        return "empty"
    if pd.api.types.is_bool_dtype(s) or u == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        if u == n and n > 20 and name.lower().endswith("id"):
            return "id"
        return "categorical" if (u <= 5 and (x % 1 == 0).all()) else "numeric"
    if u == n and n > 20:
        return "id"
    return "categorical" if u <= 15 else "text"


def check_normal(x):
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    if n < 3 or x.nunique() == 1:
        return {"ok": False, "p": None, "W": None, "n": n}
    W, p = stats.shapiro(x.sample(5000, random_state=0) if n > 5000 else x)
    ok = p > 0.05 or (n >= 100 and abs(stats.skew(x)) < 1)
    return {"ok": bool(ok), "p": float(p), "W": float(W), "n": n, "skew": float(stats.skew(x))}


def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


# ---------- cleaning ----------
def fill_missing(df, col, method, label="Not applicable"):
    df = df.copy()
    n = int(df[col].isna().sum())
    if method == "drop":
        df = df.dropna(subset=[col])
        return df, f"Dropped {n} rows with missing '{col}'."
    if method == "na":
        df[col] = df[col].astype(object).fillna(label)
        return df, f"Labelled {n} blank answers in '{col}' as '{label}' (skip question)."
    val = {"median": lambda s: s.median(), "mean": lambda s: s.mean(), "mode": lambda s: s.mode().iloc[0]}[method](df[col])
    df[col] = df[col].fillna(val)
    return df, f"Filled {n} missing values in '{col}' with {method} ({val if isinstance(val, str) else round(float(val), 2)})."


def cap_outliers(df, col):
    lo, hi = iqr_bounds(df[col].dropna())
    n = int(((df[col] < lo) | (df[col] > hi)).sum())
    df = df.copy()
    df[col] = df[col].clip(lo, hi)
    return df, f"Capped {n} outliers in '{col}' to [{lo:.1f}, {hi:.1f}]."


# ---------- the recommender ----------
def _out(test, reasons, stat_label, stat, p, alpha, df=None, effect=None, effect_label="", ci=None,
         alt="", interp="", check=None, posthoc=None, plot=None, warnings=None):
    sig = p < alpha
    return dict(test=test, reasons=reasons, stat_label=stat_label, stat=stat, df=df, p=p, alpha=alpha,
                effect=effect, effect_label=effect_label, ci=ci, alternative=alt, interpretation=interp,
                check=check, posthoc=posthoc, plot=plot, warnings=warnings or [], significant=sig,
                decision=f"{fmt_p(p)}: {'statistically significant' if sig else 'not statistically significant'} at the {alpha} level.")


def analyze(df, v1, v2, paired=False, alpha=0.05, prefer=None):
    r = _analyze(df, v1, v2, paired, alpha, prefer)
    if prefer:
        a = _analyze(df, v1, v2, paired, alpha, None)
        if a["test"] != r["test"]:
            r["warnings"].insert(0, f"You chose a different test from the recommended one ({a['test']}). The assumption checks above explain the recommendation. Interpret with care.")
    return r


def _analyze(df, v1, v2, paired, alpha, prefer):
    k1, k2 = classify(df[v1], v1), classify(df[v2], v2)
    if v1 == v2:
        raise ValueError("Please choose two different variables.")
    if {k1, k2} & {"id", "text", "empty"}:
        raise ValueError("ID or free text columns cannot be analysed. Choose clinical variables.")
    cat = {"binary", "categorical"}
    if k1 == k2 == "numeric":
        return _paired(df, v1, v2, alpha, prefer) if paired else _corr(df, v1, v2, alpha, prefer)
    if paired and k1 == k2 == "binary":
        return _mcnemar(df, v1, v2, alpha)
    if paired:
        raise ValueError("A paired analysis needs two numeric measurements on the same patients.")
    if k1 == "numeric":
        return _groups(df, v1, v2, alpha, prefer)
    if k2 == "numeric":
        return _groups(df, v2, v1, alpha, prefer)
    return _assoc(df, v1, v2, alpha, prefer)


def _groups(df, m, g, alpha, prefer=None):
    d = df[[m, g]].dropna()
    grp = {str(k): x[m].astype(float).values for k, x in d.groupby(g) if len(x) >= 2}
    names, vals = list(grp), list(grp.values())
    k = len(names)
    if k < 2:
        raise ValueError("Each group needs at least 2 observations, and at least 2 groups.")
    nm = [check_normal(v) for v in vals]
    all_ok = {"parametric": True, "nonparametric": False}.get(prefer, all(x["ok"] for x in nm))
    lev = float(stats.levene(*vals, center="median")[1])
    reasons = [f"'{m}' is numeric and '{g}' splits patients into {k} independent groups."]
    reasons += [f"{n}: n = {len(v)}, Shapiro Wilk {fmt_p(x['p']) if x['p'] is not None else 'not available'}, "
                f"{'looks normal' if x['ok'] else 'not normal'}." for n, v, x in zip(names, vals, nm)]
    reasons.append(f"Equal variance check (Levene): {fmt_p(lev)}.")
    plot = {"type": "box", "groups": grp, "ylabel": m}
    pct = int((1 - alpha) * 100)
    if k == 2:
        a, b = vals
        if all_ok:
            res = stats.ttest_ind(a, b, equal_var=lev > alpha)
            ci = res.confidence_interval(1 - alpha)
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            crit = stats.t.ppf(1 - alpha / 2, res.df)
            return _out("Independent samples t test" + ("" if lev > alpha else " (Welch)"), reasons, "t", res.statistic,
                        res.pvalue, alpha, res.df, (a.mean() - b.mean()) / sp, "Cohen's d",
                        (f"{pct}% CI for mean difference", (ci.low, ci.high)),
                        "Mann Whitney U test if you doubt the normality assumption.",
                        f"Mean {m} was {a.mean():.2f} in {names[0]} and {b.mean():.2f} in {names[1]} (difference {a.mean() - b.mean():.2f}).",
                        {"name": f"Critical t (two tailed, df = {res.df:.1f})", "crit": crit, "hit": abs(res.statistic) > crit},
                        plot=plot)
        res = stats.mannwhitneyu(a, b)
        return _out("Mann Whitney U test", reasons, "U", res.statistic, res.pvalue, alpha, None,
                    2 * res.statistic / (len(a) * len(b)) - 1, "Rank biserial correlation", None,
                    "Independent samples t test if the data were approximately normal.",
                    f"Median {m} was {np.median(a):.2f} in {names[0]} and {np.median(b):.2f} in {names[1]}.", plot=plot)
    N = sum(len(v) for v in vals)
    if all_ok and (lev > alpha or prefer == "parametric"):
        F, p = stats.f_oneway(*vals)
        allv = np.concatenate(vals)
        ssb = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in vals)
        res = stats.tukey_hsd(*vals)
        ph = pd.DataFrame([{"Comparison": f"{names[i]} vs {names[j]}", "Mean difference": vals[i].mean() - vals[j].mean(),
                            "Adjusted p": res.pvalue[i, j]} for i in range(k) for j in range(i + 1, k)])
        crit = stats.f.ppf(1 - alpha, k - 1, N - k)
        return _out("One way ANOVA", reasons, "F", F, p, alpha, (k - 1, N - k), ssb / ((allv - allv.mean()) ** 2).sum(),
                    "Eta squared", None, "Kruskal Wallis test if normality or equal variance fails.",
                    "Group means: " + ", ".join(f"{n} {v.mean():.2f}" for n, v in zip(names, vals)) + ".",
                    {"name": f"Critical F (df = {k - 1}, {N - k})", "crit": crit, "hit": F > crit},
                    ph, plot, ["ANOVA only says that at least one group differs. Use the post hoc table to see which."])
    H, p = stats.kruskal(*vals)
    m_pairs = k * (k - 1) // 2
    ph = pd.DataFrame([{"Comparison": f"{names[i]} vs {names[j]}", "Adjusted p": min(1, stats.mannwhitneyu(vals[i], vals[j])[1] * m_pairs)}
                       for i in range(k) for j in range(i + 1, k)])
    crit = stats.chi2.ppf(1 - alpha, k - 1)
    return _out("Kruskal Wallis test", reasons, "H", H, p, alpha, k - 1, (H - k + 1) / (N - k), "Epsilon squared", None,
                "One way ANOVA if the data were normal with equal variances.",
                "Group medians: " + ", ".join(f"{n} {np.median(v):.2f}" for n, v in zip(names, vals)) + ".",
                {"name": f"Critical chi square (df = {k - 1})", "crit": crit, "hit": H > crit}, ph, plot,
                ["Post hoc uses pairwise Mann Whitney tests with Bonferroni adjustment."])


def _paired(df, a, b, alpha, prefer=None):
    x = df[[a, b]].dropna().astype(float)
    if len(x) < 3:
        raise ValueError("Need at least 3 complete pairs.")
    d = x[a] - x[b]
    nm = check_normal(d)
    reasons = [f"'{a}' and '{b}' are two numeric measurements on the same patients (paired).",
               f"Differences: n = {nm['n']}, Shapiro Wilk {fmt_p(nm['p']) if nm['p'] is not None else 'not available'}, "
               f"{'look normal' if nm['ok'] else 'not normal'}."]
    plot = {"type": "paired", "x": x[a].values, "y": x[b].values, "xlabel": a, "ylabel": b}
    interp = f"Mean {a} minus {b} was {d.mean():.2f} (n = {len(x)} pairs)."
    if {"parametric": True, "nonparametric": False}.get(prefer, nm["ok"]):
        res = stats.ttest_rel(x[a], x[b])
        ci = res.confidence_interval(1 - alpha)
        crit = stats.t.ppf(1 - alpha / 2, res.df)
        return _out("Paired t test", reasons, "t", res.statistic, res.pvalue, alpha, res.df, d.mean() / d.std(ddof=1),
                    "Cohen's d (paired)", (f"{int((1 - alpha) * 100)}% CI for mean difference", (ci.low, ci.high)),
                    "Wilcoxon signed rank test if differences are not normal.", interp,
                    {"name": f"Critical t (two tailed, df = {res.df:.0f})", "crit": crit, "hit": abs(res.statistic) > crit}, plot=plot)
    dd = d[d != 0]
    r = stats.rankdata(dd.abs())
    wp, wm = r[dd > 0].sum(), r[dd < 0].sum()
    res = stats.wilcoxon(d)
    return _out("Wilcoxon signed rank test", reasons, "W", res.statistic, res.pvalue, alpha, None, (wp - wm) / (wp + wm),
                "Matched pairs rank biserial correlation", None, "Paired t test if differences were normal.",
                f"Median difference ({a} minus {b}) was {d.median():.2f} (n = {len(x)} pairs).", plot=plot)


def _corr(df, a, b, alpha, prefer=None):
    x = df[[a, b]].dropna().astype(float)
    n = len(x)
    if n < 4:
        raise ValueError("Need at least 4 complete pairs.")
    na, nb = check_normal(x[a]), check_normal(x[b])
    reasons = [f"'{a}' and '{b}' are both numeric, so we look for an association.",
               f"'{a}': {'looks normal' if na['ok'] else 'not normal'}. '{b}': {'looks normal' if nb['ok'] else 'not normal'}.",
               "Check the scatterplot: Pearson assumes a roughly straight line relationship."]
    pct = int((1 - alpha) * 100)
    if {"parametric": True, "nonparametric": False}.get(prefer, na["ok"] and nb["ok"]):
        res = stats.pearsonr(x[a], x[b])
        r, p = res.statistic, res.pvalue
        ci = res.confidence_interval(1 - alpha)
        ci, test, lab, alt = (ci.low, ci.high), "Pearson correlation", "r", "Spearman correlation if the relationship is not linear or data are skewed."
    else:
        r, p = stats.spearmanr(x[a], x[b])
        z, se = np.arctanh(r), np.sqrt((1 + r ** 2 / 2) / (n - 3))
        zc = stats.norm.ppf(1 - alpha / 2)
        ci, test, lab, alt = (np.tanh(z - zc * se), np.tanh(z + zc * se)), "Spearman rank correlation", "rho", "Pearson correlation if both variables were normal and linear."
    tc = stats.t.ppf(1 - alpha / 2, n - 2)
    lr = stats.linregress(x[a], x[b])
    strength = "negligible" if abs(r) < 0.1 else "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.5 else "strong"
    return _out(test, reasons, lab, r, p, alpha, n - 2, r, "Correlation coefficient", (f"{pct}% CI", ci), alt,
                f"A {strength} {'positive' if r > 0 else 'negative'} relationship. Simple regression: {b} = {lr.intercept:.2f} {lr.slope:+.3f} x {a}. Correlation is not causation.",
                {"name": f"Critical r (two tailed, df = {n - 2})", "crit": tc / np.sqrt(n - 2 + tc ** 2), "hit": abs(r) > tc / np.sqrt(n - 2 + tc ** 2)},
                plot={"type": "scatter", "x": x[a].values, "y": x[b].values, "xlabel": a, "ylabel": b, "fit": (lr.intercept, lr.slope)})


def _assoc(df, a, b, alpha, prefer=None):
    tab = pd.crosstab(df[a], df[b])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        raise ValueError("Both variables need at least 2 levels with data.")
    chi, p, dof, exp = stats.chi2_contingency(tab, correction=False)
    n = tab.values.sum()
    low = float((exp < 5).mean())
    v = float(np.sqrt(chi / (n * (min(tab.shape) - 1))))
    pct = tab.div(tab.sum(1), axis=0) * 100
    reasons = [f"'{a}' and '{b}' are both categories, so we compare proportions.",
               f"Smallest expected count is {exp.min():.1f}; {low * 100:.0f}% of cells are below 5."]
    plot = {"type": "stack", "pct": pct, "ylabel": f"% within {a}"}
    warn = ["If the same patients are measured twice (before and after), use McNemar's test instead."]
    if tab.shape == (2, 2):
        A, B, C, D = (tab.values.ravel() + (0.5 if (tab.values == 0).any() else 0)).tolist()
        orr = A * D / (B * C)
        se = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)
        zc = stats.norm.ppf(1 - alpha / 2)
        ci = (f"{int((1 - alpha) * 100)}% CI for odds ratio", (np.exp(np.log(orr) - zc * se), np.exp(np.log(orr) + zc * se)))
        r0, r1, c1 = tab.index[0], tab.index[1], tab.columns[1]
        interp = (f"{pct.iloc[0, 1]:.1f}% of '{r0}' had '{c1}' compared with {pct.iloc[1, 1]:.1f}% of '{r1}'. "
                  f"Odds ratio {orr:.2f} (odds of '{c1}' in '{r1}' versus '{r0}'). Cramer's V = {v:.2f}.")
        if (exp.min() >= 5 and prefer != "exact") or prefer == "chisquare":
            return _out("Chi square test of independence", reasons, "chi square", chi, p, alpha, dof, orr, "Odds ratio", ci,
                        "Fisher's exact test if any expected count falls below 5.", interp,
                        {"name": f"Critical chi square (df = {dof})", "crit": stats.chi2.ppf(1 - alpha, dof), "hit": chi > stats.chi2.ppf(1 - alpha, dof)},
                        plot=plot, warnings=warn)
        reasons.append("Fisher's exact test is used (an expected count is below 5, or you chose it).")
        _, pf = stats.fisher_exact(tab.values)
        return _out("Fisher's exact test", reasons, "odds ratio", orr, pf, alpha, None, orr, "Odds ratio", ci,
                    "Chi square test if all expected counts were 5 or more.", interp, plot=plot, warnings=warn)
    w = warn + (["Many cells have small expected counts. Combine sparse categories or use an exact test in R or SPSS."] if low > 0.2 else [])
    return _out("Chi square test of independence", reasons, "chi square", chi, p, alpha, dof, v, "Cramer's V", None,
                "Fisher Freeman Halton exact test if expected counts are small.",
                f"This tests whether the distribution of '{b}' differs across '{a}' groups. Cramer's V = {v:.2f}. See the percentages in the chart.",
                {"name": f"Critical chi square (df = {dof})", "crit": stats.chi2.ppf(1 - alpha, dof), "hit": chi > stats.chi2.ppf(1 - alpha, dof)},
                plot=plot, warnings=w)


# ---------- Table 1 ----------
def _levels(s):
    u = set(s.dropna().unique())
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if c in u]
    return sorted(u, key=str)


def table_one(df, group=None, alpha=0.05):
    levels = _levels(df[group]) if group else []
    rows = []
    for c in df.columns:
        k = classify(df[c], c)
        if c == group or k in ("id", "text", "empty"):
            continue
        pv, tn = "", ""
        if group:
            try:
                r = analyze(df, c, group, False, alpha)
                pv, tn = fmt_p(r["p"]), r["test"] + f" ({r['stat_label']} = {r['stat']:.2f}" + (f", df = {r['df']:g}" if isinstance(r["df"], (int, float, np.floating)) and r["df"] is not None else "") + ")"
            except Exception:
                pass
        subsets = {"Overall": df} | {str(lv): df[df[group] == lv] for lv in levels}
        if k == "numeric":
            ok = check_normal(df[c])["ok"]
            f = (lambda s: f"{s.mean():.1f} ({s.std():.1f})") if ok else (lambda s: f"{s.median():.1f} ({s.quantile(.25):.1f} to {s.quantile(.75):.1f})")
            rows.append({"Variable": f"{c}, {'mean (SD)' if ok else 'median (IQR)'}", **{n: f(s[c].dropna()) for n, s in subsets.items()}, "p value": pv, "Test": tn})
        else:
            rows.append({"Variable": f"{c}, n (%)", **{n: "" for n in subsets}, "p value": pv, "Test": tn})
            for lv in _levels(df[c]):
                rows.append({"Variable": f"   {lv}", **{n: f"{(s[c] == lv).sum()} ({(s[c] == lv).sum() / max(s[c].notna().sum(), 1) * 100:.1f}%)" for n, s in subsets.items()}, "p value": "", "Test": ""})
    return pd.DataFrame(rows)


# ---------- demo data ----------
def make_demo(seed=7, n=320):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(52, 15, n), 18, 90).round()
    bmi = np.clip(rng.normal(27.5, 5, n), 16, 50).round(1)
    tx = rng.choice(["Drug A", "Drug B", "Placebo"], n)
    eff = pd.Series(tx).map({"Drug A": -14, "Drug B": -8, "Placebo": -2}).values
    base = np.clip(rng.normal(150, 15, n) + 0.2 * (age - 52), 100, 220).round()
    smoker = rng.choice(["Yes", "No"], n, p=[0.22, 0.78])
    df = pd.DataFrame({
        "patient_id": [f"P{1000 + i}" for i in range(n)], "age": age, "sex": rng.choice(["Male", "Female"], n),
        "bmi": bmi, "smoker": smoker, "treatment_group": tx, "sbp_baseline": base,
        "sbp_week12": (base + eff + rng.normal(0, 9, n)).round(),
        "hba1c": np.clip(rng.lognormal(np.log(6.4), 0.22, n), 4, 15).round(1),
        "diabetes": np.where(rng.random(n) < 1 / (1 + np.exp(-(bmi - 30) / 3)), "Yes", "No"),
        "length_of_stay_days": np.clip(rng.gamma(2, 2.2, n) + (smoker == "Yes") * 1.5, 1, 60).round(),
        "improved": np.where(rng.random(n) < pd.Series(tx).map({"Drug A": .72, "Drug B": .6, "Placebo": .4}).values, "Yes", "No")})
    df.loc[[5, 40], "bmi"] = [68.0, 71.5]
    for col, fr in [("age", .04), ("bmi", .05), ("hba1c", .06), ("smoker", .03), ("sbp_week12", .04)]:
        df.loc[rng.choice(n, int(fr * n), replace=False), col] = np.nan
    return pd.concat([df, df.sample(6, random_state=1)]).reset_index(drop=True)


# ---------- extra tests: McNemar, one sample, regression ----------
def _mcnemar(df, a, b, alpha):
    d = df[[a, b]].dropna()
    tab = pd.crosstab(d[a], d[b])
    if tab.shape != (2, 2):
        raise ValueError("Both measurements need the same two levels (for example Yes and No).")
    x, y = int(tab.iloc[0, 1]), int(tab.iloc[1, 0])
    if x + y == 0:
        raise ValueError("No patients changed between the two measurements.")
    p = stats.binomtest(x, x + y, 0.5).pvalue
    reasons = [f"'{a}' and '{b}' are two yes/no measurements on the same patients (paired).",
               f"Only the {x + y} patients whose answer changed carry information: {x} changed one way, {y} the other way."]
    return _out("McNemar's exact test", reasons, "changed one way", x, p, alpha, None, (x / y if y else np.inf), "Odds ratio of discordant pairs", None,
                "Chi square test would be wrong here because the groups are not independent.",
                f"{x} patients changed from '{tab.index[0]}' to '{tab.columns[1]}' and {y} changed the opposite way.", plot={"type": "stack", "pct": tab.div(tab.sum(1), axis=0) * 100, "ylabel": f"% within {a}"})


def one_sample(df, v, mu, alpha=0.05):
    x = df[v].dropna().astype(float)
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 values.")
    nm = check_normal(x)
    reasons = [f"One numeric variable '{v}' is compared with a known value ({mu}), for example a population mean.",
               f"n = {n}; population standard deviation unknown, so the t statistic is used for small samples (n of 30 or less).",
               f"Shapiro Wilk {fmt_p(nm['p']) if nm['p'] is not None else 'not available'}: {'looks normal' if nm['ok'] else 'not normal'}."]
    plot = {"type": "hist", "x": x.values, "mu": mu, "xlabel": v}
    if nm["ok"]:
        res = stats.ttest_1samp(x, mu)
        ci = res.confidence_interval(1 - alpha)
        crit = stats.t.ppf(1 - alpha / 2, res.df)
        return _out("One sample t test", reasons, "t", res.statistic, res.pvalue, alpha, res.df, (x.mean() - mu) / x.std(ddof=1), "Cohen's d",
                    (f"{int((1 - alpha) * 100)}% CI for the mean", (ci.low, ci.high)), "Wilcoxon signed rank test if the data are skewed.",
                    f"Sample mean {x.mean():.2f} versus {mu}.", {"name": f"Critical t (two tailed, df = {res.df})", "crit": crit, "hit": abs(res.statistic) > crit}, plot=plot)
    res = stats.wilcoxon(x - mu)
    return _out("Wilcoxon signed rank test (one sample)", reasons, "W", res.statistic, res.pvalue, alpha, None, 0.0, "Effect size not computed", None,
                "One sample t test if the data were normal.", f"Sample median {x.median():.2f} versus {mu}.", plot=plot)


def _design(df, y, xs):
    d = df[[y] + xs].dropna()
    cols, names = [np.ones(len(d))], ["Intercept"]
    for x in xs:
        if classify(d[x], x) == "numeric":
            cols.append(d[x].astype(float).values)
            names.append(f"{x} (per unit)")
        else:
            lv = sorted(d[x].unique(), key=str)
            for l in lv[1:]:
                cols.append((d[x] == l).astype(float).values)
                names.append(f"{x}: {l} vs {lv[0]}")
    return d, np.column_stack(cols), names


def regress(df, y, xs, alpha=0.05):
    """Binary logistic regression for a yes/no outcome, multiple linear regression for a numeric outcome."""
    if not xs or y in xs:
        raise ValueError("Choose an outcome and at least one different predictor.")
    ky = classify(df[y], y)
    d, A, names = _design(df, y, xs)
    n, k = A.shape
    if n <= k + 5:
        raise ValueError(f"Too few complete rows ({n}) for {k - 1} predictor terms.")
    warn, zc = [], stats.norm.ppf(1 - alpha / 2)
    if ky == "binary":
        lv = sorted(d[y].unique(), key=str)
        yy = (d[y] == lv[1]).astype(float).values
        beta = np.zeros(k)
        for _ in range(100):
            p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
            H = A.T @ (A * (p * (1 - p))[:, None])
            try:
                step = np.linalg.solve(H, A.T @ (yy - p))
            except np.linalg.LinAlgError:
                raise ValueError("Predictors are perfectly related to each other. Remove one.")
            beta += step
            if np.abs(step).max() < 1e-8:
                break
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        z = beta / se
        pv = 2 * stats.norm.sf(abs(z))
        ev = int(min(yy.sum(), n - yy.sum()))
        ll = float((yy * np.log(p + 1e-12) + (1 - yy) * np.log(1 - p + 1e-12)).sum())
        p0 = yy.mean()
        ll0 = float((yy * np.log(p0) + (1 - yy) * np.log(1 - p0)).sum())
        lr = 2 * (ll - ll0)
        lrp = float(stats.chi2.sf(lr, k - 1))
        if ev / (k - 1) < 10:
            warn.append(f"Only {ev} events for {k - 1} predictor terms ({ev / (k - 1):.1f} per term). About 10 or more per term is advised, otherwise estimates are unstable.")
        if np.abs(beta).max() > 15:
            warn.append("Some coefficients are extreme, which suggests complete separation. Combine categories or remove a predictor.")
        tab = pd.DataFrame({"Predictor": names, "Odds ratio": np.exp(beta), "CI low": np.exp(beta - zc * se), "CI high": np.exp(beta + zc * se),
                            "Coefficient": beta, "SE": se, "z": z, "p value": pv}).iloc[1:]
        return dict(test="Binary logistic regression", table=tab, warnings=warn, n=n, p=lrp,
                    reasons=[f"Outcome '{y}' is binary, so a logistic model estimates the odds of '{lv[1]}' versus '{lv[0]}' for each predictor, adjusted for the others.",
                             f"{n} complete rows; {int(yy.sum())} with '{lv[1]}'."],
                    fit=f"Model chi square = {lr:.2f}, df = {k - 1}, {fmt_p(lrp)}; McFadden R squared = {1 - ll / ll0:.3f}.",
                    interpretation="An odds ratio above 1 means higher odds of the outcome; below 1 means lower odds. If the confidence interval includes 1, the predictor is not statistically significant after adjustment.")
    if ky != "numeric":
        raise ValueError("The outcome must be numeric (linear regression) or yes/no (logistic regression).")
    yy = d[y].astype(float).values
    beta = np.linalg.lstsq(A, yy, rcond=None)[0]
    res = yy - A @ beta
    dfr = n - k
    s2 = (res ** 2).sum() / dfr
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    t = beta / se
    pv = 2 * stats.t.sf(abs(t), dfr)
    tc = stats.t.ppf(1 - alpha / 2, dfr)
    tss, rss = ((yy - yy.mean()) ** 2).sum(), (res ** 2).sum()
    F = ((tss - rss) / (k - 1)) / (rss / dfr)
    Fp = float(stats.f.sf(F, k - 1, dfr))
    r2 = 1 - rss / tss
    nm = check_normal(res)
    if not nm["ok"]:
        warn.append("Residuals are not normal (Shapiro Wilk " + fmt_p(nm["p"]) + "). Consider transforming the outcome, for example with a log.")
    tab = pd.DataFrame({"Predictor": names, "Coefficient": beta, "CI low": beta - tc * se, "CI high": beta + tc * se, "SE": se, "t": t, "p value": pv}).iloc[1:]
    return dict(test="Multiple linear regression", table=tab, warnings=warn, n=n, p=Fp,
                reasons=[f"Outcome '{y}' is numeric, so a linear model estimates how much it changes per unit of each predictor, adjusted for the others.", f"{n} complete rows."],
                fit=f"F({k - 1}, {dfr}) = {F:.2f}, {fmt_p(Fp)}; R squared = {r2:.3f}; adjusted R squared = {1 - (1 - r2) * (n - 1) / dfr:.3f}.",
                interpretation="Each coefficient is the average change in the outcome for a one unit increase in that predictor (or versus the reference group), holding the other predictors constant.")


# ---------- objectives and analysis plan checker ----------
OBJ_RULES = [
    ("predict", ["predictor", "determinant", "factors associated", "risk factor", "factors influencing", "independent", "adjust", "predict"],
     "Binary logistic regression (yes/no outcome) or multiple linear regression (numeric outcome)", "Several factors act together, and one test at a time cannot separate their effects."),
    ("compare", ["compare", "comparison", "difference", "differ", "between groups"],
     "t test (2 groups), ANOVA (3 or more groups), chi square or Fisher's exact (proportions)", "Group comparisons depend on the outcome type and the number of groups."),
    ("change", ["before and after", "pre and post", "effectiveness", "effect of", "impact of", "improvement"],
     "Paired t test or Wilcoxon signed rank (same patients), or t test, ANOVA or chi square between intervention and control groups", "The question is about change, so compare the same patients over time or compare groups."),
    ("associate", ["association", "associated", "relationship", "correlat", "related to"],
     "Chi square or Fisher's exact for categories; Pearson or Spearman correlation for numeric variables", "Association tests depend on whether each variable is numeric or categorical."),
    ("survival", ["survival", "time to", "hazard"],
     "Kaplan Meier curves, log rank test and Cox regression (not yet built into this lab)", "Time to event data need methods that handle patients who have not yet had the event."),
    ("describe", ["prevalence", "proportion", "describe", "determine the level", "pattern", "knowledge", "awareness", "distribution", "incidence", "frequency", "assess"],
     "Descriptive statistics: n (%) with 95% confidence intervals, mean (SD) or median (IQR)", "Descriptive objectives need summaries, not significance tests."),
]
COVER = {"predict": {"logistic regression", "linear regression"}, "change": {"paired t test", "wilcoxon", "t test", "anova", "mcnemar", "chi square"},
         "compare": {"t test", "anova", "chi square", "fisher", "mann whitney", "kruskal"}, "associate": {"chi square", "fisher", "pearson", "spearman", "correlation", "logistic regression", "linear regression"},
         "describe": {"descriptive"}}
TESTS = {"paired t test": ["paired t"], "t test": ["t test", "student", "ttest"], "anova": ["anova", "analysis of variance", "f test"], "chi square": ["chi square", "chi2", "x2"],
         "fisher": ["fisher"], "mann whitney": ["mann whitney", "rank sum"], "wilcoxon": ["wilcoxon"], "kruskal": ["kruskal"], "pearson": ["pearson"], "spearman": ["spearman"],
         "correlation": ["correlation"], "linear regression": ["linear regression"], "logistic regression": ["logistic regression"], "mcnemar": ["mcnemar"],
         "descriptive": ["descriptive", "frequenc", "percentage", "mean", "median", "standard deviation"]}
KEY = {"t test": ["t test"], "paired t test": ["paired t"], "anova": ["anova"], "chi square": ["chi square"], "fisher": ["fisher"], "mann whitney": ["mann whitney"], "wilcoxon": ["wilcoxon"],
       "kruskal": ["kruskal"], "pearson": ["pearson"], "spearman": ["spearman"], "correlation": ["pearson", "spearman"], "mcnemar": ["mcnemar"]}
GENERIC = {
    "t test": "Needs a numeric outcome, exactly 2 independent groups and roughly normal data (check with Shapiro Wilk). For 3 or more groups use ANOVA, because repeated t tests raise the chance of a false positive. For skewed data use Mann Whitney U.",
    "paired t test": "Needs the same patients measured twice and normally distributed differences. Otherwise use Wilcoxon signed rank.",
    "anova": "Needs a numeric outcome, 3 or more independent groups, normal data and equal variances, followed by a post hoc test (Tukey). Otherwise use Kruskal Wallis.",
    "chi square": "Needs two categorical variables and expected counts of 5 or more in the cells. Otherwise use Fisher's exact test.",
    "fisher": "An exact test for categorical variables with small expected counts. It is valid for any sample size, so it is a safe choice.",
    "mann whitney": "A nonparametric alternative to the t test for skewed or ordinal outcomes with 2 independent groups. Slightly less powerful if data are truly normal.",
    "wilcoxon": "Nonparametric test for paired data (or one sample against a value) when differences are not normal.",
    "kruskal": "Nonparametric alternative to one way ANOVA for 3 or more independent groups with skewed data.",
    "pearson": "Needs two numeric variables that are normal with a straight line relationship. Otherwise use Spearman.",
    "spearman": "Rank based correlation for skewed, ordinal or non linear (but steadily rising or falling) relationships.",
    "correlation": "Choose Pearson for normal, linear data and Spearman otherwise. Correlation does not show causation.",
    "linear regression": "Needs a numeric outcome. Check that residuals are roughly normal and the relationship is linear. Use it when you want adjusted effects of several predictors.",
    "logistic regression": "Needs a yes/no outcome and about 10 events per predictor. Report odds ratios with 95% confidence intervals.",
    "mcnemar": "For paired yes/no data on the same patients, for example before and after.",
    "descriptive": "Good for describing the sample. Report mean (SD) for normal data, median (IQR) for skewed data, and n (%) for categories.",
}


SPLIT = re.compile(r"(?<!\d)\.|\.(?!\d)|[\n;]+")


def classify_objectives(lines):
    out = []
    for o in [l.strip() for l in lines if l.strip()]:
        low = o.lower().replace("-", " ")
        hit = [r for r in OBJ_RULES if any(k in low for k in r[1])]
        if not hit:
            out.append({"Objective": o, "Type": "unclear", "Suggested analysis": "Rewrite the objective with a verb such as describe, compare, determine association, or identify predictors.", "Why": "The analysis follows the objective."})
        else:
            out.append({"Objective": o, "Type": " + ".join(h[0] for h in hit), "Suggested analysis": "; ".join(h[2] for h in hit), "Why": hit[0][3]})
    return out


FRIENDLY = {"predict": "Finding factors that predict an outcome", "compare": "Comparing groups", "change": "Measuring change over time",
            "associate": "Finding a relationship between variables", "survival": "Time to an event", "describe": "Describing the participants", "unclear": "Not clear yet"}


def _why(planned, rec):
    rt, r0 = rec["test"].lower(), rec["reasons"][0]
    m = re.search(r"into (\d+) independent groups", r0)
    if m and int(m.group(1)) >= 3 and "t test" in planned:
        return f"A t test compares only two groups, but this variable has {m.group(1)} groups."
    if "fisher" in rt:
        return "Some groups are small (fewer than 5 people expected in a cell), so a chi square test would be unreliable."
    if any(k in rt for k in ("mann whitney", "kruskal", "wilcoxon", "spearman")):
        return "The values are not bell shaped (Shapiro Wilk test), so a test that does not assume this is safer."
    if any(k in rt for k in ("t test", "anova", "pearson")):
        return "The values are roughly bell shaped, so this test is suitable and has more power."
    return " ".join(rec["reasons"][:2])


def check_plan(objectives, plan, df=None, alpha=0.05):
    rows, used = [], set()

    def add(status, head, why, do, item=None):
        rows.append({"Item": item or head, "Status": status, "Headline": head, "Why": why, "Do": do, "Advice": f"{why} {do}"})

    for sent in [x.strip() for x in SPLIT.split(plan) if x.strip()]:
        low = sent.lower().replace("-", " ")
        found = [t for t, kws in TESTS.items() if any(k in low for k in kws)]
        if "paired t test" in found and "t test" in found:
            found.remove("t test")
        if "correlation" in found and ({"pearson", "spearman"} & set(found)):
            found.remove("correlation")
        if len(found) > 1 and "descriptive" in found:
            found.remove("descriptive")
        cols = [c for c in df.columns if re.search(r"\b" + re.escape(c.lower()) + r"\b", low) or re.search(r"\b" + re.escape(c.lower().replace("_", " ")) + r"\b", low)] if df is not None else []
        for t in found:
            used.add(t)
            first = GENERIC[t].split(". ")[0].rstrip(".") + "."
            if t == "descriptive":
                add("Appropriate", "Descriptive statistics", first, "Keep it. Describe every variable before you test anything.")
            elif t in ("logistic regression", "linear regression") and cols:
                need, label = ("binary", "yes/no") if t == "logistic regression" else ("numeric", "numeric")
                ok = [c for c in cols if classify(df[c], c) == need]
                if ok:
                    add("Appropriate", f"{t.title()} with outcome {ok[0]}", f"'{ok[0]}' is a {label} variable, which is what a {t} needs for its outcome.", "Keep it. Remember about 10 events per predictor.")
                else:
                    add("Better option", f"Planned: {t}. The outcome type does not match", f"A {t} needs a {label} outcome, but none of the variables named ({', '.join(cols)}) is {label}.",
                        "Use logistic regression for a yes/no outcome and linear regression for a measurement.")
            elif t in KEY and len(cols) >= 2:
                try:
                    r = analyze(df, cols[0], cols[1], paired=t in ("paired t test", "wilcoxon", "mcnemar"), alpha=alpha)
                    if any(k in r["test"].lower() for k in KEY[t]):
                        add("Appropriate", f"{t.title()} for {cols[0]} and {cols[1]}", "Your data fit this test. " + first, "Keep it in your plan.")
                    else:
                        add("Better option", f"Planned: {t}. Suggested: {r['test']}", _why(t, r),
                            f"Use the {r['test']} instead and give the reason in your plan. You can still run your original test in Step 7.")
                except Exception:
                    add("Check", f"{t.title()} for {cols[0]} and {cols[1]}", "The lab could not run this pair automatically.", first)
            else:
                add("Check", f"{t.title()}: {sent[:70]}", first, "Load your data and name the variables in this sentence so the lab can check it against the real data." if df is None
                    else "Name two variables from your dataset in this sentence so the lab can check it.")
    for o in classify_objectives(objectives):
        for ty in o["Type"].split(" + "):
            if ty in COVER and not (COVER[ty] & used):
                add("Gap", f"No test in your plan answers: {o['Objective'][:80]}", f"This is a '{FRIENDLY[ty].lower()}' question, and the plan names no test for it.",
                    f"Add: {o['Suggested analysis']}.", item=f"Objective: {o['Objective'][:80]}")
    low = plan.lower().replace("-", " ")
    if used & {"t test", "paired t test", "anova", "pearson", "linear regression"} and "shapiro" not in low and "normal" not in low:
        add("Gap", "Say how you will check normality", "Tests such as the t test, ANOVA and Pearson correlation assume the values are roughly bell shaped. Examiners expect you to say how you checked.",
            "Add a sentence such as: Normality will be checked with the Shapiro Wilk test and histograms.", item="Normality")
    if "confidence" not in low and "effect size" not in low and "odds ratio" not in low:
        add("Gap", "Report confidence intervals and effect sizes", "A p value only says whether a result is likely due to chance. A confidence interval and an effect size show how big the difference is.",
            "Add a sentence such as: 95% confidence intervals and effect sizes (for example odds ratios) will be reported.", item="Reporting")
    if len(used - {"descriptive"}) >= 5:
        add("Check", "Many tests will be run", "Each test has about a 5% chance of a false alarm, so running many tests makes at least one false alarm likely.",
            "State how you will handle this, for example a stricter p value (Bonferroni) or naming your main outcomes.", item="Many tests")
    if not rows:
        add("Check", "No statistical tests were recognised", "The lab looks for test names such as chi square, t test or logistic regression.", "Name each test, for example: chi square test for the association between sex and outcome.")
    return pd.DataFrame(rows)


# ---------- guided suggestions (objective to variables and planned tests) ----------
STOP = {"the", "and", "between", "among", "with", "after", "before", "determine", "identify", "factors", "level", "levels", "mean", "change", "association", "associated",
        "independently", "describe", "compare", "participants", "profile", "characteristics", "sociodemographic", "for", "from", "that", "this", "among", "who", "are", "was", "were"}
SYN = {"sbp": ["systolic", "blood", "pressure"], "dbp": ["diastolic", "blood", "pressure"], "bp": ["blood", "pressure"], "bmi": ["body", "mass", "index", "weight"],
       "hba1c": ["glycated", "haemoglobin", "glucose"], "fbs": ["fasting", "glucose"], "los": ["length", "stay"], "hb": ["haemoglobin", "anaemia"], "htn": ["hypertension"], "dm": ["diabetes"]}
PRE = {"pre", "preop", "before", "baseline", "initial"}
POST = {"post", "after", "followup", "follow", "final", "endline"}
PLAN_PREF = {"t test": "parametric", "paired t test": "parametric", "anova": "parametric", "pearson": "parametric", "mann whitney": "nonparametric",
             "wilcoxon": "nonparametric", "kruskal": "nonparametric", "spearman": "nonparametric", "fisher": "exact", "chi square": "chisquare"}


def _toks(name):
    return [x for x in re.split(r"[^a-z0-9]+", name.lower()) if x]


def _stems(words):
    ex = set(words)
    for w in words:
        ex |= set(SYN.get(w, []))
    return {w[:5] for w in ex if len(w) >= 3}


def related(a, b):
    """True if two column names describe the same thing (for example sbp and bp_controlled)."""
    return bool(_stems(_toks(a)) & _stems(_toks(b)))


def suggest(objective, df):
    low = objective.lower().replace("-", " ")
    otoks = {w[:5] for w in set(_toks(low)) - STOP if len(w) >= 3}
    post = any(w in low for w in ("after", "follow", "week", "post"))
    pre = any(w in low for w in ("before", "baseline"))
    grp = "group" in low or " arm" in low
    score = {}
    for c in df.columns:
        k = classify(df[c], c)
        if k in ("id", "text", "empty"):
            continue
        ct = _toks(c)
        sc = len(_stems(ct) & otoks)
        if grp and k in ("binary", "categorical"):
            for lv in df[c].dropna().unique():
                v = str(lv).lower()
                if len(v) >= 3 and re.search(r"\b" + re.escape(v) + r"\b", low):
                    sc += 2
        if sc:
            sc += (post and bool(set(ct) & POST)) + (pre and bool(set(ct) & PRE))
            score[c] = sc
    nums = [c for c in df.columns if classify(df[c], c) == "numeric"]
    def is_post(ts):
        return any(x in POST or re.fullmatch(r"\d+(m|w|d)(po)?|postop", x) for x in ts)

    def core(ts):
        return " ".join(x for x in ts if x not in PRE and x not in POST and not re.fullmatch(r"\d+(m|w|d)(po)?|postop", x))

    pairs = [(a, b) for a in nums for b in nums if set(_toks(a)) & PRE and is_post(_toks(b))
             and difflib.SequenceMatcher(None, core(_toks(a)), core(_toks(b))).ratio() >= 0.75]
    pairs.sort(key=lambda p: -(score.get(p[0], 0) + score.get(p[1], 0)))
    return {"matched": sorted(score, key=lambda c: -score[c]), "pairs": pairs}


def planned_tests(plan, cols=(), ty=None):
    """Tests named in the proposal's analysis plan for sentences that mention the chosen variables."""
    out = []
    for sent in SPLIT.split(plan):
        low = sent.lower().replace("-", " ")
        found = [t for t, kws in TESTS.items() if any(k in low for k in kws)]
        if "paired t test" in found and "t test" in found:
            found.remove("t test")
        if len(found) > 1 and "descriptive" in found:
            found.remove("descriptive")
        if found and cols and all(re.search(r"\b" + re.escape(c.lower()) + r"\b", low) for c in cols):
            out += [t for t in found if t not in out]
    return out


# ---------- real world data preparation ----------
def skip_parent(df, col):
    """Detect a skip question: blanks in `col` sit almost entirely under one answer of another question."""
    miss = df[col].isna()
    if miss.sum() < 3:
        return None
    best = None
    for p in df.columns:
        if p == col or classify(df[p], p) not in ("binary", "categorical"):
            continue
        ok = df[p].notna()
        for lv in df[p].dropna().unique():
            sel = df[p] == lv
            if sel.sum() < 3:
                continue
            share = (miss & sel).sum() / miss.sum()
            other = (miss & ~sel & ok).sum() / max((~sel & ok).sum(), 1)
            within = (miss & sel).sum() / sel.sum()
            if share >= 0.9 and other <= 0.05 and within >= 0.8 and (best is None or share > best[2]):
                best = (p, lv, float(share))
    return best


def categorise(s, cuts):
    cuts = sorted({float(c) for c in cuts})
    x = pd.to_numeric(s, errors="coerce")
    ints = bool(x.dropna().mod(1).eq(0).all()) and all(c == int(c) for c in cuts)
    f = (lambda v: str(int(v))) if ints else (lambda v: f"{v:g}")
    labels = [f"Under {f(cuts[0])}"] + [f"{f(a)} to {f(b - 1)}" if ints else f"{f(a)} to under {f(b)}" for a, b in zip(cuts, cuts[1:])] + [f"{f(cuts[-1])} and above"]
    return pd.cut(x, [-np.inf] + cuts + [np.inf], labels=labels, right=False, ordered=True)


def suggest_cuts(s, method, n=4):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if method == "quartiles":
        return [round(v) for v in x.quantile([.25, .5, .75])]
    return [round(v) for v in np.linspace(x.min(), x.max(), n + 1)[1:-1]]


def make_codes(cols, scheme, kap=None):
    used, out, cnt = set(), [], {"K": 0, "A": 0, "P": 0}
    for n, c in enumerate(cols, 1):
        if scheme == "short":
            w = [x for x in re.findall(r"[a-z0-9]+", str(c).lower()) if x not in STOP][:4]
            code = "_".join(w) or f"q{n}"
        elif scheme == "kap" and kap and kap.get(c) in ("Knowledge", "Attitude", "Practice"):
            l = kap[c][0]
            cnt[l] += 1
            code = f"{l}{cnt[l]}"
        else:
            code = f"Q{n}"
        base, k = code, 2
        while code in used:
            code, k = f"{base}_{k}", k + 1
        used.add(code)
        out.append(code)
    return out


KAP_WORDS = {
    "Knowledge": ["know", "aware", "heard", "cause", "symptom", "sign", "risk factor", "transmit", "spread", "prevent", "true or false", "which of", "defin", "complication", "cure"],
    "Attitude": ["agree", "disagree", "believe", "feel", "think", "opinion", "should", "willing", "comfortable", "important", "afraid", "fear", "likely", "support", "worried", "ashamed", "attitude"],
    "Practice": ["have you", "do you", "did you", "was the", "were you", "how often", "how many times", "practice", "usually", "regularly", "in the last", "ever been", "wash", "screened", "tested", "visited", "attend", "exercise", "checked"]}
LIKMAP = {"sd": 1, "d": 2, "n": 3, "a": 4, "sa": 5, "strongly disagree": 1, "disagree": 2, "neutral": 3, "undecided": 3, "not sure": 3, "agree": 4, "strongly agree": 5}


def detect_kap(df):
    rows = []
    for c in df.columns:
        if classify(df[c], c) not in ("binary", "categorical"):
            continue
        low = str(c).lower()
        vals = {str(v).strip().lower() for v in df[c].dropna().unique()}
        sc = {d: sum(w in low for w in ws) for d, ws in KAP_WORDS.items()}
        lik = len(vals & set(LIKMAP)) >= 3
        if lik:
            sc["Attitude"] += 3
        elif vals & {"i don't know", "i dont know", "don't know", "dont know", "not sure"}:
            sc["Knowledge"] += 3
        if "true or false" in low or vals <= {"true", "false"}:
            sc["Knowledge"] += 1
        best = max(sc, key=sc.get)
        demo = any(re.search(r"\b" + w + r"\b", low) for w in ("sex", "gender", "age", "education", "marital", "religion", "occupation", "income", "ethnic", "tribe"))
        rows.append({"Column": c, "Section": best if sc[best] > (1 if demo else 0) else "Other", "_v": frozenset(vals)})
    for _ in range(2):  # questionnaires come in blocks: a neighbour with the same answer set shares the section
        for n, r in enumerate(rows):
            if r["Section"] == "Other":
                for nb in (rows[n + 1] if n + 1 < len(rows) else None, rows[n - 1] if n else None):
                    if nb and nb["Section"] != "Other" and nb["_v"] == r["_v"] and len(r["_v"]) >= 2:
                        r["Section"] = nb["Section"]
                        break
    for n, r in enumerate(rows):  # fill gaps between matching neighbours
        if r["Section"] == "Other":
            prev = next((x["Section"] for x in reversed(rows[:n]) if x["Section"] != "Other"), None)
            nxt = next((x["Section"] for x in rows[n + 1:] if x["Section"] != "Other"), None)
            if prev and prev == nxt:
                r["Section"] = prev
    return pd.DataFrame(rows, columns=["Column", "Section"])


def kap_score(df, items, domain, cutoff="Median"):
    """items: {column: {"correct": answer, "reverse": bool}}. Returns percentage score, level, cutoff used."""
    pts, mx = [], []
    for c, o in items.items():
        s = df[c].astype(object).where(df[c].notna()).map(lambda v: str(v).strip().lower() if pd.notna(v) else np.nan)
        if domain == "Attitude" and s.notna().any() and set(s.dropna().unique()) <= set(LIKMAP):
            p = s.map(LIKMAP).astype(float)
            p = 6 - p if o.get("reverse") else p
            m = 5.0
        else:
            p = (s == str(o.get("correct", "")).strip().lower()).astype(float).where(s.notna())
            m = 1.0
        pts.append(p)
        mx.append(m)
    P = pd.concat(pts, axis=1)
    ans = P.notna()
    pct = P.fillna(0).sum(axis=1) / (ans * np.array(mx)).sum(axis=1).replace(0, np.nan) * 100
    pct[ans.mean(axis=1) < 0.8] = np.nan
    cut = {"Median": pct.median(), "Mean": pct.mean()}.get(cutoff, None)
    if cut is None:
        cut = float(cutoff.split("%")[0])
    hi, lo = ("Positive", "Negative") if domain == "Attitude" else ("Good", "Poor")
    level = pd.Series(np.where(pct >= cut, hi, lo), index=df.index).where(pct.notna())
    return pct.round(1), level, float(cut)


def review_rows(df, rows, alpha=0.05):
    """Compare a user's earlier tests with what the lab recommends."""
    out = []
    for r in rows:
        v1, v2, used, p_user, paired = r.get("v1"), r.get("v2"), r.get("test"), r.get("p"), bool(r.get("paired"))
        op = r.get("op", "=")
        if not v1 or not v2 or not used or v1 not in df.columns or v2 not in df.columns:
            continue
        item = {"Variables": f"{v1} and {v2}", "Test you used": used}
        try:
            rec = analyze(df, v1, v2, paired, alpha)
        except Exception as e:
            out.append({**item, "Verdict": "Could not check", "Lab recommends": "", "Advice": str(e)})
            continue
        match = any(k in rec["test"].lower() for k in KEY.get(used, [used]))
        item.update({"Lab recommends": rec["test"], "Lab p value (recommended)": round(float(rec["p"]), 4)})
        advice = []
        pref = PLAN_PREF.get(used)
        try:
            mine = analyze(df, v1, v2, paired, alpha, pref) if pref else rec
            if match or any(k in mine["test"].lower() for k in KEY.get(used, [used])):
                item["Lab p value (your test)"] = round(float(mine["p"]), 4)
                if p_user not in (None, "") and not pd.isna(p_user):
                    gap = abs(float(p_user) - float(mine["p"]))
                    same = (float(mine["p"]) < float(p_user)) if op == "<" else (gap <= max(0.005, 0.1 * float(mine["p"])))
                    advice.append("The reported p value matches the lab's." if same else
                                  f"The reported p value ({float(p_user):.3f}) differs from the lab's ({float(mine['p']):.3f}) for the same test. Ask which rows were included and how missing values were handled.")
            else:
                advice.append(f"A {used} cannot be run on these variables the way it was described.")
        except Exception:
            pass
        item["Verdict"] = "Appropriate" if match else "Better option"
        if not match:
            advice.insert(0, "Why the lab suggests a different test: " + " ".join(rec["reasons"][:3]))
        if rec["significant"] != (p_user not in (None, "") and not pd.isna(p_user) and float(p_user) < alpha) and p_user not in (None, "") and not pd.isna(p_user):
            advice.append("The conclusion (significant or not) differs between your result and the recommended test.")
        item["Advice"] = " ".join(advice)
        out.append(item)
    return pd.DataFrame(out)


# ---------- KAP scoring system (uploaded or edited in the lab) ----------
NEG_CUES = ["no need", "not necessary", "okay to", "ok to", "should be able to", "need to keep", "i should start", "i should take", "rather", "skipping",
            "safe drug", "go to another", "ashamed", "can continue", "does not matter"]


def kap_template(df, kap, attitude_mode="scale"):
    """One row per item and answer, with starting points the user must check. Grid items ([option] columns) share a Group."""
    rows, pre = [], {}
    for c in kap:
        if str(c).rstrip().endswith("]") and "[" in str(c):
            pre.setdefault(str(c)[:str(c).rindex("[")].strip(), []).append(c)
    gid = {c: k for k, v in pre.items() if len(v) >= 2 for c in v}
    for c, sec in kap.items():
        if sec not in ("Knowledge", "Attitude", "Practice") or c not in df.columns:
            continue
        levels = sorted(df[c].dropna().astype(str).str.strip().unique(), key=str)
        lik = len({l.lower() for l in levels} & set(LIKMAP)) >= 3
        neg = any(k in str(c).lower() for k in NEG_CUES)
        for l in levels:
            ll = l.lower()
            if sec == "Attitude" and lik:
                if attitude_mode == "dichotomised":
                    pts = int(ll in (("sd", "d", "strongly disagree", "disagree") if neg else ("sa", "a", "strongly agree", "agree")))
                else:
                    pts = LIKMAP.get(ll, 0)
            else:
                pts = 1 if ll in ("yes", "true", "correct") else 0
            rows.append({"Item": c, "Domain": sec, "Answer": l, "Points": pts, "Group": gid.get(c, "")})
    return pd.DataFrame(rows, columns=["Item", "Domain", "Answer", "Points", "Group"])


def match_item(item, cols, codebook=None):
    norm = lambda x: re.sub(r"\s+", " ", str(x).strip().lower())
    cmap = {norm(c): c for c in cols}
    for code, q in (codebook or {}).items():
        if code in cols:
            cmap.setdefault(norm(q), code)
    k = norm(item)
    if k in cmap:
        return cmap[k]
    m = difflib.get_close_matches(k, list(cmap), n=1, cutoff=0.85)
    return cmap[m[0]] if m else None


def score_system(df, table, codebook=None, missing_as_zero=True):
    """Sum the points of every item in each domain. Items sharing a Group (a tick all that apply grid) earn the group's points only if every item is answered correctly."""
    t = table.copy()
    t["col"] = t["Item"].map(lambda i: match_item(i, list(df.columns), codebook))
    unmatched = sorted(t.loc[t["col"].isna(), "Item"].astype(str).unique())
    t = t.dropna(subset=["col"])
    t["Points"] = pd.to_numeric(t["Points"], errors="coerce").fillna(0)
    if "Domain" not in t:
        t["Domain"] = "Score"
    if "Group" not in t:
        t["Group"] = ""
    t["Domain"], t["Group"] = t["Domain"].fillna("Score").astype(str), t["Group"].fillna("").astype(str)
    out = {}
    for dom, g in t.groupby("Domain"):
        pts, mx = [], 0.0
        def answered(c, gi):
            mp = {str(a).strip().lower(): float(p) for a, p in zip(gi["Answer"], gi["Points"])}
            s_ = df[c].map(lambda v: str(v).strip().lower() if pd.notna(v) else np.nan)
            return s_, mp
        for c, gi in g[g["Group"] == ""].groupby("col"):
            s_, mp = answered(c, gi)
            p = s_.map(mp)
            pts.append(p.where(~(s_.notna() & p.isna()), 0.0))
            mx += max(mp.values())
        for gname, gg in g[g["Group"] != ""].groupby("Group"):
            oks = []
            for c, gi in gg.groupby("col"):
                s_, mp = answered(c, gi)
                p = s_.map(mp).where(s_.notna()).fillna(-1) if False else s_.map(mp)
                oks.append((p.where(~(s_.notna() & p.isna()), 0.0) > 0).astype(float).where(s_.notna()))
            top = gg["Points"].max()
            pts.append(pd.concat(oks, axis=1).min(axis=1, skipna=False) * top)
            mx += top
        P = pd.concat(pts, axis=1)
        raw = P.fillna(0).sum(axis=1) if missing_as_zero else P.sum(axis=1).where(P.notna().all(axis=1))
        raw = raw.where(P.notna().any(axis=1))
        out[dom] = (raw, raw / mx * 100 if mx else raw * np.nan, mx, P.shape[1])
    return out, unmatched


def assign_levels(score, cats):
    """cats: DataFrame with Category and Minimum. A person gets the highest category whose minimum they reach."""
    c = cats.dropna().sort_values("Minimum")
    mins, labs = c["Minimum"].astype(float).values, c["Category"].astype(str).values
    idx = np.searchsorted(mins, score.values.astype(float), side="right") - 1
    vals = [labs[i] if (i >= 0 and pd.notna(v)) else None for i, v in zip(idx, score.values)]
    return pd.Categorical(vals, categories=list(labs), ordered=True)


PVAL = re.compile(r"\bp\s*(?:value)?\s*(<|=|>|\u2264|\u2265)?\s*(0?\.\d+|\d(?:\.\d+)?(?:e-\d+)?)", re.I)


def review_text(df, text, alpha=0.05, codebook=None):
    """Read an analyst's methods and results text, find each test, the variables and the p value, and check them."""
    rows, loose, notes = [], [], []
    names = {c: [c.lower(), c.lower().replace("_", " ")] + ([str(codebook[c]).lower()] if codebook and c in codebook else []) for c in df.columns}
    for sent in [x.strip() for x in SPLIT.split(text) if x.strip()]:
        low = sent.lower().replace("-", " ")
        found = [t for t, kws in TESTS.items() if t != "descriptive" and any(k in low for k in kws)]
        if "paired t test" in found and "t test" in found:
            found.remove("t test")
        if "correlation" in found and ({"pearson", "spearman"} & set(found)):
            found.remove("correlation")
        if not found:
            continue
        cols = [c for c, ns in names.items() if any(re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", low) for n in ns if n)]
        m = PVAL.search(sent)
        p, op = (float(m.group(2)), "<" if m.group(1) == "<" else "=") if m else (None, "=")
        for t in found:
            if t in ("logistic regression", "linear regression"):
                kinds = [classify(df[c], c) for c in cols]
                need, label = ("binary", "yes/no") if t == "logistic regression" else ("numeric", "numeric")
                ok = need in kinds
                rows.append({"Variables": ", ".join(cols) or "not found", "Test you used": t, "Verdict": ("Appropriate" if ok else "Better option") if cols else "Could not check",
                             "Lab recommends": "", "Advice": (f"The model needs a {label} outcome and one of the named variables is {label}." if ok else
                              (f"A {t} needs a {label} outcome, but none of the variables named is {label}. Check which variable is the outcome." if cols else "No variable from your data was found in this sentence.")) + " " + GENERIC[t]})
            elif len(cols) >= 2 and t in KEY:
                rows.append({"v1": cols[0], "v2": cols[1], "test": t, "p": p, "op": op, "paired": t in ("paired t test", "wilcoxon", "mcnemar") or "before and after" in low})
            else:
                loose.append(sent[:140])
    chk = review_rows(df, [r for r in rows if "v1" in r], alpha)
    extra = pd.DataFrame([r for r in rows if "v1" not in r], columns=["Variables", "Test you used", "Verdict", "Lab recommends", "Advice"])
    out = pd.concat([chk, extra], ignore_index=True) if len(extra) else chk
    g = check_plan([], text, None, alpha)
    notes = [r["Headline"] + ": " + r["Why"] for _, r in g.iterrows() if r["Item"] in ("Normality", "Reporting", "Many tests")]
    return out, loose, notes


# ---------- repeated measurements, within group analysis, numbers stored as text ----------
TIME_RE = re.compile(r"^\s*(preop|preoperative|baseline|pre|\d+\s*(?:m|w|d)(?:po)?|\d+\s*(?:month|week|day)s?)\b[\s_:.\-]*", re.I)


def time_families(df):
    """Find measurements repeated over time from column prefixes such as PREOP, 3MPO, 6MPO. Returns {parameter: {time label: column}}."""
    parts = {}
    for c in df.columns:
        m = TIME_RE.match(str(c))
        if m and classify(df[c], c) == "numeric":
            parts.setdefault(m.group(1).lower().replace(" ", ""), {})[c] = re.sub(r"[^a-z0-9]+", " ", str(c)[m.end():].lower()).strip()
    if len(parts) < 2:
        return {}
    base_names = ("preop", "preoperative", "baseline", "pre")
    keys = sorted(parts, key=lambda k: (0 if k in base_names else 1, int(re.match(r"\d+", k).group()) if re.match(r"\d+", k) else 0))
    fams = {}
    for bc, bname in parts[keys[0]].items():
        fam = {keys[0]: bc}
        for k in keys[1:]:
            best = max(parts[k].items(), key=lambda kv: difflib.SequenceMatcher(None, bname, kv[1]).ratio(), default=None)
            if best and difflib.SequenceMatcher(None, bname, best[1]).ratio() >= 0.6:
                fam[k] = best[0]
        if len(fam) >= 2:
            fams[bname] = fam
    return fams


def _order(s):
    return _levels(s) if isinstance(s.dtype, pd.CategoricalDtype) else list(pd.unique(s.dropna()))


def _spread(x, center):
    x = pd.Series(x).dropna()
    if not len(x):
        return ""
    return f"{x.median():.2f} ({x.quantile(.25):.2f} to {x.quantile(.75):.2f})" if center == "median" else f"{x.mean():.2f} ({x.std():.2f})"


def repeated_table(df, group, fams, alpha=0.05, prefer=None, center="median"):
    """Dissertation style table: each parameter by group at every time point, within group p (versus baseline) and between group p at each time point."""
    levels = _order(df[group])
    rows = []
    for fam, cols in fams.items():
        times = list(cols)
        for gi, lv in enumerate(levels):
            sub = df[df[group] == lv]
            row = {"Parameter": fam if gi == 0 else "", "Group": str(lv)}
            for k in times:
                row[f"{k} {center}"] = _spread(sub[cols[k]], center)
            for k in times[1:]:
                try:
                    row[f"p {times[0]} vs {k}"] = round(float(analyze(sub, cols[times[0]], cols[k], True, alpha, prefer)["p"]), 4)
                except Exception:
                    row[f"p {times[0]} vs {k}"] = None
            for k in times:
                try:
                    row[f"p between groups at {k}"] = round(float(analyze(df, cols[k], group, False, alpha, prefer)["p"]), 4) if gi == 0 else None
                except Exception:
                    row[f"p between groups at {k}"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def by_group(df, a, b, paired, by, alpha=0.05, prefer=None):
    """Run the same test separately within each level of another variable."""
    rows = []
    for lv in _order(df[by]):
        sub = df[df[by] == lv]
        try:
            r = analyze(sub, a, b, paired, alpha, prefer)
            rows.append({by: str(lv), "n": len(sub), "Test": r["test"], "Statistic": f"{r['stat_label']} = {r['stat']:.2f}", "p value": round(float(r["p"]), 4), "Significant": "Yes" if r["significant"] else "No"})
        except Exception as e:
            rows.append({by: str(lv), "n": len(sub), "Test": "could not run", "Statistic": "", "p value": None, "Significant": str(e)[:60]})
    return pd.DataFrame(rows)


def numeric_like(df, share=0.8):
    """Text columns where most entries are numbers (for example a few entries typed as text)."""
    out = []
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) in ("str", "string"):
            x = df[c].dropna()
            n = pd.to_numeric(x, errors="coerce")
            if len(x) and n.notna().mean() >= share and n.notna().mean() < 1:
                out.append(c)
    return out


# ---------- privacy check ----------
ID_WORDS = ["name", "phone", "mobile", "tel", "email", "e mail", "address", "hospital number", "hospital no", "folder", "dob", "date of birth", "birth date",
            "matric", "nin", "bvn", "signature", "initials", "timestamp"]


def identifier_check(df):
    """Flag columns that may identify a person: names, contacts, exact entry times, free text that is unique for almost every row."""
    rows = []
    for c in df.columns:
        low = str(c).lower().replace("_", " ")
        why = []
        if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in ID_WORDS):
            why.append("the column name suggests an identifier")
        s = df[c].dropna()
        if pd.api.types.is_datetime64_any_dtype(df[c]) and len(s) and (s.dt.hour.nunique() > 1 or s.dt.minute.nunique() > 1):
            why.append("exact date and time of each entry")
        elif (df[c].dtype == object or str(df[c].dtype) in ("str", "string")) and len(s):
            v = s.astype(str)
            if v.str.contains(r"@[\w.]+\.\w+|(?:\+?234|0)[789][01]\d{8}\b").any():
                why.append("contains email addresses or phone numbers")
            elif len(s) > 20 and v.nunique() / len(s) > 0.9 and v.str.len().mean() > 6:
                why.append("free text that is different in almost every row")
        if why:
            rows.append({"Column": c, "Concern": "; ".join(why)})
    return pd.DataFrame(rows, columns=["Column", "Concern"])


def rare_groups(df, min_n=3):
    """Answers given by only a few people, which can make someone recognisable in a small sample."""
    rows = []
    for c in df.columns:
        if classify(df[c], c) in ("binary", "categorical"):
            vc = df[c].value_counts()
            rows += [{"Column": c, "Answer": str(k), "People": int(n)} for k, n in vc.items() if n <= min_n]
    return pd.DataFrame(rows, columns=["Column", "Answer", "People"])
