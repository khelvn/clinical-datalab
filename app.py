import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stats_engine as se

st.set_page_config(page_title="Clinical DataLab", page_icon="🩺", layout="centered")
st.markdown("<style>h1,h2,h3{color:#0b2a5b} .stButton>button{background:#c8102e;color:white;border:0;border-radius:8px}</style>",
            unsafe_allow_html=True)

STEPS = ["1. Study Plan", "2. Load Data", "3. Prepare Variables", "4. Explore", "5. Clean", "6. Describe (Table 1)", "7. Choose and Run Test", "8. Review Previous Analysis", "9. Report"]
DESIGNS = ["Not stated", "Cross sectional descriptive", "Cross sectional analytical", "Case control", "Cohort", "Randomized controlled trial", "Community or field trial", "Other"]
for k, v in {"df": None, "raw": None, "log": [], "results": [], "flash": None, "step": STEPS[0], "objectives": "", "plan": "", "design": DESIGNS[0],
             "plan_check": None, "codebook": {}, "kap": {}, "kap_table": None, "review": None}.items():
    st.session_state.setdefault(k, v)

CONCEPTS = {
    STEPS[0]: ("Objectives decide the analysis", "Each objective points to a type of analysis: describing (frequencies, means), comparing groups, testing association, or finding predictors. Write objectives first, then choose tests that answer them, never the other way round."),
    STEPS[1]: ("Why does data quality start here?", "Every result depends on how the data were collected. Check that each row is one patient (or one visit) and each column is one variable, with a header row."),
    STEPS[2]: ("Codes and groups make analysis easier", "Questionnaire headings are long, so short codes keep tables readable while a codebook keeps the original questions. Groups such as age bands should be chosen after you look at how the data are spread. For KAP studies, answers are scored with your scoring system and graded into levels."),
    STEPS[3]: ("Look before you test", "Most analysis mistakes are visible in a first look: a skewed variable, a numeric column that is really a code, an outcome with very few events. The Shapiro Wilk test checks normality: a small p value (below 0.05) means the data are not normal."),
    STEPS[4]: ("There is no single correct way to handle missing data", "Some blanks are expected, for example follow up questions that were skipped. Do not fill those. Filling with the median is easy but understates real variation, and dropping rows shrinks your sample. Extreme values may be real findings, so keep them unless you confirm they are errors."),
    STEPS[5]: ("Describing before comparing", "Use mean and SD for roughly normal numeric data, median and IQR for skewed data, and n (%) for categories. Table 1 is usually the first table in any paper."),
    STEPS[6]: ("Choosing a test", "The test depends on the type of your variables, whether groups are independent or paired, and whether assumptions hold (normality, equal variance, expected counts). The p value is the chance of results this extreme if there were truly no effect. It does not measure importance."),
    STEPS[7]: ("Checking an analysis done by someone else", "An analyst may choose a test that does not fit the data, for example a t test for several groups or a chi square test with small counts. Re running the tests shows whether each choice was suitable and whether the p values can be reproduced. A different p value usually means different rows were included or missing values were handled differently."),
    STEPS[8]: ("Reporting honestly", "Report the test, the exact p value, the effect size and the confidence interval. Say what was cleaned and how."),
}


def fmt(c):
    q = st.session_state.codebook.get(c)
    return f"{c}: {q[:70]}" if q and q != c else str(c)


def go(step):
    st.session_state.step = step


def header(step):
    t, body = CONCEPTS[step]
    st.subheader(step)
    with st.expander("📘 Concept Corner: " + t):
        st.write(body)


def next_btn(i):
    if i + 1 < len(STEPS):
        st.button("Next: " + STEPS[i + 1], on_click=go, args=(STEPS[i + 1],), key=f"next{i}")


def draw(pl):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    t = pl["type"]
    if t == "box":
        names = list(pl["groups"])
        ax.boxplot([pl["groups"][n] for n in names], showfliers=False)
        for i, n in enumerate(names):
            v = pl["groups"][n]
            ax.scatter(np.random.default_rng(0).normal(i + 1, 0.05, len(v)), v, s=8, alpha=0.3)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names)
        ax.set_ylabel(pl["ylabel"])
    elif t in ("paired", "scatter"):
        ax.scatter(pl["x"], pl["y"], s=12, alpha=0.5)
        lo, hi = min(pl["x"].min(), pl["y"].min()), max(pl["x"].max(), pl["y"].max())
        if t == "paired":
            ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--")
            ax.set_title("Points below the dashed line fell between the two measurements", fontsize=9)
        else:
            xs = np.array([pl["x"].min(), pl["x"].max()])
            ax.plot(xs, pl["fit"][0] + pl["fit"][1] * xs, color="#c8102e")
        ax.set_xlabel(pl["xlabel"])
        ax.set_ylabel(pl["ylabel"])
    elif t == "hist":
        ax.hist(pl["x"], bins=20, color="#0b2a5b")
        ax.axvline(pl["mu"], color="#c8102e")
        ax.set_xlabel(pl["xlabel"])
    else:
        pl["pct"].plot(kind="bar", stacked=True, ax=ax, rot=0)
        ax.set_ylabel(pl["ylabel"])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _df(d):
    return "" if d is None else ("df = " + (f"{d[0]}, {d[1]}" if isinstance(d, tuple) else f"{d:.1f}".rstrip("0").rstrip(".")))


def ten_steps(r, v1, v2):
    k = r["check"]
    rule = (f"Reject the null hypothesis if the statistic is beyond the critical value ({k['crit']:.3f})." if k else "This test uses the exact p value directly, so no table is needed.")
    dec = ("Reject the null hypothesis." if r["significant"] else "Do not reject the null hypothesis.")
    return [f"**1. Data:** {v1} and {v2}.", "**2. Assumptions:** " + " ".join(r["reasons"][:3]),
            f"**3. Hypotheses:** H0: no statistically significant difference or association between {v1} and {v2}. HA: there is one. Level of significance = {r['alpha']}.",
            f"**4. Test statistic:** {r['stat_label']} ({r['test']}).", f"**5. Distribution:** {_df(r['df']) or 'exact distribution'}.",
            f"**6. Calculated value:** {r['stat_label']} = {r['stat']:.3f}.", f"**7. Decision rule:** {rule}", f"**8. Statistical decision:** {dec}",
            f"**9. Conclusion:** " + ("the data support a real difference or association." if r["significant"] else "the data do not show a difference or association."),
            f"**10. p value:** {se.fmt_p(r['p'])}."]


def show_result(r, v1, v2, key="add"):
    st.success(f"Recommended: **{r['test']}**")
    with st.expander("Why this test?", expanded=True):
        for x in r["reasons"]:
            st.write("✅ " + x)
        st.write("🔁 **Alternative:** " + r["alternative"])
    a, b, c = st.columns(3)
    a.metric(r["stat_label"], f"{r['stat']:.2f}")
    b.metric("p value", "< 0.001" if r["p"] < 0.001 else f"{r['p']:.3f}")
    c.metric(r["effect_label"], f"{r['effect']:.2f}")
    if r["ci"]:
        st.write(f"**{r['ci'][0]}:** {r['ci'][1][0]:.2f} to {r['ci'][1][1]:.2f}")
    (st.success if r["significant"] else st.warning)(r["decision"])
    st.write(r["interpretation"])
    st.caption("Statistical significance is not the same as clinical importance. Look at the effect size and confidence interval. A non significant result does not prove there is no difference.")
    for w in r["warnings"]:
        st.warning(w)
    if r["posthoc"] is not None:
        st.write("**Post hoc comparisons**")
        st.dataframe(r["posthoc"].round(4))
    if learning:
        with st.expander("Check it step by step (hypothesis testing procedure)"):
            if r["check"]:
                st.write(f"{r['check']['name']} at alpha {r['alpha']} = **{r['check']['crit']:.3f}**, compared with your statistic **{abs(r['stat']):.3f}**.")
            for line in ten_steps(r, v1, v2):
                st.markdown(line)
    draw(r["plot"])
    if st.button("Add this result to my report", key=key):
        st.session_state.results.append({"vars": f"{v1} and {v2}", "test": r["test"], "line": f"{r['stat_label']} = {r['stat']:.2f}, {se.fmt_p(r['p'])}, {r['effect_label']} = {r['effect']:.2f}", "text": r["interpretation"]})
        st.session_state.flash = "Added to your report."
        st.rerun()


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🩺 Clinical DataLab")
    st.caption("A guided, no code statistics lab for medical students and resident doctors.")
    st.radio("Workflow", STEPS, key="step")
    alpha = st.selectbox("Significance level (alpha)", [0.05, 0.01, 0.10])
    learning = st.checkbox("Learning mode", value=False, help="Shows the step by step hypothesis testing walkthrough and the critical value check. Useful for students.")
    st.caption("No dataset loaded." if st.session_state.df is None else f"Dataset: {len(st.session_state.df)} rows, {st.session_state.df.shape[1]} columns")
    if st.button("Reset everything"):
        st.session_state.clear()
        st.rerun()

step = st.session_state.step
i = STEPS.index(step)
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None
header(step)
df = st.session_state.df
if i > 1 and df is None:
    st.info("Load a dataset in Step 2 first.")
    st.stop()
if df is not None:
    kinds = {c: se.classify(df[c], c) for c in df.columns}
    nums = [c for c in df.columns if kinds[c] == "numeric"]
    usable = [c for c in df.columns if kinds[c] in ("numeric", "binary", "categorical")]
    cats = [c for c in df.columns if kinds[c] in ("binary", "categorical")]

def show_regression(y, xs, key):
    try:
        g = se.regress(df, y, xs, alpha)
    except Exception as e:
        st.error(str(e))
        return
    st.success(f"Recommended: **{g['test']}**")
    with st.expander("Why this test?", expanded=True):
        for x in g["reasons"]:
            st.write("✅ " + x)
    st.write(g["fit"])
    st.dataframe(g["table"].round(3))
    for w in g["warnings"]:
        st.warning(w)
    sig = g["table"][g["table"]["p value"] < alpha]["Predictor"].tolist()
    st.write("Statistically significant predictors: " + (", ".join(sig) if sig else "none") + ".")
    st.write(g["interpretation"])
    if st.button("Add this model to my report", key=key):
        st.session_state.results.append({"vars": f"{y} predicted by {', '.join(xs)}", "test": g["test"], "line": g["fit"], "text": "Significant predictors: " + (", ".join(sig) if sig else "none") + "."})
        st.session_state.flash = "Added to your report."
        st.rerun()


def free_choice():
    mode = st.radio("What do you want to do?", ["Compare or relate two variables", "Predict an outcome from several variables (regression)", "Compare one variable with a known value", "Repeated measurements by group (results table)"])
    if mode.startswith("Predict"):
        y = st.selectbox("Outcome (what you want to predict)", usable, format_func=fmt)
        xs = st.multiselect("Predictors", [c for c in usable if c != y], format_func=fmt)
        st.caption(f"Outcome detected as {kinds[y]}. A yes/no outcome gives binary logistic regression; a numeric outcome gives multiple linear regression.")
        if not xs:
            st.stop()
        show_regression(y, xs, "freereg")
    elif mode.startswith("Repeated"):
        fams = se.time_families(df)
        if not fams:
            st.info("No repeated measurements were found. Column names should start with a time label such as PREOP, 3MPO and 6MPO, or Baseline and 3M, with the same measurement name after it.")
            return
        grp = st.selectbox("Compare between which groups?", cats, format_func=fmt)
        names = st.multiselect("Parameters to include", list(fams), default=list(fams)[:6])
        tests = st.radio("Tests", ["Let the lab decide for each parameter", "Nonparametric for all (Wilcoxon signed rank and Mann Whitney)", "Parametric for all (paired and independent t tests)"])
        pf = None if tests.startswith("Let") else ("nonparametric" if tests.startswith("Nonp") else "parametric")
        cen = "median" if st.radio("Show", ["median (IQR)", "mean (SD)"]).startswith("median") else "mean"
        if names:
            tb = se.repeated_table(df, grp, {k: fams[k] for k in names}, alpha, pf, cen)
            st.dataframe(tb)
            st.download_button("Download table", tb.to_csv(index=False).encode(), "repeated_measurements_table.csv", "text/csv")
            with st.expander("Columns used for each parameter"):
                st.dataframe(pd.DataFrame({k: fams[k] for k in names}).T)
            st.caption("p values on the left compare each time point with the first within each group. p values between groups are shown on the first row of each parameter. Check the columns matched for each parameter above.")
    elif mode.startswith("Compare one"):
        v = st.selectbox("Numeric variable", nums)
        mu = st.number_input("Known value (for example a population mean)", value=0.0)
        try:
            show_result(se.one_sample(df, v, mu, alpha), v, f"the value {mu}")
        except Exception as e:
            st.error(str(e))
    else:
        v1 = st.selectbox("Variable 1", usable, format_func=fmt)
        v2 = st.selectbox("Variable 2", [c for c in usable if c != v1], format_func=fmt)
        st.caption(f"Detected: {v1} is {kinds[v1]}; {v2} is {kinds[v2]}.")
        paired = False
        if kinds[v1] == kinds[v2] and kinds[v1] in ("numeric", "binary"):
            paired = st.radio("Are these two measurements on the same patients (for example before and after treatment)?",
                              ["No, they are different patients or I want to see if they are related", "Yes, same patients (paired)"]).startswith("Yes")
        PREF = {"Let the lab decide (recommended)": None, "Parametric (t test, ANOVA, Pearson)": "parametric",
                "Nonparametric (Mann Whitney, Kruskal Wallis, Spearman, Wilcoxon)": "nonparametric", "Exact test (Fisher) for categories": "exact"}
        pref = PREF[st.selectbox("Your preferred test family (optional, for example the one in your proposal)", list(PREF))]
        by = st.selectbox("Repeat separately within each group of (optional)", ["None"] + [c for c in cats if c not in (v1, v2)], format_func=lambda c: c if c == "None" else fmt(c), key="byfree")
        if by != "None":
            st.write(f"**Separately within each group of {by}**")
            st.dataframe(se.by_group(df, v1, v2, paired, by, alpha, pref))
        try:
            r = se.analyze(df, v1, v2, paired, alpha, pref)
        except Exception as e:
            st.error(str(e))
            st.stop()
        show_result(r, v1, v2)


KINDS = ["Describe my participants", "Compare groups", "Find a relationship between two variables", "Measure change before and after", "Predict an outcome from several factors", "Time to event (not built yet)"]
KEYMAP = {"describe": 0, "compare": 1, "associate": 2, "change": 3, "predict": 4, "survival": 5}
KHELP = ["You want to summarise who the participants are. This needs counts, percentages, means and medians, not a test. Use Table 1 in Step 6.",
         "You want to know if a measurement (or an outcome such as yes/no) is different between groups, for example intervention versus control.",
         "You want to know if two things go together, for example education level and blood pressure control.",
         "You measured the same patients twice and want to know if the measurement changed.",
         "You want to find which factors are linked to an outcome while adjusting for the others. This is called regression."]


def effect_words(r):
    e, lab = r["effect"], r["effect_label"]
    if e is None or not np.isfinite(e):
        return ""
    if lab.startswith("Cohen"):
        w = "negligible" if abs(e) < 0.2 else "small" if abs(e) < 0.5 else "medium" if abs(e) < 0.8 else "large"
        return f"**How big is the effect?** {w.capitalize()} (Cohen's d = {e:.2f})."
    if lab == "Odds ratio":
        return f"**How big is the effect?** Odds ratio = {e:.2f}. Above 1 means higher odds, below 1 means lower odds."
    return f"**{lab}:** {e:.2f}."


def simple_result(r, v1, v2, key):
    p = se.fmt_p(r["p"])
    if r["significant"]:
        st.success(f"**Statistically significant ({p}).** This pattern is unlikely to be due to chance alone.")
    else:
        st.info(f"**Not statistically significant ({p}).** The data do not show a clear difference or relationship. This does not prove there is none.")
    st.write(r["interpretation"])
    st.write(effect_words(r))
    st.caption(f"Test used: {r['test']}. Why: {r['reasons'][0]}")
    for w in r["warnings"]:
        st.warning(w)
    if st.checkbox("Show full statistics, chart and details", key="full" + key):
        show_result(r, v1, v2, key)
    elif st.button("Add this result to my report", key="s" + key):
        st.session_state.results.append({"vars": f"{v1} and {v2}", "test": r["test"], "line": f"{r['stat_label']} = {r['stat']:.2f}, {p}, {r['effect_label']} = {r['effect']:.2f}", "text": r["interpretation"]})
        st.session_state.flash = "Added to your report."
        st.rerun()


def guided():
    objs = [o.strip() for o in st.session_state.objectives.splitlines() if o.strip()]
    plan = st.session_state.plan
    cls = se.classify_objectives(objs)
    idx = min(st.session_state.get("obj_idx", 0), len(objs) - 1)
    with st.expander("Your plan from Step 1"):
        st.write("**Study design:** " + st.session_state.design)
        for n, o in enumerate(objs, 1):
            st.write(f"{n}. {o}")
        st.write("**Analysis plan:** " + (plan or "not entered"))
    st.progress((idx + 1) / len(objs), text=f"Objective {idx + 1} of {len(objs)}")
    c1, c2 = st.columns(2)
    c1.button("Previous objective", disabled=idx == 0, on_click=lambda: st.session_state.update(obj_idx=idx - 1), key=f"prev{idx}")
    c2.button("Next objective", disabled=idx == len(objs) - 1, on_click=lambda: st.session_state.update(obj_idx=idx + 1), key=f"nxt{idx}")
    st.markdown(f"### {objs[idx]}")
    primary = cls[idx]["Type"].split(" + ")[0]
    kind = st.radio("What kind of question is this? (guessed from your wording, change it if needed)", KINDS, index=KEYMAP.get(primary, 1), key=f"kind{idx}")
    k = KINDS.index(kind)
    if k < 5:
        st.caption(KHELP[k])
    sug = se.suggest(objs[idx], df)
    m = sug["matched"]
    catm = [c for c in m if kinds[c] in ("binary", "categorical")]
    numm = [c for c in m if kinds[c] == "numeric"]
    if m:
        st.caption("Variables in your data that match this objective: " + ", ".join(m[:6]))

    def pick(label, default, key, options=None):
        opts = options or usable
        return st.selectbox(label, opts, index=opts.index(default) if default in opts else 0, key=f"{key}{idx}_{k}", format_func=fmt)

    if k == 0:
        st.button("Open Table 1 (Step 6)", on_click=go, args=(STEPS[5],), key=f"t1{idx}")
        return
    if k == 5:
        st.info("Survival analysis (Kaplan Meier, log rank, Cox regression) is not built into this lab yet.")
        return
    if k == 4:
        y = pick("What outcome do you want to predict?", next((c for c in m if kinds[c] in ("binary", "numeric")), usable[0]), "y")
        cand = [c for c in usable if c != y and not se.related(c, y)]
        ev = int(df[y].value_counts().min()) if kinds[y] == "binary" else len(df) // 10
        budget, dflt = max(2, ev // 10), []
        for c in cand:
            t = 1 if kinds[c] == "numeric" else max(1, df[c].nunique() - 1)
            if sum(1 if kinds[x] == "numeric" else max(1, df[x].nunique() - 1) for x in dflt) + t <= budget:
                dflt.append(c)
        xs = st.multiselect("Which factors do you want to include?", [c for c in usable if c != y], default=dflt, key=f"x{idx}_{y}", format_func=fmt)
        st.caption(f"We started with {len(dflt)} factors because about 10 outcome events are needed for each model term. Add or remove as you like. Do not include anything that is part of the outcome itself, such as the reading used to define it.")
        pl = se.planned_tests(plan, [y])
        if pl:
            st.write("📋 **Your proposal planned:** " + ", ".join(t.title() for t in pl))
            if kinds[y] == "binary" and "linear regression" in pl:
                st.warning("Your proposal planned linear regression, but this outcome is yes/no, so **logistic regression** is the right model.")
            if kinds[y] == "numeric" and "logistic regression" in pl:
                st.warning("Your proposal planned logistic regression, but this outcome is a measurement, so **linear regression** is the right model.")
        if xs:
            show_regression(y, xs, f"reg{idx}")
        return
    paired = k == 3
    if k == 1:
        a = pick("What do you want to compare? (a measurement or an outcome)", (numm or catm or usable)[0], "a")
        b = pick("Between which groups?", next((c for c in catm if c != a), next((c for c in usable if c != a), usable[0])), "b", [c for c in usable if c != a])
    elif k == 2:
        a = pick("First variable", m[0] if m else usable[0], "a")
        b = pick("Second variable", next((c for c in m if c != a and not se.related(c, a)), next((c for c in usable if c != a), usable[0])), "b", [c for c in usable if c != a])
    else:
        pa, pb = sug["pairs"][0] if sug["pairs"] else (nums[0], nums[-1])
        a = pick("Measurement before", pa, "a", nums)
        b = pick("Measurement after", pb, "b", [c for c in nums if c != a])
    by = st.selectbox("Repeat separately within each group of (optional)", ["None"] + [c for c in cats if c not in (a, b)], format_func=lambda c: c if c == "None" else fmt(c), key=f"byg{idx}_{k}")
    if by != "None":
        st.write(f"**Separately within each group of {by}**")
        st.dataframe(se.by_group(df, a, b, paired, by, alpha))
    try:
        rec = se.analyze(df, a, b, paired, alpha)
    except Exception as e:
        st.error(str(e))
        return
    r = rec
    planned = se.planned_tests(plan, [a, b])
    if planned:
        st.write("📋 **Your proposal planned:** " + ", ".join(t.title() for t in planned))
        t0 = planned[0]
        if t0 in se.KEY and not any(x in rec["test"].lower() for x in se.KEY[t0]):
            st.warning(f"Your proposal planned a {t0}, but for these variables the data suggest **{rec['test']}**. " + " ".join(rec["reasons"][:2]))
        pref = next((se.PLAN_PREF[t] for t in planned if t in se.PLAN_PREF), None)
        if pref:
            try:
                alt = se.analyze(df, a, b, paired, alpha, pref)
                if alt["test"] != rec["test"]:
                    ch = st.radio("Which test do you want to run?", [f"Recommended: {rec['test']}", f"From my proposal: {alt['test']}"], key=f"run{idx}")
                    r = alt if ch.startswith("From") else rec
            except Exception:
                pass
    st.success(f"Recommended test: **{rec['test']}**" if r is rec else f"Running the test from your proposal: **{r['test']}**")
    simple_result(r, a, b, f"g{idx}")


# ---------------- 1 Study Plan ----------------
if i == 0:
    st.info("**What to do:** state your study objectives and paste the analysis plan from your proposal. The lab checks whether the planned tests fit your objectives (and your data once loaded) and suggests better options with reasons. You always keep the final choice in Step 7.")
    st.session_state.design = st.selectbox("Study design", DESIGNS, index=DESIGNS.index(st.session_state.design))
    st.session_state.objectives = st.text_area("Study objectives (one per line)", st.session_state.objectives, height=130,
                                               placeholder="To determine the prevalence of ...\nTo compare ... between ...\nTo identify factors associated with ...")
    objs = st.session_state.objectives.splitlines()
    if any(o.strip() for o in objs):
        st.markdown("#### What your objectives call for")
        for n, o in enumerate(se.classify_objectives(objs), 1):
            with st.container(border=True):
                st.markdown(f"**Objective {n}:** {o['Objective']}")
                st.write("Type of question: " + " and ".join(se.FRIENDLY.get(t, t) for t in o["Type"].split(" + ")))
                st.write("Suggested analysis: " + o["Suggested analysis"])
    st.session_state.plan = st.text_area("Data analysis plan from your proposal", st.session_state.plan, height=160,
                                         placeholder="Example: Data will be analysed using ... Chi square test will be used for the association between sex and outcome ...")
    st.caption("Tip: name your variables exactly as in your dataset so the lab can check each planned test against the real data.")
    if st.session_state.plan.strip() and st.button("Check my analysis plan"):
        st.session_state.plan_check = se.check_plan(objs, st.session_state.plan, st.session_state.df, alpha)
    pc = st.session_state.plan_check
    if pc is not None:
        st.markdown("#### Plan review")
        cnt = pc["Status"].value_counts()
        st.info(f"✅ {cnt.get('Appropriate', 0)} good choices    ⚠️ {cnt.get('Better option', 0)} to reconsider    ❗ {cnt.get('Gap', 0)} missing    ℹ️ {cnt.get('Check', 0)} to confirm")
        for key, title in [("Appropriate", "✅ Good choices"), ("Better option", "⚠️ Consider changing"), ("Gap", "❗ Missing from your plan"), ("Check", "ℹ️ To confirm")]:
            sub = pc[pc["Status"] == key]
            if len(sub):
                st.markdown(f"**{title}**")
                for _, r in sub.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**{r['Headline']}**")
                        st.write("Why: " + r["Why"])
                        st.write("What to do: " + r["Do"])
        st.caption("These are suggestions only. In Step 7 you can still run the test from your proposal by choosing your preferred test family.")
    next_btn(i)

# ---------------- 2 Load ----------------
elif i == 1:
    st.info("**What to do:** choose a demo dataset or upload your own CSV or Excel file (one row per patient, one column per variable, header in the first row). Your file is held in memory for this session only.")
    if st.button("📥 Load demo clinical dataset"):
        d = se.make_demo()
        st.session_state.update(df=d, raw=d.copy(), log=[], results=[], flash="Demo dataset loaded. It has missing values, duplicates and outliers for you to practise cleaning.")
        st.rerun()
    up = st.file_uploader("Or upload a .csv or .xlsx file", type=["csv", "xlsx"])
    if up is not None and st.session_state.get("src") != up.name:
        NA = ["", "NA", "N/A", "NaN", "nan", "NULL", "#N/A"]
        d = pd.read_csv(up, keep_default_na=False, na_values=NA) if up.name.endswith("csv") else pd.read_excel(up, keep_default_na=False, na_values=NA)
        st.session_state.update(df=d, raw=d.copy(), log=[], results=[], src=up.name, flash=f"Loaded {up.name}.")
        st.rerun()
    if df is not None:
        st.write(f"**{len(df)} rows and {df.shape[1]} columns**")
        st.dataframe(df.head(10))
        st.markdown("#### Privacy check")
        idc = se.identifier_check(df)
        if len(idc):
            st.warning("These columns may identify participants. Remove them unless you really need them for the analysis.")
            st.dataframe(idc)
            if st.button("Remove these columns"):
                st.session_state.update(df=df.drop(columns=list(idc["Column"])), flash=f"Removed {len(idc)} possible identifier columns.",
                                        log=st.session_state.log + [f"Removed possible identifier columns: {', '.join(map(str, idc['Column']))}."])
                st.rerun()
        else:
            st.success("No names, phone numbers, emails or exact entry times were found.")
        st.caption("A study number is fine, as long as the list linking it to names or hospital numbers is kept separately and never uploaded. Upload anonymised data only to an online app.")
        next_btn(i)

# ---------------- 2 Explore ----------------
elif i == 2:
    t1, t2, t3, t4 = st.tabs(["Column codes", "Create categories", "KAP scoring", "Number columns"])
    with t1:
        st.info("**What to do:** replace long questionnaire headings with short codes. The original questions stay in a codebook and appear next to each code when you pick variables.")
        scheme = st.radio("Coding style", ["Q1, Q2, Q3 (simple)", "Short names from the first words of each question", "KAP codes (K1, A1, P1) and Q for the rest"], key="scheme")
        sch = "q" if scheme.startswith("Q1") else "short" if scheme.startswith("Short") else "kap"
        if sch == "kap" and not st.session_state.kap:
            st.warning("Detect KAP sections in the KAP scoring tab first. Until then every column gets a simple Q code.")
        codes = se.make_codes(list(df.columns), sch, st.session_state.kap)
        ed = st.data_editor(pd.DataFrame({"Original question": list(df.columns), "New code": codes}), disabled=["Original question"], hide_index=True, key=f"codes{sch}{len(df.columns)}")
        if st.button("Apply codes"):
            new = [str(x).strip() for x in ed["New code"]]
            if len(set(new)) != len(new) or "" in new:
                st.error("Each code must be unique and not empty.")
            else:
                cb = st.session_state.codebook
                mp = dict(zip(df.columns, new))
                st.session_state.codebook = {mp[c]: cb.get(c, c) for c in df.columns}
                st.session_state.kap = {mp.get(c, c): v for c, v in st.session_state.kap.items()}
                st.session_state.update(df=df.rename(columns=mp), flash=f"Renamed {len(new)} columns. The codebook is shown below.", log=st.session_state.log + [f"Renamed {len(new)} columns to codes."], kap_table=None)
                st.rerun()
        if st.session_state.codebook:
            cbk = pd.DataFrame({"Code": list(st.session_state.codebook), "Original question": list(st.session_state.codebook.values())})
            st.dataframe(cbk)
            st.download_button("Download codebook", cbk.to_csv(index=False).encode(), "codebook.csv", "text/csv")
    with t2:
        st.info("**What to do:** turn a measurement such as age into groups. Look at the spread first, then choose cut points that suit your study.")
        if not nums:
            st.write("There are no numeric variables.")
        else:
            c = st.selectbox("Numeric variable", nums, index=next((n for n, x in enumerate(nums) if "age" in x.lower()), 0), format_func=fmt, key="catv")
            x = df[c].dropna()
            fig, ax = plt.subplots(figsize=(6, 2.8))
            ax.hist(x, bins=25, color="#0b2a5b")
            ax.set_xlabel(c[:60])
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(x.describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_frame("value").T)
            how = st.radio("How do you want to group it?", ["Cut points I choose", "Equal width groups", "Quartiles (four equal sized groups)"], key="cathow")
            ng = st.slider("Number of groups", 2, 8, 4) if how.startswith("Equal") else 4
            c0 = se.suggest_cuts(x, "equal", ng) if how.startswith("Equal") else se.suggest_cuts(x, "quartiles")
            txt = st.text_input("Cut points (separate with commas)", ", ".join(str(int(v)) for v in c0), key=f"cuts{c}{how[:3]}{ng}")
            name = st.text_input("Name of the new column", f"{c}_group", key=f"cn{c}")
            try:
                cuts = [float(v) for v in txt.replace(";", ",").split(",") if v.strip()]
                g = se.categorise(df[c], cuts)
                cnt = g.value_counts(sort=False)
                st.dataframe(pd.DataFrame({"n": cnt, "%": (cnt / cnt.sum() * 100).round(1)}))
                if st.button("Create grouped column"):
                    d2 = df.copy()
                    d2[name] = g
                    st.session_state.update(df=d2, flash=f"Created '{name}'.", log=st.session_state.log + [f"Grouped '{c}' into '{name}' using cut points {txt}."])
                    st.rerun()
            except Exception:
                st.error("Enter numbers separated by commas, for example 30, 45, 60.")
    with t3:
        st.info("**What to do:** for KAP studies, detect the Knowledge, Attitude and Practice questions, then score them with your scoring system and grade the totals (for example Poor, Fair, Good).")
        if st.button("Detect KAP sections"):
            st.session_state.kap_table = se.detect_kap(df)
        kt = st.session_state.kap_table
        if kt is not None:
            ed = st.data_editor(kt, column_config={"Section": st.column_config.SelectboxColumn("Section", options=["Knowledge", "Attitude", "Practice", "Other"], required=True)},
                                disabled=["Column"], hide_index=True, key=f"kapedit{len(kt)}")
            kap = dict(zip(ed["Column"], ed["Section"]))
            st.session_state.kap = kap
            st.caption("Detected from the wording of each question and the type of answers. Correct any section that is wrong.")
            amode = "dichotomised" if st.radio("Attitude scoring", ["5 point scale (1 to 5 points)", "Dichotomised (1 point for the favourable side, 0 otherwise)"], key="amode").startswith("Dich") else "scale"
            tmpl = se.kap_template(df, kap, amode)
            if tmpl.empty:
                st.warning("No Knowledge, Attitude or Practice items selected.")
            else:
                st.markdown("#### Scoring system")
                st.write("Items in a tick all that apply grid share a **Group** and earn points only if every part is correct. Upload your scoring system (a table with the columns **Item**, **Answer**, **Points** and optionally **Domain** and **Group**). Item can be the question or its code. Or edit the starting table below, which is filled with guesses you must check.")
                st.download_button("Download scoring template", tmpl.to_csv(index=False).encode(), "kap_scoring_template.csv", "text/csv")
                up = st.file_uploader("Upload scoring system (csv or xlsx)", type=["csv", "xlsx"], key="kapup")
                base = tmpl
                if up is not None:
                    base = pd.read_csv(up, keep_default_na=False, na_values=[""]) if up.name.endswith("csv") else pd.read_excel(up, keep_default_na=False, na_values=[""])
                    if not {"Item", "Answer", "Points"} <= set(base.columns):
                        st.error("The file needs the columns Item, Answer and Points.")
                        base = tmpl
                table = st.data_editor(base, hide_index=True, num_rows="dynamic", key=f"score{up.name if up else 'tmpl'}{len(kap)}{amode}")
                zero = st.radio("Blank answers", ["Score a blank as 0 points (usual for skip questions)", "Leave the score blank if any answer is missing"], key="kapzero").startswith("Score")
                res, unmatched = se.score_system(df, table, st.session_state.codebook, zero)
                if unmatched:
                    st.warning("These items in the scoring system were not found in your data: " + "; ".join(u[:60] for u in unmatched[:6]))
                outs = {}
                pct_mode = st.radio("Grade categories are based on", ["Percent of the maximum score", "Points"], key="kapmode").startswith("Percent")
                for dom, (raw, pct, mx, ni) in res.items():
                    sc = pct if pct_mode else raw
                    st.markdown(f"**{dom}**: {ni} items, maximum {mx:g} points, mean {raw.mean():.1f} points ({pct.mean():.1f}%).")
                    dflt = pd.DataFrame({"Category": ["Poor", "Good"], "Minimum": [0.0, 50.0 if pct_mode else round(mx / 2, 1)]})
                    cats = st.data_editor(dflt, num_rows="dynamic", hide_index=True, key=f"cats{dom}{pct_mode}")
                    lv = pd.Series(se.assign_levels(sc, cats), index=df.index)
                    cnt = lv.value_counts(sort=False)
                    st.dataframe(pd.DataFrame({"n": cnt, "%": (cnt / max(cnt.sum(), 1) * 100).round(1)}))
                    outs[dom] = (raw, pct, lv)
                if outs and st.button("Create score and grade columns"):
                    d2 = df.copy()
                    for dom, (raw, pct, lv) in outs.items():
                        nm = dom.replace(" ", "_")
                        d2[f"{nm}_score"], d2[f"{nm}_percent"], d2[f"{nm}_level"] = raw, pct.round(1), lv
                    st.session_state.update(df=d2, flash="Created score, percent and level columns for each domain.", log=st.session_state.log + ["Scored " + ", ".join(outs) + " using the scoring system."])
                    st.rerun()
    with t4:
        st.info("**What to do:** some number columns are stored as text because a few cells contain words or symbols (for example a value like <0.1). Convert them so they can be analysed. Entries that are not numbers become blank.")
        nl = se.numeric_like(df)
        if not nl:
            st.success("No number columns stored as text were found.")
        else:
            pn = st.multiselect("Columns to convert", nl, default=nl, format_func=fmt, key="numfix")
            for c in pn:
                bad = pd.to_numeric(df[c], errors="coerce").isna() & df[c].notna()
                st.caption(f"{fmt(c)[:60]}: {int(bad.sum())} entries are not numbers, for example {', '.join(map(str, df.loc[bad, c].unique()[:4]))}")
            if pn and st.button("Convert to numbers"):
                d2, lines = df.copy(), []
                for c in pn:
                    bad = int((pd.to_numeric(d2[c], errors="coerce").isna() & d2[c].notna()).sum())
                    d2[c] = pd.to_numeric(d2[c], errors="coerce")
                    lines.append(f"Converted '{c}' to numbers ({bad} non numeric entries became blank).")
                st.session_state.update(df=d2, flash=f"Converted {len(pn)} columns.", log=st.session_state.log + lines)
                st.rerun()
    next_btn(i)

elif i == 3:
    st.info("**What to do:** check the detected type of each variable, the missing values, and the shape of each numeric variable. **Why:** the variable type and distribution decide which test is valid.")
    t1, t2, t3, t4 = st.tabs(["Variable types", "Missing values", "Distributions", "Normality table"])
    with t1:
        st.dataframe(pd.DataFrame({"Detected type": kinds, "Unique values": df.nunique(), "Example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else "")}).astype(str))
        st.caption("numeric = measured on a scale; binary = two groups; categorical = three or more groups; id or text = not analysed.")
    with t2:
        m = df.isna().sum()
        st.dataframe(pd.DataFrame({"Missing": m, "Missing %": (m / len(df) * 100).round(1)})[m > 0])
        st.write(f"Duplicate rows: **{int(df.duplicated().sum())}**")
    with t3:
        if nums:
            c = st.selectbox("Numeric variable", nums)
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.hist(df[c].dropna(), bins=25, color="#0b2a5b")
            ax.set_xlabel(c)
            st.pyplot(fig)
            plt.close(fig)
            nm = se.check_normal(df[c])
            if nm["p"] is not None:
                st.write(f"**Shapiro Wilk:** W = {nm['W']:.3f}, {se.fmt_p(nm['p'])}, skewness = {nm['skew']:.2f}. "
                         + ("Looks approximately normal: summarise with **mean and SD**, and parametric tests are suitable." if nm["ok"] else
                            "Not normal: summarise with **median and IQR**, and prefer nonparametric tests."))
                fig, ax = plt.subplots(figsize=(4.5, 3.2))
                stats.probplot(df[c].dropna(), dist="norm", plot=ax)
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Q Q plot: points close to the straight line mean the data are close to normal.")
            with st.expander("What is the Shapiro Wilk test?"):
                st.write("It tests whether a numeric variable could have come from a normal (bell shaped) distribution. W is close to 1 for normal data. The null hypothesis is that the data are normal, so a p value below 0.05 means they are not normal. It guides the choice between parametric tests (t test, ANOVA, Pearson) and nonparametric tests (Mann Whitney, Kruskal Wallis, Spearman). In very large samples it flags tiny, harmless deviations, so always look at the histogram too.")
    with t4:
        st.write("Only continuous measurements (age, blood pressure, scores) need a normality check. Pick just the ones you plan to analyse.")
        pn = st.multiselect("Numeric variables to test", nums, format_func=fmt, key="nsel")
        if pn:
            rows = []
            for c in pn:
                nm = se.check_normal(df[c])
                rows.append({"Variable": c, "n": nm["n"], "W": None if nm["W"] is None else round(nm["W"], 3), "p value": None if nm["p"] is None else round(nm["p"], 4),
                             "Skewness": None if nm.get("skew") is None else round(nm["skew"], 2), "Verdict": "Approximately normal" if nm["ok"] else "Not normal",
                             "Summarise with": "mean (SD)" if nm["ok"] else "median (IQR)"})
            st.dataframe(pd.DataFrame(rows))
    next_btn(i)

# ---------------- 3 Clean ----------------
elif i == 4:
    st.markdown("### 3.1 Missing values")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        st.success("No missing values.")
    else:
        L = {"Leave as is": "leave", "Fill with median": "median", "Fill with mean": "mean", "Fill with mode (most common)": "mode",
             "Label blanks as Not applicable (skip question)": "na", "Drop rows with missing": "drop"}
        info = {}
        for c, n in miss.items():
            sp = se.skip_parent(df, c)
            if kinds[c] == "numeric":
                opts, sug = ["Leave as is", "Fill with median", "Fill with mean", "Drop rows with missing"], ("Fill with mean" if se.check_normal(df[c])["ok"] else "Fill with median")
            else:
                opts, sug = ["Leave as is", "Fill with mode (most common)", "Label blanks as Not applicable (skip question)", "Drop rows with missing"], "Fill with mode (most common)"
            info[c] = (opts, sug, sp, n)

        def set_all(mode):
            for c, (opts, sug, sp, n) in info.items():
                st.session_state[f"mv{c}"] = "Leave as is" if (mode == "leave" or sp) else sug

        st.caption(f"{len(miss)} columns have blanks. The default is to leave them as is, because many blanks are expected (follow up questions are skipped) and filling them can distort results. Statistical tests ignore blanks automatically.")
        b1, b2 = st.columns(2)
        b1.button("Leave ALL as is", on_click=set_all, args=("leave",))
        b2.button("Use suggestions for all (skip questions stay as is)", on_click=set_all, args=("suggest",))
        na_label = st.text_input("Label for skip question blanks (if you choose that option)", "Not applicable", key="nalabel")
        ch = {}
        with st.expander(f"Choose column by column ({len(miss)} columns)"):
            for c, (opts, sug, sp, n) in info.items():
                if sp:
                    st.caption(f"'{fmt(c)[:60]}' is blank when '{fmt(sp[0])[:50]}' is '{sp[1]}'. This looks like a skip question, so blanks are expected.")
                ch[c] = st.selectbox(f"{fmt(c)[:80]}: {n} missing ({n / len(df) * 100:.1f}%). Suggested: {'leave as is (skip question)' if sp else sug}", opts, index=0, key=f"mv{c}")
        if st.button("Apply missing value choices"):
            d, log = df, []
            for c, o in ch.items():
                if L[o] != "leave":
                    d, line = se.fill_missing(d, c, L[o], na_label)
                    log.append(line)
            st.session_state.update(df=d, flash=f"Applied choices to {len(log)} columns." if log else "Nothing to change: all columns left as is.", log=st.session_state.log + log)
            st.rerun()
    st.markdown("### 3.2 Duplicate rows")
    st.write(f"Duplicate rows found: **{int(df.duplicated().sum())}**")
    if df.duplicated().sum() and st.button("Remove duplicate rows"):
        n = int(df.duplicated().sum())
        st.session_state.update(df=df.drop_duplicates().reset_index(drop=True), flash=f"Removed {n} duplicate rows.", log=st.session_state.log + [f"Removed {n} duplicate rows."])
        st.rerun()
    st.markdown("### 3.3 Outliers (review, do not change by default)")
    st.caption("Extreme values can be data entry errors or real patients. In research, changing them can bias your results, so the default is to keep every value. Check the source records and correct genuine errors there.")
    if nums:
        c = st.selectbox("Numeric variable to inspect", nums)
        lo, hi = se.iqr_bounds(df[c].dropna())
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.boxplot(df[c].dropna(), vert=False)
        st.pyplot(fig)
        plt.close(fig)
        rows = df[(df[c] < lo) | (df[c] > hi)]
        st.write(f"Values outside {lo:.1f} to {hi:.1f}: **{len(rows)}** (kept as they are)")
        if len(rows):
            st.dataframe(rows)
            with st.expander("Advanced and optional: cap these values"):
                st.warning("Capping replaces extreme values with the limits and changes your data. Use it only with a stated reason, and report it in your methods.")
                if st.checkbox("I understand and want to cap outliers in this variable", key=f"capok{c}") and st.button("Cap outliers to bounds"):
                    d, line = se.cap_outliers(df, c)
                    st.session_state.update(df=d, flash=line, log=st.session_state.log + [line])
                    st.rerun()
    st.markdown("### 3.4 Cleaning log")
    for line in st.session_state.log or ["No changes yet."]:
        st.write("• " + line)
    raw = st.session_state.raw
    st.dataframe(pd.DataFrame({"Original": [len(raw), int(raw.isna().sum().sum()), int(raw.duplicated().sum())],
                               "Now": [len(df), int(df.isna().sum().sum()), int(df.duplicated().sum())]}, index=["Total rows", "Missing values", "Duplicate rows"]))
    st.download_button("Download cleaned CSV", df.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")
    next_btn(i)

# ---------------- 4 Table 1 ----------------
elif i == 5:
    st.info("**What to do:** build Table 1, optionally split by a group such as treatment. **Tool and why:** mean (SD) for normal numeric data, median (IQR) for skewed data, n (%) for categories. With a group chosen, the app also picks and runs the right comparison test for each row.")
    g = st.selectbox("Split by (optional)", ["None"] + cats, format_func=lambda c: c if c == "None" else fmt(c))
    pick_t = st.multiselect("Variables to include (choose the ones you need)", [c for c in usable if c != g], default=[c for c in usable if c != g][:15], format_func=fmt)
    t = se.table_one(df[pick_t + ([] if g == "None" else [g])], None if g == "None" else g, alpha)
    st.dataframe(t)
    st.download_button("Download Table 1", t.to_csv(index=False).encode(), "table1.csv", "text/csv")
    next_btn(i)

# ---------------- 6 Test ----------------
# ---------------- 6 Test ----------------
elif i == 6:
    if st.session_state.objectives.strip():
        how = st.radio("How do you want to work?", ["Guided by my study objectives (recommended)", "I will choose the variables myself"])
    else:
        how = "I will choose the variables myself"
        st.info("Tip: enter your study objectives in Step 1 to get guided, objective by objective suggestions here.")
    guided() if how.startswith("Guided") else free_choice()
    next_btn(i)

elif i == 7:
    st.info("**What to do:** this step checks an analysis that was already done by someone else, such as a statistician, or in another program. Paste the methods and results text. The lab finds each test, the variables and the p values, re runs them on this data, and says whether each test was suitable, with better options and reasons. Use the same variable names as in your dataset.")
    way = st.radio("How do you want to enter the analysis?", ["Paste the analyst's methods and results", "Enter the tests one by one"], key="revway")
    TL = ["t test", "paired t test", "anova", "chi square", "fisher", "mann whitney", "wilcoxon", "kruskal", "pearson", "spearman"]
    if way.startswith("Paste"):
        txt = st.text_area("Methods and results text", height=220, key="revtext", placeholder="Example: The association between stroke_history and group was tested with the chi square test (p = 0.62). Mean sbp_followup was compared between group using Student's t test (p < 0.001).")
        st.caption("Tip: mention variables by their dataset names (or by their original question text if you renamed columns to codes). A sentence needs the test name, two variables and, if available, the p value.")
        if st.button("Review this analysis") and txt.strip():
            st.session_state.review = se.review_text(df, txt, alpha, st.session_state.codebook)
    else:
        cc = st.column_config
        base = pd.DataFrame({"Variable 1": [None], "Variable 2": [None], "Test used": [None], "Reported p value": [None], "Paired (same patients)": [False]})
        ed = st.data_editor(base, num_rows="dynamic", hide_index=True, key="review_in", column_config={
            "Variable 1": cc.SelectboxColumn(options=list(df.columns)), "Variable 2": cc.SelectboxColumn(options=list(df.columns)),
            "Test used": cc.SelectboxColumn(options=TL), "Reported p value": cc.NumberColumn(min_value=0.0, max_value=1.0, format="%.4f")})
        if st.button("Review these tests"):
            rows = [{"v1": r["Variable 1"], "v2": r["Variable 2"], "test": r["Test used"], "p": r["Reported p value"], "paired": r["Paired (same patients)"]} for _, r in ed.iterrows()]
            st.session_state.review = (se.review_rows(df, rows, alpha), [], [])
    rv = st.session_state.review
    if rv is not None:
        res, loose, notes = rv if isinstance(rv, tuple) else (rv, [], [])
        if len(res):
            st.dataframe(res.drop(columns=["Advice"]))
            icon = {"Appropriate": "✅", "Better option": "⚠️", "Could not check": "ℹ️"}
            for _, r in res.iterrows():
                st.markdown(f"{icon.get(r['Verdict'], '')} **{r['Variables']}: {r['Verdict']}.** " + (f"Lab recommends: {r['Lab recommends']}. " if r["Lab recommends"] else "") + str(r["Advice"]))
        else:
            st.warning("No complete test was found. A sentence needs a test name and at least two variables from your dataset.")
        if loose:
            st.markdown("**Tests found but variables not recognised** (check the variable names, or use the one by one option):")
            for sline in loose:
                st.write("• " + sline)
        for n in notes:
            st.info(n)
    next_btn(i)

# ---------------- 6 Report ----------------
else:
    if not st.session_state.results:
        st.info("No results saved yet. In Step 7, press 'Add this result to my report'.")
    tests = sorted({r["test"] for r in st.session_state.results})
    methods = ("Data were analysed in the Clinical DataLab using Python (pandas and SciPy). Normality was assessed with the Shapiro Wilk test. "
               "Continuous data are presented as mean (SD) or median (IQR) as appropriate, and categorical data as counts and percentages. "
               + (f"Tests used: {', '.join(tests)}. " if tests else "") + f"A two tailed p value below {alpha} was considered statistically significant.")
    out = ["STUDY DESIGN", st.session_state.design, "", "OBJECTIVES", st.session_state.objectives or "Not entered", "", "METHODS", methods, "", "DATA CLEANING"] + (st.session_state.log or ["No changes made."]) + ["", "RESULTS"]
    out += [f"{r['vars']}: {r['test']} ({r['line']}). {r['text']}" for r in st.session_state.results]
    st.text_area("Draft report (edit before use)", "\n".join(out), height=360)
    st.download_button("Download report", "\n".join(out).encode(), "report.txt", "text/plain")
