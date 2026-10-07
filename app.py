"""Interactive demo of the GA/PSO-tuned fuzzy adversarial-input detector.

    streamlit run app.py

Everything shown is computed by the code in `advfuzz/`: the live demo attacks
real test images with FGSM and runs the saved detector on them, and the
results tab can rerun the whole experiment from scratch.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from advfuzz import plots
from advfuzz.attack import fgsm
from advfuzz.baselines import msp_detector
from advfuzz.data import GREY_LEVELS, IMAGE_SHAPE
from advfuzz.detector import AdversarialDetector
from advfuzz.fuzzy import ADVERSARIAL, DESCRIPTORS, RULES, SAFE, SUSPICIOUS, TERMS, memberships
from advfuzz.pipeline import ExperimentConfig, build_detection_data
from run_experiment import METRICS, fmt, run

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "ui_runs"

STATUS = {"Safe": "#0ca30c", "Suspicious": "#fab219", "Adversarial": "#d03b3b"}
STATUS_TEXT = {"Safe": "green", "Suspicious": "orange", "Adversarial": "red"}
CONSEQUENT = {SAFE: "Safe", SUSPICIOUS: "Suspicious", ADVERSARIAL: "Adversarial"}
DESCRIPTOR_NAMES = {"C": "Confidence", "U": "Uncertainty", "P": "Perturbation response", "F": "Feature inconsistency"}
EPS_LABELS = {0: "no attack", 1: "1/16", 2: "2/16", 3: "3/16", 4: "4/16"}

st.set_page_config(page_title="Adversarial Input Detector", layout="wide")


# ------------------------------------------------------------------ loading
def result_sources():
    dirs = [ROOT / "results"] + sorted(p for p in RUNS_DIR.glob("*") if p.is_dir())
    return [d for d in dirs if (d / "metrics.json").exists() and (d / "detector.npz").exists()]


def source_label(d):
    return "results/  (saved run)" if d.name == "results" else f"ui_runs/{d.name}"


@st.cache_resource(show_spinner="Rebuilding the detection dataset: training the classifier and attacking it…")
def load_world(results_dir, stamp):
    results_dir = Path(results_dir)
    det = AdversarialDetector.load(results_dir / "detector.npz")
    metrics = json.loads((results_dir / "metrics.json").read_text())
    data = build_detection_data(ExperimentConfig(data_seed=metrics["config"].get("data_seed", 0)), log=lambda *a: None)
    msp = msp_detector().fit(data.D[data.train], data.label[data.train])
    same_model = all(np.allclose(a, b) for a, b in zip(det.model.W, data.model.W))
    return det, metrics, data, msp, same_model


# ------------------------------------------------------------------ drawing
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def pixels(x, scale=22):
    """8x8 image in [0, 1] -> crisp upscaled greyscale array (ink is dark)."""
    img = (255 * (1.0 - x.reshape(IMAGE_SHAPE))).astype(np.uint8)
    return np.kron(img, np.ones((scale, scale), dtype=np.uint8))


def perturbation_pixels(delta, eps, scale=22):
    unit = np.full(IMAGE_SHAPE, 0.5) if eps == 0 else (delta.reshape(IMAGE_SHAPE) / eps + 1) / 2
    rgb = (plots.DIVERGING(unit)[..., :3] * 255).astype(np.uint8)
    return np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8))


def short_rule(rule):
    i = int(rule.split(":")[0][1:]) - 1
    ante, cons = RULES[i]
    return f"R{i + 1}  " + " ∧ ".join(f"{d} {t}" for d, t in ante.items()) + f"  →  {CONSEQUENT[cons]}"


def rule_bars(rules):
    rules = rules[::-1]
    with plt.rc_context(plots.RC):
        fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(rules) + 0.7))
        shares = [r["share"] for r in rules]
        colors = [STATUS[r["rule"].rsplit("THEN ", 1)[1]] for r in rules]
        ax.barh(range(len(rules)), shares, color=colors, height=0.62)
        for k, (s, r) in enumerate(zip(shares, rules)):
            ax.text(s + 0.01, k, f"{s:.0%}   (firing {r['firing']:.2f} × weight {r['weight']:.2f})",
                    va="center", fontsize=8, color=plots.TEXT2)
        ax.set_yticks(range(len(rules)), [short_rule(r["rule"]) for r in rules], fontsize=8.5)
        ax.set_xlim(0, max(shares) + 0.45)
        ax.set_xlabel("share of the risk score")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
    return fig


def score_strip(score, theta, reject):
    with plt.rc_context(plots.RC):
        fig, ax = plt.subplots(figsize=(6.2, 1.15))
        for lo, hi, name in ((SAFE, theta, "Safe"), (theta, reject, "Suspicious"), (reject, ADVERSARIAL, "Adversarial")):
            ax.axvspan(lo, hi, color=STATUS[name], alpha=0.16, linewidth=0)
            ax.text((lo + hi) / 2, 0.86, name, ha="center", va="center", fontsize=8.5, color=plots.TEXT2)
        ax.plot([score, score], [0.0, 0.5], color=plots.TEXT, linewidth=2)
        ax.plot([score], [0.5], marker="v", markersize=9, color=plots.TEXT)
        ax.set_xlim(SAFE, ADVERSARIAL)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([SAFE, theta, reject, ADVERSARIAL],
                      ["0.10", f"θ = {theta:.2f}", f"{reject:.2f}", "0.90"], fontsize=8)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        fig.tight_layout()
    return fig


def membership_panels(fis, z):
    grid = np.linspace(0, 1, 401)
    M = memberships(np.repeat(grid[:, None], 4, axis=1), fis.breakpoints)
    at = memberships(np.atleast_2d(z), fis.breakpoints)[0]
    with plt.rc_context(plots.RC):
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.3), sharey=True)
        for j, (ax, d) in enumerate(zip(axes, DESCRIPTORS)):
            for t, term in enumerate(TERMS):
                ax.plot(grid, M[:, j, t], color=plots.TERM_COLORS[t], label=term, linewidth=1.4)
            ax.axvline(z[j], color=plots.TEXT, linewidth=1.2, linestyle="--")
            degrees = ", ".join(f"{term} {at[j, t]:.2f}" for t, term in enumerate(TERMS) if at[j, t] > 0)
            ax.set_title(f"{d}: {degrees}", fontsize=8.5, loc="left")
            ax.set_xlim(0, 1)
            ax.set_yticks([0, 0.5, 1])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def descriptor_table(det, D_list, names):
    rows = []
    for j, d in enumerate(DESCRIPTORS):
        row = {"Descriptor": f"{d}  {DESCRIPTOR_NAMES[d]}"}
        for D, name in zip(D_list, names):
            z = det.normaliser.transform(D)[0]
            mu = memberships(z[None], det.fis.breakpoints)[0, j]
            t = int(np.argmax(mu))
            row[f"{name}: raw"] = f"{D[0, j]:.4g}"
            row[f"{name}: percentile"] = f"{z[j]:.2f}"
            row[f"{name}: fuzzy term"] = f"{TERMS[t]} ({mu[t]:.2f})"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- live demo
def demo_tab(det, data, msp):
    st.markdown("Pick a digit the detector has never seen during tuning, attack it with FGSM, and compare the "
                "fuzzy detector with the confidence-only baseline.")
    test_clean = np.flatnonzero(data.test & (data.eps_level == 0))
    labels = data.pool_labels[test_clean]

    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    digit = c1.selectbox("True digit", ["any"] + list(range(10)))
    pool = test_clean if digit == "any" else test_clean[labels == digit]
    if st.session_state.get("pick") not in set(pool.tolist()):
        st.session_state.pick = int(pool[0])
    if c2.button("Random test image", width="stretch"):
        st.session_state.pick = int(np.random.default_rng().choice(pool))
    level = c3.select_slider("FGSM attack budget ε", options=list(EPS_LABELS), value=4, format_func=EPS_LABELS.get)

    i = st.session_state.pick
    x, y = data.X[i], int(data.pool_labels[i])
    eps = level / GREY_LEVELS
    x_adv = fgsm(det.model, x[None], np.array([y]), eps)[0] if level else x

    p_clean, p_adv = det.model.predict_proba(np.stack([x, x_adv]))
    pred_clean, pred_adv = int(p_clean.argmax()), int(p_adv.argmax())
    D_clean, D_adv = det.descriptors(x[None]), det.descriptors(x_adv[None])
    Z_adv = det.normaliser.transform(D_adv)
    ex = det.explain(x_adv, top=8)
    triage = det.fis.triage(Z_adv)[0]
    triage_clean = det.fis.triage(det.normaliser.transform(D_clean))[0]
    score_clean = float(det.fis.score(det.normaliser.transform(D_clean))[0])
    msp_flag = bool(msp.predict(D_adv)[0])
    fooled = pred_adv != y

    st.divider()
    a, b, c, v = st.columns([1, 1, 1, 2.4])
    with a:
        st.image(pixels(x), width=176)
        st.markdown(f"**Original** · true digit **{y}**")
        st.caption(f"classifier: {pred_clean} ({p_clean.max():.1%})")
    with b:
        st.image(perturbation_pixels(x_adv - x, eps), width=176)
        st.markdown(f"**Perturbation** · ε = {EPS_LABELS[level]}")
        st.caption("red = darkened, blue = lightened" if level else "no attack applied")
    with c:
        st.image(pixels(x_adv), width=176)
        st.markdown("**Input to the classifier**")
        verdict = f":red[fooled → {pred_adv}]" if fooled else f":green[still {pred_adv}]"
        st.caption(f"classifier: {verdict} ({p_adv.max():.1%})")
    with v:
        left, right = st.columns(2)
        with left:
            st.markdown("**Fuzzy detector (GA→PSO)**")
            st.markdown(f"### :{STATUS_TEXT[triage]}[{triage}]")
            st.caption(f"risk score {ex['score']:.3f}" + (f" · original scored {score_clean:.3f} ({triage_clean})" if level else ""))
        with right:
            st.markdown("**Confidence-only baseline (MSP)**")
            st.markdown(f"### :{'red' if msp_flag else 'green'}[{'Flagged' if msp_flag else 'Not flagged'}]")
            st.caption(f"confidence {D_adv[0, 0]:.4f} · flags when ≤ {1 - msp.threshold:.4f}")
        show(score_strip(ex["score"], det.fis.theta, det.fis.reject_threshold()))

    if level and fooled and ex["adversarial"] and not msp_flag:
        st.info("The attack fooled the classifier **and** left it confident, so the confidence baseline sees nothing "
                "wrong. The fuzzy detector still flags the input because its internal representation is atypical for "
                "the predicted class (high F), which is exactly what rule R9 captures.")
    elif level and fooled and not ex["adversarial"]:
        st.warning("This attack gets past the fuzzy detector. Try a different image or budget: detection is "
                   "statistical, not guaranteed (test recall is about 0.87).")
    elif level and not fooled:
        st.caption("The attack did not change the classifier's answer at this budget; such inputs are not part of the "
                   "detection dataset, but the detector still scores them.")

    st.divider()
    t1, t2 = st.columns([1.1, 1])
    with t1:
        st.markdown("**Descriptors**: percentiles are relative to the detector's training inputs")
        names = ["Original", "Attacked"] if level else ["Input"]
        st.dataframe(descriptor_table(det, [D_clean, D_adv] if level else [D_clean], names), hide_index=True)
    with t2:
        st.markdown("**Why: rules behind the decision**")
        if ex["rules"]:
            show(rule_bars(ex["rules"]))
        else:
            st.caption("No rule fires for this input; the score defaults to 0.5.")


# ----------------------------------------------------------- fuzzy playground
PRESETS = {
    "Typical clean input": (0.75, 0.25, 0.20, 0.20),
    "Weak attack: low confidence": (0.10, 0.90, 0.90, 0.60),
    "Strong attack: confident but atypical": (0.80, 0.20, 0.20, 0.95),
}


def set_sliders(values):
    for d, v in zip(DESCRIPTORS, values):
        st.session_state[f"pg_{d}"] = v


def playground_tab(det):
    st.markdown("Set the four **normalised** descriptors directly and watch the fuzzy system reason. Each value is a "
                "percentile among the detector's training inputs. In real data U moves almost exactly opposite to C, "
                "so some combinations never occur.")
    for d, v in zip(DESCRIPTORS, PRESETS["Strong attack: confident but atypical"]):
        st.session_state.setdefault(f"pg_{d}", v)
    cols = st.columns(len(PRESETS))
    for col, (name, values) in zip(cols, PRESETS.items()):
        col.button(name, on_click=set_sliders, args=(values,), width="stretch")

    left, right = st.columns([1, 1.4])
    with left:
        z = np.array([st.slider(f"{d}  {DESCRIPTOR_NAMES[d]}", 0.0, 1.0, step=0.01, key=f"pg_{d}")
                      for d in DESCRIPTORS])
    ex = det.fis.explain(z, top=8)
    triage = det.fis.triage(z[None])[0]
    with right:
        st.markdown(f"### :{STATUS_TEXT[triage]}[{triage}] · risk score {ex['score']:.3f}")
        show(score_strip(ex["score"], det.fis.theta, det.fis.reject_threshold()))
        if ex["rules"]:
            show(rule_bars(ex["rules"]))
        else:
            st.caption("No rule fires for this combination; the score defaults to 0.5.")
    st.markdown("**Membership degrees** under the tuned partition (dashed line = current value)")
    show(membership_panels(det.fis, z))


# ------------------------------------------------------------------ results
def section(report, title):
    """Body of one '## ...' section of report.md."""
    parts = report.split("\n## ")
    for p in parts:
        if p.startswith(title):
            return p.split("\n", 1)[1]
    return ""


def results_tab(metrics, results_dir):
    m = metrics["methods"]
    hyb, base = m["Hybrid GA→PSO"]["test"], m["MSP threshold"]["test"]
    s = metrics["statistics"]["MSP threshold"]
    cfg = metrics["config"]
    st.caption(f"Source: `{results_dir.relative_to(ROOT)}/` · data seed {cfg.get('data_seed', 0)} · "
               f"{cfg['n_seeds']} optimiser seeds · test split of {metrics['dataset']['test']} records")

    k = st.columns(4)
    k[0].metric("Hybrid GA→PSO F1", fmt(hyb["f1"]), border=True)
    k[1].metric("Confidence baseline F1", fmt(base["f1"]), border=True)
    lo, hi = s["bootstrap_ci95"]
    k[2].metric("F1 gain over baseline", f"{s['bootstrap_mean_f1_gain']:+.3f}", border=True,
                help=f"Bootstrap mean over {cfg['n_bootstrap']} resamples; 95% CI [{lo:.3f}, {hi:.3f}]")
    k[3].metric("McNemar p vs baseline", f"{s['mcnemar_p']:.1e}", border=True,
                help=f"{s['mcnemar_hybrid_better']} test inputs only the hybrid gets right, "
                     f"{s['mcnemar_other_better']} only the baseline gets right")

    st.subheader("Detection performance on the test split")
    rows = [{"Method": name, **{k_.upper() if k_ == "auc" else k_.capitalize(): fmt(v["test"][k_]) for k_ in METRICS},
             "Train F1": fmt(v["train_f1"])} for name, v in m.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Optimised rows are mean ± s.d. over optimiser seeds. Every threshold is fitted on the training split.")

    fig_dir = results_dir / "figures"
    a, b = st.columns([1, 1.3])
    with a:
        st.subheader("Detection rate by attack strength")
        st.dataframe(pd.DataFrame([{"ε": f"{r['eps_level']}/16", "n": r["n"], "Hybrid": f"{r['hybrid_recall']:.3f}",
                                    "MSP baseline": f"{r['msp_recall']:.3f}"} for r in metrics["per_budget"]]),
                     hide_index=True)
        st.caption("Representative hybrid run: the seed with the best training F1.")
    with b:
        if (fig_dir / "fig7_budget.png").exists():
            st.image(str(fig_dir / "fig7_budget.png"))

    st.subheader("Figures")
    figures = {"ROC curves": "fig6_roc.png", "Confusion matrices": "fig8_confusion.png",
               "Convergence": "fig5_convergence.png", "Membership partitions": "fig4_partitions.png",
               "Descriptors": "fig3_descriptors.png", "Attack examples": "fig2_examples.png"}
    for tab, file in zip(st.tabs(list(figures)), figures.values()):
        with tab:
            if (fig_dir / file).exists():
                st.image(str(fig_dir / file), width=900)
            else:
                st.caption("Figure not generated for this run.")

    report = (results_dir / "report.md").read_text(encoding="utf-8")
    with st.expander("Comparison with the paper"):
        st.markdown(section(report, "Comparison with the paper"))
    with st.expander("Rule base with optimised weights"):
        st.markdown(section(report, "Table II"))
    with st.expander("Full report"):
        st.markdown(report)

    with st.expander("Rerun the experiment live", expanded=False):
        st.markdown("Trains the classifier from scratch, attacks it, tunes the fuzzy system with the GA and PSO, and "
                    "recomputes every number and figure. Results go to `ui_runs/` and never overwrite `results/`.")
        c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
        seed = c1.number_input("Data seed", min_value=0, max_value=9999, value=1)
        n = c2.number_input("Optimiser seeds", min_value=1, max_value=10, value=3)
        if c3.button("Run experiment", type="primary", width="stretch"):
            out = RUNS_DIR / f"data{seed}_seeds{n}"
            with st.spinner("Running the full pipeline…"):
                run(ExperimentConfig(data_seed=int(seed), n_seeds=int(n)), out, figures=True, log=lambda *a: None)
            st.session_state.pending_source = source_label(out)
            st.rerun()


# --------------------------------------------------------------------- page
def main():
    sources = result_sources()
    with st.sidebar:
        st.title("Adversarial input detector")
        st.caption("Fuzzy inference over four inference-time descriptors, with the membership partition tuned by a "
                   "genetic algorithm and the rule weights by particle swarm optimisation.")
        if not sources:
            st.error("No results yet. Run `python run_experiment.py` first.")
            st.stop()
        labels = {source_label(d): d for d in sources}
        if "pending_source" in st.session_state:
            st.session_state.source = st.session_state.pop("pending_source")
        choice = st.selectbox("Results source", list(labels), key="source")
        results_dir = labels[choice]

        det, metrics, data, msp, same_model = load_world(str(results_dir), (results_dir / "detector.npz").stat().st_mtime)
        st.divider()
        st.markdown("**Pipeline**")
        st.markdown("1. Classifier predicts\n2. Descriptors C, U, P, F\n3. Quantile normalisation\n"
                    "4. 15-rule fuzzy inference\n5. Risk score → triage")
        st.markdown("**Detector settings**")
        st.caption(f"flag threshold θ = {det.fis.theta:.3f}  \nreject threshold = {det.fis.reject_threshold():.3f}")
    if not same_model:
        st.warning("The saved detector's classifier differs from the one rebuilt for this data seed, so demo images "
                   "may not match its training split. Rerun the experiment to refresh it.")

    st.title("Adversarial attack detection with genetic algorithms and fuzzy logic")
    demo, playground, results = st.tabs(["Live demo", "Fuzzy playground", "Results"])
    with demo:
        demo_tab(det, data, msp)
    with playground:
        playground_tab(det)
    with results:
        results_tab(metrics, results_dir)


main()
