"""Run the full experiment and write tables, figures and the trained detector.

    python run_experiment.py                 # ten seeds, as in the paper
    python run_experiment.py --seeds 3       # quicker
    python run_experiment.py --out results --no-figures
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from advfuzz import plots
from advfuzz.baselines import LogisticDetector, entropy_detector, knn_distance_detector, msp_detector
from advfuzz.data import GREY_LEVELS
from advfuzz.detector import AdversarialDetector
from advfuzz.fuzzy import DESCRIPTORS, N_RULES, RULES, FuzzyInferenceSystem, rule_text, uniform_partition
from advfuzz.metrics import (
    best_f1_threshold,
    bootstrap_f1_difference,
    confusion,
    detection_metrics,
    mcnemar,
)
from advfuzz.pipeline import ExperimentConfig, build_detection_data
from advfuzz.tuning import tune_ga, tune_hybrid, tune_pso

METRICS = ("accuracy", "precision", "recall", "f1", "auc")
OPTIMISED = ("GA-FIS", "PSO-FIS", "Hybrid GA→PSO")

# Headline numbers reported in the paper, for side-by-side comparison.
PAPER = {
    "holdout_accuracy": 0.9750,
    "f1": {"MSP threshold": 0.728, "Entropy threshold": 0.722, "kNN-distance thr.": 0.846,
           "Logistic regression": 0.852, "Expert FIS (untuned)": 0.780, "GA-FIS": 0.841,
           "PSO-FIS": 0.828, "Hybrid GA→PSO": 0.841},
    "hybrid_recall": 0.904, "hybrid_auc": 0.887,
    "recall_4_16": {"hybrid": 0.833, "msp": 0.561},
    "mcnemar_p_msp": 5.5e-8, "mcnemar_p_expert": 0.017,
    "bootstrap_msp": (0.102, 0.066, 0.139), "bootstrap_expert": (0.049, 0.020, 0.080),
    "corr": {"C-U": -1.00, "C-P": -0.944, "C-F": -0.723},
}


def evaluate(y, pred, scores):
    return detection_metrics(y, pred, scores)


def summarise(per_seed):
    return {m: {"mean": float(np.mean([r[m] for r in per_seed])),
                "sd": float(np.std([r[m] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0}
            for m in per_seed[0]}


def run(cfg, out, figures=True, log=print):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    data = build_detection_data(cfg, log)
    tr, te = data.train, data.test
    Dtr, Dte, Ztr, Zte = data.D[tr], data.D[te], data.Z[tr], data.Z[te]
    ytr, yte = data.label[tr], data.label[te]

    methods, test_scores, test_preds = {}, {}, {}

    # ---------------------------------------------------------- baselines
    for name, det, Xa, Xb in [
        ("MSP threshold", msp_detector(), Dtr, Dte),
        ("Entropy threshold", entropy_detector(), Dtr, Dte),
        ("kNN-distance thr.", knn_distance_detector(), Dtr, Dte),
        ("Logistic regression", LogisticDetector(), Ztr, Zte),
    ]:
        det.fit(Xa, ytr)
        test_scores[name], test_preds[name] = det.score(Xb), det.predict(Xb)
        methods[name] = {"test": evaluate(yte, test_preds[name], test_scores[name]),
                         "train_f1": evaluate(ytr, det.predict(Xa), det.score(Xa))["f1"]}

    expert = FuzzyInferenceSystem()
    expert.theta = best_f1_threshold(expert.score(Ztr), ytr)  # only the operating point is fitted
    name = "Expert FIS (untuned)"
    test_scores[name], test_preds[name] = expert.score(Zte), expert.predict(Zte)
    methods[name] = {"test": evaluate(yte, test_preds[name], test_scores[name]),
                     "train_f1": evaluate(ytr, expert.predict(Ztr), expert.score(Ztr))["f1"],
                     "theta": expert.theta}

    # ---------------------------------------------------------- optimisers
    tuners = {
        "GA-FIS": lambda rng: tune_ga(Ztr, ytr, cfg.ga, rng),
        "PSO-FIS": lambda rng: tune_pso(Ztr, ytr, cfg.pso, rng),
        "Hybrid GA→PSO": lambda rng: tune_hybrid(
            Ztr, ytr, replace(cfg.ga, generations=cfg.hybrid_generations),
            replace(cfg.pso, iterations=cfg.hybrid_iterations), rng),
    }
    runs, histories = {}, {}
    for k, (name, tuner) in enumerate(tuners.items(), start=1):
        runs[name] = []
        for seed in range(cfg.n_seeds):
            r = tuner(np.random.default_rng([k, seed]))
            m = evaluate(yte, r.fis.predict(Zte), r.fis.score(Zte))
            runs[name].append((r, m))
        per_seed = [m for _, m in runs[name]]
        # The representative run is chosen on training fitness, never on test results.
        rep = int(np.argmax([r.train_f1 for r, _ in runs[name]]))
        rep_fis = runs[name][rep][0].fis
        test_scores[name], test_preds[name] = rep_fis.score(Zte), rep_fis.predict(Zte)
        histories[name] = [r.history for r, _ in runs[name]]
        methods[name] = {
            "test": summarise(per_seed),
            "train_f1": summarise([{"f1": r.train_f1} for r, _ in runs[name]])["f1"],
            "per_seed": per_seed,
            "representative_seed": rep,
            "representative": {"test": per_seed[rep], "fis": rep_fis.to_dict()},
        }
        log(f"{name:15s} test F1 {methods[name]['test']['f1']['mean']:.3f} ± {methods[name]['test']['f1']['sd']:.3f} "
            f"(train {methods[name]['train_f1']['mean']:.3f}) over {cfg.n_seeds} seeds")

    hyb_name = "Hybrid GA→PSO"
    hybrid = runs[hyb_name][methods[hyb_name]["representative_seed"]][0].fis
    ga_fis = runs["GA-FIS"][methods["GA-FIS"]["representative_seed"]][0].fis
    p_hyb, p_msp, p_exp = test_preds[hyb_name], test_preds["MSP threshold"], test_preds["Expert FIS (untuned)"]

    # ---------------------------------------------------------- statistics
    rng = np.random.default_rng([99, cfg.data_seed])
    stats = {}
    for other, p_other in [("MSP threshold", p_msp), ("Expert FIS (untuned)", p_exp)]:
        b, c, p = mcnemar(yte, p_hyb, p_other)
        mean, (lo, hi) = bootstrap_f1_difference(yte, p_hyb, p_other, cfg.n_bootstrap, rng)
        stats[other] = {"mcnemar_hybrid_better": b, "mcnemar_other_better": c, "mcnemar_p": p,
                        "bootstrap_mean_f1_gain": mean, "bootstrap_ci95": [lo, hi]}

    budget_rows = []
    for k in cfg.eps_levels:
        mask = data.eps_level[te] == k
        budget_rows.append({"eps_level": k, "n": int(mask.sum()),
                            "hybrid_recall": float(p_hyb[mask].mean()),
                            "msp_recall": float(p_msp[mask].mean())})

    cms = {n: confusion(yte, test_preds[n]) for n in ("MSP threshold", "Expert FIS (untuned)", hyb_name)}
    corr = np.corrcoef(Ztr.T)

    triage = hybrid.triage(Zte)
    triage_table = {cls: {t: int(np.sum(triage[yte == lab] == t)) for t in ("Safe", "Suspicious", "Adversarial")}
                    for cls, lab in (("clean", 0), ("adversarial", 1))}
    no_rule_fires = float(np.mean((hybrid.firing(Zte) * hybrid.weights).sum(axis=1) == 0))

    # A strong attack the hybrid catches but MSP misses, and a clean input it passes.
    te_idx = np.flatnonzero(te)
    caught = np.flatnonzero((yte == 1) & (p_hyb == 1) & (p_msp == 0) & (data.eps_level[te] == max(cfg.eps_levels)))
    passed = np.flatnonzero((yte == 0) & (p_hyb == 0))
    examples = []
    for label, pos in (("adversarial, eps=4/16, missed by MSP", caught), ("clean", passed)):
        if len(pos):
            i = pos[0]
            ex = hybrid.explain(Zte[i])
            ex.update(kind=label, record=int(te_idx[i]), descriptors=dict(zip(DESCRIPTORS, map(float, Dte[i]))))
            examples.append(ex)

    # ---------------------------------------------------------- detector
    detector = AdversarialDetector(data.model, data.feature_inconsistency, data.normaliser, hybrid,
                                   cfg.noise_draws, cfg.noise_sigma, cfg.noise_seed)
    detector.save(out / "detector.npz")
    reloaded = AdversarialDetector.load(out / "detector.npz")
    consistent = bool(np.allclose(reloaded.score(data.X[te]), test_scores[hyb_name]))

    results = {
        "config": {"n_seeds": cfg.n_seeds, "ga": vars(cfg.ga), "pso": vars(cfg.pso),
                   "hybrid_generations": cfg.hybrid_generations, "hybrid_iterations": cfg.hybrid_iterations,
                   "knn_k": cfg.knn_k, "noise_draws": cfg.noise_draws, "noise_sigma": cfg.noise_sigma,
                   "n_bootstrap": cfg.n_bootstrap},
        "classifier": data.classifier_stats,
        "attack": data.attack_table,
        "dataset": {"records": int(len(data.label)), "adversarial": int(data.label.sum()),
                    "clean": int((data.label == 0).sum()), "train": int(tr.sum()), "test": int(te.sum()),
                    "test_clean": int((yte == 0).sum()), "test_adversarial": int(yte.sum())},
        "correlation": {f"{a}-{b}": float(corr[i, j]) for i, a in enumerate(DESCRIPTORS)
                        for j, b in enumerate(DESCRIPTORS) if j > i},
        "methods": methods,
        "statistics": stats,
        "per_budget": budget_rows,
        "confusion": {n: cm.tolist() for n, cm in cms.items()},
        "triage": triage_table,
        "no_rule_fires_fraction": no_rule_fires,
        "reject_threshold": hybrid.reject_threshold(),
        "partition": {"expert": uniform_partition().tolist(), "ga_tuned": ga_fis.breakpoints.tolist()},
        "examples": examples,
        "detector_reload_consistent": consistent,
        "runtime_seconds": time.time() - t0,
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out / "report.md").write_text(render_report(results), encoding="utf-8")

    if figures:
        fig_dir = out / "figures"
        fig_dir.mkdir(exist_ok=True)
        plots.fig_examples(data, fig_dir / "fig2_examples.png")
        plots.fig_descriptors(data, fig_dir / "fig3_descriptors.png")
        plots.fig_partitions(uniform_partition(), ga_fis.breakpoints, fig_dir / "fig4_partitions.png")
        plots.fig_convergence({n: histories[n] for n in OPTIMISED}, fig_dir / "fig5_convergence.png",
                              switch=cfg.hybrid_generations)
        plots.fig_roc(yte, {n: test_scores[n] for n in ("MSP threshold", "kNN-distance thr.", "Logistic regression",
                                                        "Expert FIS (untuned)", hyb_name)},
                      fig_dir / "fig6_roc.png")
        plots.fig_budget(budget_rows, data.attack_table, fig_dir / "fig7_budget.png")
        plots.fig_confusion({n: (cms[n], f1_of(methods[n])) for n in cms}, fig_dir / "fig8_confusion.png")

    log(f"done in {time.time() - t0:.1f}s -> {out}/report.md")
    return results


def f1_of(method):
    rep = method.get("representative")
    return rep["test"]["f1"] if rep else method["test"]["f1"]


# ----------------------------------------------------------------- report
def fmt(v, digits=3):
    if isinstance(v, dict):
        return f"{v['mean']:.{digits}f} ± {v['sd']:.{digits}f}"
    return f"{v:.{digits}f}"


def render_report(R):
    L = []
    a = L.append
    c, d = R["classifier"], R["dataset"]
    a("# Experiment report\n")
    a(f"Optimised rows are mean ± s.d. over {R['config']['n_seeds']} seeds; every threshold, the normaliser "
      "and all optimisation are fitted on the detector training split only.\n")

    a("## Classifier and attack\n")
    a(f"MLP 64-64-32-10: training accuracy {c['train_accuracy']:.4f}, held-out accuracy "
      f"{c['holdout_accuracy']:.4f} ({c['n_train']} / {c['n_holdout']} images). "
      f"Attack pool (correctly classified held-out images): {c['pool_size']}.\n")
    a("**Table III - FGSM attack characterisation**\n")
    a("| ε | Success rate | Classifier acc. | Mean L₂ | Used | Off-grid pixels before quantisation |")
    a("|---|---|---|---|---|---|")
    for r in R["attack"]:
        a(f"| {r['eps_level']}/{GREY_LEVELS} | {r['success_rate']:.3f} | {r['classifier_accuracy']:.3f} | "
          f"{r['mean_l2']:.3f} | {r['used']} | {r['off_grid_pixel_fraction']:.4f} |")
    a("")
    if all(r["off_grid_pixel_fraction"] == 0 for r in R["attack"]):
        a("Quantisation moved no pixels: every budget is a multiple of 1/16, so x ± ε already lies on the "
          "grey-level grid. The step only changes images for budgets that are not multiples of 1/16.\n")
    a(f"Detection dataset: {d['adversarial']} adversarial + {d['clean']} clean = {d['records']} records; "
      f"grouped split {d['train']} train / {d['test']} test ({d['test_clean']} clean, {d['test_adversarial']} adversarial).\n")

    a("## Descriptor correlation (normalised, training split)\n")
    a("| pair | ρ |")
    a("|---|---|")
    for k, v in R["correlation"].items():
        a(f"| {k} | {v:.3f} |")
    a("")

    a("## Table V - detection performance on the test split\n")
    a("| Method | Acc. | Prec. | Rec. | F1 | AUC | train F1 |")
    a("|---|---|---|---|---|---|---|")
    for name, m in R["methods"].items():
        t = m["test"]
        a(f"| {name} | " + " | ".join(fmt(t[k]) for k in METRICS) + f" | {fmt(m['train_f1'])} |")
    a("")

    a("## Table VI - detection rate (recall) by attack strength\n")
    a("Representative hybrid run (best training F1) against the MSP baseline.\n")
    a("| ε | n | Hybrid GA→PSO | MSP baseline |")
    a("|---|---|---|---|")
    for r in R["per_budget"]:
        a(f"| {r['eps_level']}/{GREY_LEVELS} | {r['n']} | {r['hybrid_recall']:.3f} | {r['msp_recall']:.3f} |")
    a("")

    a("## Significance (representative hybrid run)\n")
    a("| Hybrid vs. | McNemar discordant (hybrid better / other better) | exact p | bootstrap ΔF1 | 95% CI |")
    a("|---|---|---|---|---|")
    for other, s in R["statistics"].items():
        a(f"| {other} | {s['mcnemar_hybrid_better']} / {s['mcnemar_other_better']} | {s['mcnemar_p']:.2e} | "
          f"{s['bootstrap_mean_f1_gain']:+.3f} | [{s['bootstrap_ci95'][0]:.3f}, {s['bootstrap_ci95'][1]:.3f}] |")
    a("")

    a("## Confusion matrices (rows = true clean / adversarial)\n")
    a("| Method | TN | FP | FN | TP |")
    a("|---|---|---|---|---|")
    for name, cm in R["confusion"].items():
        a(f"| {name} | {cm[0][0]} | {cm[0][1]} | {cm[1][0]} | {cm[1][1]} |")
    a("")

    hyb = R["methods"]["Hybrid GA→PSO"]["representative"]["fis"]
    a("## Table II - rule base with optimised weights (representative hybrid run)\n")
    a(f"Decision threshold θ = {hyb['theta']:.3f}.\n")
    a("| # | Rule | wᵣ |")
    a("|---|---|---|")
    for i in range(N_RULES):
        text = rule_text(i).split(": ", 1)[1]
        a(f"| R{i + 1} | {text} | {hyb['weights'][f'R{i + 1}']:.3f} |")
    a("")
    a(f"Fraction of test inputs where no rule fires (score defaults to 0.5): {R['no_rule_fires_fraction']:.4f}.\n")

    a("## GA-tuned membership partition (representative GA-FIS run)\n")
    a("| Descriptor | expert p₁, p₂, p₃ | GA-tuned p₁, p₂, p₃ |")
    a("|---|---|---|")
    for j, name in enumerate(DESCRIPTORS):
        e = ", ".join(f"{v:.2f}" for v in R["partition"]["expert"][j])
        g = ", ".join(f"{v:.2f}" for v in R["partition"]["ga_tuned"][j])
        a(f"| {name} | {e} | {g} |")
    a("")

    a("## Three-way triage on the test split (representative hybrid)\n")
    a(f"Safe: r < θ = {hyb['theta']:.3f} (accept). Suspicious: θ ≤ r < {R['reject_threshold']:.3f} (secondary "
      "screening). Adversarial: r ≥ the midpoint between θ and the Adversarial consequent 0.9 (reject).\n")
    a("| True class | Safe | Suspicious | Adversarial |")
    a("|---|---|---|---|")
    for cls, row in R["triage"].items():
        a(f"| {cls} | {row['Safe']} | {row['Suspicious']} | {row['Adversarial']} |")
    a("")

    a("## Example explanations\n")
    for ex in R["examples"]:
        desc = ", ".join(f"{k} = {v:.4g}" for k, v in ex["descriptors"].items())
        verdict = "ADVERSARIAL" if ex["adversarial"] else "clean"
        a(f"**{ex['kind']}** (record {ex['record']}): risk score {ex['score']:.3f} -> {verdict}. Raw descriptors: {desc}.\n")
        for rr in ex["rules"]:
            a(f"- {rr['rule']} - firing {rr['firing']:.2f}, weight {rr['weight']:.2f}, share {rr['share']:.0%}")
        a("")

    a("## Comparison with the paper\n")
    a("| Quantity | Paper | This run |")
    a("|---|---|---|")
    a(f"| Held-out classifier accuracy | {PAPER['holdout_accuracy']:.4f} | {c['holdout_accuracy']:.4f} |")
    for name, v in PAPER["f1"].items():
        a(f"| F1, {name} | {v:.3f} | {fmt(R['methods'][name]['test']['f1'])} |")
    h = R["methods"]["Hybrid GA→PSO"]["test"]
    a(f"| Hybrid recall | {PAPER['hybrid_recall']:.3f} | {fmt(h['recall'])} |")
    a(f"| Hybrid AUC | {PAPER['hybrid_auc']:.3f} | {fmt(h['auc'])} |")
    last = R["per_budget"][-1]
    a(f"| Recall at ε = 4/16, hybrid / MSP | {PAPER['recall_4_16']['hybrid']:.3f} / {PAPER['recall_4_16']['msp']:.3f} | "
      f"{last['hybrid_recall']:.3f} / {last['msp_recall']:.3f} |")
    s_msp, s_exp = R["statistics"]["MSP threshold"], R["statistics"]["Expert FIS (untuned)"]
    a(f"| McNemar p, hybrid vs. MSP | {PAPER['mcnemar_p_msp']:.1e} | {s_msp['mcnemar_p']:.1e} |")
    a(f"| McNemar p, hybrid vs. expert FIS | {PAPER['mcnemar_p_expert']:.3f} | {s_exp['mcnemar_p']:.3f} |")
    for key, s in (("bootstrap_msp", s_msp), ("bootstrap_expert", s_exp)):
        p = PAPER[key]
        a(f"| Bootstrap ΔF1 {'vs. MSP' if key.endswith('msp') else 'vs. expert'} | {p[0]:+.3f} [{p[1]:.3f}, {p[2]:.3f}] | "
          f"{s['bootstrap_mean_f1_gain']:+.3f} [{s['bootstrap_ci95'][0]:.3f}, {s['bootstrap_ci95'][1]:.3f}] |")
    for k, v in PAPER["corr"].items():
        a(f"| ρ({k}) | {v:.3f} | {R['correlation'][k]:.3f} |")
    a("")
    a(f"Saved detector reloads and reproduces the test scores exactly: {R['detector_reload_consistent']}. "
      f"Runtime {R['runtime_seconds']:.0f} s.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="independent optimiser runs per configuration")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--data-seed", type=int, default=0, help="seed for the classifier, attack sampling and split")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    cfg = ExperimentConfig(data_seed=args.data_seed, n_seeds=args.seeds, n_bootstrap=args.bootstrap)
    run(cfg, args.out, figures=not args.no_figures)


if __name__ == "__main__":
    main()
