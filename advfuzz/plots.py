"""Figures 2-8 of the paper, rendered with matplotlib."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from .data import GREY_LEVELS, IMAGE_SHAPE  # noqa: E402
from .fuzzy import DESCRIPTORS, TERMS, memberships  # noqa: E402
from .metrics import roc_auc, roc_curve  # noqa: E402

# Chart chrome and ink.
SURFACE, TEXT, TEXT2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Each method keeps one colour in every figure. Slots follow the validated
# categorical order so that the series drawn together stay distinguishable
# under colour-vision deficiency.
SERIES = {
    "Hybrid GA→PSO": "#2a78d6",
    "MSP threshold": "#eb6834",
    "kNN-distance thr.": "#1baf7a",
    "Logistic regression": "#eda100",
    "Expert FIS (untuned)": "#e87ba4",
    "GA-FIS": "#008300",
    "PSO-FIS": "#4a3aa7",
    "Entropy threshold": "#e34948",
}
CLEAN, ADVERSARIAL = "#2a78d6", "#eb6834"
TERM_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # Low, Medium, High

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a6a5", "#e34948", "#9c2526"]
)

DESCRIPTOR_TITLES = {
    "C": "C  confidence",
    "U": "U  uncertainty",
    "P": "P  perturbation response",
    "F": "F  feature inconsistency",
}

RC = {
    # Listing families directly lets matplotlib fall back per glyph (e.g. for "→").
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": TEXT2,
    "axes.titlecolor": TEXT,
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": TEXT2,
    "ytick.labelcolor": TEXT2,
    "legend.frameon": False,
    "legend.labelcolor": TEXT2,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _eps_label(level):
    return f"{level}/{GREY_LEVELS}"


# ------------------------------------------------------------------- Fig. 2
def fig_examples(data, path, eps_level=3, n=5):
    """Clean images, FGSM perturbations and quantised adversarial images."""
    picked, seen = [], set()
    for i in np.flatnonzero(data.eps_level == eps_level):
        if data.pool_labels[i] not in seen:
            picked.append(i)
            seen.add(data.pool_labels[i])
        if len(picked) == n:
            break
    preds = data.model.predict(data.X[picked])
    eps = eps_level / GREY_LEVELS

    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, n, figsize=(1.55 * n + 0.6, 5.0))
        for col, (i, pred) in enumerate(zip(picked, preds)):
            clean, adv = data.X_clean[i], data.X[i]
            axes[0, col].imshow(clean.reshape(IMAGE_SHAPE), cmap="gray_r", vmin=0, vmax=1)
            axes[1, col].imshow((adv - clean).reshape(IMAGE_SHAPE), cmap=DIVERGING, vmin=-eps, vmax=eps)
            axes[2, col].imshow(adv.reshape(IMAGE_SHAPE), cmap="gray_r", vmin=0, vmax=1)
            axes[0, col].set_title(f"true '{data.pool_labels[i]}'", fontsize=9)
            axes[2, col].set_xlabel(f"predicted '{pred}'", color=TEXT, fontsize=9)
        for row, name in enumerate(["clean", "perturbation δ", "adversarial"]):
            axes[row, 0].set_ylabel(name, fontsize=9)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(GRID)
        fig.suptitle(f"FGSM at ε = {_eps_label(eps_level)}: δ is red where a pixel was darkened, blue where lightened",
                     fontsize=9, color=TEXT2, y=0.995)
        fig.tight_layout()
        _save(fig, path)


# ------------------------------------------------------------------- Fig. 3
def fig_descriptors(data, path):
    """Raw descriptor distributions and correlation of normalised descriptors."""
    tr = data.train
    D, Z, y = data.D[tr], data.Z[tr], data.label[tr]
    corr = np.corrcoef(Z.T)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(10, 7.4))
        gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.25], hspace=0.45, wspace=0.45)
        for j, name in enumerate(DESCRIPTORS):
            ax = fig.add_subplot(gs[0, j])
            bp = ax.boxplot(
                [D[y == 0, j], D[y == 1, j]], widths=0.55, patch_artist=True,
                medianprops={"color": TEXT, "linewidth": 1.4},
                whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "none",
                            "markeredgecolor": MUTED, "alpha": 0.5},
            )
            for patch, color in zip(bp["boxes"], (CLEAN, ADVERSARIAL)):
                patch.set_facecolor(color + "55")
                patch.set_edgecolor(color)
            ax.set_xticks([1, 2], ["clean", "adversarial"])
            ax.set_title(DESCRIPTOR_TITLES[name], fontsize=9.5, loc="left")
            ax.grid(axis="x", visible=False)

        ax = fig.add_subplot(gs[1, 1:3])
        im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(4), DESCRIPTORS)
        ax.set_yticks(range(4), DESCRIPTORS)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        for a in range(4):
            for b in range(4):
                ax.text(b, a, f"{corr[a, b]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(corr[a, b]) > 0.6 else TEXT)
        ax.set_title("Pearson correlation of quantile-normalised descriptors (training split)", fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=TEXT2)
        _save(fig, path)


# ------------------------------------------------------------------- Fig. 4
def fig_partitions(expert_bp, tuned_bp, path):
    """Membership partitions before and after GA tuning."""
    z = np.linspace(0, 1, 801)
    Zgrid = np.repeat(z[:, None], len(DESCRIPTORS), axis=1)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(4, 2, figsize=(8, 8.6), sharex=True, sharey=True)
        for col, (bp, title) in enumerate([(expert_bp, "expert partition"), (tuned_bp, "GA-tuned partition")]):
            M = memberships(Zgrid, bp)
            for j, name in enumerate(DESCRIPTORS):
                ax = axes[j, col]
                for t, term in enumerate(TERMS):
                    ax.plot(z, M[:, j, t], color=TERM_COLORS[t], label=term)
                ax.set_title("p = " + ", ".join(f"{p:.2f}" for p in bp[j]), loc="right", fontsize=8, color=TEXT2)
                ax.set_ylim(-0.03, 1.05)
                ax.set_yticks([0, 0.5, 1])
                if col == 0:
                    ax.set_ylabel(f"{name}   μ")
            axes[0, col].set_title(title, loc="left", fontsize=10)
            axes[-1, col].set_xlabel("normalised descriptor value")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        _save(fig, path)


# ------------------------------------------------------------------- Fig. 5
def fig_convergence(histories, path, switch=40):
    """Best training F1 against generation / iteration, mean over seeds."""
    offsets = {"GA-FIS": ("bottom", 0.0012), "PSO-FIS": ("top", -0.0012), "Hybrid GA→PSO": ("top", -0.0012)}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for name, H in histories.items():
            H = np.asarray(H)
            mean = H.mean(axis=0)
            x = np.arange(len(mean))
            ax.plot(x, mean, color=SERIES[name], label=name)
            if name == "GA-FIS" and len(H) > 1:
                sd = H.std(axis=0, ddof=1)
                ax.fill_between(x, mean - sd, mean + sd, color=SERIES[name], alpha=0.12, linewidth=0)
            va, dy = offsets.get(name, ("bottom", 0.001))
            ax.text(x[-1], mean[-1] + dy, f"{name}  {mean[-1]:.3f}", ha="right", va=va, fontsize=8.5, color=TEXT2)
        ax.axvline(switch, color=MUTED, linestyle=":", linewidth=1)
        ax.text(switch + 0.8, ax.get_ylim()[0], "hybrid: PSO stage", color=MUTED, fontsize=8, va="bottom")
        ax.set_xlabel("generation / iteration")
        ax.set_ylabel("best training F1")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06))
        fig.tight_layout()
        _save(fig, path)


# ------------------------------------------------------------------- Fig. 6
def fig_roc(y, scores, path):
    """ROC curves on the test split."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        ax.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1)
        for name, s in scores.items():
            fpr, tpr = roc_curve(y, s)
            ax.plot(fpr, tpr, color=SERIES[name], label=f"{name}  (AUC {roc_auc(y, s):.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Entropy threshold omitted: its ranking equals MSP's", fontsize=8.5, color=TEXT2, loc="left")
        fig.tight_layout()
        _save(fig, path)


# ------------------------------------------------------------------- Fig. 7
def fig_budget(budget_rows, attack_table, path):
    """Detection rate against attack strength, and the attack's effect on the classifier."""
    levels = [r["eps_level"] for r in budget_rows]
    eps = np.array(levels) / GREY_LEVELS
    with plt.rc_context(RC):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True,
                                          gridspec_kw={"height_ratios": [1.15, 1.0]})
        series = [("Hybrid GA→PSO", "hybrid_recall", "o", "-"), ("MSP threshold", "msp_recall", "s", "--")]
        for name, key, marker, ls in series:
            vals = [r[key] for r in budget_rows]
            top.plot(eps, vals, color=SERIES[name], marker=marker, linestyle=ls, label=name,
                     markeredgecolor=SURFACE, markeredgewidth=1.2)
            top.annotate(f"{vals[-1]:.3f}", (eps[-1], vals[-1]), xytext=(8, 0), textcoords="offset points",
                         va="center", fontsize=8.5, color=TEXT)
        top.set_ylabel("detection rate (recall)")
        top.legend(loc="lower left")

        acc = [r["classifier_accuracy"] for r in attack_table]
        succ = [r["success_rate"] for r in attack_table]
        bottom.plot(eps, acc, color=TEXT2, marker="^", label="classifier accuracy",
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        bottom.plot(eps, succ, color=MUTED, marker="v", linestyle="--", label="attack success rate",
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        bottom.text(eps[0], acc[0] + 0.06, "classifier accuracy", fontsize=8.5, color=TEXT2)
        bottom.text(eps[0], succ[0] - 0.1, "attack success rate", fontsize=8.5, color=TEXT2, va="top")
        bottom.set_ylim(-0.05, 1.08)
        bottom.set_ylabel("rate on the attack pool")
        bottom.set_xticks(eps, [f"ε = {_eps_label(k)}" for k in levels])
        bottom.set_xlabel("attack strength")
        bottom.set_xlim(eps[0] - 0.015, eps[-1] + 0.03)
        fig.tight_layout()
        _save(fig, path)


# ------------------------------------------------------------------- Fig. 8
def fig_confusion(matrices, path):
    """Test-set confusion matrices, rows = true class."""
    vmax = max(cm.max() for cm, _ in matrices.values())
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(matrices), figsize=(3.0 * len(matrices), 3.3))
        for ax, (name, (cm, f1)) in zip(np.atleast_1d(axes), matrices.items()):
            ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=vmax)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, str(cm[a, b]), ha="center", va="center", fontsize=11,
                            color="white" if cm[a, b] > 0.5 * vmax else TEXT)
            ax.set_xticks([0, 1], ["clean", "adversarial"])
            ax.set_yticks([0, 1], ["clean", "adversarial"])
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{name}\nF1 = {f1:.3f}", fontsize=9.5)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        fig.tight_layout()
        _save(fig, path)
