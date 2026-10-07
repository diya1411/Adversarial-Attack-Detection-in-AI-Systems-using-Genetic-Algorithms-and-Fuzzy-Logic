# Experiment report

Optimised rows are mean ± s.d. over 10 seeds; every threshold, the normaliser and all optimisation are fitted on the detector training split only.

## Classifier and attack

MLP 64-64-32-10: training accuracy 1.0000, held-out accuracy 0.9847 (1078 / 719 images). Attack pool (correctly classified held-out images): 708.

**Table III - FGSM attack characterisation**

| ε | Success rate | Classifier acc. | Mean L₂ | Used | Off-grid pixels before quantisation |
|---|---|---|---|---|---|
| 1/16 | 0.225 | 0.775 | 0.431 | 159 | 0.0000 |
| 2/16 | 0.767 | 0.233 | 0.846 | 159 | 0.0000 |
| 3/16 | 0.984 | 0.016 | 1.251 | 159 | 0.0000 |
| 4/16 | 1.000 | 0.000 | 1.645 | 159 | 0.0000 |

Quantisation moved no pixels: every budget is a multiple of 1/16, so x ± ε already lies on the grey-level grid. The step only changes images for budgets that are not multiples of 1/16.

Detection dataset: 636 adversarial + 708 clean = 1344 records; grouped split 817 train / 527 test (284 clean, 243 adversarial).

## Descriptor correlation (normalised, training split)

| pair | ρ |
|---|---|
| C-U | -1.000 |
| C-P | -0.945 |
| C-F | -0.654 |
| U-P | 0.946 |
| U-F | 0.655 |
| P-F | 0.621 |

## Table V - detection performance on the test split

| Method | Acc. | Prec. | Rec. | F1 | AUC | train F1 |
|---|---|---|---|---|---|---|
| MSP threshold | 0.706 | 0.633 | 0.864 | 0.730 | 0.821 | 0.725 |
| Entropy threshold | 0.706 | 0.633 | 0.864 | 0.730 | 0.821 | 0.725 |
| kNN-distance thr. | 0.837 | 0.796 | 0.868 | 0.831 | 0.921 | 0.834 |
| Logistic regression | 0.837 | 0.803 | 0.856 | 0.829 | 0.922 | 0.832 |
| Expert FIS (untuned) | 0.786 | 0.724 | 0.864 | 0.788 | 0.865 | 0.799 |
| GA-FIS | 0.836 ± 0.004 | 0.791 ± 0.014 | 0.877 ± 0.024 | 0.831 ± 0.005 | 0.879 ± 0.015 | 0.837 ± 0.001 |
| PSO-FIS | 0.837 ± 0.006 | 0.820 ± 0.012 | 0.828 ± 0.018 | 0.824 ± 0.007 | 0.882 ± 0.003 | 0.835 ± 0.001 |
| Hybrid GA→PSO | 0.837 ± 0.003 | 0.796 ± 0.011 | 0.871 ± 0.017 | 0.831 ± 0.003 | 0.877 ± 0.012 | 0.838 ± 0.001 |

## Table VI - detection rate (recall) by attack strength

Representative hybrid run (best training F1) against the MSP baseline.

| ε | n | Hybrid GA→PSO | MSP baseline |
|---|---|---|---|
| 1/16 | 57 | 0.965 | 1.000 |
| 2/16 | 54 | 0.852 | 0.963 |
| 3/16 | 65 | 0.877 | 0.846 |
| 4/16 | 67 | 0.776 | 0.687 |

## Significance (representative hybrid run)

| Hybrid vs. | McNemar discordant (hybrid better / other better) | exact p | bootstrap ΔF1 | 95% CI |
|---|---|---|---|---|
| MSP threshold | 95 / 25 | 8.44e-11 | +0.102 | [0.065, 0.139] |
| Expert FIS (untuned) | 38 / 10 | 6.17e-05 | +0.044 | [0.020, 0.067] |

## Confusion matrices (rows = true clean / adversarial)

| Method | TN | FP | FN | TP |
|---|---|---|---|---|
| MSP threshold | 162 | 122 | 33 | 210 |
| Expert FIS (untuned) | 204 | 80 | 33 | 210 |
| Hybrid GA→PSO | 232 | 52 | 33 | 210 |

## Table II - rule base with optimised weights (representative hybrid run)

Decision threshold θ = 0.306.

| # | Rule | wᵣ |
|---|---|---|
| R1 | IF C is High and P is Low THEN Safe | 0.449 |
| R2 | IF C is High and U is Low and F is Low THEN Safe | 0.329 |
| R3 | IF C is Med and P is Low and F is Low THEN Safe | 0.164 |
| R4 | IF U is Low and F is Low THEN Safe | 0.886 |
| R5 | IF C is Low and U is High THEN Adversarial | 0.028 |
| R6 | IF C is Low and P is High THEN Adversarial | 0.673 |
| R7 | IF U is High and F is High THEN Adversarial | 0.586 |
| R8 | IF P is High and F is High THEN Adversarial | 0.168 |
| R9 | IF C is High and F is High THEN Adversarial | 0.409 |
| R10 | IF C is High and P is High THEN Suspicious | 0.304 |
| R11 | IF C is Med and U is Med THEN Suspicious | 0.825 |
| R12 | IF C is Low and P is Low and F is Low THEN Suspicious | 0.787 |
| R13 | IF U is Med and P is Med THEN Suspicious | 0.642 |
| R14 | IF C is Med and F is High THEN Adversarial | 0.677 |
| R15 | IF P is Med and F is Med THEN Suspicious | 0.026 |

Fraction of test inputs where no rule fires (score defaults to 0.5): 0.0000.

## GA-tuned membership partition (representative GA-FIS run)

| Descriptor | expert p₁, p₂, p₃ | GA-tuned p₁, p₂, p₃ |
|---|---|---|
| C | 0.25, 0.50, 0.75 | 0.24, 0.30, 0.66 |
| U | 0.25, 0.50, 0.75 | 0.53, 0.76, 0.93 |
| P | 0.25, 0.50, 0.75 | 0.63, 0.76, 0.86 |
| F | 0.25, 0.50, 0.75 | 0.16, 0.48, 0.54 |

## Three-way triage on the test split (representative hybrid)

Safe: r < θ = 0.306 (accept). Suspicious: θ ≤ r < 0.603 (secondary screening). Adversarial: r ≥ the midpoint between θ and the Adversarial consequent 0.9 (reject).

| True class | Safe | Suspicious | Adversarial |
|---|---|---|---|
| clean | 232 | 38 | 14 |
| adversarial | 33 | 90 | 120 |

## Example explanations

**adversarial, eps=4/16, missed by MSP** (record 1190): risk score 0.482 -> ADVERSARIAL. Raw descriptors: C = 1, U = 4.788e-06, P = 0.001056, F = 8.93.

- R1: IF C is High and P is Low THEN Safe - firing 1.00, weight 0.45, share 52%
- R9: IF C is High and F is High THEN Adversarial - firing 1.00, weight 0.41, share 48%

**clean** (record 1): risk score 0.100 -> clean. Raw descriptors: C = 1, U = 2.756e-05, P = 0.0001597, F = 4.525.

- R4: IF U is Low and F is Low THEN Safe - firing 1.00, weight 0.89, share 53%
- R1: IF C is High and P is Low THEN Safe - firing 1.00, weight 0.45, share 27%
- R2: IF C is High and U is Low and F is Low THEN Safe - firing 1.00, weight 0.33, share 20%

## Comparison with the paper

| Quantity | Paper | This run |
|---|---|---|
| Held-out classifier accuracy | 0.9750 | 0.9847 |
| F1, MSP threshold | 0.728 | 0.730 |
| F1, Entropy threshold | 0.722 | 0.730 |
| F1, kNN-distance thr. | 0.846 | 0.831 |
| F1, Logistic regression | 0.852 | 0.829 |
| F1, Expert FIS (untuned) | 0.780 | 0.788 |
| F1, GA-FIS | 0.841 | 0.831 ± 0.005 |
| F1, PSO-FIS | 0.828 | 0.824 ± 0.007 |
| F1, Hybrid GA→PSO | 0.841 | 0.831 ± 0.003 |
| Hybrid recall | 0.904 | 0.871 ± 0.017 |
| Hybrid AUC | 0.887 | 0.877 ± 0.012 |
| Recall at ε = 4/16, hybrid / MSP | 0.833 / 0.561 | 0.776 / 0.687 |
| McNemar p, hybrid vs. MSP | 5.5e-08 | 8.4e-11 |
| McNemar p, hybrid vs. expert FIS | 0.017 | 0.000 |
| Bootstrap ΔF1 vs. MSP | +0.102 [0.066, 0.139] | +0.102 [0.065, 0.139] |
| Bootstrap ΔF1 vs. expert | +0.049 [0.020, 0.080] | +0.044 [0.020, 0.067] |
| ρ(C-U) | -1.000 | -1.000 |
| ρ(C-P) | -0.944 | -0.945 |
| ρ(C-F) | -0.723 | -0.654 |

Saved detector reloads and reproduces the test scores exactly: True. Runtime 8 s.
