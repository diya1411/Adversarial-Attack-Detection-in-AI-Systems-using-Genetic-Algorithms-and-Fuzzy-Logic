"""Mamdani-style fuzzy inference system with singleton consequents (Section IV.C.4).

Each of the four quantile-normalised descriptors C, U, P, F carries three
linguistic terms Low / Medium / High defined by ordered breakpoints
0 <= p1 < p2 < p3 <= 1 (Eq. 7). Rule firing uses the product t-norm with an
arity correction (Eq. 8) and the risk score is the weighted average of the
singleton consequents (Eq. 9).
"""

from dataclasses import dataclass, field

import numpy as np

DESCRIPTORS = ("C", "U", "P", "F")
TERMS = ("Low", "Med", "High")

SAFE, SUSPICIOUS, ADVERSARIAL = 0.10, 0.50, 0.90
CONSEQUENT_NAMES = {SAFE: "Safe", SUSPICIOUS: "Suspicious", ADVERSARIAL: "Adversarial"}

# Table II. Each rule is (antecedents {descriptor: term}, consequent).
RULES = (
    ({"C": "High", "P": "Low"}, SAFE),                       # R1
    ({"C": "High", "U": "Low", "F": "Low"}, SAFE),            # R2
    ({"C": "Med", "P": "Low", "F": "Low"}, SAFE),             # R3
    ({"U": "Low", "F": "Low"}, SAFE),                         # R4
    ({"C": "Low", "U": "High"}, ADVERSARIAL),                 # R5
    ({"C": "Low", "P": "High"}, ADVERSARIAL),                 # R6
    ({"U": "High", "F": "High"}, ADVERSARIAL),                # R7
    ({"P": "High", "F": "High"}, ADVERSARIAL),                # R8
    ({"C": "High", "F": "High"}, ADVERSARIAL),                # R9: confident yet atypical representation
    ({"C": "High", "P": "High"}, SUSPICIOUS),                 # R10
    ({"C": "Med", "U": "Med"}, SUSPICIOUS),                   # R11
    ({"C": "Low", "P": "Low", "F": "Low"}, SUSPICIOUS),       # R12
    ({"U": "Med", "P": "Med"}, SUSPICIOUS),                   # R13
    ({"C": "Med", "F": "High"}, ADVERSARIAL),                 # R14
    ({"P": "Med", "F": "Med"}, SUSPICIOUS),                   # R15
)
N_RULES = len(RULES)
N_DESCRIPTORS = len(DESCRIPTORS)
N_BREAKPOINTS = 3 * N_DESCRIPTORS
CONSEQUENTS = np.array([c for _, c in RULES])

# Antecedents as indices into the flattened (descriptor, term) membership
# vector of length 12. Rules with fewer than three antecedents are padded with
# index 12, which points at a constant column of ones and so leaves the
# product unchanged.
_PAD = N_BREAKPOINTS
_MAX_ARITY = max(len(a) for a, _ in RULES)
_ANTECEDENT_IDX = np.full((N_RULES, _MAX_ARITY), _PAD)
for _r, (_ante, _) in enumerate(RULES):
    for _k, (_d, _t) in enumerate(_ante.items()):
        _ANTECEDENT_IDX[_r, _k] = 3 * DESCRIPTORS.index(_d) + TERMS.index(_t)
ARITY = np.array([len(a) for a, _ in RULES], dtype=np.float64)

MIN_SEPARATION = 0.03


def rule_text(r):
    ante, cons = RULES[r]
    cond = " and ".join(f"{d} is {t}" for d, t in ante.items())
    return f"R{r + 1}: IF {cond} THEN {CONSEQUENT_NAMES[cons]}"


def uniform_partition():
    """The expert partition: breakpoints 0.25, 0.50, 0.75 for every descriptor."""
    return np.tile([0.25, 0.50, 0.75], (N_DESCRIPTORS, 1))


def decode_partition(genes, min_sep=MIN_SEPARATION):
    """Map 12 genes in [0, 1] to a valid (4, 3) breakpoint array.

    Each triple is sorted and pushed apart to a minimum separation, which
    repairs any ordering violation introduced by crossover or mutation.
    """
    bp = np.sort(np.asarray(genes, dtype=np.float64).reshape(N_DESCRIPTORS, 3), axis=1)
    p1 = np.minimum(bp[:, 0], 1.0 - 2 * min_sep)
    p2 = np.minimum(np.maximum(bp[:, 1], p1 + min_sep), 1.0 - min_sep)
    p3 = np.minimum(np.maximum(bp[:, 2], p2 + min_sep), 1.0)
    return np.stack([p1, p2, p3], axis=1)


def memberships(Z, breakpoints):
    """Membership degrees, shape (n, 4, 3), of normalised descriptors Z (n, 4).

    Low = trap(0, 0, p1, p2), Med = tri(p1, p2, p3), High = trap(p2, p3, 1, 1).
    The three terms sum to one everywhere, so the partition is complete.
    """
    p1, p2, p3 = breakpoints[:, 0], breakpoints[:, 1], breakpoints[:, 2]
    rise = (Z - p1) / (p2 - p1)
    fall = (p3 - Z) / (p3 - p2)
    low = np.clip(1.0 - rise, 0.0, 1.0)
    med = np.clip(np.minimum(rise, fall), 0.0, 1.0)
    high = np.clip(1.0 - fall, 0.0, 1.0)
    return np.stack([low, med, high], axis=-1)


def firing_strengths(M):
    """Arity-corrected product t-norm (Eq. 8): geometric mean of antecedent degrees.

    Without the 1/|A_r| exponent, two-antecedent rules would systematically
    outfire three-antecedent rules because every factor lies in [0, 1].
    """
    flat = M.reshape(len(M), N_BREAKPOINTS)
    padded = np.concatenate([flat, np.ones((len(M), 1))], axis=1)
    prod = padded[:, _ANTECEDENT_IDX].prod(axis=2)
    return prod ** (1.0 / ARITY)


def defuzzify(alpha, weights):
    """Weighted-average defuzzification (Eq. 9). Returns 0.5 if no rule fires."""
    wa = alpha * weights
    den = wa.sum(axis=-1)
    num = wa @ CONSEQUENTS
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), SUSPICIOUS)


@dataclass
class FuzzyInferenceSystem:
    breakpoints: np.ndarray = field(default_factory=uniform_partition)
    weights: np.ndarray = field(default_factory=lambda: np.ones(N_RULES))
    theta: float = 0.5

    def firing(self, Z):
        return firing_strengths(memberships(Z, self.breakpoints))

    def score(self, Z):
        """Risk score r(x) in [0.1, 0.9]."""
        return defuzzify(self.firing(Z), self.weights)

    def predict(self, Z):
        """1 = adversarial (r >= theta), 0 = clean."""
        return (self.score(Z) >= self.theta).astype(np.int64)

    def reject_threshold(self):
        """Midpoint between the decision threshold and the Adversarial consequent."""
        return 0.5 * (self.theta + ADVERSARIAL)

    def triage(self, Z, reject_at=None):
        """Three-way routing consistent with the binary decision: scores below theta
        are accepted (Safe), flagged scores go to secondary screening (Suspicious)
        unless they reach `reject_at`, where they are rejected (Adversarial)."""
        reject_at = self.reject_threshold() if reject_at is None else reject_at
        r = self.score(Z)
        out = np.full(len(r), "Suspicious", dtype=object)
        out[r < self.theta] = "Safe"
        out[r >= max(reject_at, self.theta)] = "Adversarial"
        return out

    def explain(self, z, top=5):
        """Rules ranked by their share of the defuzzified output for one input z (4,)."""
        alpha = self.firing(np.atleast_2d(z))[0]
        wa = alpha * self.weights
        total = wa.sum()
        r = float(wa @ CONSEQUENTS / total) if total > 0 else SUSPICIOUS
        share = wa / total if total > 0 else np.zeros(N_RULES)
        order = np.argsort(-share)[:top]
        rules = [
            {"rule": rule_text(i), "firing": float(alpha[i]), "weight": float(self.weights[i]), "share": float(share[i])}
            for i in order if share[i] > 0
        ]
        return {"score": r, "adversarial": r >= self.theta, "rules": rules}

    def to_dict(self):
        return {
            "breakpoints": {d: self.breakpoints[j].tolist() for j, d in enumerate(DESCRIPTORS)},
            "weights": {f"R{i + 1}": float(w) for i, w in enumerate(self.weights)},
            "theta": float(self.theta),
        }

    @classmethod
    def from_dict(cls, d):
        bp = np.array([d["breakpoints"][k] for k in DESCRIPTORS])
        w = np.array([d["weights"][f"R{i + 1}"] for i in range(N_RULES)])
        return cls(bp, w, float(d["theta"]))
