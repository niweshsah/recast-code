"""(ii) Interaction-phase segmentation with a left-to-right HMM, and phase-conditioned gripper labels.

States: idle -> approach -> grasp -> carry -> release -> idle (cyclic, persistent self-transitions), so a recording
can contain several interaction cycles. Each state emits the 4-vector x_t = (d_t, kappa_t, v_t, dd_t) under a
diagonal Gaussian. Parameters are fitted from a few manually labelled videos per task; Viterbi decodes new ones.

Gripper label (paper eq. 3), 1 = fully open:
    g_t = 1                for idle, approach
    g_t = 0                for grasp, carry
    g_t = ramp 0 -> 1      over the release segment

    python -m recast.phases fit   --labels labels/cube/*.json --out models/phase_hmm_cube.json
    python -m recast.phases decode runs/cube --model models/phase_hmm_cube.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recast import PHASES

IDLE, APPROACH, GRASP, CARRY, RELEASE = range(5)
N_STATES = len(PHASES)
FEATURES = ("d", "kappa", "v", "dd")
FAR_M = 0.5          # distance used when no hand is visible
VAR_FLOOR = 1e-4


def observation_matrix(d: np.ndarray, kappa: np.ndarray, v: np.ndarray, dd: np.ndarray) -> np.ndarray:
    """(T, 4) observations; frames without a hand count as far away and not in contact."""
    x = np.stack([d, kappa, v, dd], axis=1).astype(float)
    x[np.isnan(x[:, 0]), 0] = FAR_M
    x[np.isnan(x[:, 1]), 1] = 0.0
    x[np.isnan(x[:, 2]), 2] = 0.0
    x[np.isnan(x[:, 3]), 3] = 0.0
    x[:, 0] = np.minimum(x[:, 0], FAR_M)
    return x


def allowed_transitions() -> np.ndarray:
    A = np.eye(N_STATES, dtype=bool)
    for s in range(N_STATES):
        A[s, (s + 1) % N_STATES] = True
    return A


@dataclass
class PhaseHMM:
    means: np.ndarray                     # (S, F)
    variances: np.ndarray                 # (S, F)
    transitions: np.ndarray = field(default=None)   # (S, S), rows sum to 1, zero outside the allowed structure
    initial: np.ndarray = field(default=None)       # (S,)

    def __post_init__(self):
        if self.transitions is None:
            self.transitions = self.left_to_right(0.95)
        if self.initial is None:
            self.initial = np.array([0.8, 0.2, 0.0, 0.0, 0.0])

    @staticmethod
    def left_to_right(stay: float) -> np.ndarray:
        A = np.zeros((N_STATES, N_STATES))
        for s in range(N_STATES):
            A[s, s], A[s, (s + 1) % N_STATES] = stay, 1.0 - stay
        return A

    @classmethod
    def prior(cls, stay: float = 0.95) -> PhaseHMM:
        """Hand-set emissions following the paper's phase definitions; use until labelled videos are available."""
        means = np.array([[0.40, 0.00, 0.00, 0.00],     # idle: hand far, no contact
                          [0.08, 0.10, 0.00, -0.25],    # approach: shrinking gap, weak contact
                          [0.00, 0.80, 0.02, 0.00],     # grasp: strong contact, object nearly still
                          [0.00, 0.80, 0.20, 0.00],     # carry: sustained contact, object moving
                          [0.03, 0.20, 0.01, 0.25]])    # release: contact fading, hand moving away, object at rest
        sd = np.array([[0.15, 0.10, 0.05, 0.15],
                       [0.05, 0.15, 0.05, 0.20],
                       [0.02, 0.25, 0.04, 0.10],
                       [0.02, 0.25, 0.12, 0.15],
                       [0.03, 0.20, 0.05, 0.20]])
        return cls(means, sd ** 2, cls.left_to_right(stay))

    # ----------------------------------------------------------------------------------------- learning
    @classmethod
    def fit(cls, sequences: list[np.ndarray], labels: list[np.ndarray], smoothing: float = 1.0) -> PhaseHMM:
        """Maximum-likelihood fit from labelled sequences (supervised: counts and per-state moments)."""
        X, Y = np.concatenate(sequences), np.concatenate(labels)
        prior = cls.prior()
        means, variances = prior.means.copy(), prior.variances.copy()
        for s in range(N_STATES):
            xs = X[Y == s]
            if len(xs) >= 2:
                means[s], variances[s] = xs.mean(0), np.maximum(xs.var(0), VAR_FLOOR)
        allowed = allowed_transitions()
        counts = np.where(allowed, smoothing, 0.0)
        init = np.full(N_STATES, smoothing)
        for y in labels:
            init[y[0]] += 1
            for a, b in zip(y[:-1], y[1:]):
                if not allowed[a, b]:
                    raise ValueError(f"label sequence has a forbidden transition {PHASES[a]} -> {PHASES[b]}")
                counts[a, b] += 1
        return cls(means, variances, counts / counts.sum(1, keepdims=True), init / init.sum())

    # ----------------------------------------------------------------------------------------- inference
    def log_emission(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.means[None]
        return -0.5 * (np.log(2 * np.pi * self.variances)[None] + diff ** 2 / self.variances[None]).sum(-1)

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most probable state sequence (T,)."""
        with np.errstate(divide="ignore"):
            logA, logpi = np.log(self.transitions), np.log(self.initial)
        logB = self.log_emission(X)
        T = len(X)
        delta = logpi + logB[0]
        back = np.zeros((T, N_STATES), int)
        for t in range(1, T):
            scores = delta[:, None] + logA
            back[t] = scores.argmax(0)
            delta = scores.max(0) + logB[t]
        path = np.zeros(T, int)
        path[-1] = int(delta.argmax())
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    # ----------------------------------------------------------------------------------------- io
    def to_json(self) -> dict:
        return {"states": list(PHASES), "features": list(FEATURES), "means": self.means.tolist(),
                "variances": self.variances.tolist(), "transitions": self.transitions.tolist(),
                "initial": self.initial.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> PhaseHMM:
        return cls(np.array(d["means"]), np.array(d["variances"]), np.array(d["transitions"]),
                   np.array(d["initial"]))


def gripper_labels(phases: np.ndarray) -> np.ndarray:
    """Paper eq. (3): 1 open for idle/approach, 0 closed for grasp/carry, linear 0 -> 1 ramp over each release."""
    g = np.where(np.isin(phases, [GRASP, CARRY]), 0.0, 1.0)
    t = 0
    while t < len(phases):
        if phases[t] == RELEASE:
            end = t
            while end + 1 < len(phases) and phases[end + 1] == RELEASE:
                end += 1
            n = end - t + 1
            g[t:end + 1] = np.arange(1, n + 1) / n
            t = end + 1
        else:
            t += 1
    return g


def segments(phases: np.ndarray) -> list[tuple[int, int, int]]:
    """[(state, first, last)] runs of constant phase."""
    out, start = [], 0
    for t in range(1, len(phases) + 1):
        if t == len(phases) or phases[t] != phases[start]:
            out.append((int(phases[start]), start, t - 1))
            start = t
    return out


def frame_f1(pred: np.ndarray, true: np.ndarray) -> float:
    """Macro-averaged frame-level F1 over the phases present in ``true`` (phase-accuracy metric)."""
    scores = []
    for s in np.unique(true):
        tp = np.sum((pred == s) & (true == s))
        fp, fn = np.sum((pred == s) & (true != s)), np.sum((pred != s) & (true == s))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


# --------------------------------------------------------------------------------------------- CLI
def _signals(run: Path) -> np.ndarray:
    with np.load(run / "contact" / "signals.npz") as f:
        return observation_matrix(f["d"], f["kappa"], f["v"], f["dd"])


def _labels(path: Path) -> tuple[Path, np.ndarray]:
    """Label file: {"run": "<run dir>", "phases": ["idle", ...] one per frame}."""
    d = json.loads(path.read_text())
    return Path(d["run"]), np.array([PHASES.index(p) for p in d["phases"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--labels", type=Path, nargs="+", required=True)
    f.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("decode")
    d.add_argument("run", type=Path)
    d.add_argument("--model", type=Path, help="fitted HMM JSON (default: hand-set prior)")
    a = ap.parse_args()
    if a.cmd == "fit":
        pairs = [_labels(p) for p in a.labels]
        hmm = PhaseHMM.fit([_signals(r) for r, _ in pairs], [y for _, y in pairs])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(hmm.to_json(), indent=2))
        print(f"fitted on {len(pairs)} videos -> {a.out}")
    else:
        hmm = PhaseHMM.from_json(json.loads(a.model.read_text())) if a.model else PhaseHMM.prior()
        path = hmm.viterbi(_signals(a.run))
        out = {"phases": [PHASES[s] for s in path], "gripper": gripper_labels(path).round(4).tolist(),
               "segments": [(PHASES[s], i, j) for s, i, j in segments(path)]}
        (a.run / "phases").mkdir(parents=True, exist_ok=True)
        (a.run / "phases" / "phases.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out["segments"]))


if __name__ == "__main__":
    main()
