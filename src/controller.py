"""
controller.py
-------------
Proportional CLF state feedback controller on the unsafe SAE feature subspace.

Control law:
    w_t = -α · e_t,   e_t = W_dec[:,I] @ f[I] ∈ range(Π)

Lyapunov function:
    V(e) = e^T P e,   P = 2·Π + 0.1·I   (positive definite, P ≻ 0)

    On range(Π): P_u = (2 + 0.1)·I = 2.1·I, so V(e) = 2.1·‖e‖²

Formal properties (proofs in RESEARCH_NOTE.md):
  Theorem 1 — Zero Alignment Tax:
      ⟨w_t, v_s⟩ = 0 for all v_s ⊥ range(Π), exactly, for all t.
  Theorem 2 — CLF Exponential Decay (d_t = 0, n_t = 0):
      V(e_{t+1}) ≤ (1-α)² · ρ_u² · V(e_t)              — V's own rate is ρ² = (1-α)²ρ_u²
      ‖e‖ rate ρ = (1-α)·ρ_u ≤ 0.08 × 0.75 = 0.06  (fast convergence; this ρ is what
      Proposition 1 below calls ρ — V's rate is ρ², not ρ itself)
  Proposition 1 — ISS under bounded disturbance ‖Π d_t‖ ≤ D:
      V(e_{t+1}) ≤ 2ρ² · V(e_t) + 4.2 · D²
      V_∞ ≤ 4.2D²/(1-2ρ²) ≈ 26.4  (Corollary tight: V_∞ ≤ 2.1D²/(1-ρ)² ≈ 14.78)

Note on naming: NOT H∞-optimal (no Riccati equation, no H∞ norm bound).
The design shares one structural property with H∞ loop-shaping: zero gain on
safe-subspace directions, proportional gain on the unsafe subspace only.
A precise name: proportional CLF state feedback on the unsafe SAE subspace.
See RESEARCH_NOTE.md §5 for the full comparison.
"""

import numpy as np
from src.sae_mock import STATE_DIM, I_UNSAFE, encode

LYAPUNOV_BOUNDARY = 0.4   # c — ISS detection threshold; invariant boundary when d_t=0 under
                          # Thm 2's full conditions (linear plant, n_t=0, AΠ=ΠA) — the mock's
                          # own commutativity gap (0.68) and process noise mean this is an
                          # empirically observed threshold here, not a proven invariant (see
                          # RESEARCH_NOTE.md §7.1, README.md §Motivation)
SUPPRESSION_GAIN  = 0.92  # α — fraction of unsafe component cancelled per step


class AlignmentBreach(Exception):
    def __init__(self, t: int, v: float):
        super().__init__(f"[t={t}] V={v:.4f} >= c={LYAPUNOV_BOUNDARY}")
        self.t, self.v = t, v


class SafeSubspaceClfController:
    def __init__(self, sae: dict, raise_on_breach: bool = False):
        self.sae = sae
        self.raise_on_breach = raise_on_breach
        self.breach_log: list[dict] = []
        self._step = 0

        d = STATE_DIM
        # Orthogonal projector onto unsafe decoder subspace (QR, reduced mode)
        V_unsafe = sae["W_dec"][:, I_UNSAFE]          # (d, k)
        Q_orth, _ = np.linalg.qr(V_unsafe, mode='reduced')
        self.Pi = Q_orth @ Q_orth.T                    # (d, d), rank k

        # P = c1·Π + c2·I, c1 > c2 > 0 → P ≻ 0, block-diagonal in (Π, I-Π) basis
        self.P = 2.0 * self.Pi + 0.1 * np.eye(d)

    def compute(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        f   = encode(x, self.sae)
        # Direct feature computation: e_t ∈ range(Π) exactly, no ReLU gap
        # (see RESEARCH_NOTE §2 — consistent with RealSafeSubspaceClfController)
        e_t = self.sae["W_dec"][:, I_UNSAFE] @ f[I_UNSAFE]

        v = self.lyapunov(e_t)
        if v >= LYAPUNOV_BOUNDARY:
            self.breach_log.append({"t": self._step, "V": v})
            if self.raise_on_breach:
                raise AlignmentBreach(self._step, v)

        self._step += 1
        return -SUPPRESSION_GAIN * e_t, v   # w_t ∈ range(Π) — zero tax on safe dims

    def lyapunov(self, e: np.ndarray) -> float:
        return float(e @ self.P @ e)

    def reset(self):
        self._step = 0
        self.breach_log.clear()
