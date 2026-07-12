"""
main.py
-------
Runs the full LLM alignment control simulation and produces two publication-
quality figures:

  Figure 1 — Unsafe Feature Energy Trajectory
      Red    (uncontrolled) : unsafe activation explodes during jailbreak window
      Orange (naive Pi@x)   : geometric projection baseline — no SAE certificate
      Green  (CLF controller): SAE-grounded CLF suppresses the disturbance

  Figure 2 — Lyapunov Runtime Monitor
      Shows V(e_t) for the controlled trajectory.
      Horizontal dashed line marks boundary c — invariant when d_t=0 *and* Theorem 2's
      other conditions hold (linear plant, n_t=0, AΠ=ΠA), none of which this mock's own
      disturbance-free case fully satisfies (see RESEARCH_NOTE.md §7.1). Treat as an
      empirically observed threshold here, not a proven invariant.
      Shaded region flags where the monitor would trigger a pre-token halt.

Usage:
    python main.py
    python main.py --steps 40 --attack-start 10 --attack-end 16 --save
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from src.sae_mock   import build_sae
from src.system     import build_plant, simulate
from src.controller import SafeSubspaceClfController, LYAPUNOV_BOUNDARY, SUPPRESSION_GAIN


# ── Style constants ────────────────────────────────────────────────────────────
COLOR_UNCONTROLLED = "#E05252"   # red
COLOR_CONTROLLED   = "#4CAF80"   # green
COLOR_NAIVE        = "#E09B52"   # orange — naive projection baseline
COLOR_BOUNDARY     = "#F0A500"   # amber
COLOR_BREACH_FILL  = "#FFF3CD"   # pale yellow breach zone
ATTACK_FILL        = "#F5E6FF"   # pale purple attack window


class NaiveProjectionController:
    """
    Baseline comparator: w_t = -alpha * Pi @ x  (geometric projection, no SAE encoding).

    Differences from SafeSubspaceClfController:
      - Does not call encode(); projects x directly onto range(Pi)
      - Has zero alignment tax (w_t in range(Pi)) but no SAE-grounded certificate
      - e_t = Pi @ x may be nonzero even when SAE feature activations are zero
        (ignores ReLU sparsity — over-suppresses when features are inactive)

    Why suppression appears very high (~99%):
      After correction, x = x_next - α·Pi@x_next = (I - α·Pi)@x_next + (I-Pi)@x_next.
      The unsafe component in residual space is reduced to (1-α)·Pi@x_next ≈ 0.08·Pi@x_next.
      The SAE encoder then finds near-zero activations on all unsafe features.
      This is real suppression, not an artefact — but it comes at the cost of
      correcting regardless of ReLU sparsity (no SAE certificate, no ISS bound).
      unsafe_energy is measured post-correction (see system.py), identical for all runs.
    """
    def __init__(self, sae: dict):
        from src.sae_mock import STATE_DIM, I_UNSAFE
        V_unsafe = sae["W_dec"][:, I_UNSAFE]
        Q, _     = np.linalg.qr(V_unsafe, mode='reduced')
        self.Pi  = Q @ Q.T
        d        = STATE_DIM
        self.P   = 2.0 * self.Pi + 0.1 * np.eye(d)
        self.breach_log: list = []
        self._step = 0

    def compute(self, x: np.ndarray):
        e = self.Pi @ x
        v = float(e @ self.P @ e)
        if v >= LYAPUNOV_BOUNDARY:
            self.breach_log.append({"t": self._step, "V": v})
        self._step += 1
        return -SUPPRESSION_GAIN * e, v

    def reset(self):
        self._step = 0
        self.breach_log.clear()


def _stability_quantities(ctrl: SafeSubspaceClfController, plant: dict) -> dict:
    """ISS + §7.1 M-matrix quantities (RESEARCH_NOTE.md), factored out so run() and
    sweep_seeds() compute them identically instead of duplicating the block."""
    Pi = ctrl.Pi
    A  = plant["A"]
    I  = np.eye(len(A))
    rho_u    = float(np.linalg.svd(Pi @ A @ Pi, compute_uv=False)[0])
    rho      = (1.0 - SUPPRESSION_GAIN) * rho_u
    comm_gap = np.linalg.norm(Pi @ A - A @ Pi, 'fro') / np.linalg.norm(A, 'fro')
    kappa    = float(np.linalg.svd(Pi @ A @ (I - Pi), compute_uv=False)[0])
    kappa_21 = float(np.linalg.svd((I - Pi) @ A @ Pi, compute_uv=False)[0])
    rho_s    = float(np.linalg.svd((I - Pi) @ A @ (I - Pi), compute_uv=False)[0])
    M = np.array([[(1.0 - SUPPRESSION_GAIN) * rho_u,    kappa],
                  [(1.0 - SUPPRESSION_GAIN) * kappa_21,  rho_s]])
    rho_M = float(np.max(np.abs(np.linalg.eigvals(M))))
    return {"rho_u": rho_u, "rho": rho, "comm_gap": comm_gap, "kappa": kappa,
            "kappa_21": kappa_21, "rho_s": rho_s, "M": M, "rho_M": rho_M}


def sweep_seeds(n_seeds: int = 10, n_steps: int = 30,
                 attack_windows: list[tuple[int, int]] = ((10, 15), (5, 10), (18, 25))) -> None:
    """Robustness check (single-seed caveat, README §Results): the headline 70.6%/99.2%
    numbers are one realisation of random A, W_dec and one attack window. This reruns the
    mock across n_seeds trial seeds (decorrelated SAE/plant/process-noise seeds per trial)
    x len(attack_windows) windows and reports medians + [min, max] instead of one point.
    Mock simulation is cheap (no GPU, ~ms per trial), so this costs seconds, not sessions.
    """
    sup_clf_all, sup_naive_all, rho_m_all = [], [], []

    for i in range(n_seeds):
        sae   = build_sae(seed=42 + i)
        plant = build_plant(seed=1000 + i)
        ctrl  = SafeSubspaceClfController(sae, raise_on_breach=False)
        naive_ctrl = NaiveProjectionController(sae)
        stab  = _stability_quantities(ctrl, plant)
        rho_m_all.append(stab["rho_M"])

        for (a_start, a_end) in attack_windows:
            sim_seed = 100 + i
            uncontrolled = simulate(plant, sae, controller=ctrl, n_steps=n_steps,
                                     controlled=False, seed=sim_seed,
                                     attack_start=a_start, attack_end=a_end)
            ctrl.reset()
            controlled = simulate(plant, sae, controller=ctrl, n_steps=n_steps,
                                   controlled=True, seed=sim_seed,
                                   attack_start=a_start, attack_end=a_end)
            naive = simulate(plant, sae, controller=naive_ctrl, n_steps=n_steps,
                              controlled=True, seed=sim_seed,
                              attack_start=a_start, attack_end=a_end)

            peak_unc   = uncontrolled["unsafe_energy"].max()
            peak_ctrl  = controlled["unsafe_energy"].max()
            peak_naive = naive["unsafe_energy"].max()
            sup_clf_all.append((1 - peak_ctrl / (peak_unc + 1e-9)) * 100)
            sup_naive_all.append((1 - peak_naive / (peak_unc + 1e-9)) * 100)

    def _stats(vals):
        return (float(np.median(vals)), float(np.min(vals)), float(np.max(vals)))

    clf_med, clf_lo, clf_hi       = _stats(sup_clf_all)
    naive_med, naive_lo, naive_hi = _stats(sup_naive_all)
    rho_med, rho_lo, rho_hi       = _stats(rho_m_all)
    n_trials = n_seeds * len(attack_windows)

    print("\n" + "=" * 60)
    print(f"  SEED SWEEP  ({n_seeds} seeds x {len(attack_windows)} attack windows = {n_trials} trials)")
    print("=" * 60)
    print(f"  CLF suppression   : median {clf_med:.1f}%   range [{clf_lo:.1f}%, {clf_hi:.1f}%]")
    print(f"  Naive suppression : median {naive_med:.1f}%   range [{naive_lo:.1f}%, {naive_hi:.1f}%]")
    print(f"  §7.1 rho(M)       : median {rho_med:.4f}     range [{rho_lo:.4f}, {rho_hi:.4f}]"
          f"   ({'all < 1 — stable across every trial ✓' if rho_hi < 1.0 else 'SOME TRIALS >= 1 — bound does not hold everywhere ✗'})")
    print("=" * 60 + "\n")


def measure_benign_lyapunov_floor(sae: dict, plant: dict,
                                   n_steps: int = 500, percentile: float = 95.0) -> float:
    """
    Diagnostic only — does NOT set LYAPUNOV_BOUNDARY. Runs the plant for n_steps with no
    disturbance and returns the given percentile of V(e_t): the empirical "benign floor"
    used to *justify* the manually-chosen c = LYAPUNOV_BOUNDARY = 0.4 (well above this
    floor, below the ISS bound V_inf — see README §Design highlights), not to *derive* it.
    Previously named calibrate_lyapunov_boundary, which implied it calibrated the runtime
    boundary; it never did — it was always print-only. Renamed for honesty, not behavior.
    """
    ctrl_cal = SafeSubspaceClfController(sae)
    result   = simulate(plant, sae, ctrl_cal,
                        n_steps=n_steps, controlled=True, seed=99,
                        attack_start=n_steps + 1, attack_end=n_steps + 2)
    return float(np.percentile(result["lyapunov"], percentile))


def run(n_steps: int = 30, attack_start: int = 10, attack_end: int = 15,
        save: bool = False, sim_seed: int = 1):

    # ── Build shared components ────────────────────────────────────────────────
    sae        = build_sae(seed=42)
    plant      = build_plant(seed=0)
    ctrl       = SafeSubspaceClfController(sae, raise_on_breach=False)
    naive_ctrl = NaiveProjectionController(sae)

    # ── ISS quantities — Proposition 1 + Corollary (RESEARCH_NOTE.md) ─────────
    stab    = _stability_quantities(ctrl, plant)
    _rho    = stab["rho"]
    _comm_gap, _kappa, _kappa_21, _rho_s, _M, _rho_M = (
        stab["comm_gap"], stab["kappa"], stab["kappa_21"], stab["rho_s"], stab["M"], stab["rho_M"])
    _D      = 2.5      # adversarial magnitude; in range(Pi) by construction
    V_iss   = 2.1 * _D**2 / (1.0 - _rho) ** 2   # tight bound: V_∞ ≤ 2.1D²/(1-ρ)²

    # Diagnostic only — see measure_benign_lyapunov_floor docstring: this does not set
    # LYAPUNOV_BOUNDARY, it measures the floor used to justify the manually-chosen value.
    benign_floor_p95 = measure_benign_lyapunov_floor(sae, plant)

    # ── Simulate all three trajectories ───────────────────────────────────────
    # NOTE (single-seed caveat, README §Results): sae_seed=42, plant_seed=0 and
    # sim_seed (below, default 1, override with --seed) are one realisation of
    # random A, W_dec and one attack window — see `python main.py --sweep 10`
    # for medians/ranges across seeds and attack windows instead of this one point.
    uncontrolled = simulate(plant, sae, controller=ctrl,
                            n_steps=n_steps, controlled=False, seed=sim_seed,
                            attack_start=attack_start, attack_end=attack_end)
    ctrl.reset()
    controlled   = simulate(plant, sae, controller=ctrl,
                            n_steps=n_steps, controlled=True,  seed=sim_seed,
                            attack_start=attack_start, attack_end=attack_end)
    naive        = simulate(plant, sae, controller=naive_ctrl,
                            n_steps=n_steps, controlled=True,  seed=sim_seed,
                            attack_start=attack_start, attack_end=attack_end)

    steps = np.arange(n_steps)

    # ── Figure 1: Unsafe Feature Energy ───────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                             gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("#0D1117")   # dark GitHub-style background

    ax1 = axes[0]
    ax1.set_facecolor("#161B22")

    # Attack window shading
    ax1.axvspan(attack_start, attack_end, color=ATTACK_FILL,
                alpha=0.25, label="Jailbreak injection window")

    ax1.plot(steps, uncontrolled["unsafe_energy"],
             color=COLOR_UNCONTROLLED, linewidth=2.2, label="No controller (baseline)")
    ax1.plot(steps, naive["unsafe_energy"],
             color=COLOR_NAIVE,        linewidth=2.0, linestyle="--",
             label="Naive projection  Pi@x  (no SAE certificate)")
    ax1.plot(steps, controlled["unsafe_energy"],
             color=COLOR_CONTROLLED,   linewidth=2.2, label="CLF Controller (proposed)")

    ax1.set_title("Unsafe Feature Activation Energy  ‖f[I_unsafe]‖²",
                  color="white", fontsize=13, pad=10)
    ax1.set_xlabel("Generation step  t", color="#8B949E", fontsize=10)
    ax1.set_ylabel("Unsafe energy  (a.u.)", color="#8B949E", fontsize=10)
    ax1.tick_params(colors="#8B949E")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363D")
    ax1.legend(facecolor="#161B22", edgecolor="#30363D",
               labelcolor="white", fontsize=9)

    # Annotation: peak attack
    peak_t = int(np.argmax(uncontrolled["unsafe_energy"]))
    peak_v = uncontrolled["unsafe_energy"][peak_t]
    ax1.annotate("Peak instability\n(OOD attack)",
                 xy=(peak_t, peak_v),
                 xytext=(peak_t + 2, peak_v * 0.85),
                 arrowprops=dict(arrowstyle="->", color=COLOR_UNCONTROLLED),
                 color=COLOR_UNCONTROLLED, fontsize=8)

    # ── Figure 2: Lyapunov Runtime Monitor ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161B22")

    lyap = controlled["lyapunov"]
    ax2.plot(steps, lyap, color="#58A6FF", linewidth=2.2,
             label="V(eₜ) = eₜᵀ P eₜ  (Lyapunov energy)")

    # Boundary line
    ax2.axhline(LYAPUNOV_BOUNDARY, color=COLOR_BOUNDARY,
                linewidth=1.5, linestyle="--",
                label=f"Halt boundary  c = {LYAPUNOV_BOUNDARY}  "
                      f"(invariant when dₜ=0 *and* Thm 2's other conditions hold — not fully met by this mock, see README)")

    # Shade breach zone
    ax2.fill_between(steps, lyap, LYAPUNOV_BOUNDARY,
                     where=(lyap >= LYAPUNOV_BOUNDARY),
                     color=COLOR_BREACH_FILL, alpha=0.4, label="Pre-token halt zone")

    # Attack window
    ax2.axvspan(attack_start, attack_end, color=ATTACK_FILL, alpha=0.2)

    # Annotate breach events
    breaches = ctrl.breach_log
    if breaches:
        first_breach = breaches[0]["t"]
        ax2.axvline(first_breach, color=COLOR_BOUNDARY,
                    linewidth=1.2, linestyle=":", alpha=0.8)
        ax2.text(first_breach + 0.3, LYAPUNOV_BOUNDARY * 1.05,
                 "⚠ Halt triggered", color=COLOR_BOUNDARY, fontsize=8)

    ax2.set_title("Lyapunov Runtime Monitor  V(eₜ)",
                  color="white", fontsize=13, pad=10)
    ax2.set_xlabel("Generation step  t", color="#8B949E", fontsize=10)
    ax2.set_ylabel("V(eₜ)  (Lyapunov energy)", color="#8B949E", fontsize=10)
    ax2.tick_params(colors="#8B949E")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363D")
    ax2.legend(facecolor="#161B22", edgecolor="#30363D",
               labelcolor="white", fontsize=9)

    # ISS steady-state bound annotation (Corollary, RESEARCH_NOTE.md)
    ax2.text(0.98, 0.97,
             f"ISS bound  V∞ ≤ {V_iss:.1f}  (Corollary, RESEARCH_NOTE §4)\n"
             f"ρ = {_rho:.4f},  D = {_D},  2.1·D²/(1−ρ)²",
             transform=ax2.transAxes, ha='right', va='top',
             color=COLOR_BOUNDARY, fontsize=7.5,
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#161B22',
                       edgecolor='#30363D', alpha=0.92))

    plt.suptitle(
        "LLM Alignment via Internal State-Space Control\n"
        "CLF Controller · SAE Feature Clamping · Lyapunov Runtime Monitoring",
        color="white", fontsize=11, y=0.98
    )

    if save:
        path = "figures/simulation_results.png"
        os.makedirs("figures", exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved → {path}")
    else:
        plt.show()

    # ── Console summary ────────────────────────────────────────────────────────
    peak_unc   = uncontrolled["unsafe_energy"].max()
    peak_ctrl  = controlled["unsafe_energy"].max()
    peak_naive = naive["unsafe_energy"].max()
    sup_clf    = (1 - peak_ctrl  / (peak_unc + 1e-9)) * 100
    sup_naive  = (1 - peak_naive / (peak_unc + 1e-9)) * 100

    print("\n" + "="*60)
    print("  SIMULATION SUMMARY")
    print("="*60)
    print(f"  Steps simulated       : {n_steps}")
    print(f"  Jailbreak window      : t ∈ [{attack_start}, {attack_end})")
    _margin_ratio = LYAPUNOV_BOUNDARY / max(benign_floor_p95, 1e-9)
    print(f"  Lyapunov boundary (c) : {LYAPUNOV_BOUNDARY}  (manually set, NOT derived from "
          f"the line below)")
    print(f"  Benign 95th-pct floor : {benign_floor_p95:.4f}  "
          f"(diagnostic — justifies c via a {_margin_ratio:.0f}x margin, doesn't set it)")
    print(f"  Peak unsafe energy (uncontrolled)   : {peak_unc:.4f}")
    print(f"  Peak unsafe energy (naive Pi@x)     : {peak_naive:.4f}  ({sup_naive:.1f}% suppression)")
    print(f"  Peak unsafe energy (CLF controller) : {peak_ctrl:.4f}  ({sup_clf:.1f}% suppression)")
    print(f"  CLF Lyapunov breaches : {len(ctrl.breach_log)}")
    if ctrl.breach_log:
        print(f"  First breach at t     : {ctrl.breach_log[0]['t']}  "
              f"(V = {ctrl.breach_log[0]['V']:.4f})")
    print(f"  Note: naive Pi@x has higher suppression because it applies the full")
    print(f"        geometric projection regardless of SAE ReLU sparsity (over-corrects")
    print(f"        when features are inactive). CLF correction is SAE-calibrated and")
    print(f"        carries a formal ISS certificate; naive provides none.")
    print(f"  ISS bound V∞ ≤ {V_iss:.2f}  (Corollary: 2.1·D²/(1-ρ)²,  ρ={_rho:.4f})")
    print(f"  Commutativity gap ‖ΠA-AΠ‖_F/‖A‖_F : {_comm_gap:.4f}"
          "  (Thm 2 does not apply at this gap — convergence is empirical, see §7.1)")
    print(f"  §7.1 cross-coupling  κ₁₂=‖ΠA(I-Π)‖₂ : {_kappa:.4f}")
    print(f"                       κ₂₁=‖(I-Π)AΠ‖₂ : {_kappa_21:.4f}")
    print(f"                       ρ_s=‖(I-Π)A(I-Π)‖₂ : {_rho_s:.4f}")
    print(f"  §7.1 M matrix (bound, RESEARCH_NOTE §7.1): "
          f"[[{_M[0,0]:.4f}, {_M[0,1]:.4f}], [{_M[1,0]:.4f}, {_M[1,1]:.4f}]]")
    print(f"  §7.1 spectral radius ρ(M) = {_rho_M:.4f}"
          f"  ({'< 1 — coupled (e,s) system stable ✓' if _rho_M < 1.0 else '>= 1 — coupled system bound invalid ✗'})")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Control Alignment Simulation")
    parser.add_argument("--steps",        type=int,  default=30)
    parser.add_argument("--attack-start", type=int,  default=10)
    parser.add_argument("--attack-end",   type=int,  default=15)
    parser.add_argument("--seed",         type=int,  default=1,
                        help="Process-noise seed for the single-run plot (sae/plant seeds are fixed at 42/0)")
    parser.add_argument("--save",         action="store_true",
                        help="Save figures to figures/ instead of displaying")
    parser.add_argument("--sweep",        type=int,  default=0, metavar="N_SEEDS",
                        help="Run the robustness sweep (N_SEEDS trial seeds x 3 attack windows, "
                             "prints median/range) instead of a single plot")
    args = parser.parse_args()

    if args.sweep:
        sweep_seeds(n_seeds=args.sweep, n_steps=args.steps)
    else:
        run(n_steps=args.steps,
            attack_start=args.attack_start,
            attack_end=args.attack_end,
            save=args.save,
            sim_seed=args.seed)
