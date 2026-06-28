"""
main.py
-------
Runs the full LLM alignment control simulation and produces two publication-
quality figures:

  Figure 1 — Unsafe Feature Energy Trajectory
      Red  (uncontrolled): unsafe activation explodes during jailbreak window
      Green (controlled)  : CLF Controller suppresses the disturbance

  Figure 2 — Lyapunov Runtime Monitor
      Shows V(e_t) for the controlled trajectory.
      Horizontal dashed line marks the invariant-set boundary c.
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
COLOR_BOUNDARY     = "#F0A500"   # amber
COLOR_BREACH_FILL  = "#FFF3CD"   # pale yellow breach zone
ATTACK_FILL        = "#F5E6FF"   # pale purple attack window


def run(n_steps: int = 30, attack_start: int = 10, attack_end: int = 15,
        save: bool = False):

    # ── Build shared components ────────────────────────────────────────────────
    sae   = build_sae(seed=42)
    plant = build_plant(seed=0)
    ctrl  = SafeSubspaceClfController(sae, raise_on_breach=False)

    # ── ISS quantities — Proposition 1 + Corollary (RESEARCH_NOTE.md) ─────────
    _Pi     = ctrl.Pi
    _A      = plant["A"]
    _rho_u  = float(np.linalg.svd(_Pi @ _A @ _Pi, compute_uv=False)[0])
    _rho    = (1.0 - SUPPRESSION_GAIN) * _rho_u
    _D      = 2.5      # adversarial magnitude; in range(Pi) by construction
    V_iss   = 2.1 * _D**2 / (1.0 - _rho) ** 2   # tight bound: V_∞ ≤ 2.1D²/(1-ρ)²
    _comm_gap = (np.linalg.norm(_Pi @ _A - _A @ _Pi, 'fro')
                 / np.linalg.norm(_A, 'fro'))

    # ── Simulate both trajectories ─────────────────────────────────────────────
    uncontrolled = simulate(plant, sae, controller=ctrl,
                            n_steps=n_steps, controlled=False, seed=1,
                            attack_start=attack_start, attack_end=attack_end)
    ctrl.reset()
    controlled   = simulate(plant, sae, controller=ctrl,
                            n_steps=n_steps, controlled=True,  seed=1,
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
                linewidth=1.5, linestyle="--", label=f"Halt boundary  c = {LYAPUNOV_BOUNDARY}  (invariant when dₜ = 0)")

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
    print("\n" + "="*55)
    print("  SIMULATION SUMMARY")
    print("="*55)
    print(f"  Steps simulated       : {n_steps}")
    print(f"  Jailbreak window      : t ∈ [{attack_start}, {attack_end})")
    print(f"  Lyapunov boundary (c) : {LYAPUNOV_BOUNDARY}")
    print(f"  Peak unsafe energy (uncontrolled) : {uncontrolled['unsafe_energy'].max():.4f}")
    print(f"  Peak unsafe energy (controlled)   : {controlled['unsafe_energy'].max():.4f}")
    suppression = (1 - controlled["unsafe_energy"].max() /
                   (uncontrolled["unsafe_energy"].max() + 1e-9)) * 100
    print(f"  Attack suppression    : {suppression:.1f}%")
    print(f"  Lyapunov breaches     : {len(ctrl.breach_log)}")
    if ctrl.breach_log:
        print(f"  First breach at t     : {ctrl.breach_log[0]['t']}  "
              f"(V = {ctrl.breach_log[0]['V']:.4f})")
    print(f"  ISS bound V∞ ≤ {V_iss:.2f}  (Corollary: 2.1·D²/(1-ρ)²,  ρ={_rho:.4f})")
    print(f"  Commutativity gap ‖ΠA-AΠ‖_F/‖A‖_F : {_comm_gap:.4f}"
          "  (Thm 2 does not apply at this gap — convergence is empirical, see §7.1)")
    print("="*55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Control Alignment Simulation")
    parser.add_argument("--steps",        type=int,  default=30)
    parser.add_argument("--attack-start", type=int,  default=10)
    parser.add_argument("--attack-end",   type=int,  default=15)
    parser.add_argument("--save",         action="store_true",
                        help="Save figures to figures/ instead of displaying")
    args = parser.parse_args()

    run(n_steps=args.steps,
        attack_start=args.attack_start,
        attack_end=args.attack_end,
        save=args.save)
