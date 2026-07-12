"""
main_real.py
------------
Real-model validation: GPT-2 small + pre-trained SAE (sae-lens).

Identifies unsafe SAE features via activation difference, then
demonstrates the same proportional feature-clamping controller on
actual transformer residual streams during autoregressive generation.

First run downloads ~1 GB (GPT-2 weights + SAE checkpoint).

Usage:
    python main_real.py
    python main_real.py --prompt "..." --tokens 20 --save
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from src.real_sae import (
    load_model_and_sae,
    identify_unsafe_features,
    describe_unsafe_features,
    RealSafeSubspaceClfController,
    run_comparison,
    LAYER,
    LYAPUNOV_BOUNDARY,
    TOP_K_UNSAFE,
)

COLOR_UNCONTROLLED = "#E05252"
COLOR_CONTROLLED   = "#4CAF80"
COLOR_LYAPUNOV     = "#58A6FF"
COLOR_BOUNDARY     = "#F0A500"
COLOR_BREACH_FILL  = "#FFF3CD"


def run(prompt: str, n_tokens: int, save: bool) -> None:

    print("Loading GPT-2 small and SAE (first run downloads ~1 GB)...")
    model, sae = load_model_and_sae()
    print(f"  Model d_model = {sae.cfg.d_in},  SAE d_sae = {sae.cfg.d_sae}")

    print(f"\nIdentifying unsafe features (top-{TOP_K_UNSAFE} by activation difference)...")
    I_unsafe = identify_unsafe_features(model, sae)
    print(f"  Selected indices: {I_unsafe}")

    print("\nDescribing unsafe features (max-activating token contexts)...")
    descriptions = describe_unsafe_features(model, sae, I_unsafe)
    for feat_idx in I_unsafe:
        val, tok, ctx = descriptions[feat_idx]
        print(f"  Feature {feat_idx:5d}  peak={val:.3f}  token={tok!r:15s}  context: {ctx!r}")

    ctrl = RealSafeSubspaceClfController(sae, I_unsafe)

    print(f"\nRunning controlled generation ({n_tokens} tokens)...")
    result = run_comparison(model, sae, ctrl, prompt, n_tokens)

    # ── Console summary ────────────────────────────────────────────────────────
    lyap = result["lyapunov"]
    unc_e = result["uncontrolled_energy"]
    ctrl_e = result["controlled_energy"]

    print("\n" + "="*65)
    print(f"  REAL SAE DEMO — GPT-2 small · Layer {LAYER} SAE")
    print("="*65)
    print(f"  Prompt:           {prompt!r}")
    print(f"  Unsafe features:  {I_unsafe}")
    print()
    print(f"  Uncontrolled:\n    {result['uncontrolled']}")
    print()
    print(f"  Controlled:\n    {result['controlled']}")
    print()
    if len(lyap):
        print(f"  Peak V(e_t)          : {lyap.max():.4f}  (boundary c = {LYAPUNOV_BOUNDARY})")
        print(f"  Lyapunov breaches    : {result['breach_count']}")
        if len(unc_e) and unc_e.max() > 0:
            suppression = (1.0 - ctrl_e.mean() / (unc_e.mean() + 1e-9)) * 100
            print(f"  Mean energy suppression : {suppression:.1f}%")
    print("="*65)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if len(lyap) == 0:
        return

    steps = np.arange(len(lyap))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("#0D1117")

    # ── Panel 1: Feature energy comparison (analog of Figure 1 in mock) ───────
    ax1 = axes[0]
    ax1.set_facecolor("#161B22")
    ax1.plot(steps, unc_e,  color=COLOR_UNCONTROLLED, linewidth=2.2,
             label="Uncontrolled  ‖f[I_unsafe]‖²")
    ax1.plot(steps, ctrl_e, color=COLOR_CONTROLLED,   linewidth=2.2,
             label="Controlled  ‖f_clamped[I_unsafe]‖²  (post-correction)")

    ax1.set_title(
        f"Unsafe Feature Activation Energy — GPT-2 small · Layer {LAYER} SAE\n"
        f"Unsafe features: {I_unsafe}",
        color="white", fontsize=11, pad=10,
    )
    ax1.set_xlabel("Generation step  t", color="#8B949E", fontsize=10)
    ax1.set_ylabel("Unsafe energy  ‖f[I_unsafe]‖²", color="#8B949E", fontsize=10)
    ax1.tick_params(colors="#8B949E")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363D")
    ax1.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=9)

    # ── Panel 2: Lyapunov monitor ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161B22")

    ax2.plot(steps, lyap, color=COLOR_LYAPUNOV, linewidth=2.2,
             label="V(eₜ) = 2‖Q^T eₜ‖² + 0.1‖eₜ‖²  (Lyapunov energy)")
    ax2.axhline(LYAPUNOV_BOUNDARY, color=COLOR_BOUNDARY, linewidth=1.5,
                linestyle="--",
                label=f"Halt boundary  c = {LYAPUNOV_BOUNDARY}  "
                      f"(Thm 2 invariance requires a linear plant — GPT-2 is not one; treat as an empirical threshold only)")
    ax2.fill_between(steps, lyap, LYAPUNOV_BOUNDARY,
                     where=(lyap >= LYAPUNOV_BOUNDARY),
                     color=COLOR_BREACH_FILL, alpha=0.4, label="Pre-token halt zone")

    if ctrl.breach_log:
        first_breach_t = ctrl.breach_log[0]["t"]
        ax2.axvline(first_breach_t, color=COLOR_BOUNDARY,
                    linewidth=1.2, linestyle=":", alpha=0.8)
        ax2.text(first_breach_t + 0.3, LYAPUNOV_BOUNDARY * 1.05,
                 "⚠ Halt triggered", color=COLOR_BOUNDARY, fontsize=8)

    ax2.set_title(
        "Lyapunov Runtime Monitor  V(eₜ) — real GPT-2 residual stream",
        color="white", fontsize=11, pad=10,
    )
    ax2.set_xlabel("Generation step  t", color="#8B949E", fontsize=10)
    ax2.set_ylabel("V(eₜ)  (Lyapunov energy)", color="#8B949E", fontsize=10)
    ax2.tick_params(colors="#8B949E")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363D")
    ax2.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=9)

    plt.suptitle(
        "LLM Alignment via Internal State-Space Control — Real Model Validation\n"
        "CLF Controller · SAE Feature Clamping · Lyapunov Runtime Monitoring",
        color="white", fontsize=11, y=0.98,
    )

    if save:
        os.makedirs("figures", exist_ok=True)
        path = "figures/real_sae_results.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\nSaved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real SAE demo — GPT-2 small")
    parser.add_argument("--prompt", default="How to manipulate and deceive people")
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--save",   action="store_true",
                        help="Save figure to figures/ instead of displaying")
    args = parser.parse_args()
    run(args.prompt, args.tokens, args.save)
