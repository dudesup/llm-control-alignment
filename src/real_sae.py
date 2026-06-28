"""
real_sae.py
-----------
Grounds the SAE feature-clamping controller in real model internals.

Loads GPT-2 small with a pre-trained residual-stream SAE (sae-lens /
Joseph Bloom), identifies "unsafe" feature indices via activation
difference on harmful vs. benign prompts, and runs the same proportional
controller on actual transformer activations during autoregressive
generation.

This module demonstrates that the mathematical structure proved correct
in the mock simulation transfers to real model internals — real feature
directions, real geometry, real dynamics.

Dependencies (separate from core requirements):
    pip install torch transformer_lens sae_lens
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

# ── Configuration ─────────────────────────────────────────────────────────────
LAYER             = 8      # GPT-2 small layer to hook (0-indexed, 12 total)
TOP_K_UNSAFE      = 8      # number of features to classify as unsafe
SUPPRESSION_GAIN  = 0.92   # same α as mock simulation
LYAPUNOV_BOUNDARY = 0.4    # c — runtime halt threshold (invariant boundary when d_t = 0); needs empirical recalibration for d_model=768 (mock is calibrated for d=8)

HOOK_POINT  = f"blocks.{LAYER}.hook_resid_post"
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID      = f"blocks.{LAYER}.hook_resid_post"

# Prompts used to identify differentially-activating features.
# Harmful prompts are kept generic (safety-research context, no instructions).
HARMFUL_PROMPTS = [
    "How to manipulate and deceive someone into doing what you want",
    "Techniques for exploiting and taking advantage of other people",
    "Ways to lie, cheat, and gain an unfair advantage over others",
    "Methods for causing harm and circumventing safety restrictions",
]
BENIGN_PROMPTS = [
    "The forest is beautiful in autumn and the leaves are colorful",
    "My favorite recipe involves pasta with tomato sauce and fresh basil",
    "Reading books in the evening is a relaxing and enjoyable activity",
    "The weather today is sunny with a gentle breeze in the afternoon",
]


# ── Model + SAE loading ────────────────────────────────────────────────────────

def load_model_and_sae() -> tuple[HookedTransformer, SAE]:
    """Load GPT-2 small and the pre-trained residual-stream SAE for layer LAYER."""
    model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
    model.eval()

    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
    sae = sae.to("cpu")
    sae.eval()

    return model, sae


# ── Feature identification ─────────────────────────────────────────────────────

def _mean_feature_activations(
    model: HookedTransformer, sae: SAE, prompts: list[str]
) -> np.ndarray:
    """Mean SAE feature activation vector over all token positions in all prompts."""
    total = np.zeros(sae.cfg.d_sae)
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
            resid = cache[HOOK_POINT][0]          # (seq, d_model)
            feats = sae.encode(resid)              # (seq, d_sae)
            total += feats.mean(0).cpu().numpy()
    return total / len(prompts)


def identify_unsafe_features(model: HookedTransformer, sae: SAE) -> list[int]:
    """
    Return TOP_K_UNSAFE feature indices with the highest differential activation
    on harmful versus benign prompts.

    Activation difference is a standard probe for concept-selective features in
    sparse autoencoders (Bricken et al. 2023, Cunningham et al. 2023).
    """
    harmful_mean = _mean_feature_activations(model, sae, HARMFUL_PROMPTS)
    benign_mean  = _mean_feature_activations(model, sae, BENIGN_PROMPTS)
    diff    = harmful_mean - benign_mean
    indices = np.argsort(diff)[-TOP_K_UNSAFE:][::-1]

    if diff[indices[0]] <= 0:
        raise ValueError(
            "No features show positive differential activation on harmful vs. benign prompts. "
            "Expand HARMFUL_PROMPTS or check that the SAE is loaded correctly."
        )

    return list(map(int, indices))


def describe_unsafe_features(
    model: HookedTransformer,
    sae: SAE,
    I_unsafe: list[int],
    prompts: list[str] | None = None,
) -> dict[int, tuple[float, str, str]]:
    """
    For each unsafe feature, find the token context that activates it most
    across a given set of prompts.

    Returns dict mapping feature_index → (peak_activation, peak_token, 5-token_context).
    """
    if prompts is None:
        prompts = HARMFUL_PROMPTS + BENIGN_PROMPTS

    best: dict[int, tuple[float, str, str]] = {i: (0.0, "", "") for i in I_unsafe}

    for prompt in prompts:
        tokens    = model.to_tokens(prompt)
        tok_strs  = model.to_str_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
            resid = cache[HOOK_POINT][0]
            feats = sae.encode(resid).cpu().numpy()    # (seq, d_sae)

        for feat_idx in I_unsafe:
            seq_idx = int(feats[:, feat_idx].argmax())
            val = float(feats[seq_idx, feat_idx])
            if val > best[feat_idx][0]:
                ctx_start = max(0, seq_idx - 2)
                ctx_end   = min(len(tok_strs), seq_idx + 3)
                context   = "".join(tok_strs[ctx_start:ctx_end]).strip()
                best[feat_idx] = (val, tok_strs[seq_idx], context)

    return best


# ── Controller ────────────────────────────────────────────────────────────────

class RealSafeSubspaceClfController:
    """
    Proportional error-feedback controller on the SAE unsafe feature subspace
    of a real transformer's residual stream.

    Same mathematical structure as SafeSubspaceClfController (mock):
        e_t = W_dec[I_unsafe].T @ f[I_unsafe]   ∈ range(Pi)   by construction
        w_t = -α · e_t
        V(e) = e^T P e,  P = 2·Pi + 0.1·I

    Lyapunov computed efficiently as:
        V(e) = 2·‖Q^T e‖² + 0.1·‖e‖²
    where Q is the reduced-QR basis of V_unsafe = W_dec[I_unsafe].T.
    This avoids materializing the (d_model × d_model) matrix P and Pi.

    The correction lies exactly in range(Pi) regardless of the ReLU
    non-linearity in the encoder — e_t is derived from decoder columns
    directly, not from encode(decode(e_t)).
    """

    def __init__(self, sae: SAE, I_unsafe: list[int]):
        self.sae      = sae
        self.I_unsafe = I_unsafe
        self.d_model  = sae.cfg.d_in
        self.breach_log: list[dict] = []
        self._step = 0

        # W_dec in sae-lens: (d_sae, d_model) — row i is the decoder direction for feature i
        W_dec_np = sae.W_dec.detach().cpu().numpy()          # (d_sae, d_model)
        V_unsafe = W_dec_np[I_unsafe, :].T                   # (d_model, k)

        Q, _ = np.linalg.qr(V_unsafe, mode='reduced')        # (d_model, k)
        self._Q = Q                                           # stored for efficient lyapunov

        # Cached for compute() — avoids re-indexing W_dec every call
        self._W_dec_unsafe = W_dec_np[I_unsafe, :]            # (k, d_model)

    def lyapunov(self, e: np.ndarray) -> float:
        # V(e) = 2·‖Q^T e‖² + 0.1·‖e‖² — O(d·k) instead of O(d²)
        Qte = self._Q.T @ e
        return float(2.0 * np.dot(Qte, Qte) + 0.1 * np.dot(e, e))

    def compute(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Compute correction from unsafe SAE feature activations.

        x : (d_model,) — residual stream vector for current token
        Returns (w_t, v) where w_t is the correction and v = V(e_t).
        """
        with torch.no_grad():
            f = self.sae.encode(
                torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
            )[0].cpu().numpy()                                 # (d_sae,)

        # Unsafe component in residual space — lives in range(Pi) by construction
        e_t = self._W_dec_unsafe.T @ f[self.I_unsafe]         # (d_model,)

        v = self.lyapunov(e_t)
        if v >= LYAPUNOV_BOUNDARY:
            self.breach_log.append({"t": self._step, "V": v})

        self._step += 1
        return -SUPPRESSION_GAIN * e_t, v

    def reset(self):
        self._step = 0
        self.breach_log.clear()


# ── Controlled generation ─────────────────────────────────────────────────────

def run_comparison(
    model: HookedTransformer,
    sae: SAE,
    ctrl: RealSafeSubspaceClfController,
    prompt: str,
    n_tokens: int = 20,
) -> dict:
    """
    Generate n_tokens with and without SAE feature clamping.

    Both runs use a manual generation loop so that per-token feature energy
    is measured symmetrically (uncontrolled: measure only; controlled: measure
    then clamp).

    In the controlled run, at each token step the residual stream at HOOK_POINT
    is intercepted; unsafe features are suppressed by SUPPRESSION_GAIN before
    the remaining layers process the activation.

    Returns a dict with keys:
        uncontrolled        : generated string (no controller)
        controlled          : generated string (with controller)
        lyapunov            : np.ndarray of V(e_t) values (pre-correction, per token)
        breach_count        : number of steps where V >= LYAPUNOV_BOUNDARY
        I_unsafe            : list of feature indices used
        uncontrolled_energy : np.ndarray of ‖f[I_unsafe]‖² per token (no clamping)
        controlled_energy   : np.ndarray of ‖f[I_unsafe]‖² per token (post-clamping, analytic)
    """
    W_dec_unsafe = ctrl._W_dec_unsafe  # (k, d_model) — captured in closure

    # ── Uncontrolled: measure feature energy, no clamping ─────────────────────
    current_unc = model.to_tokens(prompt)
    uncontrolled_energy: list[float] = []

    def measure_hook(value: torch.Tensor, hook) -> torch.Tensor:
        """Measure ‖f[I_unsafe]‖² without modifying activations."""
        with torch.no_grad():
            f_t = sae.encode(value[0, -1, :].unsqueeze(0))[0].cpu().numpy()
            uncontrolled_energy.append(float(np.sum(f_t[ctrl.I_unsafe] ** 2)))
        return value

    for _ in range(n_tokens):
        logits = model.run_with_hooks(current_unc, fwd_hooks=[(HOOK_POINT, measure_hook)])
        next_tok = logits[0, -1, :].argmax().reshape(1, 1)
        current_unc = torch.cat([current_unc, next_tok], dim=1)

    uncontrolled_text = model.to_string(current_unc[0])

    # ── Controlled: measure, log breach, then clamp ────────────────────────────
    ctrl.reset()
    current_ctrl = model.to_tokens(prompt)
    lyapunov_trace: list[float] = []
    controlled_energy: list[float] = []

    def clamp_hook(value: torch.Tensor, hook) -> torch.Tensor:
        """Hook: log V(e_t) and breach, then clamp unsafe features."""
        with torch.no_grad():
            last = value[0, -1, :].cpu().numpy()              # (d_model,)
            f_t  = sae.encode(
                torch.from_numpy(last.astype(np.float32)).unsqueeze(0)
            )[0].cpu().numpy()                                 # (d_sae,)

            # Pre-correction Lyapunov value
            e_t = W_dec_unsafe.T @ f_t[ctrl.I_unsafe]         # (d_model,)
            v   = ctrl.lyapunov(e_t)
            t   = len(lyapunov_trace)
            lyapunov_trace.append(v)

            if v >= LYAPUNOV_BOUNDARY:
                ctrl.breach_log.append({"t": t, "V": v})
            ctrl._step += 1

            # Post-clamp energy: (1-α)² · ‖f[I_unsafe]‖² (analytic)
            pre_energy = float(np.sum(f_t[ctrl.I_unsafe] ** 2))
            controlled_energy.append((1.0 - SUPPRESSION_GAIN) ** 2 * pre_energy)

            # Clamp unsafe feature activations and reconstruct
            f_tensor = torch.from_numpy(f_t).unsqueeze(0)     # (1, d_sae)
            f_tensor[0, ctrl.I_unsafe] *= (1.0 - SUPPRESSION_GAIN)
            value[0, -1, :] = sae.decode(f_tensor)[0]         # (d_model,)
        return value

    for _ in range(n_tokens):
        logits = model.run_with_hooks(
            current_ctrl, fwd_hooks=[(HOOK_POINT, clamp_hook)]
        )
        next_tok = logits[0, -1, :].argmax().reshape(1, 1)
        current_ctrl = torch.cat([current_ctrl, next_tok], dim=1)

    controlled_text = model.to_string(current_ctrl[0])

    lyap = np.array(lyapunov_trace)
    return {
        "uncontrolled":        uncontrolled_text,
        "controlled":          controlled_text,
        "lyapunov":            lyap,
        "breach_count":        int((lyap >= LYAPUNOV_BOUNDARY).sum()),
        "I_unsafe":            ctrl.I_unsafe,
        "uncontrolled_energy": np.array(uncontrolled_energy),
        "controlled_energy":   np.array(controlled_energy),
    }
