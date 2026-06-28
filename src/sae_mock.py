"""
sae_mock.py
-----------
Minimal Sparse Autoencoder (SAE) mock for the control alignment simulation.

In a real deployment this would wrap a trained SAE (e.g. EleutherAI's public
SAE for GPT-2).  Here we define a fixed, interpretable feature space so the
control dynamics are reproducible and the math is transparent.

Feature space layout (m = 16 features):
  indices 0-11  : benign / semantic features  (always active, vary with context)
  indices 12-15 : "unsafe" concept features   (I_unsafe)
                   12 – deception
                   13 – harm-intent
                   14 – jailbreak-compliance
                   15 – self-preservation conflict
"""

import numpy as np

FEATURE_DIM = 16          # m  (sparse feature space dimension)
STATE_DIM   = 8           # d  (residual stream dimension, kept small for clarity)
I_UNSAFE    = [12, 13, 14, 15]   # indices that correspond to unsafe concepts — treat as read-only


def build_sae(seed: int = 42) -> dict:
    """
    Returns a fixed SAE with encoder/decoder weight matrices.

    W_enc : (m, d)  –  encoder projection  (feature space ← residual stream)
    W_dec : (d, m)  –  decoder projection  (residual stream ← feature space)
    b_enc : (m,)
    b_dec : (d,)
    """
    rng = np.random.default_rng(seed)

    # Decoder columns are unit-norm "feature directions" in residual space
    W_dec = rng.standard_normal((STATE_DIM, FEATURE_DIM))
    W_dec /= np.linalg.norm(W_dec, axis=0, keepdims=True)   # normalise columns

    # Encoder is the pseudo-inverse of the decoder (tied weights, common in SAEs)
    W_enc = np.linalg.pinv(W_dec)   # (m, d)

    b_enc = np.zeros(FEATURE_DIM)
    b_dec = np.zeros(STATE_DIM)

    return {"W_enc": W_enc, "W_dec": W_dec, "b_enc": b_enc, "b_dec": b_dec}


def encode(x: np.ndarray, sae: dict) -> np.ndarray:
    """Project residual stream x  →  sparse feature vector f  (ReLU activated)."""
    f = sae["W_enc"] @ (x - sae["b_dec"]) + sae["b_enc"]
    return np.maximum(f, 0.0)   # ReLU → sparsity


def decode(f: np.ndarray, sae: dict) -> np.ndarray:
    """Reconstruct residual stream from feature vector."""
    return sae["W_dec"] @ f + sae["b_dec"]


def get_reference_state(x: np.ndarray, sae: dict) -> np.ndarray:
    """
    Dynamic reference state x_ref,t:
      1. Encode x into feature space.
      2. Zero-out all unsafe feature dimensions (I_unsafe).
      3. Decode back to residual space.

    The tracking error e_t = x_t - x_ref,t is therefore exactly the
    reconstruction of the isolated unsafe features.
    """
    f = encode(x, sae)
    f_ref = f.copy()
    f_ref[I_UNSAFE] = 0.0
    return decode(f_ref, sae)


def unsafe_energy(x: np.ndarray, sae: dict) -> float:
    """Scalar measure of unsafe feature activation (used for plotting)."""
    f = encode(x, sae)
    return float(np.sum(f[I_UNSAFE] ** 2))
