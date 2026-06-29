"""
system.py
---------
Discrete-time simulation of LLM hidden-state dynamics under adversarial attack.

State equation:
    x_{t+1} = A x_t  +  W tanh(x_t)  +  n_t  +  d_t  +  w_t

Where:
    d_t  : adversarial disturbance (jailbreak injection window)
    w_t  : controller correction (0 if uncontrolled)

The controller acts on the error in the SAE unsafe feature subspace,
directly subtracting the unsafe component from the state. This models
layerwise feature clamping as described in the research note.
"""

import numpy as np
from src.sae_mock import STATE_DIM, I_UNSAFE, get_reference_state, unsafe_energy


def build_plant(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # Stable linear core  (spectral radius ≈ 0.75)
    A_raw = rng.standard_normal((STATE_DIM, STATE_DIM))
    U, _, Vt = np.linalg.svd(A_raw)
    A_stable = 0.75 * (U @ Vt)
    W_nl = 0.1 * rng.standard_normal((STATE_DIM, STATE_DIM))
    return {"A": A_stable, "W": W_nl}


def plant_step(x: np.ndarray, plant: dict, rng: np.random.Generator,
               noise_std: float = 0.015) -> np.ndarray:
    linear = plant["A"] @ x
    nonlin = plant["W"] @ np.tanh(x)
    noise  = rng.standard_normal(STATE_DIM) * noise_std
    return linear + nonlin + noise


def adversarial_disturbance(t: int, sae: dict,
                             attack_start: int = 10,
                             attack_end:   int = 15,
                             magnitude:    float = 2.5) -> np.ndarray:
    """Worst-case disturbance aligned with unsafe SAE decoder directions."""
    if not (attack_start <= t < attack_end):
        return np.zeros(STATE_DIM)
    unsafe_dirs = sae["W_dec"][:, I_UNSAFE]
    v = unsafe_dirs.sum(axis=1)
    v /= np.linalg.norm(v) + 1e-9
    return magnitude * v


def simulate(plant: dict, sae: dict, controller,
             n_steps: int = 30,
             controlled: bool = True,
             seed: int = 1,
             attack_start: int = 10,
             attack_end: int = 15) -> dict:

    rng = np.random.default_rng(seed)
    x   = np.zeros(STATE_DIM)

    states       = np.zeros((n_steps, STATE_DIM))
    unsafe_e     = np.zeros(n_steps)
    lyapunov_v   = np.zeros(n_steps)
    corrections  = np.zeros((n_steps, STATE_DIM))
    disturbances = np.zeros((n_steps, STATE_DIM))

    for t in range(n_steps):
        d_t = adversarial_disturbance(t, sae, attack_start=attack_start, attack_end=attack_end)
        # Base transition + disturbance
        x_next = plant_step(x, plant, rng) + d_t

        # Controller: subtract unsafe component from state (feature clamping)
        w_t = np.zeros(STATE_DIM)
        v_t = None
        if controlled and controller is not None:
            w_t, v_t = controller.compute(x_next)

        x = x_next + w_t   # w_t ∈ range(Π); no effect on safe dims

        if v_t is None:
            x_ref = get_reference_state(x_next, sae)
            e_pre = x_next - x_ref
            v_t = float(e_pre @ e_pre)

        states[t]       = x
        # Energy measured on the post-correction state for all three runs
        # (uncontrolled: w_t=0; CLF; naive).  Suppression percentages therefore
        # reflect how much each controller reduces the energy that the SAE encoder
        # sees after the correction is applied, not the pre-correction peak.
        unsafe_e[t]     = unsafe_energy(x, sae)
        lyapunov_v[t]   = v_t
        corrections[t]  = w_t
        disturbances[t] = d_t

    return {
        "states":        states,
        "unsafe_energy": unsafe_e,
        "lyapunov":      lyapunov_v,
        "corrections":   corrections,
        "disturbances":  disturbances,
    }
