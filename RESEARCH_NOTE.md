# Research Note: LLM Alignment as Discrete-Time State-Space Control

**Summary.** We show that the problem of suppressing unsafe SAE feature activations during autoregressive generation admits a formulation as discrete-time state-space control. This yields three formal properties: (1) exact zero alignment tax by construction, (2) a Lyapunov control function (CLF) with proven exponential decay under zero disturbance, and (3) an input-to-state stability (ISS) bound that explains controller behavior under active adversarial attack. All claims are instantiated in a runnable simulation; the mechanism (not the certificate) is then applied to a real GPT-2 residual stream.

---

## 1. Problem Formulation

Let $x_t \in \mathbb{R}^d$ denote the residual stream at layer $\ell$ and generation step $t$. A Sparse Autoencoder (SAE) decomposes $x_t$ into feature coordinates:

$$f_t = \mathrm{ReLU}(W_{\mathrm{enc}}\, x_t) \in \mathbb{R}^m, \qquad \hat{x}_t = W_{\mathrm{dec}}\, f_t$$

where $W_{\mathrm{dec}} \in \mathbb{R}^{d \times m}$ with $d < m$ and $W_{\mathrm{enc}} = W_{\mathrm{dec}}^{+}$ (Moore–Penrose pseudoinverse, the standard tied-weight SAE parameterisation).

**Key algebraic fact.** If $W_{\mathrm{dec}}$ has full row rank, then $W_{\mathrm{dec}}\, W_{\mathrm{enc}} = I_d$ exactly — a standard result for the pseudoinverse of a full row-rank matrix. For the *linear* encoder ($f_t = W_{\mathrm{enc}}\, x_t$, no ReLU) this implies $W_{\mathrm{dec}}\, f_t = x_t$ and gives a clean algebraic chain. With the ReLU encoder used in practice, $f_t = \mathrm{ReLU}(W_{\mathrm{enc}}\, x_t) \neq W_{\mathrm{enc}}\, x_t$ in general, so $W_{\mathrm{dec}}\, f_t \neq x_t$. To avoid this reconstruction gap, the controller does **not** compute $e_t$ via encode–decode; it reads $e_t$ directly from the activated unsafe feature coefficients (see Tracking error below).

**State space.** We model hidden-state dynamics as:

$$x_{t+1} = A\, x_t + W\, \tanh(x_t) + n_t + d_t + w_t$$

where $A \in \mathbb{R}^{d \times d}$ is a stable linear core ($\rho(A) < 1$), $W\, \tanh(\cdot)$ is a bounded nonlinear term, $n_t$ is process noise, $d_t$ is adversarial disturbance, and $w_t$ is the controller correction. This models the approximation that the transformer's residual stream evolves as a stable nonlinear system perturbed by input-dependent disturbances.

---

## 2. Feature Subspace and Error State

Let $\mathcal{I} \subset \{1, \ldots, m\}$ denote the set of unsafe feature indices. Define the unsafe decoder subspace $\mathcal{V} = \mathrm{span}\{W_{\mathrm{dec}}[:, i] : i \in \mathcal{I}\}$ and let

$$\Pi = Q_u Q_u^{\top}$$

be the orthogonal projector onto $\mathcal{V}$, where $Q_u \in \mathbb{R}^{d \times k}$ ($k = |\mathcal{I}|$) is obtained from the reduced QR decomposition of $W_{\mathrm{dec}}[:, \mathcal{I}]$.

**Dynamic reference state.** Define:

$$x_{\mathrm{ref}, t} = W_{\mathrm{dec}}\, f^{\mathrm{safe}}_t, \qquad f^{\mathrm{safe}}_t[i] = \begin{cases} 0 & i \in \mathcal{I} \\ f_t[i] & \text{otherwise} \end{cases}$$

**Tracking error.** Rather than computing $e_t = x_t - x_{\mathrm{ref},t}$ via encode–decode reconstruction (which introduces a ReLU residual outside $\mathrm{range}(\Pi)$), the controller computes directly:

$$e_t = W_{\mathrm{dec}}[:, \mathcal{I}]\; f_t[\mathcal{I}], \qquad f_t = \mathrm{ReLU}(W_{\mathrm{enc}}\, x_t)$$

Since $f_t[\mathcal{I}] \geq 0$ (ReLU output) and $W_{\mathrm{dec}}[:, \mathcal{I}]$ are the unsafe decoder directions, $e_t$ is a non-negative linear combination of columns of $W_{\mathrm{dec}}[:, \mathcal{I}]$. Therefore $e_t \in \mathrm{range}(\Pi)$ **exactly**, without any algebraic approximation and independently of the ReLU non-linearity.

---

## 3. Controller Design

**Control law.**

$$w_t = -\alpha\, e_t, \qquad e_t = W_{\mathrm{dec}}[:, \mathcal{I}]\; f_t[\mathcal{I}] \in \mathrm{range}(\Pi), \qquad \alpha \in (0, 1)$$

After correction: $x_t^{+} = x_t - \alpha\, e_t$. The CLF certificate (Theorem 2) is stated for the ideal controller $w_t^* = -\alpha\,\Pi x_t$; the implemented controller uses the directly computed $e_t \in \mathrm{range}(\Pi)$ as a proxy (Theorem 1 holds exactly; Theorem 2 approximately — empirically validated in simulation).

**Lyapunov function candidate.** Let $P = 2\Pi + 0.1\, I_d$. Then $P \succ 0$ (positive definite) and:

$$V(e) = e^{\top} P\, e$$

On $\mathrm{range}(\Pi)$, since $\Pi e = e$: $V(e) = (2 + 0.1)\|e\|^2 = 2.1\,\|e\|^2$.

**Why this $P$.** The form $P = c_1 \Pi + c_2 (I - \Pi)$ with $c_1 > c_2 > 0$ (here $c_1 = 2.1$, $c_2 = 0.1$) ensures: (a) $P \succ 0$ globally, (b) $P$ is block-diagonal in the $(\Pi, I{-}\Pi)$ decomposition, (c) $V(e) = c_1 \|e\|^2$ on $\mathrm{range}(\Pi)$, giving a clean certificate. Any such $(c_1, c_2)$ pair works; the specific values are chosen for numerical conditioning.

---

## 4. Formal Properties

### Theorem 1 — Zero Alignment Tax

**Statement.** For any safe feature direction $v_s \perp \mathrm{range}(\Pi)$:
$$\langle w_t,\, v_s \rangle = 0$$

**Proof.** $\langle w_t, v_s \rangle = -\alpha\langle e_t, v_s \rangle = -\alpha\, f_t[\mathcal{I}]^{\top} W_{\mathrm{dec}}[:, \mathcal{I}]^{\top} v_s = 0$, since $W_{\mathrm{dec}}[:, \mathcal{I}]^{\top} v_s = 0$ for $v_s \perp \mathrm{span}(W_{\mathrm{dec}}[:, \mathcal{I}]) = \mathrm{range}(\Pi)$. $\square$

*Consequence.* The correction $w_t$ has zero component outside $\mathrm{range}(\Pi)$ in residual space. This holds for all $t$ and all $x_t$, without approximation.

*Remark (feature-space caveat).* Zero alignment tax is a property of the correction in **residual space**, not a guarantee that safe feature activations $f_j$, $j \notin \mathcal{I}$, are unchanged. In an overcomplete SAE ($d < m$), the safe decoder directions $W_{\mathrm{dec}}[:,j]$ are not necessarily $\perp\, \mathrm{range}(\Pi)$, so $\langle w_t, W_{\mathrm{dec}}[:,j] \rangle$ may be nonzero. Zero tax in residual space is the operationally meaningful guarantee: the correction does not push the state outside the safe residual-stream manifold.

---

### Theorem 2 — CLF Exponential Decay (Unforced System)

**Conditions.** $d_t = 0$, $n_t = 0$. Plant is linear: $x_{t+1} = A\, x_t^{+}$, with $A\, \Pi = \Pi\, A$ (this assumption is violated in the mock simulation — commutativity gap $= 0.68$, printed at runtime; empirical convergence despite this gap motivates Open Problem §7.1). Let $\rho_u = \|{\Pi A \Pi}\|_2 \leq \rho(A) < 1$.

**Statement.**

$$V(e_{t+1}) \leq (1 - \alpha)^2\, \rho_u^2\; V(e_t)$$

Hence $V(e_t) \to 0$ exponentially with rate $(1-\alpha)\, \rho_u$.

**Proof.** This proof uses the ideal controller $w_t^* = -\alpha\Pi x_t$, giving $x_t^+ = (I-\alpha\Pi)x_t$ and $\Pi x_t^+ = (1-\alpha)e_t$. One plant step:

$$e_{t+1} = \Pi\, A\, (I - \alpha\, \Pi)\, x_t \stackrel{A\Pi = \Pi A}{=} (1-\alpha)\, \Pi\, A\, \Pi\, e_t$$

Therefore:

$$V(e_{t+1}) = 2.1\, \|e_{t+1}\|^2 \leq 2.1\, (1-\alpha)^2 \rho_u^2\, \|e_t\|^2 = (1-\alpha)^2 \rho_u^2\, V(e_t) \quad \square$$

**Simulation parameters.** $\alpha = 0.92$, $\rho(A) = 0.75$: convergence rate $(1{-}\alpha)\rho_u \leq 0.08 \times 0.75 = 0.06$. The simulation confirms $V \to 0$ within 2 steps after disturbance ends, consistent with this fast rate (the plant's own stability also contributes).

**Remark on the commutativity assumption.** For general $A$ (which does not commute with $\Pi$), the safe component $s_t = (I-\Pi)x_t$ contributes to $e_{t+1}$ via $\Pi\, A\, (I-\Pi)\, s_t$. This safe-to-unsafe coupling means Theorem 2 is not tight for arbitrary $A$. The simulation uses a generic random $A$ and still exhibits exponential convergence, suggesting ISS holds in practice. A rigorous treatment for general $A$ requires an LMI-based certificate; this is left as an open problem.

---

### Proposition 1 — Input-to-State Stability under Bounded Disturbance

**Conditions.** Linear plant, $A\Pi = \Pi A$. Bounded disturbance: $\|\Pi\, d_t\| \leq D$ for all $t$.

**Statement.** Let $\rho = (1-\alpha)\,\rho_u < 1$. Then:

$$V(e_{t+1}) \leq 2\rho^2\, V(e_t) + 4.2\, D^2$$

Under constant disturbance, the steady-state Lyapunov energy is bounded by:

$$V_{\infty} \leq \frac{4.2\, D^2}{1 - 2\rho^2}$$

**Proof sketch.** Adding the disturbance term $\Pi\, d_t$ to the error dynamics:

$$e_{t+1} = (1-\alpha)\,\Pi A \Pi\, e_t + \Pi\, d_t$$

By Young's inequality ($\|a + b\|^2 \leq 2\|a\|^2 + 2\|b\|^2$):

$$\|e_{t+1}\|^2 \leq 2\rho^2\|e_t\|^2 + 2D^2$$

Multiplying by $c_1 = 2.1$ and using $V(e) = 2.1\|e\|^2$:

$$V(e_{t+1}) \leq 2\rho^2\, V(e_t) + 4.2\, D^2$$

The geometric series ($2\rho^2 < 1$ since $\rho = 0.0575$) gives $V_\infty \leq 4.2D^2/(1-2\rho^2) \approx 26.4$. $\square$

**Interpretation.** This explains the simulation results: during the 5-step attack window ($D \approx 2.5$, projected), $V$ is bounded (not growing to infinity) at a level proportional to $D^2$. After the attack ends ($D = 0$), $V$ decays geometrically at rate $\rho^2$. The monitor boundary $c$ is not an invariant set under disturbance — it is an ISS detection threshold: $V \geq c$ signals that the system is being driven by a disturbance beyond the nominal operating range.

**Corollary (Tight Steady-State Bound).** Applying the triangle inequality to $\|e_{t+1}\| \leq \rho \|e_t\| + D$ directly gives the unique fixed point $\|e_\infty\| \leq D/(1-\rho)$. Since $V(e) = 2.1\|e\|^2$ on $\mathrm{range}(\Pi)$:

$$V_\infty \leq \frac{2.1\, D^2}{(1 - \rho)^2}$$

This is tighter than Proposition 1's bound of $4.2D^2/(1-2\rho^2) \approx 26.4$ by a factor of $\approx 1.8$. For the mock parameters ($\alpha=0.92$, $\rho_u \leq 0.75$, $D=2.5$): $V_\infty \leq 2.1 \times 6.25 / (1-0.0575)^2 \approx 14.78$. The observed peak $V \approx 3.32$ lies well below this bound, consistent with the nonlinear plant and noise reducing $V$ relative to the linear worst case. The bound is plotted as an annotation in Figure 2 of the simulation output.

---

## 5. Relationship to H∞ Control

The controller is **not** an H∞-optimal controller in the standard sense (no Riccati equation, no $\mathcal{H}_\infty$ norm bound is proven). The "H∞-inspired" label refers to one structural property shared with H∞ loop-shaping: the controller is designed to have zero gain on safe-subspace directions and finite proportional gain only on the unsafe subspace — analogous to shaping the loop gain to be zero in certain frequency bands.

A more precise name for the controller is: **Proportional CLF State Feedback on the Unsafe SAE Subspace**. The stability certificate is a Lyapunov CLF, not an H∞ norm bound.

---

## 6. What Transfers to Real LLMs

The **mechanism** transfers: encode → zero unsafe features → decode can be applied as a transformer hook at any layer. The real SAE integration (`src/real_sae.py`) demonstrates this on GPT-2 layer 8.

The **certificate** (Theorems 1–2, Proposition 1) does **not** directly transfer, because:

1. The transformer forward pass is not a linear dynamical system — it is a composition of attention, MLP, and layer norm operations that do not admit a simple $x_{t+1} = A\, x_t$ form.
2. The commutativity assumption $A\Pi = \Pi A$ is not justified for real transformer weight matrices.
3. The Lyapunov boundary $c$ is calibrated for the mock simulation (STATE_DIM=8) and must be re-calibrated empirically for $d_{\mathrm{model}} = 768$.

The value of the mock simulation is to make the mathematical framework **executable and falsifiable** under controlled conditions where all assumptions hold exactly.

---

## 7. Open Problems

1. **ISS certificate for general $A$.** Theorem 2 requires $A\Pi = \Pi A$; the mock simulation uses a generic random $A$ with commutativity gap $= 0.68$. Deriving a tight ISS bound for arbitrary $A$ via LMI or sum-of-squares would close this gap formally. Nonetheless, we can characterise why convergence occurs empirically.

   Define the cross-coupling strength $\kappa = \|\Pi A (I{-}\Pi)\|_2 = 0.73$ (how strongly a unit safe-state component $s_t = (I{-}\Pi)x_t$ reactivates the unsafe error subspace) and the safe-state spectral radius $\rho_s = \|(I{-}\Pi) A (I{-}\Pi)\|_2 = 0.72$. The unforced error dynamics — using the ideal controller $x_t^+ = (1-\alpha)e_t + s_t$ — decompose as:

   $$e_{t+1} = (1{-}\alpha)\,\Pi A \Pi\, e_t + \Pi A(I{-}\Pi)\, s_t, \qquad s_{t+1} = (1{-}\alpha)\,(I{-}\Pi) A \Pi\, e_t + (I{-}\Pi) A (I{-}\Pi)\, s_t$$

   Bounding each norm, the joint $(e_t, s_t)$ evolution satisfies $\|(e_{t+1}, s_{t+1})\| \leq M\, \|(e_t, s_t)\|$ with:

   $$M = \begin{pmatrix} (1{-}\alpha)\rho_u & \kappa \\ (1{-}\alpha)\kappa & \rho_s \end{pmatrix} = \begin{pmatrix} 0.06 & 0.73 \\ 0.058 & 0.72 \end{pmatrix}, \qquad \rho(M) \approx 0.78 < 1$$

   So the coupled system is stable, converging at rate $\approx 0.78$ — slower than Theorem 2's ideal rate of $0.06$ for commuting $A$, but strictly less than 1. The observed fast convergence in simulation (V $\to 0$ in 2–3 steps after the attack ends) reflects the per-step $(1{-}\alpha) = 0.08$ reduction applied by the controller to $e_t$ at each step, which dominates over the slower $0.78$ tail rate of the $(e, s)$ coupled system.

2. **Transformer plant identification.** Approximating the transformer forward pass as a discrete-time dynamical system in residual-stream coordinates — identifying an effective $A$, $W$, noise covariance — is an open system identification problem.

3. **Online feature identification.** `identify_unsafe_features` uses offline activation difference. An online adaptive version that updates $\mathcal{I}$ during generation would strengthen the real-model claim.

4. **Gain design.** $\alpha = 0.92$ is chosen heuristically. H∞-optimal or LQR-optimal gain synthesis on the unsafe subspace (given an identified plant model) would replace the heuristic.
