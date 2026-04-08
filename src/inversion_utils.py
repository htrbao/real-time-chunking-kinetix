import jax
import jax.numpy as jnp

def _invert_rfm(
    self,
    obs: jax.Array,
    x_1_target: jax.Array,
    num_steps: int,          # unused — single model call regardless
) -> jax.Array:
    """
    RFM single-step inversion (arxiv:2601.08136 Remark 5).

    Derivation
    ──────────
    Flow matching linear interpolant: x_t = (1-t)·x_0 + t·x_1
    Velocity target:                  v = x_1 - x_0
    Noise prediction at t=1:          x̂_0 = x_1 - v_θ(x_1, t=1)

    This is exact for a perfectly straight flow (OT-FM).
    For a real model it has error proportional to flow curvature
    (how non-straight the learned trajectories are), NOT to dt.

    Cost: 1 model call.
    Error: O(flow curvature) — not O(dt), so increasing N does not help.
    Best when: flow is well-trained with OT objective (straightest trajectories).
    """
    t_one    = jnp.ones((obs.shape[0],))
    v_at_1   = self(obs, x_1_target, t_one)
    x_0_star = x_1_target - v_at_1
    return x_0_star

def _invert_euler(
    self,
    obs: jax.Array,
    x_1_target: jax.Array,
    num_steps: int,
) -> jax.Array:
    """
    Naive backward Euler inversion.

    Algorithm per step (t goes from 1-dt down to 0):
        x_{t-dt} = x_t - dt · v_θ(x_t, t)

    Cost: N model calls.
    Error: O(dt) per step = O(1/N) total.
    At N=5, dt=0.2: error O(0.2) per step — significant.
    Increasing N improves quality but adds cost proportionally.

    This is the baseline — other methods should outperform it.
    """
    dt = 1.0 / num_steps

    def backward_step(carry, _):
        x_t, time = carry
        v_t = self(obs, x_t, time)
        return (x_t - dt * v_t, time - dt), None

    (x_0_star, _), _ = jax.lax.scan(
        backward_step, (x_1_target, 1.0 - dt), length=num_steps
    )
    return x_0_star

def _invert_dpm2(
    self,
    obs: jax.Array,
    x_1_target: jax.Array,
    num_steps: int,
) -> jax.Array:
    """
    DPM2 / Heun second-order backward inversion.

    Algorithm per step (backward, t: 1 → 0):
        k1   = v_θ(x_t, t)              — velocity at current point
        x_mid = x_t - (dt/2) · k1       — half-step predictor (backward)
        k2   = v_θ(x_mid, t - dt/2)     — velocity at midpoint
        x_{t-dt} = x_t - dt · k2        — corrected full step

    Why this works
    ──────────────
    Euler uses velocity at the START of each step (t).
    DPM2 uses velocity at the MIDPOINT (t - dt/2).
    The midpoint estimate cancels the first-order error term:

        Error (Euler): O(dt)    at N=5 → O(0.2) per step
        Error (DPM2):  O(dt²)   at N=5 → O(0.04) per step  ←  5× better

    This is equivalent to Runge-Kutta RK2 applied to the backward ODE.

    Cost: 2N model calls (2 per step).
    Error: O(dt²) per step — 4× more accurate than Euler at same N.
    Tradeoff: 2× cost of Euler, same quality as Euler with 2× more steps.

    Best when: N is small (5-10), where Euler error is large but
                doubling N would be too expensive.
    """
    dt = 1.0 / num_steps

    def backward_step_dpm2(carry, _):
        x_t, time = carry

        # predictor: half backward Euler step
        k1    = self(obs, x_t, time)
        x_mid = x_t - 0.5 * dt * k1

        # corrector: evaluate at midpoint time
        k2    = self(obs, x_mid, time - 0.5 * dt)

        # full step using midpoint velocity
        x_prev = x_t - dt * k2
        return (x_prev, time - dt), None

    (x_0_star, _), _ = jax.lax.scan(
        backward_step_dpm2, (x_1_target, 1.0 - dt), length=num_steps
    )
    return x_0_star

def _invert_optimization(
    self,
    obs: jax.Array,
    x_1_target: jax.Array,
    num_steps: int,
    num_iters: int = 5,
    lr: float = 0.1,
    init_method: str = "rfm",    # "rfm" | "euler" | "zeros" | "random"
    rng: jax.Array | None = None,
) -> jax.Array:
    """
    Optimization-based inversion — most accurate, most expensive.

    Solves:  x_0* = argmin_{x_0} ||forward_ODE(x_0, obs) - x_1_target||²

    This is exact up to optimization convergence — error is not tied to dt
    but to how many gradient steps we take and how well-conditioned the
    forward ODE is as a function of x_0.

    Algorithm
    ──────────
    1. Initialize x_0 (warm-start from RFM or Euler for faster convergence)
    2. For each iteration:
            x_1_hat = forward_ODE(x_0, obs)      [N model calls, differentiable]
            loss    = ||x_1_hat - x_1_target||²
            x_0    -= lr · ∇_{x_0} loss          [backprop through ODE]

    Warm-start initialization
    ──────────────────────────
    Starting from the RFM estimate (1 model call) dramatically reduces
    the number of gradient iterations needed — typically 3-5 iterations
    from RFM init achieves better quality than 20+ from random init.

    Cost: init_cost + num_iters × N model calls (forward) + backprop.
            At num_iters=5, N=5: 25 forward + backprop ≈ 50 effective calls.
    Error: converges to zero with enough iterations (given sufficient lr tuning).
    Warning: jax.grad through jax.lax.scan — memory cost O(N) for gradient tape.

    Best when: quality is critical and compute budget allows it.
                Use as an upper bound on what inversion can achieve.
    """
    dt = 1.0 / num_steps

    # ── differentiable forward ODE (needed for grad) ──────────────────────
    def forward_ode(x_0):
        def step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t + dt * v_t, time + dt), None
        (x_1, _), _ = jax.lax.scan(step, (x_0, 0.0), length=num_steps)
        return x_1

    # ── loss: squared distance to target ─────────────────────────────────
    def loss_fn(x_0):
        x_1_hat = forward_ode(x_0)
        return jnp.mean(jnp.sum((x_1_hat - x_1_target) ** 2, axis=(-2, -1)))

    # ── warm-start initialization ─────────────────────────────────────────
    if init_method == "rfm":
        # best init: RFM single-step (1 call, close to solution)
        x_0 = self._invert_rfm(obs, x_1_target, num_steps)
    elif init_method == "euler":
        # decent init: backward Euler (N calls)
        x_0 = self._invert_euler(obs, x_1_target, num_steps)
    elif init_method == "zeros":
        x_0 = jnp.zeros_like(x_1_target)
    else:
        # random: requires rng
        assert rng is not None, "rng required for random init"
        x_0 = jax.random.normal(rng, shape=x_1_target.shape)

    # ── gradient descent ──────────────────────────────────────────────────
    # use jax.grad for clean gradient computation through the ODE
    grad_fn = jax.grad(loss_fn)

    def opt_step(x_0, _):
        g    = grad_fn(x_0)
        x_0  = x_0 - lr * g
        return x_0, None

    x_0_star, _ = jax.lax.scan(opt_step, x_0, None, length=num_iters)
    return x_0_star
