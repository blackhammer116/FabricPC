import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple


class HJOTState(NamedTuple):
    """State for the HJ-OT optimizer, tracks parameter momentum (velocity field), second moments, and step count."""

    momentum: optax.Updates
    nu: optax.Updates
    count: jnp.ndarray


def scale_by_hj_ot(
    viscosity: float = 0.9,
    transport_cost: float = 1e-3,
    dt: float = 1.0,
    mass: float = 1.0,
    viscosity_decay: float = 1.0,
    viscosity_min: float = 0.1,
    beta2: float = 0.999,
    eps: float = 1e-8,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    """
    Scales updates by simulating Hamilton-Jacobi Optimal Transport with adaptive scaling.

    This interprets the weights as a fluid distribution flowing to minimize the
    free energy, subject to a Wasserstein transport cost and kinematic viscosity.
    The gradient field \nabla F acts as the force, and the parameter momentum
    acts as the velocity field governed by a viscous Burgers-style equation.

    This adaptive version also tracks second-order moments of the gradients to
    scale the effective transport cost and velocity, similar to RMSProp.

    Args:
        viscosity: Decay factor analogous to classical momentum.
        transport_cost: Coefficient for the non-linear convective transport penalty.
        dt: Integration time step for the transport.
        mass: Inertia coefficient. Higher mass makes parameters harder to accelerate.
        viscosity_decay: Exponential decay rate for viscosity per step.
        viscosity_min: The lowest allowed value for viscosity during decay.
        beta2: Exponential decay rate for the second-moment estimate.
        eps: Small constant for numerical stability during scaling.
        nesterov: Whether to use Nesterov-style momentum.
    """

    def init_fn(params):
        return HJOTState(
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params),
            nu=jax.tree_util.tree_map(jnp.zeros_like, params),
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(updates, state, params=None):
        # Calculate current dynamic viscosity
        current_visc = jnp.maximum(
            viscosity_min,
            viscosity * jnp.power(viscosity_decay, state.count.astype(jnp.float32)),
        )

        # Adaptive scaling: update second-moment estimates
        def _update_nu(g, n):
            return beta2 * n + (1 - beta2) * jnp.square(g)

        new_nu = jax.tree_util.tree_map(_update_nu, updates, state.nu)

        # Bias correction for second-moment estimate (like Adam)
        count_float = state.count.astype(jnp.float32) + 1
        bias_corr = 1 - jnp.power(beta2, count_float)

        def _update_momentum(g, m, n):
            # Normalizing gradient by root-mean-square for adaptive dynamics
            # This turns the transport into a "preconditioned" flow
            g_scaled = g / (jnp.sqrt(n / bias_corr) + eps)

            # 1. Viscous drag using dynamic viscosity
            drag = (1.0 - current_visc) * m

            # 2. Convective transport cost (non-linear dissipation).
            convective = transport_cost * m * jnp.abs(m)

            # 3. Semi-implicit integration:
            # Solve for new_m: new_m = m + (dt/M) * (g_scaled - DRAG(new_m) - CONV(new_m))
            denominator = 1.0 + (dt / mass) * (
                1.0 - current_visc + transport_cost * jnp.abs(m)
            )
            new_m = (m + (dt / mass) * g_scaled) / denominator

            if nesterov:
                # Nesterov lookahead: return the velocity evaluated at the next step
                return current_visc * new_m + (dt / mass) * g_scaled

            return new_m

        new_momentum = jax.tree_util.tree_map(
            _update_momentum, updates, state.momentum, new_nu
        )

        return new_momentum, HJOTState(
            momentum=new_momentum, nu=new_nu, count=state.count + 1
        )

    return optax.GradientTransformation(init_fn, update_fn)


def hj_ot_optimizer(
    learning_rate: float,
    viscosity: float = 0.9,
    transport_cost: float = 1e-4,
    dt: float = 1.0,
    mass: float = 1.0,
    viscosity_decay: float = 1.0,
    viscosity_min: float = 0.1,
    weight_decay: float = 1e-4,
    beta2: float = 0.999,
    eps: float = 1e-8,
    gradient_centering: bool = True,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    """
    Creates an optimizer based on Hamilton-Jacobi Optimal Transport.

    Args:
        learning_rate: Step size scaling the final transport velocity.
        viscosity: Parameter for momentum retention (0 to 1).
        transport_cost: Coefficient for the non-linear transport penalty.
        dt: Internal integration timestep.
        mass: Inertia coefficient (default: 1.0).
        viscosity_decay: Rate of viscosity decay (default: 1.0).
        viscosity_min: Minimum viscosity floor.
        weight_decay: L2 regularization strength.
        beta2: Decay rate for the second-moment adaptive scaling.
        eps: Epsilon for adaptive scaling stability.
        gradient_centering: Whether to center gradients (improves stability).
        nesterov: Whether to use Nesterov momentum.
    """
    chain = []
    if gradient_centering:
        chain.append(optax.centralize())

    chain.extend(
        [
            optax.clip_by_global_norm(1.0),
            scale_by_hj_ot(
                viscosity=viscosity,
                transport_cost=transport_cost,
                dt=dt,
                mass=mass,
                viscosity_decay=viscosity_decay,
                viscosity_min=viscosity_min,
                beta2=beta2,
                eps=eps,
                nesterov=nesterov,
            ),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        ]
    )

    return optax.chain(*chain)
