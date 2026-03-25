"""
IFN Transformer Block implementation for JAX predictive coding networks.

This implements a transformer block where the latent state (Sequence, Dimension)
is constrained by Navier-Stokes fluid dynamics (Incompressible Flow).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING
import jax
import jax.numpy as jnp

from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.core.types import NodeState, NodeInfo, NodeParams
from fabricpc.core.energy import NavierStokesEnergy, GaussianEnergy
from fabricpc.core.activations import IdentityActivation

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional


class IFNTransformerBlock(TransformerBlock):
    """
    Incompressible Flow Network (IFN) Transformer Block.

    This block applies Navier-Stokes energy to the (Seq, Dim) latent field.
    The 'u' and 'v' velocities can be interpreted as the token-to-token
    and dimension-to-dimension information flux.
    """

    def __init__(
        self,
        shape: Tuple[int, int],  # (seq_len, embed_dim)
        name: str,
        viscosity: float = 0.1,
        ns_weight: float = 0.1,
        momentum_weight: float = 1.0,
        divergence_weight: float = 1.0,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy_type: str = "navier_stokes",  # 'navier_stokes' or 'hybrid'
        **kwargs,
    ):
        # We wrap a NavierStokesEnergy if requested
        if energy_type == "navier_stokes":
            energy = NavierStokesEnergy(
                viscosity=viscosity,
                latent_ns_weight=ns_weight,
                prediction_ns_weight=ns_weight,
                momentum_weight=momentum_weight,
                divergence_weight=divergence_weight,
                data_weight=1.0,  # Alignment with prediction
                channel_map={"u": 0, "v": 1, "p": 2},  # Custom mapping later?
            )
        else:
            energy = GaussianEnergy()

        super().__init__(
            shape=shape, name=name, activation=activation, energy=energy, **kwargs
        )
        self.energy_type = energy_type

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        """
        Forward pass with IFN fluid-dynamical constraints.

        For NavierStokesEnergy to work, the state (Batch, Seq, Dim) must be
        interpreted as a 4D tensor (Batch, H, W, C).
        Since we only have Seq x Dim, we can treat it as (Batch, Seq, Dim/3, 3)
        to have u,v,p channels, OR we can use the first 3 channels and Gaussian for the rest.
        """
        # 1. Standard Transformer Forward (gets z_mu)
        # We reuse the logic from TransformerBlock.forward by calling it.
        # However, we need to handle the 4D transformation for the energy call.

        batch_size, seq_len, embed_dim = state.z_latent.shape

        # Call standard forward logic
        # Note: TransformerBlock.forward is a static method
        total_energy, new_state = TransformerBlock.forward(
            params, inputs, state, node_info
        )

        # If using NavierStokes, we need to ensure the energy functional sees 4D tensors
        if isinstance(node_info.energy, NavierStokesEnergy):
            # Reshape (B, S, D) -> (B, S, D/3, 3)
            # This assumes D is a multiple of 3. If not, we could pad.
            # For simplicity, we'll try to use a 2D grid like (B, S, D, 1) and use
            # different channels for u,v,p.

            # Actually, the simplest way is to treat the whole (S, D) as a 2D field of 'p' (pressure)
            # and let 'u', 'v' be zero or derived.
            # But the user wants IFN (Incompressible Flow).
            # I will implement a wrapper that reshapes specifically for the energy computation.
            pass

        return total_energy, new_state

    # We override the energy_functional to handle the 4D transformation
    @staticmethod
    def energy_functional(state: NodeState, node_info: NodeInfo) -> NodeState:
        """Reshapes text sequences into 2D 'flow fields' for Navier-Stokes energy."""
        energy_obj = node_info.energy
        if not isinstance(energy_obj, NavierStokesEnergy):
            return super(IFNTransformerBlock, IFNTransformerBlock).energy_functional(
                state, node_info
            )

        # Reshape (B, S, D) -> (B, S, D//3, 3) if D >= 3
        B, S, D = state.z_latent.shape
        C = 3
        W = D // C
        if D % C != 0:
            # Pad if necessary
            padding = C - (D % C)
            z_latent_4d = jnp.pad(
                state.z_latent, ((0, 0), (0, 0), (0, padding))
            ).reshape(B, S, -1, C)
            z_mu_4d = jnp.pad(state.z_mu, ((0, 0), (0, 0), (0, padding))).reshape(
                B, S, -1, C
            )
        else:
            z_latent_4d = state.z_latent.reshape(B, S, W, C)
            z_mu_4d = state.z_mu.reshape(B, S, W, C)

        # Compute energy and its derivative w.r.t. the node's latent state.
        context = {"node_info": node_info}
        energy = type(energy_obj).energy(
            z_latent_4d, z_mu_4d, energy_obj.config, context=context
        )
        grad_4d = type(energy_obj).grad_latent(
            z_latent_4d, z_mu_4d, energy_obj.config, context=context
        )

        # Reshape back to (B, S, D)
        grad = grad_4d.reshape(B, S, -1)
        if grad.shape[2] > D:
            grad = grad[:, :, :D]  # Remove padding

        latent_grad = state.latent_grad + grad
        return state._replace(energy=energy, latent_grad=latent_grad)
