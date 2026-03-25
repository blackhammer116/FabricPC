"""
Conv2D node implementation for JAX predictive coding networks.

This implements a 2D convolutional node that supports:
- Filters, kernel_size, strides, padding
- PC updates optimized for convolutional feature maps
- Input shape: (Batch, H, W, C)
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import (
    NodeBase,
    SlotSpec,
    FlattenInputMixin,
)
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import KaimingInitializer, NormalInitializer

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class Conv2D(FlattenInputMixin, NodeBase):
    """
    2D Convolutional node: y = activation(conv(x, W) + b)
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],  # (H, W, C)
        name: str,
        filters: int,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
    ):
        """
        Args:
            shape: Output field shape (H, W, C)
            name: Node name
            filters: Number of output channels
            kernel_size: 2D kernel dimensions (H, W)
            strides: 2D stride (H, W)
            padding: Padding type ('SAME' or 'VALID')
            activation: Activation function
            energy: Energy functional
        """
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            weight_init=weight_init,
            latent_init=latent_init,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        from fabricpc.core.initializers import initialize

        filters = config.get("filters")
        kernel_size = config.get("kernel_size", (3, 3))

        # Split keys
        key_w, key_b = jax.random.split(key)

        weights_dict = {}
        rand_keys = jax.random.split(key_w, len(input_shapes))

        for (edge_key, in_shape), k in zip(input_shapes.items(), rand_keys):
            # Conv weight shape: (out_c, in_c, k_h, k_w) or (k_h, k_w, in_c, out_c)
            # lax.conv_general_dilated dimensions default: ('NHWC', 'HWIO', 'NHWC')
            in_channels = in_shape[-1]
            weight_shape = kernel_size + (in_channels, filters)  # HWIO
            weights_dict[edge_key] = initialize(k, weight_shape, weight_init)

        use_bias = config.get("use_bias", True)
        biases = {}
        if use_bias:
            biases["b"] = jnp.zeros((1, 1, 1, filters))

        return NodeParams(weights=weights_dict, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        config = node_info.node_config
        strides = config.get("strides", (1, 1))
        padding = config.get("padding", "SAME")
        activation = node_info.activation

        # Batch size
        batch_size = state.z_latent.shape[0]

        # Accumulate convolutions from all inputs
        # We assume 'NHWC' format
        pre_activation = jnp.zeros((batch_size,) + node_info.shape)

        for edge_key, x in inputs.items():
            W = params.weights[edge_key]
            # dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            out = jax.lax.conv_general_dilated(
                lhs=x,
                rhs=W,
                window_strides=strides,
                padding=padding,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            pre_activation = pre_activation + out

        if "b" in params.biases:
            pre_activation = pre_activation + params.biases["b"]

        z_mu = type(activation).forward(pre_activation, activation.config)
        error = state.z_latent - z_mu

        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        return jnp.sum(state.energy), state
