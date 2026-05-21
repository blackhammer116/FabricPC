"""
Test suite for FluidActivation — physics-informed activation for NS energy layers.

Tests shape preservation, channel-aware behavior, derivative correctness,
configuration, sharpening, and integration with NavierStokesEnergy in a graph.
"""

import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.activations import FluidActivation
from fabricpc.core.energy import NavierStokesEnergy, GaussianEnergy
from fabricpc.core.activations import IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params, initialize_graph_state
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.graph.graph_net import compute_local_weight_gradients


class TestFluidActivationForward:
    """Test FluidActivation forward pass."""

    def test_forward_shape_preservation(self):
        """Output shape must match input shape."""
        act = FluidActivation()
        x = jnp.ones((2, 4, 4, 8))
        y = FluidActivation.forward(x, act.config)
        assert y.shape == x.shape

    def test_forward_1d_shape(self):
        """Works with simple 2D (batch, channels) tensors too."""
        act = FluidActivation()
        x = jnp.ones((4, 5))
        y = FluidActivation.forward(x, act.config)
        assert y.shape == x.shape

    def test_velocity_channels_bounded(self):
        """Velocity channels (tanh) should be bounded in [-1, 1]."""
        act = FluidActivation()
        x = jnp.array([[100.0, -100.0, 0.0, 0.0, 0.0]])
        y = FluidActivation.forward(x, act.config)
        # First 2 channels are velocity (tanh)
        assert jnp.all(y[0, :2] >= -1.0)
        assert jnp.all(y[0, :2] <= 1.0)

    def test_pressure_channel_nonnegative(self):
        """Pressure channel (softplus) should be non-negative."""
        act = FluidActivation()
        x = jnp.array([[-5.0, -5.0, -10.0]])
        y = FluidActivation.forward(x, act.config)
        # Channel 2 is pressure (softplus)
        assert y[0, 2] >= 0.0

    def test_channel_split_behavior(self):
        """Verify different non-linearities are applied to different channels."""
        act = FluidActivation(velocity_gain=1.0, pressure_gain=1.0, sharpening=0.0)
        x = jnp.array([[2.0, 2.0, 2.0]])
        y = FluidActivation.forward(x, act.config)

        # Velocity uses tanh, pressure uses softplus — outputs should differ
        vel_out = y[0, 0]  # tanh(2.0)
        pres_out = y[0, 2]  # softplus(2.0)

        expected_vel = jnp.tanh(2.0)
        expected_pres = jnp.log(1.0 + jnp.exp(2.0))

        assert jnp.allclose(vel_out, expected_vel, atol=1e-5)
        assert jnp.allclose(pres_out, expected_pres, atol=1e-5)

    def test_extra_channels_get_velocity_treatment(self):
        """Channels beyond u, v, p get velocity-like (tanh) activation."""
        act = FluidActivation(velocity_gain=1.0, pressure_gain=1.0, sharpening=0.0)
        x = jnp.array([[2.0, 2.0, 2.0, 2.0, 2.0]])  # 5 channels
        y = FluidActivation.forward(x, act.config)

        expected_tanh = jnp.tanh(2.0)
        # Channels 0,1 (velocity) and 3,4 (extra) should all be tanh
        assert jnp.allclose(y[0, 0], expected_tanh, atol=1e-5)
        assert jnp.allclose(y[0, 1], expected_tanh, atol=1e-5)
        assert jnp.allclose(y[0, 3], expected_tanh, atol=1e-5)
        assert jnp.allclose(y[0, 4], expected_tanh, atol=1e-5)


class TestFluidActivationDerivative:
    """Test FluidActivation derivative computation."""

    def test_derivative_shape_preservation(self):
        """Derivative shape must match input shape."""
        act = FluidActivation()
        x = jnp.ones((2, 4, 4, 8))
        d = FluidActivation.derivative(x, act.config)
        assert d.shape == x.shape

    def test_derivative_matches_autodiff(self):
        """Analytical derivative should match JAX autodiff on element-wise part."""
        act = FluidActivation(sharpening=0.0)
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (1, 5))

        # Analytical
        analytical = FluidActivation.derivative(x, act.config)

        # Autodiff: compute per-element derivatives via vmap+grad
        def scalar_forward(xi):
            """Forward for a single element — need full channel context."""
            return FluidActivation.forward(xi[None, :], act.config)[0]

        # Use jax.jacobian for full derivative
        jac = jax.jacobian(scalar_forward)(x[0])
        # Extract diagonal (element-wise derivatives)
        autodiff_diag = jnp.diag(jac)

        assert jnp.allclose(analytical[0], autodiff_diag, atol=1e-4)

    def test_derivative_positive(self):
        """Derivative should be positive for all channel types at moderate inputs."""
        act = FluidActivation(sharpening=0.0)
        x = jnp.array([[0.5, -0.5, 1.0, 0.0]])
        d = FluidActivation.derivative(x, act.config)
        assert jnp.all(d > 0)


class TestFluidActivationConfig:
    """Test configuration handling."""

    def test_default_config_values(self):
        """Default config should contain expected keys."""
        act = FluidActivation()
        assert "velocity_gain" in act.config
        assert "pressure_gain" in act.config
        assert "sharpening" in act.config
        assert "n_velocity_channels" in act.config
        assert act.config["velocity_gain"] == 1.5
        assert act.config["pressure_gain"] == 0.8
        assert act.config["sharpening"] == 0.1
        assert act.config["n_velocity_channels"] == 2

    def test_custom_config(self):
        """Custom parameters should be stored and used."""
        act = FluidActivation(velocity_gain=2.0, pressure_gain=0.5, sharpening=0.0)
        x = jnp.array([[1.0, 1.0, 1.0]])
        y = FluidActivation.forward(x, act.config)

        expected_vel = jnp.tanh(1.0 * 2.0)
        expected_pres = jnp.log(1.0 + jnp.exp(1.0 * 0.5))

        assert jnp.allclose(y[0, 0], expected_vel, atol=1e-5)
        assert jnp.allclose(y[0, 2], expected_pres, atol=1e-5)


class TestFluidActivationSharpening:
    """Test spectral sharpening behavior."""

    def test_sharpening_modifies_output(self):
        """Non-zero sharpening should change output vs zero sharpening."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 4, 4, 3))

        act_no_sharp = FluidActivation(sharpening=0.0)
        act_sharp = FluidActivation(sharpening=0.3)

        y_no = FluidActivation.forward(x, act_no_sharp.config)
        y_yes = FluidActivation.forward(x, act_sharp.config)

        assert not jnp.allclose(y_no, y_yes, atol=1e-6)

    def test_sharpening_disabled_for_1d(self):
        """Sharpening should not apply for 2D tensors (batch, channels)."""
        act = FluidActivation(sharpening=0.5)
        x = jnp.array([[1.0, 1.0, 1.0]])

        # For 2D, sharpening is skipped (ndim < 3)
        act_no = FluidActivation(sharpening=0.0)
        y_with = FluidActivation.forward(x, act.config)
        y_without = FluidActivation.forward(x, act_no.config)

        assert jnp.allclose(y_with, y_without, atol=1e-6)


class TestFluidActivationIntegration:
    """Integration tests with NavierStokesEnergy in a graph."""

    def test_graph_with_fluid_activation_and_ns_energy(self):
        """FluidActivation + NavierStokesEnergy should produce valid energy and gradients."""
        input_node = IdentityNode(shape=(4, 4, 3), name="input")
        fluid_node = Linear(
            shape=(4, 4, 3),
            name="fluid",
            activation=FluidActivation(sharpening=0.0),
            energy=NavierStokesEnergy(viscosity=0.1),
        )

        structure = graph(
            nodes=[input_node, fluid_node],
            edges=[Edge(source=input_node, target=fluid_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=fluid_node),
            inference=InferenceSGD(eta_infer=0.01, infer_steps=2),
        )

        params = initialize_params(structure, jax.random.PRNGKey(0))
        clamps = {
            structure.task_map["x"]: jnp.ones((2, 4, 4, 3), dtype=jnp.float32),
            structure.task_map["y"]: jnp.ones((2, 4, 4, 3), dtype=jnp.float32) * 0.5,
        }

        state = initialize_graph_state(
            structure,
            batch_size=2,
            rng_key=jax.random.PRNGKey(1),
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)
        grads = compute_local_weight_gradients(params, final_state, structure)

        # Energy should be finite and have correct shape
        assert final_state.nodes["fluid"].energy.shape == (2,)
        assert jnp.all(jnp.isfinite(final_state.nodes["fluid"].energy))

        # Gradients should exist and have correct shapes
        assert grads.nodes["fluid"].weights
        for key, weight in params.nodes["fluid"].weights.items():
            assert grads.nodes["fluid"].weights[key].shape == weight.shape

    def test_multi_layer_graph_with_fluid_activation(self):
        """FluidActivation in a multi-layer network with mixed activations."""
        pixels = IdentityNode(shape=(4, 4, 3), name="pixels")
        fluid = Linear(
            shape=(4, 4, 3),
            name="fluid",
            activation=FluidActivation(),
            energy=NavierStokesEnergy(viscosity=0.1),
        )
        output = Linear(
            shape=(4,),
            name="out",
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            flatten_input=True,
        )

        structure = graph(
            nodes=[pixels, fluid, output],
            edges=[
                Edge(source=pixels, target=fluid.slot("in")),
                Edge(source=fluid, target=output.slot("in")),
            ],
            task_map=TaskMap(x=pixels, y=output),
            inference=InferenceSGD(eta_infer=0.01, infer_steps=2),
        )

        params = initialize_params(structure, jax.random.PRNGKey(0))

        # Verify the graph was constructed correctly with FluidActivation
        fluid_info = structure.nodes["fluid"].node_info
        assert isinstance(fluid_info.activation, FluidActivation)
        assert isinstance(fluid_info.energy, NavierStokesEnergy)
