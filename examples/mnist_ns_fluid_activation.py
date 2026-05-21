"""
Predictive Coding Network — MNIST Navier-Stokes FluidActivation Comparison
==========================================================================

Compares training regimes on MNIST with Navier-Stokes architecture:
1. Baseline: GELU activation on NS layers + HJ-OT optimizer
2. FluidActivation: Physics-informed activation on NS layers + HJ-OT optimizer
3. Deep FluidActivation: 3 fluid layers with FluidActivation + tuned HJ-OT

Usage:
    python examples/mnist_ns_fluid_activation.py
"""

import os
import tempfile
import time
from typing import Iterator

import numpy as np
import matplotlib.pyplot as plt

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")
os.environ.setdefault(
    "TFDS_DATA_DIR", os.path.join(tempfile.gettempdir(), "fabricpc_tfds")
)

import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SoftmaxActivation,
    GeluActivation,
    FluidActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.training.hj_ot import hj_ot_optimizer
from fabricpc.graph.state_initializer import FeedforwardStateInit

jax.config.update("jax_default_prng_impl", "threefry2x32")


def mnist_to_uvp(images: np.ndarray) -> np.ndarray:
    """Map MNIST grayscale images to a simple `(u, v, p)` field."""
    zeros = np.zeros_like(images)
    return np.concatenate([images, images, zeros], axis=-1).astype(np.float32)


class MnistNavierStokesLoader:
    def __init__(self, split: str, batch_size: int, **loader_kwargs):
        self.loader = MnistLoader(
            split=split,
            batch_size=batch_size,
            tensor_format="NHWC",
            **loader_kwargs,
        )

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        for images, labels in self.loader:
            field = mnist_to_uvp(np.asarray(images))
            yield {"x": field, "y": np.asarray(labels)}

    def __len__(self) -> int:
        return len(self.loader)


def create_structure_baseline():
    """Baseline: single NS layer with GELU (from mnist_ns_linear_hj_ot.py)."""
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")

    fluid1 = Linear(
        shape=(28, 28, 32),
        activation=GeluActivation(),
        energy=NavierStokesEnergy(
            viscosity=0.5,
            data_weight=1.0,
            latent_ns_weight=0.05,
            prediction_ns_weight=0.05,
            momentum_weight=0.5,
            divergence_weight=0.5,
            channel_map={"u": 0, "v": 1, "p": 2},
        ),
        flatten_input=False,
        name="fluid1",
    )

    fluid2 = Linear(
        shape=(28, 28, 64),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
        flatten_input=False,
        name="fluid2",
    )

    fc1 = Linear(
        shape=(256,),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
        flatten_input=True,
        name="fc1",
    )

    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="class",
    )

    return graph(
        nodes=[pixels, fluid1, fluid2, fc1, output],
        edges=[
            Edge(source=pixels, target=fluid1.slot("in")),
            Edge(source=fluid1, target=fluid2.slot("in")),
            Edge(source=fluid2, target=fc1.slot("in")),
            Edge(source=fc1, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=100, max_norm=1.0),
        graph_state_initializer=FeedforwardStateInit(),
    )


def create_structure_fluid_activation():
    """FluidActivation on NS layers, GELU on dense layers."""
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")

    # NS layer with FluidActivation — physics-aware channel treatment
    fluid1 = Linear(
        shape=(28, 28, 32),
        activation=FluidActivation(
            velocity_gain=1.5,
            pressure_gain=0.8,
            sharpening=0.1,
            n_velocity_channels=2,
        ),
        energy=NavierStokesEnergy(
            viscosity=0.5,
            data_weight=1.0,
            latent_ns_weight=0.05,
            prediction_ns_weight=0.05,
            momentum_weight=0.5,
            divergence_weight=0.5,
            channel_map={"u": 0, "v": 1, "p": 2},
        ),
        flatten_input=False,
        name="fluid1",
    )

    fluid2 = Linear(
        shape=(28, 28, 64),
        activation=FluidActivation(
            velocity_gain=1.2,
            pressure_gain=0.6,
            sharpening=0.05,
            n_velocity_channels=2,
        ),
        energy=GaussianEnergy(precision=2.0),
        flatten_input=False,
        name="fluid2",
    )

    fc1 = Linear(
        shape=(256,),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
        flatten_input=True,
        name="fc1",
    )

    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="class",
    )

    return graph(
        nodes=[pixels, fluid1, fluid2, fc1, output],
        edges=[
            Edge(source=pixels, target=fluid1.slot("in")),
            Edge(source=fluid1, target=fluid2.slot("in")),
            Edge(source=fluid2, target=fc1.slot("in")),
            Edge(source=fc1, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=100, max_norm=1.0),
        graph_state_initializer=FeedforwardStateInit(),
    )


def create_structure_deep_fluid():
    """Deep architecture: 3 NS fluid layers + dense, all with FluidActivation."""
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")

    shared_ns_kwargs = dict(
        viscosity=0.5,
        data_weight=1.0,
        latent_ns_weight=0.05,
        prediction_ns_weight=0.05,
        momentum_weight=0.5,
        divergence_weight=0.5,
        channel_map={"u": 0, "v": 1, "p": 2},
    )

    fluid1 = Linear(
        shape=(28, 28, 16),
        activation=FluidActivation(
            velocity_gain=1.8, pressure_gain=0.9, sharpening=0.15
        ),
        energy=NavierStokesEnergy(**shared_ns_kwargs),
        flatten_input=False,
        name="fluid1",
    )

    fluid2 = Linear(
        shape=(28, 28, 32),
        activation=FluidActivation(
            velocity_gain=1.5, pressure_gain=0.8, sharpening=0.1
        ),
        energy=NavierStokesEnergy(**shared_ns_kwargs),
        flatten_input=False,
        name="fluid2",
    )

    fluid3 = Linear(
        shape=(28, 28, 64),
        activation=FluidActivation(
            velocity_gain=1.2, pressure_gain=0.6, sharpening=0.05
        ),
        energy=NavierStokesEnergy(**shared_ns_kwargs),
        flatten_input=False,
        name="fluid3",
    )

    fc1 = Linear(
        shape=(256,),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
        flatten_input=True,
        name="fc1",
    )

    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="class",
    )

    return graph(
        nodes=[pixels, fluid1, fluid2, fluid3, fc1, output],
        edges=[
            Edge(source=pixels, target=fluid1.slot("in")),
            Edge(source=fluid1, target=fluid2.slot("in")),
            Edge(source=fluid2, target=fluid3.slot("in")),
            Edge(source=fluid3, target=fc1.slot("in")),
            Edge(source=fc1, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=100, max_norm=1.0),
        graph_state_initializer=FeedforwardStateInit(),
    )


def train_and_eval(create_fn, optimizer, name, train_loader, test_loader, max_epochs):
    train_config = {"num_epochs": max_epochs}
    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    structure = create_fn()
    params = initialize_params(structure, graph_key)

    print(f"\n--- Training with {name} ---")
    accuracies = []

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        metrics = evaluate_pcn(params, structure, test_loader, config, rng_key)
        acc = metrics["accuracy"] * 100
        print(f"[{name}] Epoch {epoch_idx + 1} Test Accuracy: {acc:.2f}%")
        accuracies.append(acc)
        return acc

    start_time = time.time()
    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
        epoch_callback=epoch_callback,
    )
    elapsed = time.time() - start_time
    print(f"[{name}] Total training time: {elapsed:.2f}s")

    return accuracies


def main():
    num_epochs = 30
    batch_size = 128

    train_loader = MnistNavierStokesLoader(
        "train", batch_size=batch_size, shuffle=True, seed=42
    )
    test_loader = MnistNavierStokesLoader("test", batch_size=batch_size, shuffle=False)

    total_steps = num_epochs * len(train_loader)

    # --- 1. Baseline: GELU + HJ-OT ---
    lr_schedule_base = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=total_steps, alpha=0.1
    )
    opt_base = hj_ot_optimizer(
        learning_rate=lr_schedule_base,
        viscosity=0.7,
        transport_cost=1e-5,
        weight_decay=1e-4,
    )
    gelu_accs = train_and_eval(
        create_structure_baseline,
        opt_base,
        "Baseline (GELU + HJ-OT)",
        train_loader,
        test_loader,
        max_epochs=num_epochs,
    )

    # --- 2. FluidActivation + HJ-OT ---
    lr_schedule_fluid = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=total_steps, alpha=0.1
    )
    opt_fluid = hj_ot_optimizer(
        learning_rate=lr_schedule_fluid,
        viscosity=0.7,
        transport_cost=1e-5,
        weight_decay=1e-4,
    )
    fluid_accs = train_and_eval(
        create_structure_fluid_activation,
        opt_fluid,
        "FluidActivation + HJ-OT",
        train_loader,
        test_loader,
        max_epochs=num_epochs,
    )

    # --- 3. Deep FluidActivation + tuned HJ-OT ---
    lr_schedule_deep = optax.cosine_decay_schedule(
        init_value=3e-4, decay_steps=total_steps, alpha=0.05
    )
    opt_deep = hj_ot_optimizer(
        learning_rate=lr_schedule_deep,
        viscosity=0.85,
        viscosity_decay=0.999,
        viscosity_min=0.2,
        transport_cost=5e-6,
        weight_decay=5e-5,
        nesterov=True,
    )
    deep_accs = train_and_eval(
        create_structure_deep_fluid,
        opt_deep,
        "Deep FluidActivation + Tuned HJ-OT",
        train_loader,
        test_loader,
        max_epochs=num_epochs,
    )

    # --- Plotting ---
    plt.figure(figsize=(12, 8))

    def plot_arm(data, label, color, marker, linestyle="-", linewidth=2):
        if not data:
            return
        epochs = range(1, len(data) + 1)
        plt.plot(
            epochs,
            data,
            marker + linestyle,
            label=label,
            color=color,
            linewidth=linewidth,
        )

    plot_arm(gelu_accs, "Baseline (GELU + HJ-OT)", "gray", "o", linestyle=":")
    plot_arm(
        fluid_accs,
        "FluidActivation + HJ-OT",
        "blue",
        "^",
        linestyle="--",
        linewidth=2.5,
    )
    plot_arm(deep_accs, "Deep FluidActivation + Tuned HJ-OT", "cyan", "P", linewidth=3)

    plt.axhline(y=90.0, color="k", linestyle=":", alpha=0.5, label="90% Baseline")
    plt.axhline(y=95.0, color="g", linestyle="--", alpha=0.3, label="95% Target")
    plt.axhline(y=99.0, color="r", linestyle=":", alpha=0.3, label="99% Stretch Goal")

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title(
        "FluidActivation: Physics-Informed Activation for NS Energy + HJ-OT",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = os.path.join(
        os.getcwd(), "examples", "ns_fluid_activation_comparison.png"
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
