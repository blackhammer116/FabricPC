"""
Comprehensive Comparison: Standard PC vs NS-PC vs HJ-OT NS-PC vs Backprop
======================================================================

Evaluates four training regimes on a shared Navier-Stokes architecture:
1. Standard PC (Gaussian Energy) + Adam
2. Navier-Stokes PC (NS Energy) + Adam
3. HJ-OT NS-PC (NS Energy) + HJ-OT Optimizer (with LR/Viscosity decay)
4. Standard Backpropagation + Adam

Usage:
    python examples/pc_ns_hjot_backprop_compare.py
"""

import os
import tempfile
import time
from typing import Iterator

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SoftmaxActivation, IdentityActivation
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.training.train_backprop import train_backprop, evaluate_backprop
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.training.hj_ot import hj_ot_optimizer

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


def create_structure(energy_type: str = "gaussian", fluid_channels: int = 3):
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")

    if energy_type == "navier_stokes":
        fluid_energy = NavierStokesEnergy(
            viscosity=0.1,
            data_weight=1.0,
            latent_ns_weight=0.1,
            prediction_ns_weight=0.1,
            momentum_weight=1.0,
            divergence_weight=1.0,
            channel_map={"u": 0, "v": 1, "p": 2},  # Always use first 3 channels for NS
        )
    else:
        fluid_energy = GaussianEnergy()

    fluid_layer = Linear(
        shape=(28, 28, fluid_channels),
        activation=IdentityActivation(),
        energy=fluid_energy,
        name="fluid",
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        flatten_input=True,
    )

    return graph(
        nodes=[pixels, fluid_layer, output],
        edges=[
            Edge(source=pixels, target=fluid_layer.slot("in")),
            Edge(source=fluid_layer, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )


def train_and_eval_pcn(
    optimizer,
    name,
    train_loader,
    test_loader,
    energy_type="gaussian",
    fluid_channels=3,
    max_epochs=10,
):
    train_config = {"num_epochs": max_epochs}
    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, _ = jax.random.split(master_rng_key, 3)

    structure = create_structure(energy_type, fluid_channels)
    params = initialize_params(structure, graph_key)

    print(f"\n--- Training {name} ---")
    accuracies = []

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        metrics = evaluate_pcn(params, structure, test_loader, config, rng_key)
        acc = metrics["accuracy"] * 100
        print(f"[{name}] Epoch {epoch_idx + 1} Test Accuracy: {acc:.2f}%")
        accuracies.append(acc)
        return acc

    train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
        epoch_callback=epoch_callback,
    )
    return accuracies


def train_and_eval_backprop(
    optimizer,
    name,
    train_loader,
    test_loader,
    energy_type="gaussian",
    fluid_channels=3,
    max_epochs=10,
):
    train_config = {"num_epochs": max_epochs}
    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, _ = jax.random.split(master_rng_key, 3)

    structure = create_structure(energy_type, fluid_channels)
    params = initialize_params(structure, graph_key)

    print(f"\n--- Training {name} ---")
    accuracies = []

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        metrics = evaluate_backprop(params, structure, test_loader, config, rng_key)
        acc = metrics["accuracy"] * 100
        print(f"[{name}] Epoch {epoch_idx + 1} Test Accuracy: {acc:.2f}%")
        accuracies.append(acc)
        return acc

    train_backprop(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
        epoch_callback=epoch_callback,
    )
    return accuracies


import itertools


def main():
    max_epochs = 5
    batch_size = 200
    loader = MnistNavierStokesLoader(
        "train", batch_size=batch_size, shuffle=True, seed=42
    )
    # Pre-load a subset into memory to speed up training and avoid re-downloading/shuffling issues
    train_loader = list(itertools.islice(loader, 50))  # 10,000 images
    test_loader = MnistNavierStokesLoader("test", batch_size=batch_size, shuffle=False)
    total_steps = max_epochs * len(train_loader)

    # 1. Standard PC (Gaussian) + Adam
    opt_adam = optax.adam(1e-3)
    standard_pc = train_and_eval_pcn(
        opt_adam,
        "Standard PC (Gaussian)",
        train_loader,
        test_loader,
        "gaussian",
        3,
        max_epochs,
    )

    # 2. Navier-Stokes PC + Adam
    ns_pc = train_and_eval_pcn(
        opt_adam, "NS-PC", train_loader, test_loader, "navier_stokes", 3, max_epochs
    )

    # 3. HJ-OT NS-PC (Baseline 3ch)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=1e-3, decay_steps=total_steps, alpha=0.1
    )
    opt_hj_ot = hj_ot_optimizer(
        learning_rate=lr_schedule,
        viscosity=0.9,
        viscosity_decay=0.9996,
        viscosity_min=0.3,
        transport_cost=5e-5,
    )
    hj_ot_pc = train_and_eval_pcn(
        opt_hj_ot,
        "HJ-OT NS-PC (3ch)",
        train_loader,
        test_loader,
        "navier_stokes",
        3,
        max_epochs,
    )

    # 4. Super-Fluid HJ-OT NS-PC (16ch + Nesterov)
    lr_schedule_sf = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=total_steps, alpha=0.1
    )
    opt_hj_ot_sf = hj_ot_optimizer(
        learning_rate=lr_schedule_sf,
        viscosity=0.9,
        viscosity_decay=0.9996,
        viscosity_min=0.3,
        transport_cost=5e-5,
        nesterov=True,
    )
    hj_ot_sf_pc = train_and_eval_pcn(
        opt_hj_ot_sf,
        "Medium-Fluid HJ-OT (16ch)",
        train_loader,
        test_loader,
        "navier_stokes",
        16,
        max_epochs,
    )

    # 5. Standard Backprop (3ch)
    backprop_accs = train_and_eval_backprop(
        opt_adam, "Backprop (3ch)", train_loader, test_loader, "gaussian", 3, max_epochs
    )

    # 6. Super-Fluid Backprop (16ch)
    opt_adam_sf = optax.adam(5e-4)
    backprop_sf_accs = train_and_eval_backprop(
        opt_adam_sf,
        "Medium-Fluid Backprop (16ch)",
        train_loader,
        test_loader,
        "gaussian",
        16,
        max_epochs,
    )

    # Visualization
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

    plot_arm(standard_pc, "Standard PC (Gaussian)", "gray", "o", linestyle=":")
    plot_arm(ns_pc, "NS-PC (Adam)", "orange", "s", linestyle=":")
    plot_arm(hj_ot_pc, "HJ-OT NS-PC (3ch)", "blue", "^", linestyle="--")
    plot_arm(hj_ot_sf_pc, "Medium-Fluid HJ-OT (16ch)", "cyan", "P", linewidth=3)
    plot_arm(backprop_accs, "Backprop (3ch)", "red", "x", linestyle="--")
    plot_arm(
        backprop_sf_accs, "Medium-Fluid Backprop (16ch)", "magenta", "D", linewidth=3
    )

    plt.axhline(y=90.0, color="k", linestyle=":", alpha=0.5)
    plt.axhline(y=93.0, color="g", linestyle="--", alpha=0.3, label="93% Benchmark")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Scaling Predictive Coding: Super-Fluid HJ-OT vs Backprop", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = "pc_super_fluid_comparison.png"
    plt.savefig(plot_path)
    print(f"\nSaved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
