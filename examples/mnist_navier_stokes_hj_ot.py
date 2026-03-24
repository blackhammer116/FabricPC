"""
Predictive Coding Network — MNIST Navier-Stokes HW-OT Classification
=======================================

Train a predictive coding network on MNIST using the Navier-Stokes energy
on an intermediate latent field and compare HJ-OT optimizer versus Adam.
"""

import os
import tempfile
import time
from typing import Iterator

import numpy as np
import matplotlib.pyplot as plt

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

# We use CPU to align with the smoke test unless GPU is readily available without OOM.
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
from fabricpc.core.activations import SoftmaxActivation, IdentityActivation
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import train_pcn, evaluate_pcn
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


def create_structure(use_navier_stokes: bool = True, fluid_channels: int = 16):
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")

    if use_navier_stokes:
        fluid_energy = NavierStokesEnergy(
            viscosity=0.1,
            data_weight=1.0,
            latent_ns_weight=0.1,
            prediction_ns_weight=0.1,
            momentum_weight=1.0,
            divergence_weight=1.0,
            channel_map={"u": 0, "v": 1, "p": 2},  # Use first 3 channels for NS
        )
    else:
        fluid_energy = GaussianEnergy()

    # Intermediate fluid representation layer
    fluid_layer = Linear(
        shape=(28, 28, fluid_channels),
        activation=IdentityActivation(),
        energy=fluid_energy,
        name="fluid",
    )
    # Output class probabilities
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


def train_and_eval(
    optimizer,
    name,
    train_loader,
    test_loader,
    use_navier_stokes=True,
    max_epochs=2,
    fluid_channels=3,
):
    train_config = {"num_epochs": max_epochs}

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    structure = create_structure(use_navier_stokes, fluid_channels=fluid_channels)
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
    use_navier_stokes = True
    num_epochs = 20
    batch_size = 200
    fluid_channels = 16

    train_loader = MnistNavierStokesLoader(
        "train", batch_size=batch_size, shuffle=True, seed=42
    )
    test_loader = MnistNavierStokesLoader("test", batch_size=batch_size, shuffle=False)

    total_steps = num_epochs * len(train_loader)

    lr_schedule = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=total_steps, alpha=0.1
    )
    optimizer = hj_ot_optimizer(
        learning_rate=lr_schedule,
        viscosity=0.9,
        viscosity_decay=0.7,
        viscosity_min=0.3,
        transport_cost=5e-5,
        nesterov=True,
    )

    structure = create_structure(
        use_navier_stokes=use_navier_stokes, fluid_channels=fluid_channels
    )

    print(f"\n--- Training HJ-OT (Medium-Fluid 16ch, Nesterov) ---")
    accuracies = train_and_eval(
        optimizer,
        "HJ-OT-Nesterov-16ch",
        train_loader,
        test_loader,
        use_navier_stokes=use_navier_stokes,
        max_epochs=num_epochs,
        fluid_channels=fluid_channels,
    )

    # Plotting
    epochs = range(1, len(accuracies) + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(
        epochs,
        accuracies,
        "o-",
        label="Medium-Fluid HJ-OT (16ch)",
        color="blue",
        linewidth=2,
    )
    plt.axhline(y=90.0, color="r", linestyle=":", label="90% Target")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Best Variant: Medium-Fluid HJ-OT on MNIST Navier-Stokes", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = os.path.join(os.getcwd(), "examples", "navier_stokes_best_variant.png")
    plt.savefig(plot_path)
    print(f"\nSaved best-variant visualization to {plot_path}")


if __name__ == "__main__":
    main()
