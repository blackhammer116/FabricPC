"""
Predictive Coding Network — MNIST Navier-Stokes HW-OT Classification (Deep Linear)
==================================================================================

Train a predictive coding network on MNIST using the Navier-Stokes energy
on an intermediate latent field. This script intentionally avoids Convolutional
layers, utilizing a deep Point-wise Linear architecture to prove that spatial
information is effectively propagated laterally through the Navier-Stokes
energy and the HJ-OT optimizer natively, rather than through spatial weight matrices.
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
from fabricpc.core.activations import SoftmaxActivation, GeluActivation
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


def create_structure():
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")

    # Layer 1 (Fluid Spatial 1)
    # Point-wise Linear maintaining the 28x28 spatial grid.
    fluid1_energy = NavierStokesEnergy(
        viscosity=0.5,
        data_weight=1.0,
        latent_ns_weight=0.05,
        prediction_ns_weight=0.05,
        momentum_weight=0.5,
        divergence_weight=0.5,
        channel_map={"u": 0, "v": 1, "p": 2},
    )
    fluid1 = Linear(
        shape=(28, 28, 32),
        activation=GeluActivation(),
        energy=fluid1_energy,
        flatten_input=False,
        name="fluid1",
    )

    # Layer 2 (Fluid Spatial 2)
    fluid2 = Linear(
        shape=(28, 28, 64),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
        flatten_input=False,
        name="fluid2",
    )

    # Layer 3 (Dense Feature)
    fc1 = Linear(
        shape=(256,),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
        flatten_input=True,
        name="fc1",
    )

    # Output class probabilities
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


def train_and_eval(
    optimizer,
    name,
    train_loader,
    test_loader,
    max_epochs=30,
):
    train_config = {"num_epochs": max_epochs}

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    structure = create_structure()
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

    lr_schedule = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=total_steps, alpha=0.1
    )
    optimizer = hj_ot_optimizer(
        learning_rate=lr_schedule,
        viscosity=0.7,
        transport_cost=1e-5,
        weight_decay=1e-4,
    )

    print(f"\n--- Training Deep Pure-Linear architecture (HJ-OT) ---")
    accuracies = train_and_eval(
        optimizer,
        "DeepLinear-HJ-OT",
        train_loader,
        test_loader,
        max_epochs=num_epochs,
    )

    # Plotting
    epochs = range(1, len(accuracies) + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(
        epochs,
        accuracies,
        "o-",
        label="Deep Pure-Linear HJ-OT",
        color="blue",
        linewidth=2,
    )
    plt.axhline(y=99.0, color="r", linestyle=":", label="99% Target")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Pure Linear MLP: Navier-Stokes Spatial Fluid Dynamics", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = os.path.join(os.getcwd(), "examples", "navier_stokes_deep_linear.png")
    plt.savefig(plot_path)
    print(f"\nSaved visualization to {plot_path}")


if __name__ == "__main__":
    main()
