"""
Deep Convolutional Incompressible Flow Network (CIFN) Scaling for MNIST.

Targets 99.0%+ accuracy by combining:
1. Conv2D layers with Navier-Stokes energy (Incompressible Flow).
2. Deeper architecture (3 Conv + 2 Linear).
3. Advanced AHJ-OT optimizer with Cosine Decay.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from fabricpc.nodes import Conv2D, Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SoftmaxActivation,
    IdentityActivation,
    GeluActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.training.hj_ot import hj_ot_optimizer


def mnist_to_uvp(images: np.ndarray) -> jnp.ndarray:
    """Map MNIST grayscale images to a (u, v, p) field."""
    u = images
    v = images
    p = np.zeros_like(images)
    return np.concatenate([u, v, p], axis=-1).astype(np.float32)


def create_cifn_structure():
    # Input image (28x28x3)
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")

    # Layer 1: Incompressible Conv2D
    fluid1_energy = NavierStokesEnergy(
        viscosity=0.5,  # Increased for stability
        latent_ns_weight=0.05,  # Lowered initially
        prediction_ns_weight=0.05,
        momentum_weight=1.0,
        divergence_weight=1.0,
        data_weight=1.0,
    )
    fluid1 = Conv2D(
        shape=(28, 28, 16),
        name="fluid1",
        filters=16,
        kernel_size=(3, 3),
        activation=GeluActivation(),
        energy=fluid1_energy,
    )

    # Layer 2: Pooling Conv2D (Stride 2)
    fluid2 = Conv2D(
        shape=(14, 14, 32),
        name="fluid2",
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
    )

    # Layer 3: Feature Conv2D (Stride 2)
    fluid3 = Conv2D(
        shape=(7, 7, 64),
        name="fluid3",
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
    )

    # Layer 4: Dense Layer
    fc1 = Linear(
        shape=(128,),
        name="fc1",
        activation=GeluActivation(),
        flatten_input=True,
    )

    # Output: Class Projection
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        flatten_input=True,
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
        inference=InferenceSGDNormClip(eta_infer=0.01, infer_steps=20, max_norm=1.0),
    )


class UVPWrapper:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for images, labels in self.loader:
            yield {"x": mnist_to_uvp(images), "y": labels}

    def __len__(self):
        return len(self.loader)


def main():
    max_epochs = 30
    batch_size = 128

    # Optimizer with Cosine Schedule
    steps_per_epoch = 60000 // batch_size
    lr_schedule = optax.cosine_decay_schedule(
        init_value=1e-3, decay_steps=max_epochs * steps_per_epoch, alpha=0.1
    )
    optimizer = hj_ot_optimizer(
        learning_rate=lr_schedule,
        viscosity=0.7,
        transport_cost=1e-5,
        weight_decay=1e-4,
    )

    structure = create_cifn_structure()
    params = initialize_params(structure, jax.random.PRNGKey(42))

    # Loaders with simple augmentations (simulated by noise for this script)
    train_loader = MnistLoader(
        "train", batch_size=batch_size, shuffle=True, tensor_format="NHWC"
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, shuffle=False, tensor_format="NHWC"
    )

    wrapped_train = UVPWrapper(train_loader)
    wrapped_test = UVPWrapper(test_loader)

    print(f"Starting Deep CIFN training (3 Conv + 2 Linear) on {jax.devices()}")

    def epoch_callback(epoch, params, structure, config, key):
        metrics = evaluate_pcn(params, structure, wrapped_test, config, key)
        acc = metrics["accuracy"] * 100
        print(f"Epoch {epoch+1} Test Accuracy: {acc:.2f}%")
        return acc

    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=wrapped_train,
        optimizer=optimizer,
        config={"num_epochs": max_epochs},
        rng_key=jax.random.PRNGKey(0),
        epoch_callback=epoch_callback,
        verbose=True,
    )


if __name__ == "__main__":
    main()
