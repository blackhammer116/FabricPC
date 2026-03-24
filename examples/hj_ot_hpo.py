"""
HJ-OT Hyperparameter Optimization (HPO) with Optuna
==================================================

Tunes optimal viscosity, viscosity_decay, transport_cost, and learning_rate.
Results are logged to `hj_ot_tuning_results.txt`.

CUDA acceleration is enabled by default if available.
"""

import os
import argparse
import itertools
from typing import Iterator

import numpy as np
import jax
import jax.numpy as jnp
import optax
import optuna

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SoftmaxActivation, IdentityActivation
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.training.hj_ot import hj_ot_optimizer

# JAX hardware initialization - removal of CPU forced flag allows CUDA/GPU
# jax.config.update("jax_platforms", "cpu") # commented out for CUDA


def mnist_to_uvp(images: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(images)
    return np.concatenate([images, images, zeros], axis=-1).astype(np.float32)


class MnistNavierStokesLoader:
    def __init__(self, split: str, batch_size: int, **loader_kwargs):
        self.loader = MnistLoader(
            split=split, batch_size=batch_size, tensor_format="NHWC", **loader_kwargs
        )

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        for images, labels in self.loader:
            field = mnist_to_uvp(np.asarray(images))
            yield {"x": field, "y": np.asarray(labels)}

    def __len__(self) -> int:
        return len(self.loader)


def create_structure(fluid_channels: int = 16):
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")
    fluid_energy = NavierStokesEnergy(
        viscosity=0.1,
        data_weight=1.0,
        latent_ns_weight=0.1,
        prediction_ns_weight=0.1,
        momentum_weight=1.0,
        divergence_weight=1.0,
        channel_map={"u": 0, "v": 1, "p": 2},
    )
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


def logging_callback(study, trial):
    with open("hj_ot_tuning_results.txt", "a") as f:
        f.write(
            f"Trial {trial.number}: Accuracy={trial.value:.4f}, Params={trial.params}\n"
        )


def objective(trial, train_data, test_loader, n_epochs=5):
    # Search Space
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    viscosity = trial.suggest_float("viscosity", 0.5, 0.99)
    viscosity_decay = trial.suggest_float("viscosity_decay", 0.7, 1.0)
    transport_cost = trial.suggest_float("transport_cost", 1e-6, 1e-3, log=True)

    # Optimizer
    steps_per_epoch = len(train_data)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_epochs * steps_per_epoch, alpha=0.1
    )
    optimizer = hj_ot_optimizer(
        learning_rate=lr_schedule,
        viscosity=viscosity,
        viscosity_decay=viscosity_decay,
        viscosity_min=0.1,
        transport_cost=transport_cost,
        nesterov=True,
    )

    # Model
    structure = create_structure(fluid_channels=16)
    graph_key = jax.random.PRNGKey(42)
    params = initialize_params(structure, graph_key)
    train_key = jax.random.PRNGKey(trial.number)  # Varies by trial

    # Training Loop (5 epochs on subset)
    train_config = {"num_epochs": n_epochs}

    params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_data,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=False,
    )

    # Evaluate
    metrics = evaluate_pcn(params, structure, test_loader, {}, jax.random.PRNGKey(0))
    return metrics["accuracy"] * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=5)
    args = parser.parse_args()

    print(f"Starting HPO on {jax.devices()}")

    # Prepare Data Subset (10,000 images for speed)
    batch_size = 200
    loader = MnistNavierStokesLoader(
        "train", batch_size=batch_size, shuffle=True, seed=42
    )
    train_data = list(itertools.islice(loader, 50))
    test_loader = MnistNavierStokesLoader("test", batch_size=batch_size, shuffle=False)

    # Initialize Log File
    with open("hj_ot_tuning_results.txt", "w") as f:
        f.write("HJ-OT Hyperparameter Tuning Results\n")
        f.write("===================================\n")

    # Run Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, train_data, test_loader, args.n_epochs),
        n_trials=args.n_trials,
        callbacks=[logging_callback],
    )

    print("\nBest Trial:")
    print(f"  Accuracy: {study.best_trial.value:.2f}%")
    print(f"  Params: {study.best_trial.params}")

    with open("hj_ot_tuning_results.txt", "a") as f:
        f.write("\nBEST TRIAL:\n")
        f.write(f"Accuracy: {study.best_trial.value:.2f}%\n")
        f.write(f"Params: {study.best_trial.params}\n")


if __name__ == "__main__":
    main()
