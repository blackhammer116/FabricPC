"""
Fast verification of FluidActivation on MNIST.
"""

import os
import tempfile
import time
from typing import Iterator
import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")
os.environ.setdefault(
    "TFDS_DATA_DIR", os.path.join(tempfile.gettempdir(), "fabricpc_tfds")
)

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SoftmaxActivation, GeluActivation, FluidActivation
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.training.hj_ot import hj_ot_optimizer
from fabricpc.graph.state_initializer import FeedforwardStateInit


def mnist_to_uvp(images: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(images)
    return np.concatenate([images, images, zeros], axis=-1).astype(np.float32)


class MnistNavierStokesLoader:
    def __init__(self, split: str, batch_size: int, **loader_kwargs):
        self.loader = MnistLoader(
            split=split, batch_size=batch_size, tensor_format="NHWC", **loader_kwargs
        )

    def __iter__(self):
        for images, labels in self.loader:
            yield {"x": mnist_to_uvp(np.asarray(images)), "y": np.asarray(labels)}

    def __len__(self):
        return len(self.loader)


def create_structure(use_fluid_act=True):
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")
    fluid1 = Linear(
        shape=(28, 28, 8),
        activation=FluidActivation() if use_fluid_act else GeluActivation(),
        energy=NavierStokesEnergy(viscosity=0.5, channel_map={"u": 0, "v": 1, "p": 2}),
        flatten_input=False,
        name="fluid1",
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="class",
    )
    return graph(
        nodes=[pixels, fluid1, output],
        edges=[
            Edge(source=pixels, target=fluid1.slot("in")),
            Edge(source=fluid1, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=20, max_norm=1.0),
        graph_state_initializer=FeedforwardStateInit(),
    )


def main():
    train_loader = MnistNavierStokesLoader("train", batch_size=128, shuffle=True)
    test_loader = MnistNavierStokesLoader("test", batch_size=128, shuffle=False)

    # Run only 1 epoch each
    for name, use_fluid in [("GELU", False), ("Fluid", True)]:
        print(f"\n--- Testing {name} ---")
        structure = create_structure(use_fluid)
        params = initialize_params(structure, jax.random.PRNGKey(42))
        optimizer = hj_ot_optimizer(learning_rate=1e-3)
        trained_params, _, _ = train_pcn(
            params=params,
            structure=structure,
            train_loader=train_loader,
            optimizer=optimizer,
            config={"num_epochs": 1, "max_batches": 10},
            rng_key=jax.random.PRNGKey(42),
            verbose=True,
        )
        metrics = evaluate_pcn(
            trained_params,
            structure,
            test_loader,
            {"max_batches": 5},
            jax.random.PRNGKey(43),
        )
        print(f"{name} Result: {metrics['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
