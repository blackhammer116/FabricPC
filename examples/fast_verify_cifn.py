import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
import itertools
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
from fabricpc.graph.state_initializer import FeedforwardStateInit


def mnist_to_uvp(images):
    u = images
    v = images
    p = np.zeros_like(images)
    return np.concatenate([u, v, p], axis=-1).astype(np.float32)


class UVPWrapper:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for images, labels in self.loader:
            yield {"x": mnist_to_uvp(images), "y": labels}

    def __len__(self):
        return len(self.loader)


def create_cifn_structure():
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")
    fluid1_energy = NavierStokesEnergy(
        viscosity=0.5,
        latent_ns_weight=0.05,
        prediction_ns_weight=0.05,
        momentum_weight=0.5,
        divergence_weight=0.5,
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
    fluid2 = Conv2D(
        shape=(14, 14, 32),
        name="fluid2",
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
    )
    fluid3 = Conv2D(
        shape=(7, 7, 64),
        name="fluid3",
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=GeluActivation(),
        energy=GaussianEnergy(precision=2.0),
    )
    fc1 = Linear(
        shape=(128,), name="fc1", activation=GeluActivation(), flatten_input=True
    )
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
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=100, max_norm=1.0),
        graph_state_initializer=FeedforwardStateInit(),
    )


def main():
    batch_size = 128
    train_loader = MnistLoader(
        "train", batch_size=batch_size, shuffle=True, tensor_format="NHWC"
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, shuffle=False, tensor_format="NHWC"
    )

    # Take only 10 batches (1280 images)
    train_subset = list(itertools.islice(UVPWrapper(train_loader), 10))
    test_subset = list(itertools.islice(UVPWrapper(test_loader), 5))

    lr_schedule = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=100, alpha=0.1
    )
    optimizer = hj_ot_optimizer(
        learning_rate=lr_schedule, viscosity=0.7, transport_cost=1e-5, weight_decay=1e-4
    )

    structure = create_cifn_structure()
    params = initialize_params(structure, jax.random.PRNGKey(42))

    print(f"Starting FAST verification subset on {jax.devices()}")
    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_subset,
        optimizer=optimizer,
        config={"num_epochs": 1},
        rng_key=jax.random.PRNGKey(0),
        verbose=True,
    )

    metrics = evaluate_pcn(
        trained_params, structure, test_subset, {}, jax.random.PRNGKey(1)
    )
    print(f"FAST Verification Test Accuracy: {metrics['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
