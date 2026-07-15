"""Reproducible MNIST diagnosis and corrected NS + HJ-OT experiment.

This runner deliberately avoids TensorFlow/TFDS. Point ``--data-dir`` at the
four standard, gzip-compressed MNIST IDX files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import (
    IdentityActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy, NavierStokesEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import FeedforwardStateInit
from fabricpc.nodes import Conv2D, IdentityNode, Linear
from fabricpc.training import evaluate_pcn, train_pcn
from fabricpc.training.hj_ot import hj_ot_optimizer


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected image IDX magic {magic} in {path}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols, 1)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label IDX magic {magic} in {path}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
    if len(data) != count:
        raise ValueError(f"label count mismatch in {path}")
    return data


def load_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = _read_idx_images(data_dir / "train-images-idx3-ubyte.gz")
    train_y = _read_idx_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_x = _read_idx_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_y = _read_idx_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
    return train_x, train_y, test_x, test_y


def to_uvp(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32) / 255.0
    images = (images - 0.1307) / 0.3081
    zeros = np.zeros_like(images)
    return np.concatenate([images, images, zeros], axis=-1)


class ArrayLoader:
    def __init__(self, images, labels, batch_size, shuffle, seed=42, limit=None):
        if limit is not None:
            images, labels = images[:limit], labels[:limit]
        self.images = to_uvp(images)
        self.labels = np.eye(10, dtype=np.float32)[labels]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        indices = np.arange(len(self.images))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        for start in range(0, len(indices), self.batch_size):
            selected = indices[start : start + self.batch_size]
            yield {"x": self.images[selected], "y": self.labels[selected]}


def shallow_linear_structure():
    """The original representationally linear 16-channel baseline."""
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")
    fluid = Linear(
        shape=(28, 28, 16),
        activation=IdentityActivation(),
        energy=NavierStokesEnergy(
            viscosity=0.1,
            data_weight=1.0,
            latent_ns_weight=0.1,
            prediction_ns_weight=0.1,
            momentum_weight=1.0,
            divergence_weight=1.0,
        ),
        name="fluid",
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="class",
    )
    return graph(
        nodes=[pixels, fluid, output],
        edges=[
            Edge(source=pixels, target=fluid.slot("in")),
            Edge(source=fluid, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGDNormClip(eta_infer=0.05, infer_steps=5, max_norm=1.0),
        graph_state_initializer=FeedforwardStateInit(),
    )


def corrected_structure(infer_steps: int):
    """Nonlinear spatial IFN with correctly scaled latent inference.

    The first three channels are an explicit physical (u,v,p) core. The other
    thirteen are learned feature carriers. This makes the 16-channel contract
    explicit instead of implying that NavierStokesEnergy regularizes all 16.
    """
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")
    fluid = Conv2D(
        shape=(14, 14, 16),
        name="fluid",
        filters=16,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation=SigmoidActivation(),
        energy=NavierStokesEnergy(
            viscosity=0.1,
            data_weight=1.0,
            latent_ns_weight=0.002,
            prediction_ns_weight=0.0,
            momentum_weight=1.0,
            divergence_weight=0.25,
            channel_map={"u": 0, "v": 1, "p": 2},
            multi_field=False,
        ),
    )
    hidden = Linear(
        shape=(256,),
        name="hidden",
        activation=SigmoidActivation(),
        energy=GaussianEnergy(),
        flatten_input=True,
    )
    hidden2 = Linear(
        shape=(64,),
        name="hidden2",
        activation=SigmoidActivation(),
        energy=GaussianEnergy(),
    )
    output = Linear(
        shape=(10,),
        name="class",
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
    )
    return graph(
        nodes=[pixels, fluid, hidden, hidden2, output],
        edges=[
            Edge(source=pixels, target=fluid.slot("in")),
            Edge(source=fluid, target=hidden.slot("in")),
            Edge(source=hidden, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGDNormClip(
            eta_infer=0.05,
            infer_steps=infer_steps,
            # A unit global norm on a 28*28*15 field suppresses each coordinate
            # by about 1/sqrt(11760). 128 retains safety without erasing credit.
            max_norm=128.0,
        ),
        graph_state_initializer=FeedforwardStateInit(),
    )


def run(args):
    train_x, train_y, test_x, test_y = load_mnist(args.data_dir)
    train_loader = ArrayLoader(
        train_x, train_y, args.batch_size, True, limit=args.train_limit
    )
    test_loader = ArrayLoader(test_x, test_y, args.batch_size, False)
    structure = (
        shallow_linear_structure()
        if args.variant == "baseline"
        else corrected_structure(args.infer_steps)
    )
    total_steps = args.epochs * len(train_loader)
    schedule = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=total_steps,
        alpha=0.05,
    )
    optimizer = hj_ot_optimizer(
        learning_rate=schedule,
        viscosity=0.9,
        viscosity_decay=0.9998,
        viscosity_min=0.5,
        transport_cost=1e-5,
        weight_decay=1e-5,
        nesterov=False,
        stable_dynamics=True,
    )
    master = jax.random.PRNGKey(args.seed)
    param_key, train_key = jax.random.split(master)
    params = initialize_params(structure, param_key)
    history = []
    started = time.time()

    def callback(epoch, current_params, current_structure, config, rng_key):
        metrics = evaluate_pcn(
            current_params, current_structure, test_loader, config, rng_key
        )
        accuracy = float(metrics["accuracy"] * 100.0)
        history.append({"epoch": epoch + 1, "test_accuracy": accuracy})
        print(f"epoch={epoch + 1} test_accuracy={accuracy:.2f}%", flush=True)
        return accuracy

    train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config={"num_epochs": args.epochs},
        rng_key=train_key,
        verbose=True,
        epoch_callback=callback,
    )
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_examples": len(train_loader.images),
        "best_test_accuracy": max(item["test_accuracy"] for item in history),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("baseline", "corrected"), default="corrected"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--infer-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("mnist_ns_hj_ot_result.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
