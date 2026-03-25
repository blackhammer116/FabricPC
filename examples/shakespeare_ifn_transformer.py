"""
IFN Transformer for Shakespeare Character-Level Prediction.

Architecture:
- Embedding -> Positional Encoding -> Multi-layer IFN Transformer Blocks -> Vocab Projection
- Latent Constraint: Navier-Stokes energy on (Seq, Dim) grid.
"""

import os
import jax
import jax.numpy as jnp
import optax
import numpy as np

from fabricpc.nodes import (
    IFNTransformerBlock,
    Linear,
    IdentityNode,
    EmbeddingNode,
    VocabProjectionNode,
)
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
from fabricpc.utils.data.dataloader import CharDataLoader
from fabricpc.training.hj_ot import hj_ot_optimizer


def create_ifn_transformer(seq_len, vocab_size, embed_dim, num_layers=4):
    # 1. Embedding + Sequence Input
    tokens = IdentityNode(shape=(seq_len,), name="tokens")
    embedding = EmbeddingNode(
        shape=(seq_len, embed_dim),
        name="embedding",
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )

    nodes = [tokens, embedding]
    edges = [Edge(source=tokens, target=embedding.slot("in"))]

    prev_node = embedding
    for i in range(num_layers):
        block = IFNTransformerBlock(
            shape=(seq_len, embed_dim),
            name=f"ifn_block_{i}",
            viscosity=0.5,  # Increased for stability
            ns_weight=0.005,  # Lowered initially
            num_heads=8,
            ff_dim=4 * embed_dim,
            use_rope=True,
        )
        nodes.append(block)
        edges.append(Edge(source=prev_node, target=block.slot("in")))
        prev_node = block

    # Output projection
    output = VocabProjectionNode(
        shape=(seq_len, vocab_size),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        name="output",
    )
    nodes.append(output)
    edges.append(Edge(source=prev_node, target=output.slot("in")))

    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=tokens, y=output),
        inference=InferenceSGDNormClip(eta_infer=0.01, infer_steps=20, max_norm=1.0),
    )


def generate_text(params, structure, prompt, char_to_idx, idx_to_char, length=200):
    """Generate text from a prompt using the IFN Transformer."""
    seq_len = structure.nodes[0].shape[0]
    input_indices = [char_to_idx.get(c, 0) for c in prompt]

    generated = prompt
    key = jax.random.PRNGKey(42)

    from fabricpc.graph.state_initializer import initialize_graph_state
    from fabricpc.core.inference import run_inference

    curr_indices = input_indices[:]

    print(f"Generating from prompt: {prompt}")

    for i in range(length):
        # Prepare context window
        window = curr_indices[-seq_len:]
        if len(window) < seq_len:
            window = [0] * (seq_len - len(window)) + window

        x_batch = jnp.array([window])

        # Initialize state for this window
        state = initialize_graph_state(params, structure, {"x": x_batch})

        # Run inference (just forward pass since we don't have 'y' targets)
        # In PC, this minimizes the energy given 'x'
        _, final_state = run_inference(
            params=params,
            structure=structure,
            inputs={"x": x_batch},
            state=state,
            config=structure.inference.config,
            is_clamped=False,  # We don't clamp 'y' (the target) because we are generating it
        )

        # Get predictions from the output node (last node in the graph)
        # Output node 'z_mu' is (Batch, Seq, Vocab)
        output_node_name = structure.nodes[-1].name
        logits = final_state[output_node_name].z_mu[0, -1, :]  # Last token prediction

        # Greedy sampling (argmax) or probabilistic
        next_idx = int(jnp.argmax(logits))
        next_char = idx_to_char[next_idx]

        generated += next_char
        curr_indices.append(next_idx)

        if (i + 1) % 50 == 0:
            print(f"...generated {i+1} chars...")

    return generated


class LoaderWrapper:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for x, y in self.loader:
            yield {"x": x, "y": y}

    def __len__(self):
        return len(self.loader)


def main():
    seq_len = 64
    batch_size = 32
    embed_dim = 192
    num_layers = 1  # Small for initial scaling test
    max_epochs = 10

    loader = CharDataLoader(
        "train", seq_len=seq_len, batch_size=batch_size, shuffle=True
    )
    val_loader = CharDataLoader(
        "validation", seq_len=seq_len, batch_size=batch_size, shuffle=False
    )

    vocab_size = loader.vocab_size
    structure = create_ifn_transformer(
        seq_len, vocab_size, embed_dim, num_layers=num_layers
    )

    # Use AHJ-OT
    steps_per_epoch = len(loader)
    lr_schedule = optax.cosine_decay_schedule(5e-4, max_epochs * steps_per_epoch)
    optimizer = hj_ot_optimizer(learning_rate=lr_schedule, viscosity=0.8)

    params = initialize_params(structure, jax.random.PRNGKey(0))

    print(f"Training IFN Transformer (Shakespeare) on {jax.devices()}")

    wrapped_loader = LoaderWrapper(loader)
    wrapped_val = LoaderWrapper(val_loader)

    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=wrapped_loader,
        optimizer=optimizer,
        config={"num_epochs": max_epochs},
        rng_key=jax.random.PRNGKey(1),
        verbose=True,
    )

    # Generation
    print("\n--- Generating Text ---")
    char_to_idx = loader.char_to_idx
    idx_to_char = loader.idx_to_char

    generated = generate_text(
        trained_params, structure, "ROMEO: ", char_to_idx, idx_to_char, length=200
    )
    print(f"\nFINAL GENERATED TEXT:\n{generated}")


if __name__ == "__main__":
    main()
